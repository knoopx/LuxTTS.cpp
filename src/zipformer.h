// zipformer.h — TTSZipformer (Zipformer2) ggml port.
//
// Ports zipvoice/models/modules/zipformer.py (TTSZipformer + Zipformer2Encoder
// + submodules) and zipvoice/models/modules/scaling.py (SwooshR/SwooshL,
// ScaledLinear — scales are baked into the stored weights, so inference is a
// plain matmul). Both ONNX graphs in the Python pipeline are TTSZipformer
// (zipvoice/config.json: text_encoder in_dim=192, out_dim=100, 1 stack x 4
// layers, kernel 9, dim 192, ff 512, 4 heads (q/k 32, pos 4, v 12), no time
// embed; fm_decoder in_dim=300 (x|text_cond|speech_cond), out_dim=100,
// 5 stacks, downsample [1,2,4,2,1], layers [2,2,4,4,4], kernels
// [31,15,7,15,31], dim 512, ff 1536, time+guidance embed).
//
// Eval-time simplifications (verified against the Python): Balancer, Whiten,
// dropout, sequence-dropout and all ScheduledFloat skip rates are no-ops at
// inference; BypassModule reduces to `src_orig + (src - src_orig) * bypass_scale`
// (a per-channel affine residual). ScaledLinear is a plain Linear (its
// initial_scale is applied to the weights at construction and is therefore
// already inside the saved weights).

#ifndef LUXTTS_ZIPFORMER_H
#define LUXTTS_ZIPFORMER_H

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>

#include "common.h"

// ggml v0.24.0 defines these at GLOBAL scope (see <ggml.h>); the gb
// graph-builder helpers in common.h already use them unqualified.
struct ggml_context;
struct ggml_tensor;
struct ggml_backend_buffer;

namespace lux {

struct ZipformerConfig {
  int in_dim, out_dim;
  std::vector<int> downsample_factor;  // [1,2,4,2,1] decoder / [1] encoder (zipvoice/config.json)
  std::vector<int> num_layers;         // per stack
  std::vector<int> cnn_kernel;         // per stack
  int encoder_dim = 512;
  int feedforward_dim = 1536;
  int num_heads = 4;
  int query_head_dim = 32;
  int pos_head_dim = 4;
  int value_head_dim = 12;
  int pos_dim = 48;
  bool use_time_embed = false;
  int time_embed_dim = 192;
  bool use_guidance_embed = false;
};

// One loaded TTSZipformer instance (text_encoder or fm_decoder) with its
// weights resident in a ggml context.
class Zipformer {
 public:
  Zipformer() = default;
  explicit Zipformer(const ZipformerConfig& cfg);

  // zipformer.py L116 (TTSZipformer.__init__: record the weight tensors of this instance
  // (names relative to the instance prefix, e.g. "encoders.0.layers.0.norm.log_scale")
  // in the bound-tensor map (the tensors themselves stay in the caller's ggml
  // context — no copies; q8_0 weights run on the native q8_0 kernels); `named`
  // is the full name -> tensor map (names WITHOUT the instance prefix); hard
  // error on missing tensors and on a q8_0 innermost dim not a multiple of 32)
  std::string load(ggml_context* ctx,
                   const std::vector<std::pair<std::string, ggml_tensor*>>& named);

  // zipformer.py L249 (TTSZipformer.forward: forward(x, t, guidance_scale, padding_mask)
  // -> (T, out_dim) for batch 1; x is row-major (T, in_dim); t / guidance_scale are used
  // only when the instance has time / guidance embeds (fm_decoder), pass t = -1 for the
  // text encoder; pad_mask (T,): true = padding position (masked source); all per-call
  // graphs run on a persistent internal compute context)
  std::vector<float> forward(const float* x, int T,
               float t, float guidance_scale,
               const bool* pad_mask) const;

  const ZipformerConfig& cfg() const { return cfg_; }

  // ggml v0.24.0 (n_threads for the CPU backend; applied via the CPU reg's
  // ggml_backend_set_n_threads proc when the backend exists; 0 = ggml default (all cores))
  void set_n_threads(int n);

  // ggml v0.24.0 (wraps the GGUF tensor-data blob in a CPU backend buffer and
  // binds every tensor in w_ to it (the tensor's public buffer field; the CPU
  // buffer's init_tensor is a no-op); must be called once after load() bound
  // the weights, before any forward())
  void set_weight_blob(const void* blob, size_t bytes);

  // ggml v0.24.0 (one-time init: creates backend_/sched_/sched_meta_/cctx_ if
  // null (n_max_nodes = 16384, use_meta = true; cctx via
  // ggml_init({sched_meta_.size(), sched_meta_.data(), true})) and binds the
  // weight tensors to their CPU buffers; called from forward() and
  // set_weight_blob())
  void ensure_backend() const;

  ~Zipformer();
 private:
  ZipformerConfig cfg_;
  // name -> bound weight tensor (a view into the caller's context, stored
  // in a tiny local ggml context for bookkeeping)
  void* map_ctx_ = nullptr;
  std::map<std::string, ggml_tensor*> w_;
  int n_threads_ = 0;
  // ggml v0.24.0 (persistent CPU-backend compute for every graph in forward():
  // a no_alloc context whose mem_buffer is the scheduler's meta buffer (the
  // tensor OBJECTS live there; tensor DATA is auto-sized by the scheduler per
  // graph), so no static byte estimate — the graph's live T^2 attention
  // footprint can never exceed a pre-sized pool; forward() is const, hence
  // mutable)
  mutable void* backend_ = nullptr;           // ggml_backend_t (ggml_backend_cpu_init)
  mutable void* sched_ = nullptr;             // ggml_backend_sched_t
  mutable std::vector<uint8_t> sched_meta_;   // scheduler meta buffer
  mutable void* cctx_ = nullptr;              // ggml_context* (no_alloc, mem_buffer = sched_meta_)
  mutable struct ggml_backend_buffer* wbuf_ = nullptr;      // CPU buffer wrapping the GGUF weight blob (f32 + q8_0 weights)
  // ggml v0.24.0 (the GGUF tensor-data blob that set_weight_blob() wraps in wbuf_)
  const void* weight_blob_ = nullptr;
  size_t weight_blob_bytes_ = 0;
};

// Standalone attention blocks on a caller-owned ggml context (used by
// forward()). The caller ggml_reset's the context BEFORE calling; each
// function builds + computes one graph and downloads the result into the
// out buffer. Returns false on compute failure.
//
// attn_block_ggml (zipformer.py L1156): relative-position qk scores + mask +
// softmax, ALL in natural time order. The rel-pos window is built in-graph with the
// rel-shift chain (pad_ext -> reshape -> view -> cont, x2 over the (2S-1, S, H)
// pos-scores tensor), so scores[j,i,h] = q_i.k_j + pos_proj[j-i+S-1].p_i[h]
// (j = source, i = target, zipformer.py's index).
//   qkv:  (in_proj_dim, S) ch-major NATURAL order, layout
//         [q (H*dq) | k (H*dq) | p (H*dp)]
//   pep:  (2S-1, H, dp) projected compact-pe table, flat (r*H+h)*dp+d
//   mask: (S,) addend (-1000 on padded sources)
//   attn: (H, S, S) NATURAL [h][i][j], i = target, j = source
bool attn_block_ggml(ggml_context* ctx, int n_threads, const float* qkv,
                     const float* pep, const float* mask,
                     int S, int H, int dq, int dp, float* attn);

// ggml v0.24.0 (apply a (H, S, S) natural attention matrix to a (H*dv, S) ch-major
// head-major (c = h*dv+d) value tensor -> (H*dv, S); the attn leaf and the (S, dv, H)
// value leaf are byte-identical memcpy of the caller's buffers, and the batched
// mul_mat result is a byte-identical memcpy back to (H*dv, S))
bool attn_apply_ggml(ggml_context* ctx, int n_threads, const float* attn,
                     const float* v, int S, int H, int dv, float* out);

// zipformer.py L990 (CompactRelPositionalEncoding.extend_pe, parameter-free):
// builds the (2T-1, pos_dim) pe table the Python builds on the fly
std::vector<float> compact_rel_pe(int T, int pos_dim);

// zipformer.py L54 (timestep_embedding, sinusoidal, dim=time_embed_dim)
std::vector<float> timestep_embed(float t, int dim);

}

#endif // LUXTTS_ZIPFORMER_H
