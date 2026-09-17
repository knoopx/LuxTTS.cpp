// zipformer.cpp — TTSZipformer ggml graph construction + eval-time math.
//
// Both ONNX graphs in the Python pipeline (text_encoder, fm_decoder) are
// TTSZipformer (Zipformer2) encoders, so one ggml implementation serves both.
//
//   zipvoice/models/modules/zipformer.py
//     L79  (TTSZipformer), L249 (forward)
//     L307 (Zipformer2EncoderLayer), L496 (layer forward)
//     L652 (Zipformer2Encoder), L709 (forward)
//     L754 (BypassModule), L805 (forward)
//     L814 (DownsampledZipformer2Encoder), L830 (forward)
//     L880 (SimpleDownsample), L894 (forward)
//     L923 (SimpleUpsample), L932 (forward)
//     L945 (CompactRelPositionalEncoding), L1041 (forward)
//     L1066 (RelPositionMultiheadAttentionWeights), L1156 (forward)
//     L1332 (SelfAttention), L1366 (forward)
//     L1406 (FeedforwardModule), L1440 (forward)
//     L1449 (NonlinAttention), L1506 (forward)
//     L1554 (ConvolutionModule), L1645 (forward)
//   zipvoice/models/modules/scaling.py  L1103 (SwooshL), L1179 (SwooshR)
//   ggml v0.24.0 (mul_mat A-operand = the bound weight in place (f32 weights on
//   the f32 kernels; q8_0 weights on the native q8_0 kernels, f32 B quantized
//   on the fly); conv_1d_dw; soft_max)
//
// Weight tensor names bound in load() (verbatim torch state-dict keys,
// zipformer.py L116 (TTSZipformer.__init__)):
//   in_proj, out_proj, time_embed.{0,2}, guidance_scale_embed
//   encoders.{s}.layers.{i}.{bypass, bypass_mid, self_attn_weights,
//     self_attn{1,2}, feed_forward{1,2,3}, nonlin_attention,
//     conv_module{1,2}, norm}
//   encoders.{s}.time_emb.1 (simple stacks; ".encoder." prefix for
//     downsampled, zipformer.py L814)
//   encoders.{s}.{downsample.bias, out_combiner.bypass_scale} (downsampled stacks)

#include "zipformer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <deque>

#include <ggml.h>
#include <ggml-cpu.h>

namespace lux {

namespace {

// debug dump (gated by the LUX_DUMP env var, a no-op in production): write a
// host float buffer to /tmp/luxdump_<name>.bin for node-by-node comparison
// against the PyTorch reference (tools/ref_fm.py).
inline void lux_dump(const char* name, const float* buf, size_t n) {
  const char* d = std::getenv("LUX_DUMP");
  if (!d || !buf || n == 0) return;
  std::string p = std::string("/tmp/luxdump_") + name + ".bin";
  std::FILE* f = std::fopen(p.c_str(), "wb");
  if (f) { std::fwrite(buf, sizeof(float), n, f); std::fclose(f); }
}

// ---------------------------------------------------------------------------
// small ggml building blocks (ggml v0.24.0, global-scope API)
// ---------------------------------------------------------------------------

// lin: y = W @ x + b (W torch (out, in), x (in, T) -> (out, T)); b may be
// null. Returns nullptr on a quantized-weight shape violation (the caller
// surfaces it as a hard error; no silent fallback).
// ggml v0.24.0 (the bound weight tensor is passed DIRECTLY as the A operand:
// f32 weights run on the f32 kernels; q8_0 weights (32-element blocks along
// the innermost in dim) run on the SIMD q8_0 kernels — the f32 B operand is
// quantized on the fly (quantize_row_q8_0) and the dot is
// ggml_vec_dot_q8_0_q8_0 — no load-time dequantization, the weight stays
// quantized in the GGUF blob; hard-require ne[0]==in, ne[1]==out, in % 32 == 0)
// ggml v0.24.0 (shape assertion helper: grounds every in-graph input's ne against the
// reference dims (zipformer.h config, grounded in MANDATORY_REFERENCES/ysharma3501_LuxTTS
// config.json); a mismatch is a hard error, never a silent layout bug)
struct gz_shape { int64_t ne[4]; };
inline void gz_assert_shape(const char* what, ggml_tensor* t, gz_shape want, const char* file, int line) {
  if (!t) { std::fprintf(stderr, "zipformer: %s is null (%s:%d)\n", what, file, line); std::abort(); }
  for (int i = 0; i < 4; ++i) {
    if (want.ne[i] < 0) continue;  // -1 = wildcard
    if (t->ne[i] != want.ne[i]) {
      std::fprintf(stderr,
                   "zipformer: %s shape mismatch at %s:%d (ne=[%lld,%lld,%lld,%lld] want [%lld,%lld,%lld,%lld])\n",
                   what, file, line,
                   (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                   (long long)want.ne[0], (long long)want.ne[1], (long long)want.ne[2], (long long)want.ne[3]);
      std::abort();
    }
  }
}
#define GZ_ASSERT_SHAPE(what, t, ...) \
  gz_assert_shape(what, (t), gz_shape{__VA_ARGS__}, __FILE__, __LINE__)

inline ggml_tensor* lin(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
                        int in, int out, ggml_tensor* b);

inline ggml_tensor* lin(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
                        int in, int out, ggml_tensor* b) {
  // ground the shapes: w is (in, out) ggml order, x is (in, T), b is (out,)
  GZ_ASSERT_SHAPE("lin.w", w, in, out, -1, -1);
  GZ_ASSERT_SHAPE("lin.x", x, in, -1, -1, -1);
  if (b) GZ_ASSERT_SHAPE("lin.b", b, out, -1, -1, -1);
  if (ggml_is_quantized(w->type)) {
    if (w->ne[0] != in || w->ne[1] != out || (in % 32) != 0) {
      std::fprintf(stderr,
                   "zipformer: quantized weight shape violation (ne=[%lld,%lld], "
                   "want (%d,%d) with ne0 %% 32 == 0; rebuild the GGUF f32)\n",
                   (long long)w->ne[0], (long long)w->ne[1], in, out);
      return nullptr;
    }
  }
  // ggml v0.24.0 (A operand = the bound weight tensor IN PLACE: f32 -> f32
  // kernels; q8_0 -> the native q8_0 SIMD kernels (the f32 B operand x is
  // quantized on the fly). The GGUF stores the weight in ggml (in, out) order
  // (ne0=in, ne1=out); ggml_mul_mat(w, x) computes w @ x -> (out, T), the
  // channel-major layout the rest of the graph consumes. The A operand must
  // NOT be transposed (ggml v0.24.0 asserts !ggml_is_transposed(a) in
  // ggml_mul_mat); the ONNX (T, out) result is the channel-major (out, T) here.
  ggml_tensor* y = ggml_mul_mat(ctx, w, x);
  if (b) y = ggml_add(ctx, y, b);
  return y;
}

// ggml v0.24.0 (broadcast a 1-D (n,) vector over rows of an (n, T) tensor)
inline ggml_tensor* rows_view(ggml_context* ctx, ggml_tensor* v) {
  return ggml_view_2d(ctx, v, v->ne[0], 1, v->nb[0], 0);
}

// ggml v0.24.0 (no_alloc leaf fill record: the host->ggml data move for an unfilled
// leaf, performed via ggml_backend_tensor_set after the scheduler's alloc_graph)
struct leaf_fill { ggml_tensor* t; const float* src; size_t nelem; };

// ggml v0.24.0 (no_alloc-safe scalar constant: an unfilled 1-element f32 leaf; the
// value is stored in `consts` and filled via ggml_backend_tensor_set after alloc_graph
// — ggml_new_f32 asserts !no_alloc, so it cannot be used on the scheduler's ctx)
inline ggml_tensor* const_f32(ggml_context* ctx, float v,
                              std::deque<float>* consts, std::vector<leaf_fill>* fills) {
  ggml_tensor* c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  if (consts && fills) {
    consts->push_back(v);
    fills->push_back({c, &consts->back(), 1});
  }
  return c;
}

// torch.nn.functional.softplus stable form: log(1+exp(z)) = max(z,0) + log1p(exp(-|z|))
ggml_tensor* softplus_stable(ggml_context* ctx, ggml_tensor* z,
                             std::deque<float>* consts, std::vector<leaf_fill>* fills) {
  ggml_tensor* az = ggml_abs(ctx, z);
  ggml_tensor* e  = ggml_exp(ctx, ggml_neg(ctx, az));
  ggml_tensor* l  = ggml_log(ctx, ggml_add(ctx, e, const_f32(ctx, 1.0f, consts, fills)));
  ggml_tensor* m  = ggml_scale(ctx, ggml_add(ctx, z, az), 0.5f);
  return ggml_add(ctx, m, l);
}

// scaling.py L1103 (SwooshL: log(1 + exp(x-4)) - 0.08x - 0.035)
ggml_tensor* swoosh_l(ggml_context* ctx, ggml_tensor* x,
                      std::deque<float>* consts, std::vector<leaf_fill>* fills) {
  ggml_tensor* z = ggml_scale_bias(ctx, x, 1.0f, -4.0f);
  return ggml_add(ctx, softplus_stable(ctx, z, consts, fills),
                  ggml_scale_bias(ctx, x, -0.08f, -0.035f));
}

// zipformer.py L1406 (FeedforwardModule: Linear -> SwooshL -> Linear);
// returns nullptr if lin() reports a quantized-weight shape violation
ggml_tensor* ff_block(ggml_context* ctx, ggml_tensor* x, int C,
                      ggml_tensor* in_w, ggml_tensor* in_b, ggml_tensor* out_w,
                      ggml_tensor* out_b, int hidden,
                      std::deque<float>* consts, std::vector<leaf_fill>* fills) {
  ggml_tensor* h = lin(ctx, x, in_w, C, hidden, in_b);
  if (!h) return nullptr;
  h = swoosh_l(ctx, h, consts, fills);
  return lin(ctx, h, out_w, hidden, C, out_b);
}

// zipformer.py L754 (BypassModule: src_orig + (src - src_orig) * scale, per-channel (C,))
inline ggml_tensor* bypass(ggml_context* ctx, const ggml_tensor* src0,
                           ggml_tensor* src, ggml_tensor* scale) {
  ggml_tensor* s0 = const_cast<ggml_tensor*>(src0);
  return ggml_add(ctx, s0, ggml_mul(ctx, ggml_sub(ctx, src, s0), rows_view(ctx, scale)));
}

// scaling.py L407 (BiasNorm, eval-time, channel dim = ne0); ScaledLinear L351:
//   scales = mean((x - bias)^2, dim=channel)^-0.5 * exp(log_scale)
ggml_tensor* bias_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* bias,
                       ggml_tensor* log_scale, int C,
                       std::deque<float>* consts, std::vector<leaf_fill>* fills) {
  ggml_tensor* xb = ggml_sub(ctx, x, rows_view(ctx, bias));
  ggml_tensor* m = ggml_div(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, xb, xb)),
                            const_f32(ctx, (float)C, consts, fills));
  ggml_tensor* ls = const_f32(ctx, std::exp((float)ggml_get_f32_1d(log_scale, 0)), consts, fills);
  // ggml v0.24.0 (1.0/sqrt(m) via exp(-0.5 * log(m)); div(a,b) result shape = a's shape)
  ggml_tensor* inv = ggml_exp(ctx, ggml_scale_bias(ctx, ggml_log(ctx, m), -0.5f, 0.0f));
  ggml_tensor* scales = ggml_mul(ctx, inv, ls);
  return ggml_mul(ctx, x, scales);
}

// ConvolutionModule (zipformer.py L1554), host-side numeric reference
// (forward() uses conv_module_ggml below):
// in_proj -> GLU(sigmoid) -> depthwise(k, pad k/2)
// -> SwooshR -> out_proj. src is channel-major (C,S); weights are torch-layout
// raw pointers (in_w/out_w: (out,in) row-major; dw_w: (C,1,K) row-major).
// Returns channel-major (C,S) output.
[[maybe_unused]] std::vector<float> conv_host(const float* src, const float* in_w, const float* in_b,
                             const float* dw_w, const float* dw_b,
                             const float* out_w, const float* out_b,
                             int C, int K, int S) {
  // in_proj: x2[o,t] = sum_i in_w[o*C+i] * src[i,t] + in_b[o],  o in [0,2C)
  std::vector<float> x2h((size_t)2 * C * S);
  for (int t = 0; t < S; ++t) for (int o = 0; o < 2 * C; ++o) {
    float acc = in_b ? in_b[o] : 0.0f;
    for (int i = 0; i < C; ++i) acc += in_w[(size_t)o * C + i] * src[(size_t)i * S + t];
    x2h[(size_t)o * S + t] = acc;
  }
  // GLU (Python: x,s = x2.chunk(2,dim=2); s=sigmoid(s); x=x*s) -> (C,S)
  std::vector<float> gh((size_t)C * S);
  for (int c = 0; c < C; ++c) for (int t = 0; t < S; ++t) {
    const float xav = x2h[(size_t)c * S + t];
    const float sbv = x2h[(size_t)(C + c) * S + t];
    gh[(size_t)c * S + t] = xav * (1.0f / (1.0f + std::exp(-sbv)));
  }
  // zero-pad (C, S+K-1) by K/2 each side, then depthwise (groups=C)
  const int p = K / 2;
  const int SP = S + K - 1;
  std::vector<float> gpad((size_t)C * SP, 0.0f);
  for (int c = 0; c < C; ++c) for (int t = 0; t < S; ++t)
    gpad[(size_t)c * SP + (t + p)] = gh[(size_t)c * S + t];
  std::vector<float> yh((size_t)C * S);
  for (int c = 0; c < C; ++c) for (int t = 0; t < S; ++t) {
    float acc = dw_b ? dw_b[c] : 0.0f;
    for (int k = 0; k < K; ++k) acc += dw_w[(size_t)c * K + k] * gpad[(size_t)c * SP + (t + k)];
    yh[(size_t)c * S + t] = acc;
  }
  // SwooshR: softplus(x-1) - 0.08x - 0.313261687 (stable)
  for (int i = 0; i < (int)yh.size(); ++i) {
    const float x = yh[i];
    const float z = x - 1.0f;
    const float sp = (z > 0.0f) ? z + std::log1p(std::exp(-z)) : std::log1p(std::exp(z));
    yh[i] = sp - 0.08f * x - 0.313261687f;
  }
  // zipformer.py L1645 (ConvolutionModule.forward: out_proj: yo[o,t] = sum_i out_w[o*C+i] * yh[i,t] + out_b[o])
  std::vector<float> yo((size_t)C * S);
  for (int t = 0; t < S; ++t) for (int o = 0; o < C; ++o) {
    float acc = out_b ? out_b[o] : 0.0f;
    for (int i = 0; i < C; ++i) acc += out_w[(size_t)o * C + i] * yh[(size_t)i * S + t];
    yo[(size_t)o * S + t] = acc;
  }
  return yo;
}

// zipformer.py L1554 (ConvolutionModule) as ONE PURE IN-GRAPH ggml FRAGMENT
// on the caller's context (ggml v0.24.0: conv_1d_dw zero-pads K/2 in-graph):
// x is a ggml (C, S) leaf (channel-major); in_w/out_w are bound tensors
// (f32 contiguous or q8_0 dim-reversed, handled by lin()); dw_w is the
// bound f32 depthwise kernel in ggml (K, 1, C) layout. Returns the
// (C, S) output node, or nullptr on a node-construction failure (hard error).
//
// time_mask (S,) marks padded times (nonzero): the masked_fill step of
// zipformer.py L1677 (x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0))
// zeroes the GLU-activated channels at padded times BEFORE the depthwise
// conv. In-graph: a per-time (S,1,1) 0/1 mask leaf multiplies the
// (S,C,1) conv input (ggml v0.24.0: a (1,S) mask does NOT broadcast onto (S,C,1)).
// Pass nullptr when there is no padding.
ggml_tensor* conv_module_ggml(ggml_context* ctx, ggml_tensor* x,
                              ggml_tensor* in_w, ggml_tensor* in_b,
                              ggml_tensor* dw_w, ggml_tensor* dw_b,
                              ggml_tensor* out_w, ggml_tensor* out_b,
                              int C, int K, int S, const float* time_mask,
                              ggml_tensor** time_mask_leaf, std::deque<std::vector<float>>* mask_bufs,
                              std::deque<float>* consts, std::vector<leaf_fill>* fills) {
  // zipformer.py L1645 (in_proj: (C, S) -> (2C, S))
  ggml_tensor* x2 = lin(ctx, x, in_w, C, 2 * C, in_b);
  if (!x2) return nullptr;
  // zipformer.py L1664 (GLU: x,s = x2.chunk(2,dim=ch); s = sigmoid(s); x = x*s)
  ggml_tensor* xa = ggml_view_2d(ctx, x2, C, S, x2->nb[1], 0);
  ggml_tensor* xs = ggml_view_2d(ctx, x2, C, S, x2->nb[1],
                                 (size_t)C * x2->nb[0]);
  ggml_tensor* gh = ggml_mul(ctx, xa, ggml_sigmoid(ctx, xs));
  // ggml v0.24.0 (transpose (C,S) -> (S,C) t-inner, reshape to the conv (S,C,1) input)
  ggml_tensor* b3 = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, gh)), S, C, 1);
  // zipformer.py L1677 (masked_fill: padded source frames are zeroed before the depthwise conv)
  if (time_mask) {
    ggml_tensor* m = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, S);
    if (!m) return nullptr;
    if (time_mask_leaf) *time_mask_leaf = m;
    // ggml v0.24.0 (no_alloc leaf: the 0/1 mask values are collected for a
    // post-alloc ggml_backend_tensor_set fill; the leaf is created unfilled)
    if (mask_bufs) {
      std::vector<float> mv((size_t)S);
      for (int t = 0; t < S; ++t) mv[t] = (time_mask[t] == 0.0f) ? 1.0f : 0.0f;
      mask_bufs->push_back(std::move(mv));
    }
    b3 = ggml_mul(ctx, b3, ggml_reshape_3d(ctx, m, S, 1, 1));
  }
  // ggml v0.24.0 (conv_1d_dw: p0 = K/2 zero-pads both sides in-graph)
  ggml_tensor* ydw = ggml_conv_1d_dw(ctx, dw_w, b3, 1, K / 2, 1);
  if (!ydw) return nullptr;
  // ggml v0.24.0 ((S,C,1) -> (C,S) ch-major leaf: transpose + cont materialize the bytes)
  ggml_tensor* y = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_transpose(ctx, ydw)), C, S);
  // scaling.py L1179 (SwooshR: softplus(x-1) - 0.08x - 0.313261687)
  if (dw_b) y = ggml_add(ctx, y, rows_view(ctx, dw_b));
  y = ggml_add(ctx,
               softplus_stable(ctx, ggml_scale_bias(ctx, y, 1.0f, -1.0f), consts, fills),
               ggml_scale_bias(ctx, y, -0.08f, -0.313261687f));
  // zipformer.py L1645 (out_proj: (C, S) -> (C, S))
  ggml_tensor* yo = lin(ctx, y, out_w, C, C, out_b);
  return yo;
}

// Host-side f32 view of a bound 2-D weight (torch (out, in) ordering the
// host consumers index as w[o*in + i]); ggml v0.24.0 (q8_0 weights are dequantized)
std::vector<float> weight_f32(ggml_tensor* w, int in, int out) {
  std::vector<float> v((size_t)in * out);
  if (!ggml_is_quantized(w->type)) {
    std::memcpy(v.data(), w->data, (size_t)in * out * sizeof(float));
    return v;
  }
  // ggml v0.24.0 (q8_0 block: value = qs[j] * d, fp16 scale, signed int8 quants)
  struct q8_0_blk { ggml_fp16_t d; int8_t qs[32]; };
  static_assert(sizeof(q8_0_blk) == 34);
  const q8_0_blk* p = (const q8_0_blk*)w->data;
  for (int o = 0; o < out; ++o)
    for (int i0 = 0; i0 < in; i0 += 32) {
      const q8_0_blk& b = p[(size_t)o * (in / 32) + i0 / 32];
      const float d = ggml_fp16_to_fp32(b.d);
      for (int i = 0; i < 32; ++i)
        v[(size_t)o * in + i0 + i] = (float)b.qs[i] * d;
    }
  return v;
}

// ---------------------------------------------------------------------------
// host-side helpers
// ---------------------------------------------------------------------------

// zipformer.py L1156 (RelPositionMultiheadAttentionWeights.forward:
// pep[r][h][d] = sum_pe linear_pos[pe][h*dp+d] * pe[r][pe])
// host-side reference table builder (forward() projects in-graph via lin())
[[maybe_unused]] std::vector<float> rel_pos_scores_table(const float* pe,
                                        ggml_tensor* lp_w,
                                        int S, int H, int dp, int pos_dim) {
  const int R = 2 * S - 1;
  const float* lw = (const float*)lp_w->data;
  std::vector<float> pep((size_t)R * H * dp);
  #pragma omp parallel for schedule(static)
  for (int hd = 0; hd < H * dp; ++hd) {
    const int h = hd / dp, d = hd % dp;
    for (int r = 0; r < R; ++r) {
      float acc = 0.0f;
      const float* prow = pe + (size_t)r * pos_dim;
      const float* wcol = lw + (size_t)(h * dp + d) * pos_dim;
      for (int pe_ = 0; pe_ < pos_dim; ++pe_) acc += wcol[pe_] * prow[pe_];
      pep[(size_t)r * H * dp + h * dp + d] = acc;
    }
  }
  return pep;
}

// scaling.py L1103 (SwooshR, host-side scalar form: log(1+exp(x-1)) - 0.08x - 0.313261687;
// the scalar twin of the ggml SwooshL above, used for the host-side eval of the
// SwooshR gate)
float swoosh_r_host(float v) {
  const float y = v - 1.0f;
  const float l = y > 0.0f ? y + std::log1p(std::exp(-y)) : std::log1p(std::exp(y));
  return l - 0.08f * v - 0.313261687f;
}

// ggml v0.24.0 (the CPU reg exposes ggml_backend_set_n_threads via the reg proc
// address, exactly like the whisper.cpp reference; n <= 0 = ggml default)
static void cpu_set_n_threads(ggml_backend_t b, int n) {
  if (n <= 0) return;
  ggml_backend_reg_t reg = ggml_backend_cpu_reg();
  auto* f = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
  if (f) f(b, n);
}

// ggml v0.24.0 (ggml_backend_cpu_buffer_from_ptr asserts a 32-byte-aligned start, but
// the ggml pool/blob pointers are only GGML_MEM_ALIGN=16 aligned (ggml.c uses plain
// malloc); round the start UP to 32 and shrink the size — the leading bytes stay
// inside the owning allocation, and the CPU buffer's free_buffer/init_tensor are NULL
// so the wrapper never dereferences them)
static void* align32_up(void* p) {
  return (void*)(((uintptr_t)p + 31) & ~(uintptr_t)31);
}

// Persistent compute context: the per-instance cctx_ is a no_alloc ggml context
// (created in forward()), so no static byte estimate is needed — tensor data is
// malloc'd per tensor and the graph's live-tensor footprint (T^2 in the attention
// stacks) can never exceed a pre-sized pool.

} // end anonymous namespace (private ggml building blocks)

// ---------------------------------------------------------------------------
// parameter-free helpers (public, kept from the stub)
// ---------------------------------------------------------------------------

std::vector<float> timestep_embed(float t, int dim) {
  // zipformer.py L54 (timestep_embedding: half = dim//2; freqs = exp(-ln(10000)*arange/half);
  // args = t*freqs; cat([cos, sin]); float32 throughout)
  const int half = dim / 2;
  std::vector<float> out(dim);
  for (int i = 0; i < half; ++i) {
    const float f = std::expf(-std::logf(10000.0f) * (float)i / half);
    const float a = t * f;
    out[i] = std::cosf(a);
    out[half + i] = std::sinf(a);
  }
  return out;
}

// zipformer.py L945 (CompactRelPositionalEncoding.extend_pe: the rel-PE table the
// attn_block (L1156) consumes)
std::vector<float> compact_rel_pe(int T, int pos_dim) {
  //   x = -T+1..T-1 (int)
  //   compression_length = sqrt(pos_dim)
  //   x_c = cl * sign(x) * (log(|x|+cl) - log(cl))
  //   length_scale = pos_dim / (2*pi)   (length_factor = 1.0)
  //   x_atan = atan(x_c / length_scale)
  //   pe rows: even cols = cos(x_atan * f), odd cols = sin(x_atan * f),
  //   last col = 1.0 (the linear_pos bias row); f = 1..pos_dim/2.
  if (T <= 0) return {};
  const double cl = std::sqrt((double)pos_dim);
  const double length_scale = (double)pos_dim / (2.0 * M_PI);
  std::vector<float> pe((size_t)(2 * T - 1) * (pos_dim + 1));
  for (int i = 0; i < 2 * T - 1; ++i) {
    const double x = (double)i - (T - 1);            // relative offset r
    const double xc = cl * (x >= 0.0 ? 1.0 : -1.0) *
                     (std::log(std::fabs(x) + cl) - std::log(cl));
    const double a = std::atan(xc / length_scale);   // in (-pi/2, pi/2)
    for (int f = 1; f <= pos_dim / 2; ++f) {
      const int col = 2 * (f - 1);
      pe[(size_t)i * (pos_dim + 1) + col]     = (float)std::cos(a * (double)f);
      pe[(size_t)i * (pos_dim + 1) + col + 1] = (float)std::sin(a * (double)f);
    }
    // zipformer.py L1043 (extend_pe: pe[:, -1] = 1.0 for bias — this OVERWRITES
    // the last sine column; the port appended an extra column instead, leaving
    // the last sine in place).
    pe[(size_t)i * (pos_dim + 1) + pos_dim - 1] = 1.0f;
    pe[(size_t)i * (pos_dim + 1) + pos_dim] = 1.0f;
  }
  return pe;
}

// ---------------------------------------------------------------------------
// standalone attention blocks (public free functions; caller owns the ctx —
// ggml_reset it BEFORE calling; each builds + computes one graph)
// ---------------------------------------------------------------------------

bool attn_block_ggml(ggml_context* ctx, int n_threads, const float* qkv,
                     const float* pep, const float* mask,
                     int S, int H, int dq, int dp, float* attn) {
  const int ipd = (2 * dq + dp) * H;
  const int P = 2 * S - 1;

  // qkv (ipd, S) ch-major NATURAL order, uploaded as a contiguous 2-D leaf;
  // the q/k/p slices are 2-D views (nb1 = row stride, channel offsets).
  ggml_tensor* qkv_u = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ipd, S);
  std::memcpy(qkv_u->data, qkv, (size_t)ipd * S * 4);
  // head-split the natural (ipd, S) ch-major qkv into (dper, S, H) [d, t, h]:
  // transpose to a contiguous (S, ipd) t-major leaf, slice each section as a
  // 2-D view (nb1 = S*4 channel stride, nb0 = 4 time stride inherited from
  // qkvT->nb[0], offset = sec_ch*S*4), reshape (S, dper, H) [t, d, h] (ggml
  // flat layout has ne0 innermost, so the c' = h*dper+d channel split needs
  // dper as ne1), then permute to (dper, S, H). qh[d,t,h] = qkv[h*dq+d, t];
  // kh = qkv[H*dq + h*dq+d, t]; phq = qkv[2*H*dq + h*dp+d, t].
  ggml_tensor* qkvT = ggml_cont(ctx, ggml_transpose(ctx, qkv_u));
  auto heads_of = [&](int sec_ch, int nch, int dper) {
    ggml_tensor* sec = ggml_view_2d(ctx, qkvT, S, nch, (size_t)S * 4, (size_t)sec_ch * (size_t)S * 4);
    ggml_tensor* m = ggml_reshape_3d(ctx, sec, S, dper, nch / dper);   // (S, dper, H) [t, d, h]
    return ggml_cont(ctx, ggml_permute(ctx, m, 1, 0, 2, 3));          // (dper, S, H) [d, t, h]
  };
  ggml_tensor* qh   = heads_of(0, H * dq, dq);
  ggml_tensor* kh   = heads_of(H * dq, H * dq, dq);
  ggml_tensor* ph_q = heads_of(2 * H * dq, H * dp, dp);

  // pep (projected table, flat (r*H+h)*dp+d) -> (dp, P, H) leaf:
  // buf[(h*P+r)*dp+d] = pep[(r*H+h)*dp+d] (ggml (dp,P,H) contiguous is
  // h-major outer, r middle, d inner).
  std::vector<float> pep3((size_t)dp * P * H);
  for (int h = 0; h < H; ++h)
    for (int r = 0; r < P; ++r)
      for (int d = 0; d < dp; ++d)
        pep3[((size_t)h * P + r) * dp + d] = pep[(size_t)(r * H + h) * dp + d];
  ggml_tensor* pepu = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dp, P, H);
  std::memcpy(pepu->data, pep3.data(), pep3.size() * 4);

  // zipformer.py L1156 (rel-pos scores bd = pepu (dp,P,H) @ ph_q (dp,S,H) -> (P, S, H) [r, i, h]:
// bd[r,i,h] = pos_proj_r[h,:] . p_i[h,:]; the rel-shift chain (pad_ext -> reshape -> view -> cont, x2)
// turns (P, S, H) into the (S_j, S_i, H) window: final[j,i,h] = bd[j - i + S - 1, i, h] == pos_emb[j-i+S-1])
  ggml_tensor* bd = ggml_mul_mat(ctx, pepu, ph_q);
  bd = ggml_pad_ext(ctx, bd, 1, 0, 0, 0, 0, 0, 0, 0);
  bd = ggml_reshape_3d(ctx, bd, S, 2 * S, H);
  bd = ggml_view_3d(ctx, bd, S, 2 * S - 1, H, bd->nb[1], bd->nb[2], bd->nb[1]);
  bd = ggml_cont(ctx, bd);
  bd = ggml_reshape_3d(ctx, bd, 2 * S - 1, S, H);
  bd = ggml_view_3d(ctx, bd, S, S, H, bd->nb[1], bd->nb[2], 0);
  bd = ggml_cont(ctx, bd);

  // ggml v0.24.0 (qk scores: (S_j, S_i, H) [source, target, head]; mask: 1-D (S,) -> (S, 1, 1) broadcast)
  ggml_tensor* ac = ggml_mul_mat(ctx, kh, qh);
  ggml_tensor* masku = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, S);
  std::memcpy(masku->data, mask, (size_t)S * 4);
  ggml_tensor* maskb = ggml_reshape_3d(ctx, masku, S, 1, 1);
  // ggml v0.24.0 (scores = add(add(sc_qk, bd), mask); ggml_soft_max normalizes over ne[0]
  // = the source axis (the attention softmax axis))
  ggml_tensor* attn_t = ggml_soft_max(ctx, ggml_add(ctx, ggml_add(ctx, ac, bd), maskb));

  ggml_cgraph* g = ggml_new_graph_custom(ctx, 256, false);
  ggml_build_forward_expand(g, attn_t);
  ggml_set_output(attn_t);
  if (ggml_graph_compute_with_ctx(ctx, g, n_threads) != GGML_STATUS_SUCCESS) return false;

  // ggml v0.24.0 (download: attn (H,S,S) natural [h][i][j] -> (S_j, S_i, H) [j, i, h] leaf:
  // element (j, i, h) at ((h*S+i)*S+j)*4 in BOTH layouts -> byte-identical memcpy)
  std::memcpy(attn, attn_t->data, (size_t)H * S * S * 4);
  return true;
}

bool attn_apply_ggml(ggml_context* ctx, int n_threads, const float* attn,
                     const float* v, int S, int H, int dv, float* out) {
  // ggml v0.24.0 (attn (H,S,S) natural [h][i][j] -> (S_j, S_i, H) [j, i, h] leaf:
  // element (j, i, h) at ((h*S+i)*S+j)*4 in BOTH layouts -> byte-identical memcpy)
  ggml_tensor* attn3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, S, H);
  std::memcpy(attn3->data, attn, (size_t)S * S * H * 4);

  // ggml v0.24.0 (v (dvH, S) ch-major head-major (c = h*dv+d): the (S_j, dv, H) leaf
  // element (j, d, h) sits at ((h*dv+d)*S + j)*4 — byte-identical to the flat buffer)
  ggml_tensor* v3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, dv, H);
  std::memcpy(v3->data, v, (size_t)dv * H * S * 4);

  // ggml v0.24.0 (av = mul_mat(attn3 (S_j, S_i, H), v3 (S_j, dv, H)) -> (S_i, dv, H);
  // av[i][d][h] = sum_j attn3[j][i][h] * v3[j][d][h] (reduction over ne[0] = source j);
  // byte-identical to the (dvH, S) ch-major out buffer)
  ggml_tensor* av = ggml_mul_mat(ctx, attn3, v3);
  ggml_cgraph* g = ggml_new_graph_custom(ctx, 256, false);
  ggml_build_forward_expand(g, av);
  ggml_set_output(av);
  if (ggml_graph_compute_with_ctx(ctx, g, n_threads) != GGML_STATUS_SUCCESS) return false;

  std::memcpy(out, av->data, (size_t)dv * H * S * 4);
  return true;
}

// ---------------------------------------------------------------------------
// weight binding
// ---------------------------------------------------------------------------

Zipformer::Zipformer(const ZipformerConfig& cfg) : cfg_(cfg) {}

Zipformer::~Zipformer() {
  if (sched_) ggml_backend_sched_free((ggml_backend_sched_t)sched_);
  if (cctx_) ggml_free((ggml_context*)cctx_);   // mem_buffer = sched_meta_ (not owned by the ctx)
  if (wbuf_) ggml_backend_buffer_free(wbuf_);
  if (backend_) ggml_backend_free((ggml_backend_t)backend_);
}

void Zipformer::set_n_threads(int n) {
  n_threads_ = n;
  if (backend_) cpu_set_n_threads((ggml_backend_t)backend_, n);
}

void Zipformer::set_weight_blob(const void* blob, size_t bytes) {
  weight_blob_ = blob;
  weight_blob_bytes_ = bytes;
  if (wbuf_) { ggml_backend_buffer_free(wbuf_); wbuf_ = nullptr; }
  ensure_backend();
}

void Zipformer::ensure_backend() const {
  if (!cctx_) {
    ggml_backend_t b = ggml_backend_cpu_init();
    if (!b) return;
    backend_ = (void*)b;
    cpu_set_n_threads(b, n_threads_);
    // ggml v0.24.0 (single CPU backend, n_max_nodes = 16384; the meta buffer
    // holds the no_alloc tensor OBJECTS for up to n_max_nodes; the scheduler
    // auto-sizes the DATA buffers per graph — the reference whisper.cpp
    // whisper_sched_graph_init pattern)
    const int n_max_nodes = 16384;
    ggml_backend_t bk[1] = { b };
    ggml_backend_sched_t s = ggml_backend_sched_new(bk, nullptr, 1, (size_t)n_max_nodes, false, true);
    if (!s) { ggml_backend_free(b); backend_ = nullptr; return; }
    sched_ = (void*)s;
    sched_meta_.resize(ggml_tensor_overhead() * (size_t)n_max_nodes + ggml_graph_overhead());
    // ggml v0.24.0 (no_alloc context: tensor DATA is assigned by the
    // scheduler's alloc_graph; the tensor OBJECTS live in the meta buffer)
    ggml_init_params p{ sched_meta_.size(), sched_meta_.data(), true };
    cctx_ = (void*)ggml_init(p);
  }
  if (!wbuf_ && weight_blob_) {
    char* wbp = (char*)align32_up(const_cast<void*>(weight_blob_));
    size_t wbs = weight_blob_bytes_ - (size_t)(wbp - (char*)weight_blob_);
    wbuf_ = ggml_backend_cpu_buffer_from_ptr(wbp, wbs);
    // ggml v0.24.0 (mark the weight buffer WEIGHTS so the scheduler routes the
    // weight-dependent ops to the CPU backend (ggml_backend_sched_get_tensor_backend
    // checks src->buffer->usage == WEIGHTS); without it the nodes get backend_id -1)
    if (wbuf_) ggml_backend_buffer_set_usage(wbuf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  }
  // ggml v0.24.0 (bind every weight tensor to the CPU buffer that owns its
  // data: GGUF f32/q8_0 tensors all live in the blob (the q8_0 tensors point
  // INTO the blob — no dequantized copies), so a single wbuf_ wrapping the
  // blob covers every weight; the binding is the tensor's public buffer field
  // — the reference whisper.cpp sets w->buffer directly; the CPU buffer's
  // init_tensor is a no-op)
  for (auto& pr : w_) {
    ggml_tensor* t = pr.second;
    if (!t || !t->data) continue;
    const uint8_t* d = (const uint8_t*)t->data;
    const uint8_t* b0 = (const uint8_t*)weight_blob_;
    if (b0 && d >= b0 && d + ggml_nbytes(t) <= b0 + weight_blob_bytes_) {
      if (t->buffer == nullptr) t->buffer = wbuf_;
    } else {
      std::fprintf(stderr, "zipformer: weight '%s' not inside the GGUF blob (left unbound)\n", pr.first.c_str());
    }
  }
}

std::string Zipformer::load(ggml_context* ctx,
                            const std::vector<std::pair<std::string, ggml_tensor*>>& named) {
  if (map_ctx_) { ggml_free((ggml_context*)map_ctx_); map_ctx_ = nullptr; }
  w_.clear();

  // zipformer.py L116 (TTSZipformer.__init__): build the expected name list from the config.
  const int n_stacks = (int)cfg_.downsample_factor.size();
  std::vector<std::string> need;
  auto add = [&](const std::string& n) { need.push_back(n); };
  add("in_proj.weight"); add("in_proj.bias");
  add("out_proj.weight"); add("out_proj.bias");
  if (cfg_.use_time_embed) {
    add("time_embed.0.weight"); add("time_embed.0.bias");
    add("time_embed.2.weight"); add("time_embed.2.bias");
    for (int s = 0; s < n_stacks; ++s) {
      // zipformer.py L814 (DownsampledZipformer2Encoder: time_emb/layers live under
      // "encoders.{s}.encoder."; downsample/out_combiner stay at the stack level)
      const std::string ip = "encoders." + std::to_string(s) + "."
          + (cfg_.downsample_factor[s] != 1 ? "encoder." : "");
      add(ip + "time_emb.1.weight");
      add(ip + "time_emb.1.bias");
    }
  }
  if (cfg_.use_guidance_embed) add("guidance_scale_embed.weight");
  for (int s = 0; s < n_stacks; ++s) {
    const std::string inner = "encoders." + std::to_string(s) + "."
        + (cfg_.downsample_factor[s] != 1 ? "encoder." : "");
    for (int i = 0; i < cfg_.num_layers[s]; ++i) {
      const std::string p = inner + "layers." + std::to_string(i) + ".";
      add(p + "bypass.bypass_scale");
      add(p + "bypass_mid.bypass_scale");
      add(p + "self_attn_weights.in_proj.weight");
      add(p + "self_attn_weights.in_proj.bias");
      add(p + "self_attn_weights.linear_pos.weight");
      add(p + "self_attn1.in_proj.weight"); add(p + "self_attn1.in_proj.bias");
      add(p + "self_attn1.out_proj.weight"); add(p + "self_attn1.out_proj.bias");
      add(p + "self_attn2.in_proj.weight"); add(p + "self_attn2.in_proj.bias");
      add(p + "self_attn2.out_proj.weight"); add(p + "self_attn2.out_proj.bias");
      add(p + "feed_forward1.in_proj.weight"); add(p + "feed_forward1.in_proj.bias");
      add(p + "feed_forward1.out_proj.weight"); add(p + "feed_forward1.out_proj.bias");
      add(p + "feed_forward2.in_proj.weight"); add(p + "feed_forward2.in_proj.bias");
      add(p + "feed_forward2.out_proj.weight"); add(p + "feed_forward2.out_proj.bias");
      add(p + "feed_forward3.in_proj.weight"); add(p + "feed_forward3.in_proj.bias");
      add(p + "feed_forward3.out_proj.weight"); add(p + "feed_forward3.out_proj.bias");
      add(p + "nonlin_attention.in_proj.weight"); add(p + "nonlin_attention.in_proj.bias");
      add(p + "nonlin_attention.out_proj.weight"); add(p + "nonlin_attention.out_proj.bias");
      add(p + "conv_module1.in_proj.weight"); add(p + "conv_module1.in_proj.bias");
      add(p + "conv_module1.depthwise_conv.weight");
      add(p + "conv_module1.depthwise_conv.bias");
      add(p + "conv_module1.out_proj.weight"); add(p + "conv_module1.out_proj.bias");
      add(p + "conv_module2.in_proj.weight"); add(p + "conv_module2.in_proj.bias");
      add(p + "conv_module2.depthwise_conv.weight");
      add(p + "conv_module2.depthwise_conv.bias");
      add(p + "conv_module2.out_proj.weight"); add(p + "conv_module2.out_proj.bias");
      add(p + "norm.bias"); add(p + "norm.log_scale");
    }
    if (cfg_.downsample_factor[s] != 1) {
      const std::string p = "encoders." + std::to_string(s) + ".";
      add(p + "downsample.bias");
      add(p + "out_combiner.bypass_scale");
    }
  }

  std::map<std::string, ggml_tensor*> named_map;
  for (const auto& pr : named) named_map[pr.first] = pr.second;
  for (const auto& name : need) {
    auto it = named_map.find(name);
    if (it == named_map.end()) {
      std::string missing = name;
      for (const auto& n : need)
        if (n > missing && named_map.find(n) == named_map.end()) missing += ", " + n;
      return "zipformer: missing weight tensor(s): " + missing;
    }
    // ggml v0.24.0 (bind the ORIGINAL GGUF tensor directly: f32 weights run on
    // the f32 kernels; q8_0 weights run on the SIMD q8_0 kernels (lin() passes
    // the tensor in place as the mul_mat A operand) — no load-time dequantization,
    // so no f32 copy in the caller's model ggml context; the q8_0 kernels
    // require the innermost dim to be a multiple of 32)
    ggml_tensor* wt = it->second;
    if (wt->type == GGML_TYPE_Q8_0 && (wt->ne[0] % 32) != 0)
      return ("zipformer: quantized weight '" + name +
          "' ne0=" + std::to_string((int)wt->ne[0]) + " is not a multiple of 32");
    w_[name] = wt;
  }
  if (map_ctx_) { ggml_free((ggml_context*)map_ctx_); map_ctx_ = nullptr; }
  return "";
}

// ---------------------------------------------------------------------------
// forward — zipformer.py TTSZipformer.forward (the distill ODE decode path;
// per-layer bypass/ff/conv sub-blocks each cite their zipformer.py/scaling.py line)
// ---------------------------------------------------------------------------

std::vector<float> Zipformer::forward(const float* x, int T, float t, float guidance_scale,
                        const bool* pad_mask) const {
  const ZipformerConfig& cfg = cfg_;
  if (w_.empty()) return {};
  if (T <= 0 || !x) return {};

  const int C = cfg.encoder_dim;
  const int H = cfg.num_heads;
  const int dq = cfg.query_head_dim, dp = cfg.pos_head_dim, dv = cfg.value_head_dim;
  const int ff = cfg.feedforward_dim;
  const int in_proj_dim = (2 * dq + dp) * H;

  const auto W = [&](const std::string& n) -> ggml_tensor* {
    auto it = w_.find(n);
    if (it == w_.end()) { /* load() validated the set; cannot happen */ abort(); }
    return it->second;
  };

  // ggml v0.24.0 (persistent CPU-backend compute: ensure_backend() creates the
  // backend / scheduler / no_alloc cctx once (mem_buffer = sched_meta_) and binds
  // every weight tensor to its CPU buffer; tensor DATA is auto-sized by the
  // scheduler per graph, so no static byte estimate)
  ensure_backend();
  ggml_context* cctx = (ggml_context*)cctx_;

  // ggml v0.24.0 (the scheduler allocates the graph (auto-sizing the
  // intermediate buffers + assigning the no_alloc leaves' data), then the
  // leaves are filled via ggml_backend_tensor_set, then compute; the
  // reference (whisper.cpp ggml_graph_compute_helper) resets after every
  // compute so the next graph re-allocates cleanly)
  auto zalloc = [&](ggml_cgraph* g) -> bool {
    ggml_backend_sched_t s = (ggml_backend_sched_t)sched_;
    return ggml_backend_sched_alloc_graph(s, g);
  };
  auto zcompute = [&](ggml_cgraph* g) -> bool {
    ggml_backend_sched_t s = (ggml_backend_sched_t)sched_;
    const bool ok = ggml_backend_sched_graph_compute(s, g) == GGML_STATUS_SUCCESS;
    ggml_backend_sched_reset(s);
    return ok;
  };
  // ggml v0.24.0 (the scheduler sets a leaf's buffer_id via
  // ggml_backend_sched_get_tensor_backend, which returns -1 for an unbound
  // runtime input leaf (no buffer, not a weight). The gallocr then trips
  // GGML_ASSERT(buffer_id >= 0) in ggml_gallocr_reserve_n. Pre-bind every
  // unbound input tensor to a CPU buffer first (llama.cpp's "allocate the
  // inputs statically" pattern) so each leaf has a valid backend.)
  auto zprebind_t = [&](ggml_tensor* t) -> void {
    if (!t || t->data || t->view_src) return; // already bound (weight) or a view
    const size_t sz = ggml_nbytes(t);
    if (sz == 0) return;
    ggml_backend_t cpu = ggml_backend_sched_get_backend((ggml_backend_sched_t)sched_, 0);
    ggml_backend_buffer_t bf = ggml_backend_alloc_buffer(cpu, sz);
    if (bf) { t->data = ggml_backend_buffer_get_base(bf); t->buffer = bf; }
  };

  // zipformer.py L249 (TTSZipformer.forward: in_proj -> (C, T))
  std::vector<float> cur;
  {
    ggml_reset(cctx);
    ggml_tensor* xt = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, cfg.in_dim, T);
    if (!xt) return {};
    // ggml v0.24.0 (x (T, in_dim) row-major is byte-identical to the ggml (in_dim, T)
    // t-major leaf, so the fill is a post-alloc ggml_backend_tensor_set, not a memcpy)
    ggml_tensor* y = lin(cctx, xt, W("in_proj.weight"), cfg.in_dim, C, W("in_proj.bias"));
    if (!y) return {};
    ggml_cgraph* g = ggml_new_graph_custom(cctx, 4096, false);
    ggml_build_forward_expand(g, y);
    ggml_set_output(y);
    zprebind_t(xt);
    if (!zalloc(g)) return {};
    ggml_backend_tensor_set(xt, x, 0, (size_t)cfg.in_dim * (size_t)T * sizeof(float));
    if (!zcompute(g)) return {};
    cur.resize((size_t)C * T);
    {
      // ggml v0.24.0 (download the (C, T) t-major result, then transpose to
      // the channel-major cur[c*T+t] the rest of forward() consumes)
      ggml_backend_tensor_get(y, cur.data(), 0, (size_t)C * (size_t)T * sizeof(float));
      std::vector<float> tmp(cur.size());
      #pragma omp parallel for schedule(static)
      for (int t_ = 0; t_ < T; ++t_)
        for (int c = 0; c < C; ++c)
          tmp[(size_t)c * T + t_] = cur[(size_t)t_ * C + c];
      cur = std::move(tmp);
    }
    lux_dump("in_proj", cur.data(), cur.size());
  }

  // zipformer.py L276-282 (time embedding chain, fm_decoder only)
  std::vector<float> te;
  if (cfg.use_time_embed) {
    const int D = cfg.time_embed_dim;
    std::vector<float> tin = timestep_embed(t, D);
    if (cfg.use_guidance_embed) {
      std::vector<float> gin = timestep_embed(guidance_scale, D);
      // zipformer.py L282 (guidance_scale_embed) + scaling.py (ScaledLinear: 0.1 baked)
      std::vector<float> gwf = weight_f32(W("guidance_scale_embed.weight"), D, D);
      const float* lw = gwf.data();
      #pragma omp parallel for schedule(static)
      for (int o = 0; o < D; ++o) {
        float acc = 0.0f;
        for (int i = 0; i < D; ++i) acc += lw[(size_t)o * D + i] * gin[i];
        tin[o] += acc;
      }
    }
    std::vector<float> w0f = weight_f32(W("time_embed.0.weight"), D, 2 * D);
    const float* w0 = w0f.data();
    const float* b0 = (const float*)W("time_embed.0.bias")->data;
    std::vector<float> h1(2 * D);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < 2 * D; ++o) {
      float acc = b0[o];
      for (int i = 0; i < D; ++i) acc += w0[(size_t)o * D + i] * tin[i];
      h1[o] = acc;
    }
    for (auto& v : h1) {  // scaling.py L1179 (SwooshR)
      const float y = v - 1.0f;
      const float l = y > 0.0f ? y + std::log1p(std::exp(-y)) : std::log1p(std::exp(y));
      v = l - 0.08f * v - 0.313261687f;
    }
    std::vector<float> w1f = weight_f32(W("time_embed.2.weight"), 2 * D, D);
    const float* w1 = w1f.data();
    const float* b1 = (const float*)W("time_embed.2.bias")->data;
    te.resize(D);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < D; ++o) {
      float acc = b1[o];
      for (int i = 0; i < 2 * D; ++i) acc += w1[(size_t)o * (2 * D) + i] * h1[i];
      te[o] = acc;
    }
  }

  // zipformer.py L709 (Zipformer2Encoder.forward: the stacks)
  int T_cur = T;
  const int n_stacks = (int)cfg.downsample_factor.size();
  // zipformer.py L830 (DownsampledZipformer2Encoder.forward: pending out_combiner of the
// previous ds>1 stack — its post-layers output was upsampled host-side into `cur`;
// the bypass combiner (orig, up, scale) is fused into the CONSUMING graph;
// ggml v0.24.0's view assert makes an in-graph zero-stride upsample inexpressible)
  std::vector<float> comb_orig;
  std::vector<float> comb_up;   // the upsampled output of the previous ds>1 stack
  std::string comb_w;
  int comb_s = -1, comb_T = 0;
  for (int s = 0; s < n_stacks; ++s) {
    const int ds = cfg.downsample_factor[s];
    const int L = cfg.num_layers[s];
    const int K = cfg.cnn_kernel[s];
    const std::string sp = "encoders." + std::to_string(s) + ".";
    // zipformer.py L814 (DownsampledZipformer2Encoder: time_emb/layers use the inner prefix;
    // downsample/out_combiner stay at the stack level)
    const std::string ip = (ds > 1 ? sp + "encoder." : sp);

    // zipformer.py L830-858 (src_orig = src: if this stack's input is a fused
    // out_combiner from the previous ds>1 stack, the input for THIS stack's own
    // out_combiner is that fused combiner output — not the stale pre-combiner
    // host `cur`. Materialize it host-side for the comb_orig save below.)
    const bool had_comb = (comb_s >= 0);
    std::vector<float> fused_input;
    if (had_comb) {
      const float* scw = (const float*)W(comb_w)->data;
      fused_input.resize((size_t)C * (size_t)T_cur);
      #pragma omp parallel for schedule(static)
      for (int t = 0; t < T_cur; ++t)
        for (int c = 0; c < C; ++c) {
          const size_t i = (size_t)c * (size_t)T_cur + t;
          fused_input[i] = comb_orig[i] + (comb_up[i] - comb_orig[i]) * scw[c];
        }
    }

    // zipformer.py L894 (SimpleDownsample.forward: right-pad the last frame to a multiple of ds,
    // group the padded time axis into (T_ds, ds) contiguous blocks, and weighted-sum the ds
    // sub-frames with weights = softmax(bias); kept host-side: the (ds, C*T_ds) ggml view
    // from permute+reshape carries inconsistent strides, so the weighted sum is
    // done directly over the padded channel-major buffer)
    std::vector<float> orig_cur;
    std::vector<float> wds;
    int T_ds = T_cur;
    if (ds > 1) {
      orig_cur = cur;
      T_ds = (T_cur + ds - 1) / ds;
      const int Tp = T_ds * ds;
      // zipformer.py L894 (right-pad `cur` (channel-major c*T_cur + t) to Tp frames, repeating the
      // last frame: src_extra = src[-1:].expand(pad, ...))
      std::vector<float> padded((size_t)C * Tp);
      #pragma omp parallel for schedule(static)
      for (int c = 0; c < C; ++c)
        for (int i = 0; i < Tp; ++i)
          padded[(size_t)c * Tp + i] =
              cur[(size_t)c * T_cur + ((i < T_cur) ? i : T_cur - 1)];
      // zipformer.py L894 (softmax(bias) over the ds sub-frame axis)
      const float* bd = (const float*)W(sp + "downsample.bias")->data;
      wds.resize(ds);
      {
        float mx = -1e30f;
        for (int i = 0; i < ds; ++i) mx = std::max(mx, bd[i]);
        float sum = 0.0f;
        for (int i = 0; i < ds; ++i) { wds[i] = std::exp(bd[i] - mx); sum += wds[i]; }
        for (auto& v : wds) v /= sum;
      }
      // zipformer.py L894 (weighted sum: cur[c*T_ds + g] = sum_m wds[m] * padded[c*Tp + g*ds + m])
      cur.resize((size_t)C * T_ds);
      #pragma omp parallel for schedule(static)
      for (int c = 0; c < C; ++c)
        for (int g = 0; g < T_ds; ++g) {
          float acc = 0.0f;
          for (int m = 0; m < ds; ++m)
            acc += wds[m] * padded[(size_t)c * Tp + (size_t)g * ds + m];
          cur[(size_t)c * T_ds + g] = acc;
        }
    }

    // zipformer.py L709 (Zipformer2Encoder.forward: the layer stack at T_ds)
    {
      ggml_reset(cctx);
      const int h3 = 3 * C / 4;
      const int S = T_ds;

      // zipformer.py L276 (stack-level time embedding, host-side: SwooshR(te) -> Linear(D -> C))
      std::vector<float> tvec;
      if (cfg.use_time_embed) {
        const int D = cfg.time_embed_dim;
        std::vector<float> tew1 = weight_f32(W(ip + "time_emb.1.weight"), D, C);
        const float* w1 = tew1.data();
        const float* b1 = (const float*)W(ip + "time_emb.1.bias")->data;
        tvec.assign((size_t)C, 0.0f);
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < C; ++c) {
          float acc = b1[c];
          // zipformer.py L684 (per-stack time_emb = Sequential(SwooshR(), Linear):
          // the SwooshR MUST be applied to the top-level time_embed output `te`
          // before this per-stack linear — the port skipped it).
          for (int d = 0; d < D; ++d) acc += w1[(size_t)c * D + d] * swoosh_r_host(te[d]);
          tvec[c] = acc;
        }
      }

      // zipformer.py L496 (attention pad mask: -1000 on padded sources; downsampled stacks
      // stride the mask by ds: src_key_padding_mask[..., ::ds])
      std::vector<float> amask(S, 0.0f);
      if (pad_mask)
        for (int i = 0; i < S; ++i)
          if (pad_mask[std::min(i * ds, T - 1)]) amask[i] = -1000.0f;

      std::vector<float> pe;
      {
        pe = compact_rel_pe(S, cfg.pos_dim);
      }
      // ggml v0.24.0 (no_alloc leaves are created UNFILLED: the host->ggml
      // data moves happen via ggml_backend_tensor_set AFTER the scheduler's
      // alloc_graph assigns the leaf data (the reference whisper.cpp pattern))
      std::vector<leaf_fill> fills;
      // ggml v0.24.0 (deque: push_back never reallocates existing elements, so the
      // .data() pointers stored in `fills` stay valid for the whole graph)
      std::deque<std::vector<float>> tbufs;      // host buffers for transposed fills
      std::deque<std::vector<float>> mask_bufs;  // conv mask 0/1 values
      std::deque<float> consts;                   // no_alloc scalar constants (ggml_new_f32 stand-in)
      auto fill_t2d = [&](ggml_tensor* t2, const std::vector<float>& v, int rows) {
        // ggml v0.24.0 (the column count is the leaf's own ne[1]: (rows, S) for up2d
        // and (C, Tc) for the combiner leaves — not the stack's S)
        const int cols = (int)t2->ne[1];
        std::vector<float> tb((size_t)rows * cols);
        #pragma omp parallel for schedule(static)
        for (int t_ = 0; t_ < cols; ++t_)
          for (int r = 0; r < rows; ++r)
            tb[(size_t)t_ * rows + r] = v[(size_t)r * cols + t_];
        tbufs.push_back(std::move(tb));
        fills.push_back({t2, tbufs.back().data(), (size_t)rows * cols});
      };
      // ggml v0.24.0 (host->ggml channel-major transfer helper: v (rows, S) channel-major
      // (v[r*S + t]) -> ggml (rows, S) leaf (byte t*rows + r); the transposed fill is
      // collected for a post-alloc ggml_backend_tensor_set)
      auto up2d = [&](const std::vector<float>& v, int rows) -> ggml_tensor* {
        ggml_tensor* t2 = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, rows, S);
        if (!t2) return nullptr;
        fill_t2d(t2, v, rows);
        return t2;
      };
      auto up1d = [&](const std::vector<float>& v) -> ggml_tensor* {
        ggml_tensor* t2 = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, (int)v.size());
        if (!t2) return nullptr;
        fills.push_back({t2, v.data(), v.size()});
        return t2;
      };
      auto with_tvec = [&](ggml_tensor* x) -> ggml_tensor* {
        return tvec.empty() ? x : ggml_add(cctx, x, rows_view(cctx, up1d(tvec)));
      };

      // zipformer.py L830 (the previous ds>1 stack's out_combiner is fused into
      // THIS graph (up + orig leaves + bypass); else the plain (C, S) leaf;
      // every layer after the first consumes the previous layer's x_out node;
      // a downsampled (ds > 1) stack additionally runs the SimpleDownsample IN-GRAPH
      // on the combiner output, so x0 is (C, S))
      ggml_tensor* x0;
      if (comb_s >= 0) {
        const int Tc = comb_T;
        comb_s = -1;
        ggml_tensor* up_l = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, C, Tc);
        if (!up_l) return {};
        ggml_tensor* orig_l = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, C, Tc);
        if (!orig_l) return {};
        // ggml v0.24.0 (unfilled leaves: the transposed (C, Tc) fills are collected for a
        // post-alloc ggml_backend_tensor_set)
        fill_t2d(up_l, comb_up, C);
        fill_t2d(orig_l, comb_orig, C);
        x0 = bypass(cctx, orig_l, up_l, W(comb_w));
        if (!x0) return {};
        // zipformer.py L894 (the SimpleDownsample must also apply to the fused-combiner output —
        // zipformer.py L858 runs it INSIDE the encoder, before the layers, so the layers consume the
        // downsampled bypass output, not the full-length one; in-graph mirror of the host pass above)
        if (ds > 1) {
          const int Tp = T_ds * ds;
          ggml_tensor* xb = x0;
          if (Tp > Tc) {
            ggml_tensor* last = ggml_view_2d(cctx, xb, C, 1, xb->nb[1],
                                              (size_t)(Tc - 1) * xb->nb[1]);
            if (!last) return {};
            ggml_tensor* pad = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, C, Tp - Tc);
            if (!pad) return {};
            pad = ggml_repeat(cctx, last, pad);
            if (!pad) return {};
            xb = ggml_concat(cctx, xb, pad, 1);
            if (!xb) return {};
          }
          xb = ggml_reshape_3d(cctx, xb, C, ds, T_ds);
          if (!xb) return {};
          xb = ggml_permute(cctx, xb, 1, 0, 2, 3);
          if (!xb) return {};
          xb = ggml_cont(cctx, xb);
          if (!xb) return {};
          ggml_tensor* wl = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, ds);
          if (!wl) return {};
          fills.push_back({wl, wds.data(), (size_t)ds});
          xb = ggml_mul(cctx, xb, ggml_reshape_3d(cctx, wl, ds, 1, 1));
          if (!xb) return {};
          x0 = ggml_reshape_2d(cctx, ggml_cont(cctx, ggml_sum_rows(cctx, xb)), C, T_ds);
          if (!x0) return {};
        }
      } else {
        x0 = up2d(cur, C);
        if (!x0) return {};
      }

      for (int li = 0; li < L; ++li) {
        const std::string lp = ip + "layers." + std::to_string(li) + ".";

        // zipformer.py L496 (G-A nodes (in-graph): qkv, x_ff1, na_in)
        ggml_tensor* qkv = lin(cctx, x0, W(lp + "self_attn_weights.in_proj.weight"),
                               C, in_proj_dim, W(lp + "self_attn_weights.in_proj.bias"));
        if (!qkv) return {};
        ggml_tensor* x_add = with_tvec(x0);
        ggml_tensor* ff1_out = ff_block(cctx, x_add, C,
                     W(lp + "feed_forward1.in_proj.weight"),
                     W(lp + "feed_forward1.in_proj.bias"),
                     W(lp + "feed_forward1.out_proj.weight"),
                     W(lp + "feed_forward1.out_proj.bias"),
                     (ff * 3) / 4, &consts, &fills);
        if (!ff1_out) return {};
        ggml_tensor* x_ff1 = ggml_add(cctx, x_add, ff1_out);
        ggml_tensor* na_in = lin(cctx, x_ff1, W(lp + "nonlin_attention.in_proj.weight"),
                                 C, 3 * h3, W(lp + "nonlin_attention.in_proj.bias"));
        if (!na_in) return {};
        // zipformer.py L1156 (G-B nodes (in-graph): in-graph rel-shift attention + head-0
        // nonlin branch. q/k/p are 2-D views of the qkv leaf; the rel-pos window is built
        // in-graph (pe table -> linear_pos -> the rel-shift chain), so no host addend,
        // flip, or chunking; position table: peu (pos_dim, P) leaf — bytes are the pe table
        // (compact_rel_pe output, (r*pos_dim+e) layout), a plain memcpy)
        const int P = 2 * S - 1;
        ggml_tensor* peu = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, cfg.pos_dim, P);
        if (!peu) return {};
        // peu is a (pos_dim, P) [c, r] leaf (c innermost); the pe table is stored
        // (P, pos_dim) [r, c] row-major, so the fill must TRANSPOSE: peu[c, r] = pe[r, c]
        std::vector<float> pe_t((size_t)P * cfg.pos_dim);
        for (int r = 0; r < P; ++r)
          for (int c = 0; c < cfg.pos_dim; ++c)
            pe_t[(size_t)c * P + r] = pe[(size_t)r * (cfg.pos_dim + 1) + c];
        tbufs.push_back(std::move(pe_t));
        fills.push_back({peu, tbufs.back().data(), (size_t)P * cfg.pos_dim});
        // zipformer.py L1156 (ph_t = linear_pos(peu) (no bias) -> (H*dp, P); head-split to (dp,P,H))
        ggml_tensor* ph_t = lin(cctx, peu, W(lp + "self_attn_weights.linear_pos.weight"),
                                cfg.pos_dim, H * dp, nullptr);
        if (!ph_t) return {};
        // ggml v0.24.0 (ph_t3 = projected rel-pos table (dp, P, H): reshape (H*dp, P) -> (dp, H, P)
        // then permute (0,2,1,3) -> (dp, P, H), so bd = mul_mat(ph_t3 (dp,P,H), ph_q (dp,S,H)) -> (P,S,H)
        // (contract over dp, zipformer.py L1231 pos_scores = p @ pos_emb))
        // ph_t (H*dp, P) is torch head-major channel o = h*dp+d; a head-major (dp, P, H)
        // [d, r, h] leaf ([d, r, h] = ph_t[h*dp + d, r]) feeds the batched mul_mat -> (P, S, H).
        // The ggml_view_3d head-split has an nb-swap bug (crashes on the corrected strides),
        // so compute ph_t = linear_pos @ pe host-side and upload the (dp, P, H) leaf directly.
        {
          const float* lpw = (const float*)W(lp + "self_attn_weights.linear_pos.weight")->data;
          std::vector<float> ph_t3_buf((size_t)dp * P * H);
          #pragma omp parallel for schedule(static)
          for (int h = 0; h < H; ++h)
            for (int r = 0; r < P; ++r)
              for (int d = 0; d < dp; ++d) {
                const int c = h * dp + d;
                float acc = 0.0f;
                for (int i = 0; i < cfg.pos_dim; ++i)
                  acc += lpw[(size_t)c * cfg.pos_dim + i] * pe[(size_t)r * (cfg.pos_dim + 1) + i];
                ph_t3_buf[((size_t)h * P + r) * dp + d] = acc;
              }
          tbufs.push_back(std::move(ph_t3_buf));
        }
        ggml_tensor* ph_t3 = ggml_new_tensor_3d(cctx, GGML_TYPE_F32, dp, P, H);
        if (!ph_t3) return {};
        fills.push_back({ph_t3, tbufs.back().data(), (size_t)dp * P * H});
        GZ_ASSERT_SHAPE("ph_t3", ph_t3, dp, P, H, -1);
        // ggml v0.24.0 (q/k/p head-split: qkv is (in_proj_dim, S) t-major (nb0=4 c-stride,
        // nb1=in_proj_dim*4 t-stride); the torch channel order is head-major
        // c = sec_ch + h*dper + d, so a (dper, S, H) [d, t, h] VIEW reads
        // [d, t, h] = qkv[sec_ch + h*dper + d, t] (d stride nb0=4, t stride nb1,
        // h stride dper*nb0, byte offset sec_ch*nb0). A single direct view — no
        // transpose/cont/permute chain (the chain mis-strided the section slice).)
        auto heads_of = [&](int sec_ch, int nch, int dper) -> ggml_tensor* {
          // (dper, S, H) [d, t, h] reads qkv[sec_ch + h*dper + d, t]:
          //   nb0 (d) = qkv->nb[0], nb1 (t) = qkv->nb[1], nb2 (h) = dper*nb[0],
          //   base offset = sec_ch*nb[0].  ggml_view_3d HAS a base-offset arg (last).
          // A single DIRECT view of qkv (a graph node) — qkv->n_views is incremented so
          // qkv stays live in the gallocr liveness loop (a nested view_1d sub-view was
          // never in the graph node list, so qkv was freed early and its buffer reused).
          return ggml_view_3d(cctx, qkv, dper, S, H,
                              qkv->nb[1], (size_t)dper * qkv->nb[0], (size_t)sec_ch * qkv->nb[0]);
        };
        ggml_tensor* qh   = heads_of(0, H * dq, dq);
        ggml_tensor* kh   = heads_of(H * dq, H * dq, dq);
        ggml_tensor* ph_q = heads_of(2 * H * dq, H * dp, dp);
        GZ_ASSERT_SHAPE("qh", qh, dq, S, H, -1);
        GZ_ASSERT_SHAPE("kh", kh, dq, S, H, -1);
        GZ_ASSERT_SHAPE("ph_q", ph_q, dp, S, H, -1);
        // zipformer.py L1238 (bd = ph_t3 (dp,P,H) @ ph_q (dp,S,H) -> (P, S, H) [r, i, h];
        // the rel-shift (as_strided) converts the (P,) relative axis to the (S,S) absolute
        // window: window[i, j, h] = scores_flat[(S-1) + i*(S-1) + j + h*P*S] — a strided view
        // of the (P,S,H) scores with ne=(S,S,H), nb=(1,(S-1),P*S) [in elements], offset=S-1.
        // ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset) — nb0 is the element size.
        ggml_tensor* scores = ggml_cont(cctx, ggml_mul_mat(cctx, ph_t3, ph_q));  // (P, S, H) contiguous
        GZ_ASSERT_SHAPE("scores", scores, P, S, H, -1);
        // rel-shift (zipformer.py L1246 as_strided): bd[j,i,h] = scores[j-i+(S-1), i, h].
        // scores (P, S, H) [r, i, h] is at (r + i*P + h*P*S)*4 bytes; the rel-shift view
        // (S, S, H) [j, i, h] reads scores[j-i+(S-1), i, h] with nb1=(P-1)*4 (i stride),
        // nb2=P*S*4 (h stride), offset=(S-1)*4 (the as_strided storage_offset).
        ggml_tensor* bd = ggml_view_3d(cctx, scores, S, S, H,
                                        (size_t)(P - 1) * sizeof(float),  // nb1: i stride
                                        (size_t)P * S * sizeof(float),   // nb2: h stride
                                        (size_t)(S - 1) * sizeof(float)); // offset
        GZ_ASSERT_SHAPE("bd", bd, S, S, H, -1);
        // ggml v0.24.0 (scores = qk + rel-pos window + mask; softmax over the source axis i (ne[1]))
        ggml_tensor* ac = ggml_mul_mat(cctx, kh, qh);
        GZ_ASSERT_SHAPE("ac", ac, S, S, H, -1);
        ggml_tensor* masku = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, S);
        if (!masku) return {};
        fills.push_back({masku, amask.data(), (size_t)S});
        ggml_tensor* maskb = ggml_reshape_3d(cctx, masku, S, 1, 1);
        ggml_tensor* pre_softmax = ggml_add(cctx, ggml_add(cctx, ac, bd), maskb);
        // ggml_soft_max normalizes over ne[0] = source (j); pre_softmax is (S_src, S_tgt, H)
        ggml_tensor* attn_t = ggml_soft_max(cctx, pre_softmax);

        // zipformer.py L1506 (NonlinAttention.forward: each head h's (S,S) attention matrix is
        // applied to its own head_dim slice of the h3 channels; ns/nxc/nyc are the s/x/y
        // channel row-views of na_in (3h3, S); nx = x*tanh(s); nxT (S, h3) [i, c] with
        // c = h*hd+d -> nx3 (S, hd, H) [i, d, h]; na_mid3 = attn_t (S,S,H) [i,t,h] @ nx3
        // -> (S, hd, H) [t, d, h]; permute (t,d,h)->(t,h,d) and reshape to (S, h3) [t, c])
        auto na_row = [&](int r0) {
          ggml_tensor* vw = ggml_view_3d(cctx, na_in, h3, S, 1, (size_t)3 * h3 * 4, 0, (size_t)r0 * 4);
          return ggml_reshape_2d(cctx, ggml_cont(cctx, vw), h3, S);
        };
        ggml_tensor* ns   = na_row(0);
        ggml_tensor* nxc  = na_row(h3);
        ggml_tensor* nyc  = na_row(2 * h3);
        ggml_tensor* nx   = ggml_mul(cctx, nxc, ggml_tanh(cctx, ns));
        ggml_tensor* nxT  = ggml_cont(cctx, ggml_transpose(cctx, nx));
        // zipformer.py L1506 + layer forward (attn_weights[0:1] passed to NonlinAttention):
        // HEAD 0's (S,S) attention matrix is applied to ALL h3 channels; the port
        // wrongly applied per-head slices of attn_t to per-head channel slices.
        ggml_tensor* attn0 = ggml_view_3d(cctx, attn_t, S, S, 1,
                                          attn_t->nb[1], attn_t->nb[2], 0);
        ggml_tensor* na_mid2 = ggml_mul_mat(cctx, attn0, nxT);     // (S, h3) [t, c]
        ggml_tensor* na_mid  = ggml_cont(cctx, na_mid2);
        ggml_tensor* yT   = ggml_cont(cctx, ggml_transpose(cctx, nyc));
        ggml_tensor* na_xr = ggml_mul(cctx, na_mid, yT);
        // zipformer.py L1506 (na_out = nonlin.out_proj(na_xr): (S, h3) -> (h3, S) cont, then lin)
        ggml_tensor* na_xr_t = ggml_cont(cctx, ggml_transpose(cctx, na_xr));
        ggml_tensor* na_out = lin(cctx, na_xr_t,
                                  W(lp + "nonlin_attention.out_proj.weight"),
                                  h3, C, W(lp + "nonlin_attention.out_proj.bias"));
        if (!na_out) return {};
        ggml_tensor* x_dr = ggml_add(cctx, x_ff1, na_out);
        ggml_tensor* v1 = lin(cctx, x_dr, W(lp + "self_attn1.in_proj.weight"),
                               C, dv * H, W(lp + "self_attn1.in_proj.bias"));
        if (!v1) return {};
        // attn_t stays LIVE in-graph (consumed by G-C1/G-D1); no ggml_reset.
        // zipformer.py L496 (G-C1 nodes (in-graph): av1 = attn @ v1, x_F, x_G)
        // ggml v0.24.0 (v1 (dvH, S) t-major; the batched mul_mat needs the time dim as ne0
        // (the dot dim), so v1b3 is (S, dv, H) [j, d, h] with j = ne0: transpose v1 to a
        // contiguous (S, dvH) t-major leaf, then a value-preserving reshape to (S, dv, H)
        // [j, d, h] (c = h*dv + d head-major). The kernel handles the non-contiguous dot dim.)
        ggml_tensor* v1b3 = ggml_reshape_3d(cctx, ggml_cont(cctx, ggml_transpose(cctx, v1)), S, dv, H);
        GZ_ASSERT_SHAPE("v1b3", v1b3, S, dv, H, -1);
        ggml_tensor* av1 = ggml_mul_mat(cctx, attn_t, v1b3);
        GZ_ASSERT_SHAPE("av1", av1, S, dv, H, -1);
        // ggml v0.24.0 (av1 (S, dv, H) [i, d, h] -> (dvH, S) ch-major [c, i] with c = h*dv + d:
        // permute to (H, dv, S) [h, d, i] (source d->pos1, i->pos2, h->pos0), cont, then a
        // value-preserving reshape to (dvH, S). The (dvH, S) leaf feeds the out_proj lin.)
        ggml_tensor* av1u = ggml_reshape_2d(cctx, ggml_cont(cctx,
                          ggml_permute(cctx, av1, 2, 0, 1, 3)), dv * H, S);
        ggml_tensor* x_F = ggml_add(cctx, x_dr,
            lin(cctx, av1u, W(lp + "self_attn1.out_proj.weight"),
                dv * H, C, W(lp + "self_attn1.out_proj.bias")));
        if (!x_F) return {};
        ggml_tensor* x_G = with_tvec(x_F);
        // zipformer.py L1645 (conv1 runs IN-GRAPH on cctx, natural order)
        ggml_tensor* conv1;
        ggml_tensor* cmask1 = nullptr;
        {
          conv1 = conv_module_ggml(cctx, x_G,
            W(lp + "conv_module1.in_proj.weight"),
            W(lp + "conv_module1.in_proj.bias"),
            W(lp + "conv_module1.depthwise_conv.weight"),
            W(lp + "conv_module1.depthwise_conv.bias"),
            W(lp + "conv_module1.out_proj.weight"),
            W(lp + "conv_module1.out_proj.bias"),
            C, K, S, amask.data(), &cmask1, &mask_bufs, &consts, &fills);
        }
        if (!conv1) return {};
        if (cmask1) fills.push_back({cmask1, mask_bufs.back().data(), (size_t)S});
        ggml_tensor* x_H = ggml_add(cctx, x_G, conv1);
        // zipformer.py L496 (G-C2 nodes (in-graph): ff2 -> bypass_mid -> v2)
        ggml_tensor* ff2_out = ff_block(cctx, x_H, C,
                     W(lp + "feed_forward2.in_proj.weight"),
                     W(lp + "feed_forward2.in_proj.bias"),
                     W(lp + "feed_forward2.out_proj.weight"),
                     W(lp + "feed_forward2.out_proj.bias"),
                     ff, &consts, &fills);
        if (!ff2_out) return {};
        ggml_tensor* x_I = ggml_add(cctx, x_H, ff2_out);
        ggml_tensor* x_J = bypass(cctx, x0, x_I, W(lp + "bypass_mid.bypass_scale"));
        ggml_tensor* v2 = lin(cctx, x_J, W(lp + "self_attn2.in_proj.weight"),
                               C, dv * H, W(lp + "self_attn2.in_proj.bias"));
        if (!v2) return {};

        // zipformer.py L496 (G-D1 nodes (in-graph, attn from G-B still alive): av2, x_M
        // v2 (dvH, S) t-major; (S, dv, H) [j, d, h] view (time dim = ne0, the mul_mat dot dim))
        ggml_tensor* v2b3 = ggml_reshape_3d(cctx, ggml_cont(cctx, ggml_transpose(cctx, v2)), S, dv, H);
        GZ_ASSERT_SHAPE("v2b3", v2b3, S, dv, H, -1);
        ggml_tensor* av2 = ggml_mul_mat(cctx, attn_t, v2b3);
        GZ_ASSERT_SHAPE("av2", av2, S, dv, H, -1);
        ggml_tensor* av2u = ggml_reshape_2d(cctx, ggml_cont(cctx,
                          ggml_permute(cctx, av2, 2, 0, 1, 3)), dv * H, S);
        ggml_tensor* x_L = ggml_add(cctx, x_J,
            lin(cctx, av2u, W(lp + "self_attn2.out_proj.weight"),
                dv * H, C, W(lp + "self_attn2.out_proj.bias")));
        if (!x_L) return {};
        ggml_tensor* x_M = with_tvec(x_L);
        // zipformer.py L1645 (conv2 runs IN-GRAPH on cctx in natural order)
        ggml_tensor* conv2;
        ggml_tensor* cmask2 = nullptr;
        {
          conv2 = conv_module_ggml(cctx, x_M,
            W(lp + "conv_module2.in_proj.weight"),
            W(lp + "conv_module2.in_proj.bias"),
            W(lp + "conv_module2.depthwise_conv.weight"),
            W(lp + "conv_module2.depthwise_conv.bias"),
            W(lp + "conv_module2.out_proj.weight"),
            W(lp + "conv_module2.out_proj.bias"),
            C, K, S, amask.data(), &cmask2, &mask_bufs, &consts, &fills);
        }
        if (!conv2) return {};
        if (cmask2) fills.push_back({cmask2, mask_bufs.back().data(), (size_t)S});
        ggml_tensor* x_N = ggml_add(cctx, x_M, conv2);

        // zipformer.py L496 (G-D2 nodes (in-graph; no ggml_reset): ff3 -> norm -> bypass)
        ggml_tensor* ff3_out = ff_block(cctx, x_N, C,
                     W(lp + "feed_forward3.in_proj.weight"),
                     W(lp + "feed_forward3.in_proj.bias"),
                     W(lp + "feed_forward3.out_proj.weight"),
                     W(lp + "feed_forward3.out_proj.bias"),
                     (ff * 5) / 4, &consts, &fills);
        if (!ff3_out) return {};
        ggml_tensor* x_O = ggml_add(cctx, x_N, ff3_out);
        ggml_tensor* x_out = bypass(cctx, x0,
            bias_norm(cctx, x_O, W(lp + "norm.bias"), W(lp + "norm.log_scale"), C, &consts, &fills),
            W(lp + "bypass.bypass_scale"));
        x0 = x_out;
      }

      // ggml v0.24.0 (ONE graph compute for the whole stack: the ds>1 upsample/out_combiner
      // is NOT in this graph — the view assert makes the zero-stride upsample inexpressible,
      // so the graph ends at x0 (C, T_ds); the upsample is a host data-move fill and the
      // out_combiner bypass is fused into the CONSUMING graph (zipformer.py L830))
      {
        ggml_cgraph* g = ggml_new_graph_custom(cctx, L * 512 + 256, false);
        ggml_build_forward_expand(g, x0);
        ggml_set_output(x0);
        for (auto& f : fills) zprebind_t(f.t);
        if (!zalloc(g)) return {};
        for (auto& f : fills) ggml_backend_tensor_set(f.t, f.src, 0, f.nelem * 4);
        if (!zcompute(g)) return {};
      }
      // download the post-stack result (C, T_ds) ch-major.
      {
        cur.resize((size_t)C * T_ds);
        ggml_backend_tensor_get(x0, cur.data(), 0, (size_t)C * (size_t)T_ds * 4);
        std::vector<float> tmp(cur.size());
        #pragma omp parallel for schedule(static)
        for (int t_ = 0; t_ < T_ds; ++t_)
          for (int c = 0; c < C; ++c)
            tmp[(size_t)c * T_ds + t_] = cur[(size_t)t_ * C + c];
        cur = std::move(tmp);
      }
      if (ds > 1) {
        // zipformer.py L932 (SimpleUpsample.forward: host data-move fill —
        // up[c, t] = x0[c, t/ds] — repeat each frame ds times, trim to the
        // running T_cur; the out_combiner bypass is fused into the consuming graph)
        std::vector<float> up((size_t)C * T_cur);
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T_cur; ++t)
          for (int c = 0; c < C; ++c)
            up[(size_t)c * T_cur + t] = cur[(size_t)c * T_ds + t / ds];
        cur = std::move(up);
        comb_up = cur;   // save the upsampled output for the next stack's out_combiner
        comb_orig = had_comb ? fused_input : std::move(orig_cur);
        comb_w = sp + "out_combiner.bypass_scale";
        comb_s = s;
        comb_T = T_cur;
      }
      lux_dump(("stack" + std::to_string(s) + "_out").c_str(), cur.data(), cur.size());
    }
  }

  // zipformer.py L249 (out_proj -> (T, out_dim): the last stack's out_combiner (if any)
  // is fused into THIS graph (zipformer.py L830): the upsampled `cur` and comb_orig
  // feed the bypass, whose output feeds out_proj)
  std::vector<float> out((size_t)T * cfg.out_dim);
  {
    ggml_reset(cctx);
    // ggml v0.24.0 (no_alloc leaves created UNFILLED; the host->ggml data moves
    // happen via ggml_backend_tensor_set AFTER alloc_graph (the reference pattern))
    std::vector<leaf_fill> fills;
    std::deque<std::vector<float>> tbufs;
    auto fill_t2d = [&](ggml_tensor* t2, const std::vector<float>& v, int rows, int cols) {
      std::vector<float> tb((size_t)rows * cols);
      #pragma omp parallel for schedule(static)
      for (int t_ = 0; t_ < cols; ++t_)
        for (int r = 0; r < rows; ++r)
          tb[(size_t)t_ * rows + r] = v[(size_t)r * cols + t_];
      tbufs.push_back(std::move(tb));
      fills.push_back({t2, tbufs.back().data(), (size_t)rows * cols});
    };
    ggml_tensor* xt;
    ggml_tensor* y0 = nullptr;
    if (comb_s >= 0) {
      ggml_tensor* up_l = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, C, comb_T);
      if (!up_l) return {};
      ggml_tensor* orig_l = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, C, comb_T);
      if (!orig_l) return {};
      // ggml v0.24.0 (unfilled leaves: the transposed (C, comb_T) fills are collected
      // for a post-alloc ggml_backend_tensor_set)
      fill_t2d(up_l, cur, C, comb_T);
      fill_t2d(orig_l, comb_orig, C, comb_T);
      y0 = bypass(cctx, orig_l, up_l, W(comb_w));
      if (!y0) return {};
      xt = y0;
    } else {
      xt = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, C, T);
      if (!xt) return {};
      fill_t2d(xt, cur, C, T);
    }
    ggml_tensor* y = lin(cctx, xt, W("out_proj.weight"), C, cfg.out_dim, W("out_proj.bias"));
    if (!y) return {};
    ggml_cgraph* g = ggml_new_graph_custom(cctx, 4096, false);
    ggml_build_forward_expand(g, y);
    ggml_set_output(y);
    for (auto& f : fills) zprebind_t(f.t);
    if (!zalloc(g)) return {};
    for (auto& f : fills) ggml_backend_tensor_set(f.t, f.src, 0, f.nelem * 4);
    if (!zcompute(g)) return {};
    if (y0) {
      // ggml v0.24.0 (cfin is the t-major download of y0 (C, comb_T); unused downstream,
      // kept for parity with the legacy manual-transpose path)
      std::vector<float> cfin((size_t)C * comb_T);
      ggml_backend_tensor_get(y0, cfin.data(), 0, (size_t)C * (size_t)comb_T * 4);
    }
    // ggml v0.24.0 (y is a contiguous (out_dim, T) t-major leaf: element (c, t) at
    // t*out_dim + c — the SAME flat order as out (T, out_dim) row-major [t][c] ->
    // a single byte-identical memcpy)
    ggml_backend_tensor_get(y, out.data(), 0, (size_t)T * (size_t)cfg.out_dim * sizeof(float));
  }
  lux_dump("v", out.data(), out.size());
  return out;
}

}
