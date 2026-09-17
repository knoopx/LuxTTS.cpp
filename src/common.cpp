// common.cpp — shared primitives for the LuxTTS port.
#include "common.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <random>

// Real ggml engine types: the gb helpers below bind to the global-scope ggml
// API (ggml v0.24.0 defines ggml_context / ggml_tensor / ggml_cgraph and the
// ggml_* functions at global scope; ggml ships flat headers: <ggml.h>).
#include <ggml.h>

namespace lux {

// Rng — exact reproduction of PyTorch's CPU RNG (at::mt19937_engine,
// pytorch_pytorch/aten/src/ATen/core/MT19937RNGEngine.h L114 ctor / L155-160
// init_with_uint32 seeding). Combined with uniform01() + box_muller16() it
// reproduces torch.randn (CPU) bit-for-bit, so the flow-matching noise matches
// the reference (reproducible audio).
Rng::Rng(unsigned long long seed) : left_(1), next_(0), seed_(seed) {
  state_[0] = (unsigned int)seed;  // MT19937RNGEngine.h init_with_uint32
  for (int j = 1; j < 624; ++j)    // 1812433253 * (s[j-1] ^ (s[j-1]>>30)) + j, 32-bit wrap
    state_[j] = (unsigned int)(1812433253u * (unsigned int)(state_[j - 1] ^
                                                             (state_[j - 1] >> 30)) + j);
}

static inline unsigned int mt_twist(unsigned int u, unsigned int v) {
  // MT19937RNGEngine.h twist: (mix_bits(u,v) >> 1) ^ (v&1 ? MATRIX_A : 0)
  // where mix_bits(u,v) = (u & UMASK) | (v & LMASK); the check bit is v&1.
  const unsigned int m = ((u & 0x80000000u) | (v & 0x7fffffffu)) >> 1;
  return m ^ (v & 1u ? 0x9908b0dfu : 0u);  // MATRIX_A = 0x9908b0df
}

void Rng::twist() {
  // Exact copy of PyTorch MT19937RNGEngine::next_state() (pointer-based loops):
  //   for(j = N-M+1; --j; p++) *p = p[M] ^ twist(p[0], p[1]);   // 227 iters
  //   for(j = M;      --j; p++) *p = p[M-N] ^ twist(p[0], p[1]); // 397 iters
  //   *p = p[M-N] ^ twist(p[0], state_[0]);
  left_ = 624;
  next_ = 0;
  unsigned int* p = state_;
  for (int j = 624 - 397 + 1; --j; p++)
    *p = p[397] ^ mt_twist(p[0], p[1]);
  for (int j = 397; --j; p++)
    *p = p[397 - 624] ^ mt_twist(p[0], p[1]);
  *p = p[397 - 624] ^ mt_twist(p[0], state_[0]);
}

unsigned int Rng::next32() {
  if (--left_ == 0) twist();
  unsigned int y = state_[next_++];
  // MT19937RNGEngine.h operator() tempering
  y ^= y >> 11;
  y ^= (y << 7) & 0x9d2c5680u;
  y ^= (y << 15) & 0xefc60000u;
  y ^= y >> 18;
  return y;
}

// 16-wide Box-Muller in float, matching NormalFill16<opmath_t=float>
// (pytorch_pytorch/aten/src/ATen/native/cpu/DistributionTemplates.h L92;
// L101 u1=1-data[j], L102 u2=data[j+8], L103 radius, L104 theta=2*pi*u2,
// L105/L106 cos/sin): pairs (j, j+8) per block.
// theta is computed from the double product 2*pi*u2 then cast to float, exactly
// as `2.0f * c10::pi<double> * u2` (L104).
static void box_muller16f(float* d) {
  for (int j = 0; j < 8; ++j) {
    const float u1 = 1.0f - d[j];
    const float u2 = d[j + 8];
    const float radius = std::sqrtf(-2.0f * std::logf(u1));
    // PyTorch NormalFill16<opmath_t=float> uses std::cos/std::sin (double),
    // NOT std::cosf/std::sinf (float). The double product 2*pi*u2 is computed
    // in double, then cos/sin are evaluated in double, then cast to float.
    const double theta = 2.0 * M_PI * (double)u2;
    d[j] = (float)(radius * std::cos(theta));
    d[j + 8] = (float)(radius * std::sin(theta));
  }
}

// 24-bit float uniform in [0,1): uniform_real<float>(mt32, 0, 1) =
// (mt32 & (2^24-1)) * 2^-24  (numeric_limits<float>::digits == 24)
// — pytorch_pytorch/aten/src/ATen/core/DistributionsHelper.h L99
// uniform_real_distribution / L109 transformation::uniform_real
static float uniform01f(Rng& rng) {
  return (float)(rng.next32() & 0xffffffu) * (1.0f / 16777216.0f);
}

// Fill `v` with standard-normal noise bit-identical to torch.randn(n) on CPU
// after torch.manual_seed(seed) (float tensor): fill all slots with the 24-bit
// float uniform (one MT output each), Box-Muller the complete 16-blocks, then
// — if n % 16 != 0 — regenerate the last 16 with fresh uniforms.
static void fill_randn(std::vector<float>& v, unsigned long long seed) {
  Rng rng(seed);
  const std::size_t n = v.size();
  if (n == 0) return;
  std::vector<float> d(n);
  for (std::size_t i = 0; i < n; ++i) d[i] = uniform01f(rng);
  for (std::size_t i = 0; i + 15 < n; i += 16) box_muller16f(&d[i]);
  if (n % 16 != 0) {
    float buf[16];
    for (int j = 0; j < 16; ++j) buf[j] = uniform01f(rng);
    box_muller16f(buf);
    for (int j = 0; j < 16; ++j) d[n - 16 + j] = buf[j];
  }
  v = std::move(d);
}

// Seeded-noise fill helpers (deterministic per seed — reproducible audio).
void fill_noise(std::vector<float>& v, unsigned long long seed) { fill_randn(v, seed); }

// Fresh 64-bit seed for unseeded runs (config seed 0): OS entropy + clock so
// successive runs differ (the Python pipeline leaves torch.randn unseeded).
unsigned long long random_seed_u64() {
  std::random_device rd;
  const unsigned long long hi = (unsigned long long)rd();
  const unsigned long long lo = (unsigned long long)rd();
  const unsigned long long t =
      (unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count();
  return (hi << 32) ^ lo ^ t;
}

// ---------------------------------------------------------------------------
// gb — small shared ggml graph-builder helpers (context mgmt + matmul/bias/add).
// Conventions:
//   * feature tensors are (T, C) row-major; ggml 2D tensor layout is
//     (ne0 = in_dim, ne1 = T) so that ggml_mul_mat(W, X) == W @ X.
//   * W is a torch Linear(in, out) weight stored ggml-side as (in, out).
// ---------------------------------------------------------------------------
namespace gb {

// llama_cpp/ggml/include/ggml.h L823 ggml_init / L677 ggml_init_params
ggml_context* new_context(size_t mem_bytes) {
  ggml_init_params p{mem_bytes, nullptr, false};
  return ggml_init(p);
}

// llama_cpp/ggml/include/ggml.h L2889 ggml_new_graph
ggml_cgraph* new_graph(ggml_context* ctx) {
  return ggml_new_graph(ctx);
}

// llama_cpp/ggml/include/ggml.h L2873 ggml_build_forward_expand
void build(ggml_context* ctx, ggml_cgraph* cgraph, ggml_tensor* out) {
  (void)ctx;
  ggml_build_forward_expand(cgraph, out);
}

// llama_cpp/ggml/include/ggml.h L911 ggml_add
ggml_tensor* add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
  return ggml_add(ctx, a, b);
}

// llama_cpp/ggml/include/ggml.h L967 ggml_sub
ggml_tensor* sub(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
  return ggml_sub(ctx, a, b);
}

// llama_cpp/ggml/include/ggml.h L977 ggml_mul
ggml_tensor* mul(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
  return ggml_mul(ctx, a, b);
}

// A (1,) f32 constant tensor; the caller writes its value via ggml_get_data_f32.
// llama_cpp/ggml/include/ggml.h L842 ggml_new_tensor_1d / L390 GGML_TYPE_F32
ggml_tensor* scalar(ggml_context* ctx) {
  return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
}

// llama_cpp/ggml/include/ggml.h L911 (broadcast-add a (cols,) bias row to x (T, cols); ggml_add broadcasts the RHS)
ggml_tensor* add_bias(ggml_context* ctx, ggml_tensor* x, ggml_tensor* bias) {
  return ggml_add(ctx, x, bias);
}

// llama_cpp/ggml/include/ggml.h L1482 ggml_mul_mat (y = W @ x + b with W: (in, out), x: (in, T) -> y: (out, T);
// bias: (out,); bias may be null)
ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x,
                    ggml_tensor* w, ggml_tensor* bias) {
  ggml_tensor* y = ggml_mul_mat(ctx, w, x);
  if (bias) y = ggml_add(ctx, y, bias);  // L911
  return y;
}

}
}
