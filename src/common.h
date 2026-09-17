// common.h — shared primitives for the LuxTTS port.
#ifndef LUXTTS_COMMON_H
#define LUXTTS_COMMON_H

#include <cstddef>
#include <string>
#include <vector>

// Forward-declare the real (global-scope) ggml engine types. ggml v0.24.0
// defines ggml_context / ggml_tensor / ggml_cgraph at global scope (see
// <ggml.h>); the neural-module stubs (zipformer.h, vocos.h) reference them as
// ggml::ggml_context, which a later agent reconciles with the real header. The
// gb graph-builder helpers below use the real global-scope types so they bind
// to the actual ggml API.
struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace lux {

// FMat / Result removed: the port uses std::vector<float> (row-major feature
// tensors, the C++ stand-in for a torch (T, C) tensor) and std::string (empty =
// success, non-empty = error message) directly, so no named wrapper types remain.

// Deterministic CPU RNG matching torch.randn exactly (at::mt19937_engine +
// uniform_real<float> + 16-wide float Box-Muller, per
// aten/src/ATen/core/MT19937RNGEngine.h, CPUGeneratorImpl, DistributionsHelper.h,
// and DistributionTemplates.h). For a float tensor, opmath_t=float, so the
// uniform is 24-bit ((v & (2^24-1)) * 2^-24, since numeric_limits<float>::digits=24)
// and Box-Muller runs in float. Reproducible audio bit-identical to the reference.
class Rng {
 public:
  explicit Rng(unsigned long long seed = 42ull);
  unsigned long long seed() const { return seed_; }
  // Next 32-bit Mersenne-Twister output (tempered), matching at::mt19937::operator().
  unsigned int next32();

 private:
  void twist();
  unsigned int state_[624];
  int left_;
  int next_;
  unsigned long long seed_;
};

// Fill `v` with seeded standard-normal noise (deterministic per seed;
// reproduces torch.manual_seed(seed) + torch.randn).
void fill_noise(std::vector<float>& v, unsigned long long seed);

// Fresh 64-bit seed for unseeded runs (config seed 0): OS entropy + clock.
unsigned long long random_seed_u64();

// ---------------------------------------------------------------------------
// gb — small shared ggml graph-builder helpers for the neural modules.
// Keeps graph boilerplate out of zipformer/vocos. Conventions:
//   * weights are ggml tensors in a data-allocating CPU context (ggml_init
//     with mem), bound by name from the GGUF
//   * feature tensors are (T, C) row-major; ggml 2-D tensors are (ne0, ne1)
//   * `linear`: y = W @ x + b with W (in_dim, out_dim), x (in_dim, T) -> y (out_dim, T);
//     ggml_mul_mat(a,b) computes a @ b over the 2-D leading dims
// All signatures use the real global-scope ggml types (from <ggml.h>).
// ---------------------------------------------------------------------------
namespace gb {

// Allocate a data-allocating CPU ggml context (weights live here).
ggml_context* new_context(size_t mem_bytes);

// Create a forward compute graph on a context (ggml_new_graph).
ggml_cgraph* new_graph(ggml_context* ctx);
// Expand the forward graph with `out` (ggml_build_forward_expand).
void build(ggml_context* ctx, ggml_cgraph* cgraph, ggml_tensor* out);

ggml_tensor* add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);
ggml_tensor* sub(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);
ggml_tensor* mul(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);
// A (1,) f32 constant tensor (caller writes its value via ggml_get_data_f32).
ggml_tensor* scalar(ggml_context* ctx);
// Broadcast-add a (cols,) bias row to x (T, cols); bias broadcasts over T.
ggml_tensor* add_bias(ggml_context* ctx, ggml_tensor* x, ggml_tensor* bias);
// y = W @ x + b  (W: (in, out), x: (in, T) -> y: (out, T); bias: (out,)).
// bias may be null (no bias).
ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x,
                    ggml_tensor* w, ggml_tensor* bias);

}
}

#endif // LUXTTS_COMMON_H
