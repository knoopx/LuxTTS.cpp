// vocos.cpp — Vocos 48k dual-path vocoder (linacodec + upstream vocos port).
//
// Formulas verified against:
//   linacodec vocos.py L38 (Vocos), upsampler_block.py (UpSamplerBlock L98,
//   ResnetBlock L36, Snake1d L21, Normalize L62), linkwitz.py L4
//   (crossover_merge_linkwitz_riley)
//   gemelo-ai/vocos models.py L26 (VocosBackbone), modules.py L8
//   (ConvNeXtBlock), heads.py L26 (ISTFTHead), spectral_ops.py L7 (ISTFT)
//
// Torch conventions used here:
//   * Conv1d/ConvTranspose1d are plain convolutions (the weight_norm
//     parametrization is stripped before the .bin is saved, see linacodec
//     modeling_utils.py L120-121: remove_parametrizations).
//   * LayerNorm(eps=1e-6, affine), GroupNorm(32 groups, eps=1e-6, affine).
//   * GELU = 0.5x(1+erf(x/sqrt(2))) (nn.GELU default).
//   * Snake: x += 1/(alpha+1e-9) * sin(alpha*x)^2.
//   * torch.hann_window(n) is periodic: w[n] = 0.5*(1-cos(2*pi*n/n)).
//   * torch.istft(center=True) with the raw periodic Hann window: its
//     internal window normalization cancels the NOLA envelope, so the net
//     output == overlap-add of ifft(spec_t) * w (raw window) divided by the
//     NOLA envelope (overlap-add of w^2), then trims win/2 (512) samples on
//     each side -> output length (T-1)*256.
//   * torchaudio 24k->48k resample (integer 2x) returns 2L samples for L
//     input; ~= linear upsample + lowpass; we use a 15-tap Hann-windowed
//     sinc (documented approximation).

#include "vocos.h"

#include <cmath>
#include <complex>
#include <cstring>
#include <algorithm>

#include <kiss_fft.h>
#include <ggml.h>

#include <omp.h>

#include <cstdio>

namespace lux {

namespace {

constexpr int C_DIM = 512;      // backbone dim
constexpr int C_FF = 1536;      // convnext intermediate
constexpr int N_CNX = 8;        // convnext blocks
constexpr int N_NFFT = 1024;    // istft fft size
constexpr int N_HOP = 256;      // istft hop
constexpr float LN_EPS = 1e-6f;

// gemelo modules.py L8 (ConvNeXtBlock: pwconv1 GELU pwconv2; exact erf form
// 0.5x(1+erf(x/sqrt(2))))
inline float gelu(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678f));  // erf(x/sqrt(2))
}

// 1-D conv, weight (C_out, C_in, K), padding P.  Row-major (C, L).
// torch F.conv1d (upsampler_block.py L36 ResnetBlock: conv1 k3 p1 L46, conv2
// k3 p1 L63).
void conv1d(const float* in, int C_in, int L, const float* w, const float* b,
            int C_out, int K, int P, float* out) {
  #pragma omp parallel for schedule(static)
  for (int co = 0; co < C_out; ++co) {
    float* o = out + (size_t)co * L;
    for (int t = 0; t < L; ++t) {
      float acc = b ? b[co] : 0.0f;
      for (int k = 0; k < K; ++k) {
        const int s = t + k - P;
        if (s < 0 || s >= L) continue;
        for (int ci = 0; ci < C_in; ++ci)
          acc += w[((size_t)co * C_in + ci) * K + k] * in[(size_t)ci * L + s];
      }
      o[t] = acc;
    }
  }
}

// 1-D depthwise conv, weight (C, 1, K), padding P.
// torch F.conv1d(groups=C) (gemelo modules.py L8 ConvNeXtBlock dwconv k7 p3).
void conv1d_depthwise(const float* in, int C, int L, const float* w,
                      const float* b, int K, int P, float* out) {
  #pragma omp parallel for schedule(static)
  for (int c = 0; c < C; ++c) {
    const float* wp = w + (size_t)c * K;  // (C, 1, K)
    float* o = out + (size_t)c * L;
    for (int t = 0; t < L; ++t) {
      float acc = b ? b[c] : 0.0f;
      for (int k = 0; k < K; ++k) {
        const int s = t + k - P;
        if (s >= 0 && s < L) acc += wp[k] * in[(size_t)c * L + s];
      }
      o[t] = acc;
    }
  }
}

// ConvTranspose1d (weight (C_in, C_out, K), stride S, padding P).
// Exact PyTorch equivalence (torch F.conv_transpose1d, upsampler_block.py L115
// weight_norm ConvTranspose1d; float32 max diff 3.2e-04
// for k8 s2 p3 and k8 s1 p3).
// Zero-stuff the input at stride S (z[c, t*S] = x[c, t], length Lz =
// (L-1)*S + 1), then convolve with the FLIPPED kernel:
//   out[co, t] = b[co] + Σ_k W[ci, co, K-1-k] · z[ci, t - 1 + k - P],
// reading out-of-range z as zero (the boundary padding), for t = 0..Lout-1
// where Lout = (L-1)*S + K - 2P (i.e. 2L for k8 s2 p3, L for k8 s1 p3).
void conv_transpose1d(const float* in, int C_in, int L, const float* w,
                      const float* b, int C_out, int K, int S, int P,
                      float* out) {
  const int Lz = (L - 1) * S + 1;
  const int Lout = (L - 1) * S + K - 2 * P;
  std::vector<float> z((size_t)C_in * Lz, 0.0f);
  #pragma omp parallel for schedule(static)
  for (int c = 0; c < C_in; ++c)
    for (int t = 0; t < L; ++t) z[(size_t)c * Lz + (size_t)t * S] = in[(size_t)c * L + t];
  #pragma omp parallel for schedule(static)
  for (int co = 0; co < C_out; ++co) {
    float* o = out + (size_t)co * Lout;
    for (int t = 0; t < Lout; ++t) {
      float acc = b ? b[co] : 0.0f;
      for (int k = 0; k < K; ++k) {
        const int s = t - 1 + k - P;
        if (s < 0 || s >= Lz) continue;  // zero padding at the boundaries
        for (int ci = 0; ci < C_in; ++ci)
          acc += w[((size_t)ci * C_out + co) * K + (K - 1 - k)] *
                 z[(size_t)ci * Lz + s];
      }
      o[t] = acc;
    }
  }
}

// LayerNorm over the channel axis (per time frame), eps 1e-6, affine.
// torch: nn.LayerNorm(512) is applied to x.transpose(1,2) -> (B, L, 512),
// i.e. mean/var over the 512 channels at each time step (gemelo models.py
// L26 VocosBackbone `norm`/`final_layer_norm`, modules.py L8 ConvNeXtBlock).
void layer_norm_channel(const float* x, int C, int L, const float* w, const float* b,
                       float* out) {
  #pragma omp parallel for schedule(static)
  for (int t = 0; t < L; ++t) {
    double mean = 0.0;
    for (int c = 0; c < C; ++c) mean += (double)x[(size_t)c * L + t];
    mean /= (double)C;
    double var = 0.0;
    for (int c = 0; c < C; ++c) {
      const double d = (double)x[(size_t)c * L + t] - mean;
      var += d * d;
    }
    const float rstd = 1.0f / std::sqrt((float)var / (double)C + LN_EPS);
    for (int c = 0; c < C; ++c)
      out[(size_t)c * L + t] =
          ((float)x[(size_t)c * L + t] - (float)mean) * rstd * w[c] + b[c];
  }
}

// GroupNorm (groups=32), eps 1e-6, affine.  Parallel over groups (each
// group's stats accumulate over its channels/time in fixed order -> bit-exact).
// torch F.group_norm (upsampler_block.py L62 Normalize: 32 groups, eps 1e-6,
// affine; used by ResnetBlock L36 norm1/norm2).
void group_norm(const float* x, int C, int L, const float* w, const float* b,
                float* out) {
  constexpr int kGroups = 32;
  const int pg = C / kGroups;
  #pragma omp parallel for schedule(static)
  for (int g = 0; g < kGroups; ++g) {
    const int c0 = g * pg;
    double mean = 0.0;
    for (int i = 0; i < pg; ++i)
      for (int t = 0; t < L; ++t) mean += (double)x[(size_t)(c0 + i) * L + t];
    mean /= (double)(pg * L);
    double var = 0.0;
    for (int i = 0; i < pg; ++i)
      for (int t = 0; t < L; ++t) {
        const double d = (double)x[(size_t)(c0 + i) * L + t] - mean;
        var += d * d;
      }
    const float rstd = 1.0f / std::sqrt((float)var / (double)(pg * L) + LN_EPS);
    for (int i = 0; i < pg; ++i) {
      const int c = c0 + i;
      float* o = out + (size_t)c * L;
      for (int t = 0; t < L; ++t)
        o[t] = ((float)x[(size_t)c * L + t] - (float)mean) * rstd * w[c] + b[c];
    }
  }
}

// Snake: x += 1/(alpha+1e-9) * sin(alpha*x)^2 (per-channel alpha).
// Vocos Snake activation (upsampler_block.py L21 Snake1d: learnable
// per-channel alpha; formula x + 1/(a+1e-9)*sin(a*x)^2).
void snake(const float* x, int C, int L, const float* alpha, float* out) {
  #pragma omp parallel for schedule(static)
  for (int c = 0; c < C; ++c) {
    const float a = alpha[c];
    const float inv = 1.0f / (a + 1e-9f);
    const float* xr = x + (size_t)c * L;
    float* o = out + (size_t)c * L;
    for (int t = 0; t < L; ++t) {
      const float s = std::sin(a * xr[t]);
      o[t] = xr[t] + inv * s * s;
    }
  }
}

// Per-frame linear: y (out_dim, L) = W (out, in) @ x (in, L) + b.
// torch nn.Linear applied per frame (upsampler_block.py L109 out_proj
// Linear 128->512; also heads.py L40 ISTFTHead out Linear 512->1026).
void per_frame_linear(const float* x, int Cin, int L, const float* w, const float* b,
                      int Cout, float* out) {
  #pragma omp parallel for schedule(static)
  for (int o = 0; o < Cout; ++o) {
    float* oc = out + (size_t)o * L;
    for (int t = 0; t < L; ++t) {
      float acc = b ? b[o] : 0.0f;
      for (int i = 0; i < Cin; ++i)
        acc += w[(size_t)o * Cin + i] * x[(size_t)i * L + t];
      oc[t] = acc;
    }
  }
}

// ---------------------------------------------------------------------------
// torch.istft(center=True) port: spec (513, T) -> audio (T-1)*256
// ---------------------------------------------------------------------------
std::vector<float> istft_center(const float* mag, const float* ph, int T) {
  // Raw periodic hann window: torch.istft(center=True) normalizes the window
  // by sqrt(N)/||w|| internally, and that factor cancels the NOLA envelope,
  // so the net output == raw-window overlap-add divided by the raw-window NOLA
  // envelope (torch.istft vs this formula, ratio 1.0). Pre-scaling the window
  // (e.g. 1/sqrt(sum w^2)) leaves a residual factor sqrt(sum w^2) = 19.5959
  // after the NOLA division.
  std::vector<float> wn(N_NFFT);
  for (int n = 0; n < N_NFFT; ++n)
    wn[n] = 0.5f * (1.0f - std::cos(2.0f * M_PI * (double)n / N_NFFT));

  const int full = (T - 1) * N_HOP + N_NFFT;
  std::vector<float> y(full, 0.0f);
  std::vector<float> env(full, 0.0f);

  // NOLA envelope: overlap-add of wn^2 (window-only, independent of the
  // signal). Compute it first, sequentially (bit-identical to the in-loop
  // accumulation the original did). torch.istft(center=True) NOLA denominator
  // (spectral_ops.py L33 ISTFT.forward: 1/(sum of window^2 over the frame).
  for (int t = 0; t < T; ++t) {
    const size_t off = (size_t)t * N_HOP;
    for (int n = 0; n < N_NFFT; ++n) env[off + n] += wn[n] * wn[n];
  }

  // Per-frame windowed IFFT, parallel over frames (independent FFTs, one
  // scratch buffer per thread). The overlap-add into y[] is then done
  // sequentially in the SAME ascending-t order as the original, so the
  // float accumulation is bit-identical.
  kiss_fft_cfg cfg = kiss_fft_alloc(N_NFFT, 0, nullptr, nullptr);
  const float invN = 1.0f / (float)N_NFFT;
  std::vector<float> framebuf((size_t)T * N_NFFT);
  #pragma omp parallel
  {
    const int nt = omp_get_max_threads();
    std::vector<std::vector<kiss_fft_cpx>> Xs(nt, std::vector<kiss_fft_cpx>(N_NFFT));
    std::vector<std::vector<kiss_fft_cpx>> Ys(nt, std::vector<kiss_fft_cpx>(N_NFFT));
    #pragma omp for schedule(static)
    for (int t = 0; t < T; ++t) {
      const int tid = omp_get_thread_num();
      auto& X = Xs[tid];
      auto& Y = Ys[tid];
      // build Hermitian spectrum (bins 0..512), then inverse-FFT via the
      // conjugate trick: IFFT(spec) = conj(FFT(conj(spec))) / N.  kiss_fft is
      // forward-only, so conjugate the spectrum (negate imag), run the forward
      // FFT, and take the real part / N (result is real: spectrum is Hermitian).
      for (int k = 0; k <= N_NFFT / 2; ++k) {
        // heads.py L55-56 (ISTFTHead.forward: mag = torch.exp(mag);
        // mag = torch.clip(mag, max=1e2)) — clip the EXPONENTIATED magnitude,
        // not the log-magnitude (a pre-exp cap of 1e2 overflows float)
        float m = std::exp(mag[(size_t)k * T + t]);
        m = std::min(m, 1e2f);
        const double re = m * std::cos(ph[(size_t)k * T + t]);
        const double im = m * std::sin(ph[(size_t)k * T + t]);
        X[k].r = (kiss_fft_scalar)re;
        X[k].i = (kiss_fft_scalar)(-im);          // conj(spec[k])
        if (k > 0 && k < N_NFFT / 2) {  // mirror of the conjugate spectrum
          X[N_NFFT - k].r = (kiss_fft_scalar)re;
          X[N_NFFT - k].i = (kiss_fft_scalar)im;
        }
      }
      kiss_fft(cfg, X.data(), Y.data());
      const float* frame = (const float*)Y.data();  // reinterpretable: r,i,r,i
      float* fb = framebuf.data() + (size_t)t * N_NFFT;
      for (int n = 0; n < N_NFFT; ++n) fb[n] = frame[2 * n] * invN * wn[n];
    }
  }
  kiss_fft_free(cfg);
  for (int t = 0; t < T; ++t) {
    const size_t off = (size_t)t * N_HOP;
    const float* fb = framebuf.data() + (size_t)t * N_NFFT;
    for (int n = 0; n < N_NFFT; ++n) y[off + n] += fb[n];
  }
  for (int n = 0; n < full; ++n) y[n] /= (env[n] > 1e-11f ? env[n] : 1.0f);
  // trim win/2 on each side (center padding) — torch.istft(center=True) removes
  // the N//2 padding samples on each side (spectral_ops.py L33).
  const int pad = N_NFFT / 2;
  return std::vector<float>(y.begin() + pad, y.begin() + full - pad);
}

// rfft of a real signal (kiss_fft on a real input -> first n/2+1 bins).
// torch.stft magnitude/phase path (the C++ side of the ISTFT head's FFT,
// heads.py L26 ISTFTHead / L44 forward).
void rfft(const float* x, int n, float* re, float* im) {
  kiss_fft_cfg cfg = kiss_fft_alloc(n, 0, nullptr, nullptr);
  std::vector<kiss_fft_cpx> X(n), Y(n);
  for (int i = 0; i < n; ++i) {
    X[i].r = (kiss_fft_scalar)x[i];
    X[i].i = 0.0f;
  }
  kiss_fft(cfg, X.data(), Y.data());
  for (int k = 0; k <= n / 2; ++k) {
    re[k] = (float)Y[k].r;
    im[k] = (float)Y[k].i;
  }
  kiss_fft_free(cfg);
}

// irfft: (n/2+1) complex bins -> n real samples (Hermitian mirror + FFT).
// torch.istft inverse path (spectral_ops.py L33 ISTFT.forward).
void irfft(const float* re, const float* im, int n, float* out) {
  kiss_fft_cfg cfg = kiss_fft_alloc(n, 0, nullptr, nullptr);
  std::vector<kiss_fft_cpx> X(n), Y(n);
  for (int k = 0; k <= n / 2; ++k) {
    X[k].r = (kiss_fft_scalar)re[k];
    X[k].i = (kiss_fft_scalar)(-im[k]);         // conj for IFFT-via-FFT
    if (k > 0 && k < n / 2) {
      X[n - k].r = (kiss_fft_scalar)re[k];
      X[n - k].i = (kiss_fft_scalar)im[k];
    }
  }
  kiss_fft(cfg, X.data(), Y.data());
  for (int i = 0; i < n; ++i) out[i] = (float)Y[i].r / (float)n;  // inverse scaling
  kiss_fft_free(cfg);
}

} // namespace

// ---------------------------------------------------------------------------
// weight binding
// ---------------------------------------------------------------------------

namespace {

// ggml v0.24.0 (the GGUF stores the vocos.* weight tensors in the TORCH
// layout (tools/convert_to_gguf.py writes them verbatim, no value transpose) —
// the raw bytes are the torch row-major arrays the host conv kernels index
// (conv1d (C_out,C_in,K), dwconv (C,1,K), conv_transpose1d (C_in,C_out,K)).
// The memcpy below reads them directly; no un-reversing is needed.)

std::string bind_copy(const std::vector<std::pair<std::string, ggml_tensor*>>& named,
                      const char* name, std::vector<float>& dst,
                      const char* ctx_prefix) {
  for (const auto& pr : named) {
    if (pr.first != name) continue;
    const ggml_tensor* t = pr.second;
    const int64_t n = ggml_nelements(t);
    if (t->type != GGML_TYPE_F32)
      return std::string(ctx_prefix) + ": non-f32 tensor " + name;
    dst.resize((size_t)n);
    std::memcpy(dst.data(), ggml_get_data(t), (size_t)n * sizeof(float));
    return "";
  }
  return std::string(ctx_prefix) + ": missing tensor " + name;
}

} // namespace

std::string Vocos::load(ggml_context* ctx,
                       const std::vector<std::pair<std::string, ggml_tensor*>>& named) {
  (void)ctx;  // weights are copied out; the context is not retained
  auto need = [&](const char* n, std::vector<float>& dst) -> std::string {
    std::string r = bind_copy(named, n, dst, "vocos");
    if (!r.empty()) return r;
    return "";
  };
  w_.cnx.resize(N_CNX);
  w_.resnet.resize(2);
  {
    std::string r;
    r = need("backbone.embed.weight", w_.emb_w);  if (!r.empty()) return r;
    r = need("backbone.embed.bias", w_.emb_b);    if (!r.empty()) return r;
    if (w_.emb_w.size() != (size_t)C_DIM * 100 * 7)
      return "vocos: backbone.embed.weight shape";
    r = need("backbone.norm.weight", w_.bn_w);    if (!r.empty()) return r;
    r = need("backbone.norm.bias", w_.bn_b);      if (!r.empty()) return r;
    for (int i = 0; i < N_CNX; ++i) {
      const std::string p = "backbone.convnext." + std::to_string(i) + ".";
      r = need((p + "dwconv.weight").c_str(), w_.cnx[i].dw_w);  if (!r.empty()) return r;
      r = need((p + "dwconv.bias").c_str(), w_.cnx[i].dw_b);    if (!r.empty()) return r;
      r = need((p + "norm.weight").c_str(), w_.cnx[i].n_w);     if (!r.empty()) return r;
      r = need((p + "norm.bias").c_str(), w_.cnx[i].n_b);      if (!r.empty()) return r;
      r = need((p + "pwconv1.weight").c_str(), w_.cnx[i].p1_w); if (!r.empty()) return r;
      r = need((p + "pwconv1.bias").c_str(), w_.cnx[i].p1_b);   if (!r.empty()) return r;
      r = need((p + "pwconv2.weight").c_str(), w_.cnx[i].p2_w); if (!r.empty()) return r;
      r = need((p + "pwconv2.bias").c_str(), w_.cnx[i].p2_b);   if (!r.empty()) return r;
      r = need((p + "gamma").c_str(), w_.cnx[i].gamma);         if (!r.empty()) return r;
      if (w_.cnx[i].dw_w.size() != (size_t)C_DIM * 7)
        return "vocos: dwconv.weight shape";
      if (w_.cnx[i].p1_w.size() != (size_t)C_FF * C_DIM)
        return "vocos: pwconv1.weight shape";
    }
    r = need("backbone.final_layer_norm.weight", w_.fnorm_w);  if (!r.empty()) return r;
    r = need("backbone.final_layer_norm.bias", w_.fnorm_b);    if (!r.empty()) return r;
    // upsampler (upsampler_block.py L98 UpSamplerBlock: two ConvTranspose1d
    // upsample layers L115 + per-scale ResnetBlocks L118 + out_proj L109
    // + final Snake L120)
    r = need("upsampler.upsample_layers.0.weight", w_.up0_w);  if (!r.empty()) return r;
    r = need("upsampler.upsample_layers.0.bias", w_.up0_b);    if (!r.empty()) return r;
    r = need("upsampler.upsample_layers.1.weight", w_.up1_w);  if (!r.empty()) return r;
    r = need("upsampler.upsample_layers.1.bias", w_.up1_b);    if (!r.empty()) return r;
    if (w_.up0_w.size() != (size_t)C_DIM * 256 * 8)
      return "vocos: upsample_layers.0.weight shape";
    if (w_.up1_w.size() != (size_t)256 * 128 * 8)
      return "vocos: upsample_layers.1.weight shape";
    for (int i = 0; i < 2; ++i) {
      const std::string p = "upsampler.resnet_blocks." + std::to_string(i) + ".";
      const int C = i == 0 ? 256 : 128;
      auto R = w_.resnet[i];
      r = need((p + "norm1.weight").c_str(), R.n1_w);  if (!r.empty()) return r;
      r = need((p + "norm1.bias").c_str(), R.n1_b);    if (!r.empty()) return r;
      r = need((p + "snake1.alpha").c_str(), R.s1_a);   if (!r.empty()) return r;
      r = need((p + "conv1.weight").c_str(), R.c1_w);  if (!r.empty()) return r;
      r = need((p + "conv1.bias").c_str(), R.c1_b);    if (!r.empty()) return r;
      r = need((p + "norm2.weight").c_str(), R.n2_w);  if (!r.empty()) return r;
      r = need((p + "norm2.bias").c_str(), R.n2_b);    if (!r.empty()) return r;
      r = need((p + "snake2.alpha").c_str(), R.s2_a);   if (!r.empty()) return r;
      r = need((p + "conv2.weight").c_str(), R.c2_w);  if (!r.empty()) return r;
      r = need((p + "conv2.bias").c_str(), R.c2_b);    if (!r.empty()) return r;
      if (R.n1_w.size() != (size_t)C || R.c1_w.size() != (size_t)C * C * 3)
        return "vocos: resnet block shape";
      w_.resnet[i] = std::move(R);
    }
    r = need("upsampler.out_proj.weight", w_.oproj_w);  if (!r.empty()) return r;
    r = need("upsampler.out_proj.bias", w_.oproj_b);    if (!r.empty()) return r;
    r = need("upsampler.final_snake.alpha", w_.fsnake_a); if (!r.empty()) return r;
    if (w_.oproj_w.size() != (size_t)C_DIM * 128)
      return "vocos: out_proj.weight shape";
    // heads (heads.py L26 ISTFTHead: out Linear L40 -> 1026 rows = 513 mag +
    // 513 phase; the 48k head_48k is a second ISTFTHead for the dual-path
    // crossover, vocos.py L54)
    r = need("head.out.weight", w_.head_out_w);      if (!r.empty()) return r;
    r = need("head.out.bias", w_.head_out_b);        if (!r.empty()) return r;
    r = need("head_48k.out.weight", w_.head48_out_w); if (!r.empty()) return r;
    r = need("head_48k.out.bias", w_.head48_out_b);   if (!r.empty()) return r;
    if (w_.head_out_w.size() != (size_t)1026 * C_DIM ||
        w_.head48_out_w.size() != (size_t)1026 * C_DIM)
      return "vocos: head.out shape";
  }
  loaded_ = true;
  return "";
}

// ---------------------------------------------------------------------------
// pipeline
// ---------------------------------------------------------------------------

std::vector<float> Vocos::backbone(const float* feats, int T) const {
  // gemelo models.py L26 VocosBackbone.forward (L77): embed Conv1d k7 p3 ->
  // LayerNorm -> 8 ConvNeXtBlocks (modules.py L8: dwconv k7 p3 -> LayerNorm ->
  // pwconv1 GELU pwconv2 gamma -> residual add) -> final LayerNorm.
  std::vector<float> x((size_t)C_DIM * T);
  conv1d(feats, 100, T, w_.emb_w.data(), w_.emb_b.data(), C_DIM, 7, 3, x.data());
  {
    std::vector<float> y(x.size());
    layer_norm_channel(x.data(), C_DIM, T, w_.bn_w.data(), w_.bn_b.data(), y.data());
    x = std::move(y);
  }
  for (int i = 0; i < N_CNX; ++i) {
    const W::CNX& b = w_.cnx[i];
    std::vector<float> res = x;  // residual
    // Non-in-place: torch F.conv1d(groups=512) reads the ORIGINAL input, so
    // the depthwise conv must be computed into scratch. In-place, the
    // centered K=7/P=3 kernel's backward taps (s = t-1..t-3) read samples
    // already overwritten earlier in the same loop (non-in-place formula vs
    // torch, max diff 1.9e-06; in-place decorrelates through the blocks).
    std::vector<float> dw(x.size());
    conv1d_depthwise(x.data(), C_DIM, T, b.dw_w.data(), b.dw_b.data(), 7, 3, dw.data());
    {
      std::vector<float> y(x.size());
      layer_norm_channel(dw.data(), C_DIM, T, b.n_w.data(), b.n_b.data(), y.data());
      x = std::move(y);
    }
    std::vector<float> m((size_t)C_FF * T);
    per_frame_linear(x.data(), C_DIM, T, b.p1_w.data(), b.p1_b.data(), C_FF, m.data());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m.size(); ++i) m[i] = gelu(m[i]);
    {
      std::vector<float> y((size_t)C_DIM * T);
      per_frame_linear(m.data(), C_FF, T, b.p2_w.data(), b.p2_b.data(), C_DIM, y.data());
      #pragma omp parallel for schedule(static)
      for (int c = 0; c < C_DIM; ++c) {
        const float g = b.gamma[c];
        for (int t = 0; t < T; ++t) y[(size_t)c * T + t] = g * y[(size_t)c * T + t];
      }
      #pragma omp parallel for schedule(static)
      for (int i2 = 0; i2 < (int)y.size(); ++i2) y[i2] += res[i2];
      x = std::move(y);
    }
  }
  {
    std::vector<float> out(x.size());
    layer_norm_channel(x.data(), C_DIM, T, w_.fnorm_w.data(), w_.fnorm_b.data(), out.data());
    x = std::move(out);
  }
  return x;
}

std::vector<float> Vocos::upsampler(const float* feats, int T) const {
  // stage 0: ConvTranspose1d(512 -> 256, k8 s2 p3) — upsampler_block.py L115
  // upsample_layers.0; L0 = (T-1)*2 + 8 - 6 = 2T
  const int L0 = (T - 1) * 2 + 2;
  std::vector<float> x0((size_t)256 * L0);
  conv_transpose1d(feats, C_DIM, T, w_.up0_w.data(), w_.up0_b.data(), 256, 8, 2, 3,
                   x0.data());
  // ResnetBlock(256) — upsampler_block.py L36 (GroupNorm L62 -> Snake1d L21 ->
  // Conv1d k3 p1 -> GroupNorm -> Snake1d -> Conv1d k3 p1, residual add L103)
  {
    auto& R = w_.resnet[0];
    std::vector<float> h(x0.size());
    group_norm(x0.data(), 256, L0, R.n1_w.data(), R.n1_b.data(), h.data());
    snake(h.data(), 256, L0, R.s1_a.data(), h.data());
    // Non-in-place (mirrors the backbone dwconv fix): torch F.conv1d(k=3,p=1)
    // and F.group_norm read the ORIGINAL input. In-place, conv1d's backward
    // taps (s = t-1..t-3) read samples already overwritten earlier in the
    // same loop, and group_norm's per-group pass 2 overwrites a group's
    // earlier channels before pass 1 of the later channels in that group
    // consumes them (256ch: 8ch/group). In-place n2 alone is off by 2.94,
    // the in-place convs add ~5.1 on top; all non-in-place matches torch to
    // 7.6e-6, corr 1.0. Snake stays in-place (element-wise, alias-safe).
    std::vector<float> s(h.size());
    conv1d(h.data(), 256, L0, R.c1_w.data(), R.c1_b.data(), 256, 3, 1, s.data());
    h = std::move(s);
    s.resize(h.size());
    group_norm(h.data(), 256, L0, R.n2_w.data(), R.n2_b.data(), s.data());
    h = std::move(s);
    snake(h.data(), 256, L0, R.s2_a.data(), h.data());
    s.resize(h.size());
    conv1d(h.data(), 256, L0, R.c2_w.data(), R.c2_b.data(), 256, 3, 1, s.data());
    h = std::move(s);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x0.size(); ++i) h[i] += x0[i];
    x0 = std::move(h);
  }
  // stage 1: ConvTranspose1d(256 -> 128, k8 s1 p3) — upsampler_block.py L115
  // upsample_layers.1; L1 = L0 + 1
  const int L1 = L0 + 1;
  std::vector<float> x1((size_t)128 * L1);
  conv_transpose1d(x0.data(), 256, L0, w_.up1_w.data(), w_.up1_b.data(), 128, 8, 1, 3,
                   x1.data());
  // ResnetBlock(128) — upsampler_block.py L36 ResnetBlock (second scale, L118)
  {
    auto& R = w_.resnet[1];
    std::vector<float> h(x1.size());
    group_norm(x1.data(), 128, L1, R.n1_w.data(), R.n1_b.data(), h.data());
    snake(h.data(), 128, L1, R.s1_a.data(), h.data());
    // Same non-in-place scratch + swap as the 256-ch block (comment there):
    // in-place conv1d/group-n2 aliasing corrupts the block's stats/feedback.
    std::vector<float> s(h.size());
    conv1d(h.data(), 128, L1, R.c1_w.data(), R.c1_b.data(), 128, 3, 1, s.data());
    h = std::move(s);
    s.resize(h.size());
    group_norm(h.data(), 128, L1, R.n2_w.data(), R.n2_b.data(), s.data());
    h = std::move(s);
    snake(h.data(), 128, L1, R.s2_a.data(), h.data());
    s.resize(h.size());
    conv1d(h.data(), 128, L1, R.c2_w.data(), R.c2_b.data(), 128, 3, 1, s.data());
    h = std::move(s);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x1.size(); ++i) h[i] += x1[i];
    x1 = std::move(h);
  }
  // out_proj Linear(128 -> 512) per frame — upsampler_block.py L109 out_proj
  std::vector<float> y((size_t)C_DIM * L1);
  per_frame_linear(x1.data(), 128, L1, w_.oproj_w.data(), w_.oproj_b.data(), C_DIM,
                   y.data());
  // final Snake — upsampler_block.py L120 final_snake (Snake1d L21)
  snake(y.data(), C_DIM, L1, w_.fsnake_a.data(), y.data());
  return y;
}

std::vector<float> Vocos::istft_head(const std::vector<float>& feats, int T,
                                     bool head_48k) const {
  const float* w = head_48k ? w_.head48_out_w.data() : w_.head_out_w.data();
  const float* b = head_48k ? w_.head48_out_b.data() : w_.head_out_b.data();
  std::vector<float> y((size_t)1026 * T);
  per_frame_linear(feats.data(), C_DIM, T, w, b, 1026, y.data());
  // mag = rows [0, 513), phase = rows [513, 1026) — heads.py L26 ISTFTHead
  // (out_dim = n_fft + 2 = 1026, L40) splits the 1026-row per-frame linear
  // output into magnitude + phase for istft_center (heads.py L44 forward)
  auto out = istft_center(y.data(), y.data() + (size_t)513 * T, T);
  return out;
}

std::vector<float> Vocos::resample_24k_to_48k(const std::vector<float>& in) const {
  if (in.empty()) return {};
  // 2x linear upsample + 15-tap Hann-windowed-sinc lowpass (fc = Nyquist/2).
  // torchaudio.functional.resample(24000 -> 48000) returns 2L samples for L
  // input samples (verified: L=9984 -> 19968 in the parity venv), so zero-
  // stuff with a zero-padded endpoint to yield 2L, not 2L-1.
  const int L = (int)in.size();
  std::vector<float> up((size_t)(2 * L));
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < L; ++i) {
    up[2 * i] = in[i];
    up[2 * i + 1] = (i + 1 < L) ? 0.5f * (in[i] + in[i + 1]) : 0.5f * in[i];
  }
  constexpr int TAPS = 15;
  // Sinc lowpass for cutoff 0.5 (normalized by the 48k rate) — linkwitz.py
  // L4 crossover_merge_linkwitz_riley Linkwitz-Rill crossover: h[m] =
  // sinc((m - (TAPS-1)/2)) * hann; the gain sums to 1 at DC.
  std::vector<float> h(TAPS);
  {
    double gain = 0.0;
    for (int m = 0; m < TAPS; ++m) {
      const double d = m - (TAPS - 1) / 2.0;
      const float w = 0.5f * (1.0f - std::cos(2.0f * M_PI * (double)m / TAPS));
      const float s = d == 0.0 ? 1.0f : (float)(std::sin(M_PI * d) / (M_PI * d));
      h[m] = s * w;
      gain += h[m];
    }
    for (auto& v : h) v /= (float)gain;
  }
  std::vector<float> out((size_t)up.size());
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < up.size(); ++i) {
    float acc = 0.0f;
    for (int m = 0; m < TAPS; ++m) {
      const long s = (long)i + m - (TAPS - 1) / 2;
      if (s >= 0 && s < (long)up.size()) acc += h[m] * up[(size_t)s];
    }
    out[i] = acc;
  }
  return out;
}

std::vector<float> Vocos::crossover_merge(const std::vector<float>& hi,
                                          const std::vector<float>& lo) const {
  const int n = std::min((int)hi.size(), (int)lo.size());
  if (n == 0) return {};
  const int nb = n / 2 + 1;
  std::vector<float> r1(nb), i1(nb), r2(nb), i2(nb);
  rfft(hi.data(), n, r1.data(), i1.data());
  rfft(lo.data(), n, r2.data(), i2.data());

  // linkwitz.py L4 crossover_merge_linkwitz_riley: cutoff_bin = int((cutoff /
  // (sr/2)) * n_bins) = int(12000/24000*nb)
  const int cutoff_bin = (int)((12000.0 / 24000.0) * (double)nb);
  const int half = 4;  // transition_bins / 2
  const int start = std::max(0, cutoff_bin - half);
  const int end = std::min(nb, cutoff_bin + half);
  const int width = end - start;
  std::vector<float> mask(nb, 1.0f);
  for (int b = 0; b < start; ++b) mask[b] = 0.0f;
  for (int b = start; b < end; ++b) {
    const double x = width <= 1 ? 0.0 : (double)(b - start) / (width - 1) * 2.0 - 1.0;
    const double u = (x + 1.0) / 2.0;
    mask[b] = (float)(3.0 * u * u - 2.0 * u * u * u);  // cubic Hermite smoothstep
  }

  std::vector<float> mr(nb), mi(nb);
  #pragma omp parallel for schedule(static)
  for (int b = 0; b < nb; ++b) {
    mr[b] = r1[b] * mask[b] + r2[b] * (1.0f - mask[b]);
    mi[b] = i1[b] * mask[b] + i2[b] * (1.0f - mask[b]);
  }
  std::vector<float> out(n);
  irfft(mr.data(), mi.data(), n, out.data());
  return out;
}

std::vector<float> Vocos::decode(const float* feats, int T) const {
  // vocos.py L93 Vocos.forward: backbone -> upsampler -> head_48k (48k)
  // + head (24k, resampled to 48k) -> Linkwitz-Riley crossover_merge.
  if (!loaded_ || T <= 0) return {};
  const std::vector<float> bb = backbone(feats, T);
  const std::vector<float> up = upsampler(bb.data(), T);
  std::vector<float> a1 = istft_head(up, (int)up.size() / C_DIM, true);  // 48k
  std::vector<float> a2_24 = istft_head(bb, T, false);                  // 24k
  std::vector<float> a2 = resample_24k_to_48k(a2_24);
  if (a1.size() > a2.size()) a1.resize(a2.size());
  const std::vector<float> merged = crossover_merge(a1, a2);
  return merged;
}

std::vector<float> Vocos::decode_48k_raw(const float* feats, int T) const {
  // vocos.py L93 return_48k=False mode: head (24k) resampled to 48k, no
  // crossover merge.
  if (!loaded_ || T <= 0) return {};
  const std::vector<float> bb = backbone(feats, T);
  const std::vector<float> a2_24 = istft_head(bb, T, false);
  return resample_24k_to_48k(a2_24);
}

} // namespace lux
