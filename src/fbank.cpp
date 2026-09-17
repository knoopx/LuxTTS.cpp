// fbank.cpp — VocosFbank (torchaudio-compatible) + windowed-sinc resampler.
#include "fbank.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lux {

namespace {

// In-place iterative radix-2 complex FFT (n a power of two).
void fft_inplace(std::complex<float>* a, int n) {
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (int len = 2; len <= n; len <<= 1) {
    const float ang = -2.0f * (float)M_PI / (float)len;
    const std::complex<float> wl(std::cos(ang), std::sin(ang));
    for (int i = 0; i < n; i += len) {
      std::complex<float> w(1.0f, 0.0f);
      const int half = len >> 1;
      for (int k = 0; k < half; ++k) {
        const std::complex<float> u = a[i + k];
        const std::complex<float> v = a[i + k + half] * w;
        a[i + k] = u + v;
        a[i + k + half] = u - v;
        w *= wl;
      }
    }
  }
}

// HTK mel scale (torchaudio MelSpectrogram/MelScale default mel_scale="htk"):
// 2595*log10(1+hz/700); the port used a /1000 constant (non-standard) here.
float hz_to_mel(float hz) { return 2595.0f * std::log10f(1.0f + hz / 700.0f); }
float mel_to_hz(float mel) { return 700.0f * (std::powf(10.0f, mel / 2595.0f) - 1.0f); }

// Build the torchaudio MelScale filterbank: n_mels x (n_fft/2+1), row-major.
// Replicates torchaudio.transforms.MelScale (slaney, f_min=0, f_max=rate/2).
std::vector<float> build_mel_filterbank(const FbankConfig& cfg) {
  const int n_fft = cfg.n_fft, n_mels = cfg.n_mels, rate = cfg.sample_rate;
  const int n_bins = n_fft / 2 + 1;
  const float f_min = 0.0f, f_max = (float)rate / 2.0f;
  const float mel_min = hz_to_mel(f_min), mel_max = hz_to_mel(f_max);
  const int n_pts = n_mels + 2;
  std::vector<float> hz(n_pts);
  for (int i = 0; i < n_pts; ++i) {
    const float mel = mel_min + (mel_max - mel_min) * (float)i / (float)(n_pts - 1);
    hz[i] = mel_to_hz(mel);
  }
  // torchaudio: fbin = floor((n_fft+1) * hz / sample_rate)
  std::vector<int> fbin(n_pts);
  for (int i = 0; i < n_pts; ++i)
    fbin[i] = (int)std::floor((float)(n_fft + 1) * hz[i] / (float)rate);
  for (int i = 0; i < n_pts; ++i)
    fbin[i] = std::max(0, std::min(n_bins - 1, fbin[i]));

  std::vector<float> fb((size_t)n_mels * n_bins, 0.0f);
  for (int m = 0; m < n_mels; ++m) {
    const int left = fbin[m], center = fbin[m + 1], right = fbin[m + 2];
    for (int k = left; k < center; ++k)
      if (center > left) fb[(size_t)m * n_bins + k] = (float)(k - left) / (float)(center - left);
    for (int k = center; k < right; ++k)
      if (right > center) fb[(size_t)m * n_bins + k] = (float)(right - k) / (float)(right - center);
  }
  return fb;
}

// torch.hann_window(n_fft) with periodic=False: w[k]=0.5*(1-cos(2*pi*k/(n_fft-1))).
std::vector<float> hann_window(int n) {
  std::vector<float> w(n);
  for (int k = 0; k < n; ++k)
    w[k] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)k / (float)(n - 1)));
  return w;
}

} // namespace

// TEST-ONLY: expose the mel filterbank builder for cross-checking.
std::vector<float> build_mel_filterbank_test(const FbankConfig& cfg) {
  return build_mel_filterbank(cfg);
}

void resample(const float* in, int n, int in_rate, int out_rate,
              std::vector<float>& out) {
  if (in_rate == out_rate) { out.assign(in, in + n); return; }
  const double ratio = (double)out_rate / (double)in_rate;
  const int out_n = (int)std::floor((double)n * ratio);
  out.assign((size_t)out_n, 0.0f);
  if (out_n <= 0 || n <= 0) return;

  const float fc = (float)std::min(in_rate, out_rate);  // lowpass cutoff (Hz)
  const float fc_norm = fc / (float)in_rate;            // cycles per input sample
  const int W = 16;  // half-window (taps)

  for (int j = 0; j < out_n; ++j) {
    const double pos = (double)j / ratio;
    const int k0 = (int)std::floor(pos) - W;
    const int k1 = (int)std::floor(pos) + W;
    double num = 0.0, den = 0.0;
    for (int k = k0; k <= k1; ++k) {
      const double x = pos - (double)k;
      float h = 0.0f;
      if (std::fabs(x) <= (double)W) {
        const float arg = (float)(x * fc_norm);
        const float sinc = (arg == 0.0f) ? 1.0f : std::sinf((float)M_PI * arg) / ((float)M_PI * arg);
        const float win = 0.54f + 0.46f * std::cos((float)M_PI * (float)x / (float)W);
        h = sinc * win;
      }
      den += (double)h;
      if (k >= 0 && k < n) num += (double)in[k] * (double)h;
    }
    out[j] = (den != 0.0) ? (float)(num / den) : 0.0f;
  }
}

int vocos_fbank(const float* wav_24k, int n, float* out, const FbankConfig& cfg) {
  if (n <= 0) return 0;
  const int n_fft = cfg.n_fft, hop = cfg.hop_length, n_mels = cfg.n_mels;
  const int n_bins = n_fft / 2 + 1;

  const std::vector<float> win = hann_window(n_fft);
  const std::vector<float> fb = build_mel_filterbank(cfg);

  const int pad = n_fft / 2;                 // center=True padding (each side)
  const int T_full = n / hop + 1;            // torchaudio MelSpectrogram frame count
  // lhotse compute_num_frames (window_hop = hop): the reference VocosFbank.extract
  // truncates the torchaudio frames to (num_samples + hop//2) // hop (this can
  // only be <= the torchaudio count), so the port must not emit the extra frame.
  const int T = std::min(T_full, (n + hop / 2) / hop);
  if ((size_t)n_mels * (size_t)T > (size_t)-1) return 0;

  // Reusable scratch for the (complex) FFT of one frame.
  std::vector<std::complex<float>> spec(n_fft);

  for (int f = 0; f < T; ++f) {
    const int start = f * hop;  // index in the center-padded signal ([-pad, n+pad))
    // Windowed frame (zero-padded outside [0, n)).
    for (int k = 0; k < n_fft; ++k) {
      const int idx = start + k - pad;
      const float v = (idx >= 0 && idx < n) ? wav_24k[idx] * win[k] : 0.0f;
      spec[k] = std::complex<float>(v, 0.0f);
    }
    fft_inplace(spec.data(), n_fft);
    // Magnitude spectrum (power=1) of bins 0..n_bins-1.
    for (int b = 0; b < n_bins; ++b) {
      const float re = spec[b].real(), im = spec[b].imag();
      spec[b] = std::complex<float>(std::sqrt(re * re + im * im), 0.0f);
    }
    // mel = fbank @ mag ; logmel = log(clamp(mel, min=1e-7))
    for (int m = 0; m < n_mels; ++m) {
      double acc = 0.0;
      const float* row = fb.data() + (size_t)m * n_bins;
      for (int b = 0; b < n_bins; ++b) acc += (double)row[b] * (double)spec[b].real();
      const float mel = (float)acc;
      out[(size_t)f * n_mels + m] = std::log(std::max(mel, 1e-7f));
    }
  }
  return T;
}

int prompt_features(const float* wav, int n, int in_rate,
                    std::vector<float>& out, int& out_frames,
                    float feat_scale) {
  std::vector<float> wav24;
  if (in_rate != 24000) resample(wav, n, in_rate, 24000, wav24);
  else wav24.assign(wav, wav + n);

  const int T = n / 256 + 1;  // upper bound (resampled length can only grow on upsample)
  const int T2 = ((int)wav24.size()) / 256 + 1;
  out.assign((size_t)std::max(T, T2) * 100, 0.0f);
  const int frames = vocos_fbank(wav24.data(), (int)wav24.size(), out.data(), FbankConfig{});
  if (frames <= 0) return -1;
  out.resize((size_t)frames * 100);
  if (feat_scale != 0.0f)
    for (float& v : out) v *= feat_scale;
  out_frames = frames;
  return 0;
}

} // namespace lux
