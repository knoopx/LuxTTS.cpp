// fbank.h — VocosFbank mel feature extraction (torchaudio-compatible) +
// resampling, for prompt encoding. Ports zipvoice/utils/feature.py (VocosFbank:
// 24 kHz, 100-mel, n_fft 1024, hop 256, center, power=1, log) and the
// load_prompt_wav resample step.
#ifndef LUXTTS_FBANK_H
#define LUXTTS_FBANK_H

#include <cstddef>
#include <string>
#include <vector>

namespace lux {

// VocosFbank config (zipvoice/utils/feature.py VocosFbankConfig).
struct FbankConfig {
  int sample_rate = 24000;  // features are computed at 24 kHz
  int n_mels = 100;
  int n_fft = 1024;
  int hop_length = 256;
};

// Resample `in` (length n, rate in_rate) to out_rate using a windowed-sinc
// (Hamming) lowpass-interpolation kernel. in_rate == out_rate copies.
void resample(const float* in, int n, int in_rate, int out_rate,
              std::vector<float>& out);

// VocosFbank on a 24 kHz mono signal: writes log-mel features (T, n_mels)
// frame-major into `out` (size T*n_mels) and returns the frame count T.
// `out` must have capacity for at least (n/hop + 1) * n_mels floats.
int vocos_fbank(const float* wav_24k, int n, float* out, const FbankConfig& cfg);

// Full prompt-feature pipeline (the C++ of load_prompt_wav + rms_norm is done
// by the caller): resample `wav` (in_rate) to 24 kHz, run VocosFbank, and
// scale by feat_scale (0.1, the reference feat_scale). Writes frame-major
// (T, 100) into `out` and sets `out_frames = T`. Returns 0 on success.
int prompt_features(const float* wav, int n, int in_rate,
                    std::vector<float>& out, int& out_frames,
                    float feat_scale = 0.1f);

// TEST-ONLY: expose the mel filterbank builder for cross-checking.
std::vector<float> build_mel_filterbank_test(const FbankConfig& cfg);

} // namespace lux

#endif // LUXTTS_FBANK_H
