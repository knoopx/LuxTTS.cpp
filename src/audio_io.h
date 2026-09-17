// audio_io.h — minimal RIFF/WAVE I/O for the LuxTTS CLI + prompt encoding.
//
// Self-contained (no dr_wav / no external dep). Reads PCM16 / PCM32 / IEEE-float
// WAV, downmixes to mono float in [-1, 1], and writes canonical 44-byte-header
// 16-bit PCM WAV. Little-endian hosts only (all supported targets).
#ifndef LUXTTS_AUDIO_IO_H
#define LUXTTS_AUDIO_IO_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lux {

// Read a WAV file into mono float samples in [-1, 1]. Multi-channel input is
// downmixed to mono by averaging. Supports PCM16 (fmt 1/16), PCM32 (fmt 1/32)
// and IEEE-float32 (fmt 3/32). On success returns true and fills `samples`
// (mono) and `sample_rate`; on failure returns false and sets `err`.
bool read_wav(const std::string& path,
              std::vector<float>& samples,
              int& sample_rate,
              std::string* err);

// Write mono float samples (clamped to [-1, 1], scaled to s16) as a 16-bit PCM
// WAV with a canonical 44-byte header. Returns true on success, false (with
// `err` set) on I/O failure.
bool write_wav(const std::string& path,
               const float* samples,
               size_t n_samples,
               uint32_t sample_rate,
               std::string* err);

} // namespace lux

#endif // LUXTTS_AUDIO_IO_H
