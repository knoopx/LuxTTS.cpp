// audio_io.cpp — minimal RIFF/WAVE reader + 16-bit PCM writer.
#include "audio_io.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace lux {

namespace {

// Read a little-endian uint16/uint32 from a byte buffer.
uint16_t rd16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
uint32_t rd32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Parse the WAV header: locate the "fmt " and "data" chunks. Returns true on
// success and fills format/sample_rate/channels/bits_per_sample/data/data_size.
bool parse_wav_header(const uint8_t* b, size_t n,
                      uint16_t& format, uint32_t& sample_rate,
                      uint16_t& channels, uint16_t& bits,
                      const uint8_t** data, size_t* data_size) {
  if (n < 44 || std::memcmp(b, "RIFF", 4) != 0 || std::memcmp(b + 8, "WAVE", 4) != 0)
    return false;
  const uint8_t* p = b + 12;
  const uint8_t* end = b + n;
  format = 0; sample_rate = 0; channels = 0; bits = 0;
  *data = nullptr; *data_size = 0;
  bool have_fmt = false;
  while (p + 8 <= end) {
    const char id[4] = { (char)p[0], (char)p[1], (char)p[2], (char)p[3] };
    const uint32_t sz = rd32(p + 4);
    const uint8_t* body = p + 8;
    if (body + sz > end) break;  // malformed: chunk runs past EOF
    if (std::memcmp(id, "fmt ", 4) == 0) {
      if (sz < 16) return false;
      format      = rd16(body + 0);
      channels    = rd16(body + 2);
      sample_rate = rd32(body + 4);
      bits        = rd16(body + 14);
      have_fmt = true;
    } else if (std::memcmp(id, "data", 4) == 0) {
      *data = body;
      *data_size = sz;
    }
    p = body + sz + (sz & 1);  // chunks are word-aligned
  }
  return have_fmt && *data != nullptr && sample_rate > 0 && channels > 0;
}

} // namespace

bool read_wav(const std::string& path, std::vector<float>& samples,
              int& sample_rate, std::string* err) {
  auto fail = [&](const std::string& m) { if (err) *err = m; samples.clear(); return false; };

  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return fail("cannot open '" + path + "'");
  if (std::fseek(f, 0, SEEK_END) != 0) { std::fclose(f); return fail("seek failed"); }
  const long sz = std::ftell(f);
  if (std::fseek(f, 0, SEEK_SET) != 0) { std::fclose(f); return fail("seek failed"); }
  if (sz < 44) { std::fclose(f); return fail("file too small to be a WAV"); }
  std::vector<uint8_t> buf((size_t)sz);
  if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) { std::fclose(f); return fail("read failed"); }
  std::fclose(f);

  uint16_t format = 0, channels = 0, bits = 0;
  uint32_t rate = 0;
  const uint8_t* data = nullptr; size_t data_size = 0;
  if (!parse_wav_header(buf.data(), buf.size(), format, rate, channels, bits, &data, &data_size))
    return fail("not a valid RIFF/WAVE file: " + path);

  const size_t bytes_per_sample = bits / 8;
  if (bytes_per_sample < 2) return fail("unsupported bits-per-sample: " + std::to_string(bits));
  const size_t n_frames = data_size / (bytes_per_sample * channels);
  samples.assign(n_frames, 0.0f);
  if (n_frames == 0) return fail("WAV has no samples: " + path);

  for (size_t i = 0; i < n_frames; ++i) {
    double acc = 0.0;
    for (uint16_t c = 0; c < channels; ++c) {
      const uint8_t* s = data + (i * channels + c) * bytes_per_sample;
      double v = 0.0;
      if (format == 1) {  // PCM integer
        if (bits == 16) {
          int16_t x = (int16_t)rd16(s);
          v = (double)x / 32768.0;
        } else if (bits == 32) {
          int32_t x = (int32_t)rd32(s);
          v = (double)x / 2147483648.0;
        } else {
          return fail("unsupported PCM bits: " + std::to_string(bits));
        }
      } else if (format == 3) {  // IEEE float
        if (bits == 32) {
          float x; std::memcpy(&x, s, 4); v = (double)x;
        } else {
          return fail("unsupported float bits: " + std::to_string(bits));
        }
      } else {
        return fail("unsupported WAV format code: " + std::to_string(format));
      }
      acc += v;
    }
    samples[i] = (float)(acc / channels);
  }
  sample_rate = (int)rate;
  return true;
}

bool write_wav(const std::string& path, const float* samples, size_t n_samples,
               uint32_t sample_rate, std::string* err) {
  auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
  if (!samples && n_samples > 0) return fail("NULL samples");

  std::vector<int16_t> pcm(n_samples);
  for (size_t i = 0; i < n_samples; ++i) {
    const float v = std::min(1.0f, std::max(-1.0f, samples[i]));
    pcm[i] = (int16_t)std::lrintf(v * 32767.0f);
  }

  const uint32_t data_bytes = (uint32_t)(n_samples * sizeof(int16_t));
  const uint16_t channels = 1;
  const uint16_t block_align = (uint16_t)(channels * sizeof(int16_t));
  const uint32_t byte_rate = sample_rate * block_align;

  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return fail("cannot open '" + path + "' for writing");

  uint8_t hdr[44];
  uint8_t* p = hdr;
  auto put4  = [&](const char* s) { std::memcpy(p, s, 4); p += 4; };
  auto put32 = [&](uint32_t v) { std::memcpy(p, &v, 4); p += 4; };
  auto put16 = [&](uint16_t v) { std::memcpy(p, &v, 2); p += 2; };
  put4("RIFF"); put32(36 + data_bytes); put4("WAVE");
  put4("fmt "); put32(16);
  put16(1);              // PCM
  put16(channels);
  put32(sample_rate);
  put32(byte_rate);
  put16(block_align);
  put16(16);             // bits per sample
  put4("data"); put32(data_bytes);

  const bool ok = std::fwrite(hdr, 1, sizeof(hdr), f) == sizeof(hdr) &&
                  (n_samples == 0 ||
                   std::fwrite(pcm.data(), sizeof(int16_t), n_samples, f) == n_samples);
  const bool closed = std::fclose(f) == 0;
  if (!ok || !closed) return fail("write failed for '" + path + "'");
  return true;
}

} // namespace lux
