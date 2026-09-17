// luxtts.h — public C API for the self-contained LuxTTS CPU inference engine.
//
// Self-contained: no onnxruntime, no Python, no GPU. Weights come from a GGUF
// file in the model directory (see README.md "Model directory layout").
//
// All functions are synchronous and return a luxtts_status. A null status or a
// non-zero code means failure; st.message is non-null only on failure.

#ifndef LUXTTS_H
#define LUXTTS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct luxtts_ctx luxtts_ctx;

// Uniform status object (explicit error fields).
typedef struct luxtts_status {
  int   code;
  char  message[256];
} luxtts_status;

typedef struct luxtts_config {
  const char* model_path;  // path to the model GGUF file (e.g. luxtts-f32.gguf)
  int         threads;     // ggml intra/inter op threads (CPU)
  // Sampling defaults (overridable per-call); mirror the Python CLI defaults
  // (zipvoice/modeling_utils.py L47/L64, zipvoice/luxvoice.py L47):
  int   num_steps;         // ODE steps (Python default 4)
  float t_shift;           // time-shift (Python default 0.9 for distill)
  float guidance;          // classifier-free guidance scale (Python default 3.0)
  float speed;             // speech-rate multiplier (Python: speed * 1.3 internally)
  float target_rms;        // prompt RMS target (Python default 0.1)
  int   prompt_seconds;    // prompt duration cap in seconds (Python default 5)
  unsigned long long seed; // noise seed: 0 = fresh random noise per call
                           // (Python default — torch.randn is unseeded),
                           // nonzero = deterministic (torch.manual_seed(seed))
} luxtts_config;

// Encoded prompt: mel features + prompt tokens + prompt RMS (all owned by the
// encoder / luxtts_load_voice; free with luxtts_free_prompt).
typedef struct luxtts_prompt {
  float*  features;        // [T, feat_dim=100] (row-major, frame-major), already *0.1-scaled
  int     feature_frames;  // T
  int*    tokens;          // [token_len]
  int     token_len;
  float   prompt_rms;      // RMS of the (rms-normalised) prompt waveform
} luxtts_prompt;

// Generated audio: 48 kHz mono float in [-1, 1].
typedef struct luxtts_audio {
  float*  samples;         // [num_samples]
  int     num_samples;
  int     sample_rate;     // 48000
} luxtts_audio;

// Load model (GGUF) + tokenizer vocab (+G2P data). Caller frees with luxtts_free.
luxtts_ctx* luxtts_load(const luxtts_config* cfg, luxtts_status* st);

// Load a cloned-voice GGUF (the output of tools/convert_voice.py) into `out`:
// a ready-to-use prompt (features [feat_dim=100, T] already *0.1-scaled, tokens, feature_frames,
// token_len, prompt_rms). The file must carry general.architecture="lux-voice"
// (distinct from the model's "lux") with tensors voice.prompt_tokens (i32 [N]),
// voice.prompt_features (f32 [100, T]), voice.prompt_features_lens (i32 [1]),
// voice.prompt_rms (f32 [1]). The prompt is owned by the caller
// (free with luxtts_free_prompt).
int luxtts_load_voice(luxtts_ctx* ctx,
                      const char* voice_gguf_path,
                      luxtts_prompt* out,
                      luxtts_status* st);

// Encode a raw prompt WAV + its transcript into a ready-to-use prompt (the
// runtime equivalent of the Python load_prompt_wav -> rms_norm -> VocosFbank
// -> *0.1 -> texts_to_token_ids path). Reads `wav_path` (any sample rate,
// downmixed to mono), resamples to 24 kHz, computes the 100-mel log-mel
// features (scaled by 0.1), RMS-normalises to `target_rms`, and tokenises
// `prompt_text`. On success the caller owns out (free with
// luxtts_free_prompt). Returns 1 on success, 0 on failure.
int luxtts_encode_prompt(luxtts_ctx* ctx,
                         const char* wav_path,
                         const char* prompt_text,
                         float target_rms,
                         luxtts_prompt* out,
                         luxtts_status* st);

// Generate speech for `text` given an encoded prompt. Returns 48 kHz mono audio.
int luxtts_generate(luxtts_ctx* ctx,
                    const char* text,
                    const luxtts_prompt* prompt,
                    luxtts_audio* out,
                    luxtts_status* st);

// Write `audio` (48 kHz mono float in [-1,1]) to a 16-bit PCM WAV file.
// Returns 1 on success, 0 on failure.
int luxtts_write_wav(const char* path, const luxtts_audio* audio);

void luxtts_free(luxtts_ctx* ctx);
void luxtts_free_audio(luxtts_audio* a);
void luxtts_free_prompt(luxtts_prompt* p);  // frees out->features / out->tokens

#ifdef __cplusplus
}
#endif

#endif // LUXTTS_H
