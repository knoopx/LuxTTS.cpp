// luxtts_internal.h — pipeline context shared by the C API (luxtts.cpp).
#ifndef LUXTTS_INTERNAL_H
#define LUXTTS_INTERNAL_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "tokenizer.h"
#include "zipformer.h"
#include "vocos.h"

struct ggml_context;
struct gguf_context;

namespace lux {

// One loaded model: both TTSZipformer instances + Vocos + tokenizer + the
// token-embedding table, resident in the GGUF-backed ggml context.
struct Model {
  // Pipeline context — the resident model state that the Python pipeline keeps
  // alive across calls (zipvoice/modeling_utils.py L64 generate: model +
  // feature_extractor + vocos bound together).
  Zipformer text_encoder;
  Zipformer fm_decoder;
  Vocos vocos;
  Tokenizer tokenizer;

  // Token-embedding table (vocab, text_embed_dim) row-major (embed.weight).
  std::vector<float> embed;
  int embed_dim = 0;   // text_embed_dim (== text_encoder in_dim)

  // GGUF + ggml contexts owned by the loader (freed by luxtts_free).
  void* gguf_ctx = nullptr;   // gguf_context*
  void* ggml_ctx = nullptr;   // ggml_context*

  // Config carried from luxtts_config.
  int threads = 4;
  int prompt_seconds = 5;
  float target_rms = 0.1f;
};

} // namespace lux

#endif // LUXTTS_INTERNAL_H
