// luxtts.cpp — C API implementation (pipeline orchestration + GGUF loader).
//
// Pipeline (CPU-only, self-contained), grounded in:
//   zipvoice/modeling_utils.py  L47 (process_audio), L64 (generate)
//   zipvoice/onnx_modeling.py   L182 (generate_cpu)
//   zipvoice/utils/infer.py     L262 (rms_norm)
//
//   luxtts_load:
//     open the model GGUF (gguf_init_from_file; tensor data lands in one
//     host ggml context), read the architecture metadata,
//     bind the text_encoder / fm_decoder (TTSZipformer) and vocos weight
//     groups, load the embedded token table (GGUF metadata `luxtts.tokens`
//     — only the GGUF file is needed at runtime) + the embed.weight table.
//
//   luxtts_load_voice(voice_gguf):
//     open a cloned-voice GGUF (tools/convert_voice.py output, arch
//     "lux-voice") and fill a luxtts_prompt (features/tokens/lens/rms)
//     without audio decoding — the .pt prompt data packed by the converter.
//
//   luxtts_generate(text, prompt):
//     tokenize(text) -> sample_features(...) -> vocos.decode -> clamp(-1,1)
//     -> volume-match by prompt_rms -> 48 kHz mono audio.

#include "luxtts.h"
#include "luxtts_internal.h"
#include "sampling.h"
#include "audio_io.h"
#include "fbank.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <omp.h>

#include <ggml.h>
#include <ggml-cpu.h>
#include <gguf.h>

namespace {

// ---------------------------------------------------------------------------
// GGUF metadata helpers
// ---------------------------------------------------------------------------

int64_t meta_i64(const gguf_context* g, const char* key, int64_t def) {
  const int64_t id = gguf_find_key(g, key);
  if (id < 0) return def;
  const enum gguf_type t = gguf_get_kv_type(g, id);
  switch (t) {
    case GGUF_TYPE_INT32: return gguf_get_val_i32(g, id);
    case GGUF_TYPE_INT64: return gguf_get_val_i64(g, id);
    case GGUF_TYPE_FLOAT32: return (int64_t)gguf_get_val_f32(g, id);
    case GGUF_TYPE_STRING: return def;
    default: return def;
  }
}

// The converter (tools/convert_to_gguf.py) writes list metadata as space-joined strings ("1 2 4 2 1").
std::vector<int> meta_int_list(const gguf_context* g, const char* key,
                               std::vector<int> def) {  const int64_t id = gguf_find_key(g, key);
  if (id < 0) return def;
  if (gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) return def;
  const std::string s = gguf_get_val_str(g, id);
  std::vector<int> v;
  size_t pos = 0;
  while (pos < s.size()) {
    size_t sp = s.find(' ', pos);
    const size_t end = (sp == std::string::npos) ? s.size() : sp;
    if (end > pos) v.push_back(std::stoi(s.substr(pos, end - pos)));
    pos = (sp == std::string::npos) ? s.size() : sp + 1;
  }
  return v.empty() ? def : v;
}

// String metadata (the value of a GGUF_TYPE_STRING key; def when absent).
std::string meta_str(const gguf_context* g, const char* key, const char* def) {
  const int64_t id = gguf_find_key(g, key);
  if (id < 0) return def;
  if (gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) return def;
  return gguf_get_val_str(g, id);
}

std::string require_f32(const std::vector<std::pair<std::string, ggml_tensor*>>& named,
                        const char* group) {
  for (const auto& pr : named) {
    if (pr.second->type != GGML_TYPE_F32)
      return (std::string(group) + ": tensor '" + pr.first +
        "' is " + ggml_type_name(pr.second->type) +
        " — this group supports the fp32 GGUF only (rebuild without --quant)");
  }
  return "";
}

// The zipformer groups (text_encoder / fm_decoder) may be f32 or q8_0: the
// 2-D matmul weights run on ggml's q8_0 x f32 kernels directly. Any other
// dtype is a hard error naming the offending tensor.
std::string require_f32_or_q8_0(const std::vector<std::pair<std::string, ggml_tensor*>>& named,
                                const char* group) {
  for (const auto& pr : named) {
    if (pr.second->type != GGML_TYPE_F32 && pr.second->type != GGML_TYPE_Q8_0)
      return (std::string(group) + ": tensor '" + pr.first +
        "' is " + ggml_type_name(pr.second->type) +
        " — the zipformer groups support f32 or q8_0 GGUFs only");
  }
  return "";
}

}

struct CtxImpl {
  lux::Model model;
  // zipvoice/config.json (sampling defaults)
  int num_steps = 4;
  float t_shift = 0.9f;
  float guidance = 3.0f;
  float speed = 1.0f;
  float target_rms = 0.1f;
  unsigned long long seed = 0;  // 0 = fresh random noise per generate
};

namespace {

// Resolve --model: the GGUF file path itself (no directory search).
std::string find_gguf(const char* model_path) {
  struct stat sb{};
  if (stat(model_path, &sb) == 0 && S_ISREG(sb.st_mode)) return model_path;
  return {};
}

}

// luxtts_load — open the model GGUF (tools/convert_to_gguf.py layout: text_encoder /
// fm_decoder / vocos / embed / tokens groups) and bind the resident pipeline
// (zipvoice/modeling_utils.py: model + feature_extractor + vocos).
luxtts_ctx* luxtts_load(const luxtts_config* cfg, luxtts_status* st) {
  auto fail = [&](const std::string& msg) -> luxtts_ctx* {
    if (st) { st->code = -1; std::snprintf(st->message, sizeof st->message, "%s", msg.c_str()); }
    return nullptr;
  };
  if (!cfg || !cfg->model_path) return fail("null config / model_path");
  auto* impl = new CtxImpl();
  // Auto threads (cfg 0): the engine's host-loop + per-stack graph parallelism
  // plateaus at ~16 threads on a 32-thread Zen 5 (measured); extra OpenMP
  // workers only spin. Pass an explicit thread count to override.
  impl->model.threads = cfg->threads > 0 ? cfg->threads
                                         : std::min((int)omp_get_max_threads(), 16);
  impl->model.prompt_seconds = cfg->prompt_seconds > 0 ? cfg->prompt_seconds : 5;
  impl->seed = cfg->seed;
  impl->num_steps = cfg->num_steps > 0 ? cfg->num_steps : 4;
  impl->t_shift = cfg->t_shift > 0 ? cfg->t_shift : 0.9f;
  impl->guidance = cfg->guidance > 0 ? cfg->guidance : 3.0f;
  impl->speed = cfg->speed > 0 ? cfg->speed : 1.0f;
  impl->target_rms = cfg->target_rms > 0 ? cfg->target_rms : 0.1f;
  impl->model.target_rms = impl->target_rms;

  const std::string gguf_path = find_gguf(cfg->model_path);
  if (gguf_path.empty()) {
    delete impl;
    if (st) { st->code = -1; std::snprintf(st->message, sizeof st->message, "GGUF file not found: %s", cfg->model_path); }
    return nullptr;
  }

  // ggml v0.24.0 (ggml context backed by the GGUF: tensor data in one host allocation)
  ggml_init_params p{0, nullptr, false};
  ggml_context* ggml_ctx = ggml_init(p);
  if (!ggml_ctx) { delete impl; return fail("ggml_init failed"); }
  gguf_init_params gp{false, &ggml_ctx};
  gguf_context* gguf_ctx = gguf_init_from_file(gguf_path.c_str(), gp);
  if (!gguf_ctx) {
    ggml_free(ggml_ctx);
    delete impl;
    return fail("failed to open GGUF: " + gguf_path);
  }

  // ggml v0.24.0 (name -> tensor, paired by index: ggml names truncate,
  // full names come from the gguf context)
  std::vector<std::pair<std::string, ggml_tensor*>> named;
  const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
  ggml_tensor* t = ggml_get_first_tensor(ggml_ctx);
  while (t && std::strcmp(t->name, "GGUF tensor data binary blob") == 0)
    t = ggml_get_next_tensor(ggml_ctx, t);
  for (int64_t i = 0; i < n_tensors; ++i) {
    if (!t) { gguf_free(gguf_ctx); ggml_free(ggml_ctx); delete impl;
              return fail("ggml context out of tensors while reading the GGUF"); }
    const char* full = gguf_get_tensor_name(gguf_ctx, i);
    if (std::strncmp(full, t->name, GGML_MAX_NAME - 1) != 0) {
      gguf_free(gguf_ctx); ggml_free(ggml_ctx); delete impl;
      return fail("GGUF tensor order mismatch (corrupt file?)");
    }
    named.emplace_back(full, t);
    t = ggml_get_next_tensor(ggml_ctx, t);
  }

  auto find = [&](const std::string& n) -> ggml_tensor* {
    for (const auto& pr : named) if (pr.first == n) return pr.second;
    return nullptr;
  };

  // zipformer.py L116 (architecture metadata; ZipVoice defaults as fallbacks)
  lux::ZipformerConfig te_cfg;
  te_cfg.in_dim = (int)meta_i64(gguf_ctx, "text_embed_dim", 192);
  te_cfg.out_dim = (int)meta_i64(gguf_ctx, "feat_dim", 100);
  te_cfg.downsample_factor = {1};
  te_cfg.num_layers = { (int)meta_i64(gguf_ctx, "text_encoder_num_layers", 4) };
  te_cfg.cnn_kernel = { (int)meta_i64(gguf_ctx, "text_encoder_cnn_module_kernel", 9) };
  te_cfg.encoder_dim = (int)meta_i64(gguf_ctx, "text_encoder_dim", 192);
  te_cfg.feedforward_dim = (int)meta_i64(gguf_ctx, "text_encoder_feedforward_dim", 512);
  te_cfg.num_heads = (int)meta_i64(gguf_ctx, "text_encoder_num_heads", 4);
  te_cfg.query_head_dim = (int)meta_i64(gguf_ctx, "query_head_dim", 32);
  te_cfg.pos_head_dim = (int)meta_i64(gguf_ctx, "pos_head_dim", 4);
  te_cfg.value_head_dim = (int)meta_i64(gguf_ctx, "value_head_dim", 12);
  te_cfg.pos_dim = (int)meta_i64(gguf_ctx, "pos_dim", 48);
  te_cfg.use_time_embed = false;

  lux::ZipformerConfig fm_cfg;
  fm_cfg.in_dim = 3 * (int)meta_i64(gguf_ctx, "feat_dim", 100);
  fm_cfg.out_dim = (int)meta_i64(gguf_ctx, "feat_dim", 100);
  fm_cfg.downsample_factor = meta_int_list(gguf_ctx, "fm_decoder_downsampling_factor", {1, 2, 4, 2, 1});
  fm_cfg.num_layers = meta_int_list(gguf_ctx, "fm_decoder_num_layers", {2, 2, 4, 4, 4});
  fm_cfg.cnn_kernel = meta_int_list(gguf_ctx, "fm_decoder_cnn_module_kernel", {31, 15, 7, 15, 31});
  fm_cfg.encoder_dim = (int)meta_i64(gguf_ctx, "fm_decoder_dim", 512);
  fm_cfg.feedforward_dim = (int)meta_i64(gguf_ctx, "fm_decoder_feedforward_dim", 1536);
  fm_cfg.num_heads = (int)meta_i64(gguf_ctx, "fm_decoder_num_heads", 4);
  fm_cfg.query_head_dim = te_cfg.query_head_dim;
  fm_cfg.pos_head_dim = te_cfg.pos_head_dim;
  fm_cfg.value_head_dim = te_cfg.value_head_dim;
  fm_cfg.pos_dim = te_cfg.pos_dim;
  fm_cfg.use_time_embed = true;
  fm_cfg.time_embed_dim = (int)meta_i64(gguf_ctx, "time_embed_dim", 192);
  fm_cfg.use_guidance_embed = true;

  if ((int)fm_cfg.num_layers.size() != (int)fm_cfg.downsample_factor.size() ||
      (int)fm_cfg.cnn_kernel.size() != (int)fm_cfg.downsample_factor.size()) {
    gguf_free(gguf_ctx); ggml_free(ggml_ctx); delete impl;
    return fail("fm_decoder metadata lists have inconsistent lengths");
  }

  // bind the two TTSZipformer instances (zipformer.py L79) + vocos
  auto prefixed = [&](const char* prefix) {
    std::vector<std::pair<std::string, ggml_tensor*>> v;
    for (const auto& pr : named)
      if (pr.first.compare(0, std::strlen(prefix), prefix) == 0)
        v.emplace_back(pr.first.substr(std::strlen(prefix)), pr.second);
    return v;
  };
  std::string r;
  auto te_named = prefixed("text_encoder.");
  auto fm_named = prefixed("fm_decoder.");
  auto vo_named = prefixed("vocos.");
  r = require_f32_or_q8_0(te_named, "text_encoder"); if (!r.empty()) { gguf_free(gguf_ctx); ggml_free(ggml_ctx); delete impl; return fail(r.c_str()); }
  r = require_f32_or_q8_0(fm_named, "fm_decoder");   if (!r.empty()) { gguf_free(gguf_ctx); ggml_free(ggml_ctx); delete impl; return fail(r.c_str()); }
  r = require_f32(vo_named, "vocos");        if (!r.empty()) { gguf_free(gguf_ctx); ggml_free(ggml_ctx); delete impl; return fail(r.c_str()); }

  impl->model.text_encoder = lux::Zipformer(te_cfg);
  impl->model.fm_decoder = lux::Zipformer(fm_cfg);
  r = impl->model.text_encoder.load(ggml_ctx, te_named);
  if (!r.empty()) { gguf_free(gguf_ctx); ggml_free(ggml_ctx); delete impl; return fail(r.c_str()); }
  r = impl->model.fm_decoder.load(ggml_ctx, fm_named);
  if (!r.empty()) { gguf_free(gguf_ctx); ggml_free(ggml_ctx); delete impl; return fail(r.c_str()); }
  r = impl->model.vocos.load(ggml_ctx, vo_named);
  if (!r.empty()) { gguf_free(gguf_ctx); ggml_free(ggml_ctx); delete impl; return fail(r.c_str()); }
  // ggml v0.24.0 (thread wiring: the n_threads arg of every ggml_graph_compute_with_ctx in
  // the zipformer graphs; 0 = ggml default)
  impl->model.text_encoder.set_n_threads(impl->model.threads);
  impl->model.fm_decoder.set_n_threads(impl->model.threads);
  // ggml v0.24.0 (bind the zipformer weight tensors to a CPU backend buffer
  // wrapping the GGUF tensor-data blob (the first tensor of ggml_ctx; the
  // named loop above skips it), so the scheduler's CPU compute reads the
  // weights in place — no copy)
  {
    ggml_tensor* blob_t = ggml_get_first_tensor(ggml_ctx);
    if (blob_t && std::strcmp(blob_t->name, "GGUF tensor data binary blob") != 0)
      blob_t = nullptr;
    if (blob_t) {
      impl->model.text_encoder.set_weight_blob(blob_t->data, ggml_nbytes(blob_t));
      impl->model.fm_decoder.set_weight_blob(blob_t->data, ggml_nbytes(blob_t));
    }
  }

  // zipformer.py L116 (embed.weight (vocab, text_embed_dim) -> host copy)
  const ggml_tensor* emb = find("embed.weight");
  if (!emb) { gguf_free(gguf_ctx); ggml_free(ggml_ctx); delete impl;
              return fail("missing tensor 'embed.weight' (the token embedding table)"); }
  impl->model.embed_dim = te_cfg.in_dim;
  if (emb->ne[0] != te_cfg.in_dim ||
      (emb->type != GGML_TYPE_F32 && emb->type != GGML_TYPE_Q8_0)) {
    gguf_free(gguf_ctx); ggml_free(ggml_ctx); delete impl;
    return fail("embed.weight: expected (vocab, " + std::to_string(te_cfg.in_dim) +
                ") f32 or q8_0");
  }
  if (emb->type == GGML_TYPE_Q8_0) {
    // ggml v0.24.0 (q8_0 block = {ggml_fp16_t d; int8_t qs[32]}: d = amax/127 as float16,
    // val = qs[i] * d; identical to the gguf Q8_0 the converter writes.
    // ne0 = 192 (text_embed_dim) is the innermost dim, so the 32-element blocks run
    // along it (dequantize host-side into model.embed))
    const size_t vocab = emb->ne[1];
    impl->model.embed.assign((size_t)vocab * te_cfg.in_dim, 0.0f);
    const uint8_t* p = (const uint8_t*)emb->data;
    const size_t row_bytes = (size_t)(te_cfg.in_dim / 32) * 34;
    for (size_t v = 0; v < vocab; ++v) {
      const uint8_t* row = p + v * row_bytes;
      for (int c = 0; c < te_cfg.in_dim; ++c) {
        const size_t b = (size_t)(c / 32) * 34;
        const ggml_fp16_t d = ((const ggml_fp16_t*)row)[c / 32];
        const int8_t qs = ((const int8_t*)(row + b + 2))[c % 32];
        impl->model.embed[v * te_cfg.in_dim + c] =
            (float)qs * ggml_fp16_to_fp32(d);
      }
    }
  } else {
    impl->model.embed.assign((const float*)emb->data,
                             (const float*)emb->data + emb->ne[1] * emb->ne[0]);
  }

  // GGUF metadata (tokenizer vocab embedded in the GGUF; self-contained model;
  // the model dir needs ONLY the GGUF file)
  std::string terr = "";
  const std::string tok_table = meta_str(gguf_ctx, "luxtts.tokens", "");
  if (tok_table.empty()) {
    gguf_free(gguf_ctx); ggml_free(ggml_ctx); delete impl;
    return fail("GGUF is missing the embedded token table (luxtts.tokens) — "
                "regenerate with tools/convert_to_gguf.py");
  }
  impl->model.tokenizer = lux::Tokenizer::load_str(tok_table, &terr);
  if (!terr.empty()) { gguf_free(gguf_ctx); ggml_free(ggml_ctx); delete impl;
                  return fail("token table: " + terr); }

  impl->model.gguf_ctx = gguf_ctx;
  impl->model.ggml_ctx = ggml_ctx;
  if (st) { st->code = 0; st->message[0] = '\0'; }
  return (luxtts_ctx*)impl;
}

// luxtts_load_voice — load the cloned-voice prompt GGUF (tools/convert_voice.py:
// voice.prompt_features + voice.prompt_tokens + voice.prompt_rms).
int luxtts_load_voice(luxtts_ctx* ctx,
                      const char* voice_gguf_path,
                      luxtts_prompt* out,
                      luxtts_status* st) {
  if (out) { out->features = nullptr; out->tokens = nullptr; out->feature_frames = 0; out->token_len = 0; out->prompt_rms = 0.0f; }
  auto err = [&](int code, const std::string& msg) -> int {
    if (st) { st->code = code; std::snprintf(st->message, sizeof st->message, "%s", msg.c_str()); }
    return 0;
  };
  if (!ctx) return err(-1, "null ctx");
  if (!voice_gguf_path || !*voice_gguf_path) return err(-1, "null/empty voice GGUF path");

  // ggml v0.24.0 (open the voice GGUF; tensor data lands in one host ggml context)
  ggml_init_params p{0, nullptr, false};
  ggml_context* ggml_ctx = ggml_init(p);
  if (!ggml_ctx) return err(-1, "ggml_init failed");
  gguf_init_params gp{false, &ggml_ctx};
  gguf_context* gguf_ctx = gguf_init_from_file(voice_gguf_path, gp);
  auto cleanup = [&]() { gguf_free(gguf_ctx); ggml_free(ggml_ctx); };
  if (!gguf_ctx) {
    cleanup();
    return err(-1, "failed to open voice GGUF: " + std::string(voice_gguf_path));
  }

  // Architecture check: a voice file is "lux-voice" (the model is "lux");
  // reject anything else hard (a model GGUF here is a usage error, not a
  // fallback case).
  const std::string arch = meta_str(gguf_ctx, "general.architecture", "");
  if (arch != "lux-voice") {
    cleanup();
    return err(-2, "not a voice GGUF (general.architecture=\"" + arch +
                    "\", want \"lux-voice\"; produce one with tools/convert_voice.py)");
  }

  // ggml v0.24.0 (name -> tensor, paired by index: ggml names truncate,
  // full names come from the gguf context)
  std::vector<std::pair<std::string, ggml_tensor*>> named;
  const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
  ggml_tensor* t = ggml_get_first_tensor(ggml_ctx);
  while (t && std::strcmp(t->name, "GGUF tensor data binary blob") == 0)
    t = ggml_get_next_tensor(ggml_ctx, t);
  for (int64_t i = 0; i < n_tensors; ++i) {
    if (!t) { cleanup(); return err(-3, "ggml context out of tensors while reading the voice GGUF"); }
    const char* full = gguf_get_tensor_name(gguf_ctx, i);
    if (std::strncmp(full, t->name, GGML_MAX_NAME - 1) != 0) {
      cleanup();
      return err(-3, "GGUF tensor order mismatch (corrupt file?)");
    }
    named.emplace_back(full, t);
    t = ggml_get_next_tensor(ggml_ctx, t);
  }
  auto find = [&](const std::string& n) -> ggml_tensor* {
    for (const auto& pr : named) if (pr.first == n) return pr.second;
    return nullptr;
  };

  const ggml_tensor* toks = find("voice.prompt_tokens");
  const ggml_tensor* feats = find("voice.prompt_features");
  const ggml_tensor* lens_t = find("voice.prompt_features_lens");
  const ggml_tensor* rms_t = find("voice.prompt_rms");
  const char* missing = nullptr;
  if (!toks) missing = "voice.prompt_tokens";
  else if (!feats) missing = "voice.prompt_features";
  else if (!lens_t) missing = "voice.prompt_features_lens";
  else if (!rms_t) missing = "voice.prompt_rms";
  if (missing) {
    cleanup();
    return err(-4, std::string("voice GGUF missing tensor '") + missing + "'");
  }
  // tools/convert_voice.py (hard type checks, no fallback: the converter writes exactly these)
  if (toks->type != GGML_TYPE_I32) { cleanup(); return err(-5, "voice.prompt_tokens: expected i32"); }
  if (feats->type != GGML_TYPE_F32) { cleanup(); return err(-5, "voice.prompt_features: expected f32"); }
  if (lens_t->type != GGML_TYPE_I32) { cleanup(); return err(-5, "voice.prompt_features_lens: expected i32"); }
  if (rms_t->type != GGML_TYPE_F32) { cleanup(); return err(-5, "voice.prompt_rms: expected f32"); }

  const int N = (int)toks->ne[0];
  const int feat_dim = (int)feats->ne[0];
  const int T = (int)feats->ne[1];
  if (N <= 0) { cleanup(); return err(-6, "voice.prompt_tokens is empty"); }
  if (feat_dim != 100) {
    cleanup();
    return err(-7, "voice.prompt_features: expected feature dim 100, got " + std::to_string(feat_dim));
  }
  const int64_t lens0 = ((const int32_t*)lens_t->data)[0];
  if (lens0 != T) {
    cleanup();
    return err(-8, "voice.prompt_features_lens (" + std::to_string(lens0) +
                    ") != feature frame count (" + std::to_string(T) + ")");
  }
  const float rms = ((const float*)rms_t->data)[0];

  // Copy into the caller-owned prompt: the features buffer keeps the .pt flat
  // order ((T,100) row-major: 100 features contiguous per frame) — exactly
  // what luxtts_load_voice produces and sample_features consumes.
  if (out) {
    out->features = (float*)std::malloc((size_t)T * 100 * sizeof(float));
    if (!out->features) { cleanup(); return err(-9, "out of memory"); }
    std::memcpy(out->features, feats->data, (size_t)T * 100 * sizeof(float));
    out->feature_frames = T;
    out->tokens = (int*)std::malloc((size_t)N * sizeof(int));
    if (!out->tokens) { std::free(out->features); out->features = nullptr; cleanup(); return err(-9, "out of memory"); }
    const int32_t* src_toks = (const int32_t*)toks->data;
    for (int i = 0; i < N; ++i) out->tokens[i] = (int)src_toks[i];
    out->token_len = N;
    out->prompt_rms = rms;
  }
  cleanup();
  if (st) { st->code = 0; st->message[0] = '\0'; }
  return 1;
}

// luxtts_encode_prompt — runtime prompt encoding (C++ of the Python
// load_prompt_wav -> rms_norm -> VocosFbank -> *0.1 -> texts_to_token_ids).
int luxtts_encode_prompt(luxtts_ctx* ctx,
                         const char* wav_path,
                         const char* prompt_text,
                         float target_rms,
                         luxtts_prompt* out,
                         luxtts_status* st) {
  auto* impl = (CtxImpl*)ctx;
  if (out) { out->features = nullptr; out->tokens = nullptr; out->feature_frames = 0;
             out->token_len = 0; out->prompt_rms = 0.0f; }
  auto err = [&](int code, const std::string& msg) -> int {
    if (st) { st->code = code; std::snprintf(st->message, sizeof st->message, "%s", msg.c_str()); }
    return 0;
  };
  if (!ctx) return err(-1, "null ctx");
  if (!wav_path || !*wav_path) return err(-1, "null/empty wav path");
  if (!prompt_text) prompt_text = "";
  std::vector<float> wav; int sr = 0; std::string ioerr;
  if (!lux::read_wav(wav_path, wav, sr, &ioerr)) return err(-2, "read_wav: " + ioerr);
  if (wav.empty()) return err(-2, "prompt wav has no samples: " + std::string(wav_path));
  std::vector<float> wav24;
  if (sr != 24000) lux::resample(wav.data(), (int)wav.size(), sr, 24000, wav24);
  else wav24 = wav;
  // zipvoice/modeling_utils.py L48 (process_audio): librosa.load(audio,
  // sr=24000, duration=duration) — the prompt is capped to prompt_seconds
  // BEFORE rms_norm/fbank (LuxTTS default duration=5, README: set higher,
  // e.g. 1000, if artifacts are audible). --duration was previously ignored.
  const size_t max_samples = (size_t)impl->model.prompt_seconds * 24000u;
  if (max_samples > 0 && wav24.size() > max_samples) wav24.resize(max_samples);
  double acc = 0.0;
  for (float v : wav24) acc += (double)v * v;
  const float prompt_rms = (float)std::sqrt(acc / (double)std::max<size_t>(1, wav24.size()));
  if (prompt_rms > 0.0f && prompt_rms < target_rms) {
    const float s = target_rms / prompt_rms;
    for (auto& v : wav24) v *= s;
  }
  std::vector<float> feats; int frames = 0;
  if (lux::prompt_features(wav24.data(), (int)wav24.size(), 24000, feats, frames, 0.1f) != 0)
    return err(-3, "fbank feature extraction failed");
  if (frames <= 0) return err(-3, "prompt produced zero feature frames");
  std::vector<int> toks = impl->model.tokenizer.texts_to_token_ids(prompt_text);
  if (toks.empty()) return err(-4, "prompt_text tokenized to zero tokens");
  out->features = (float*)std::malloc(feats.size() * sizeof(float));
  if (!out->features) return err(-5, "out of memory (features)");
  std::memcpy(out->features, feats.data(), feats.size() * sizeof(float));
  out->feature_frames = frames;
  out->tokens = (int*)std::malloc(toks.size() * sizeof(int));
  if (!out->tokens) { std::free(out->features); out->features = nullptr; return err(-5, "out of memory (tokens)"); }
  std::memcpy(out->tokens, toks.data(), toks.size() * sizeof(int));
  out->token_len = (int)toks.size();
  out->prompt_rms = prompt_rms;
  if (st) { st->code = 0; st->message[0] = '\0'; }
  return 1;
}

// luxtts_generate — orchestrate the pipeline (onnx_modeling.py L182 generate_cpu:
// tokenize -> flow-matching sampling -> vocos decode -> rms_norm).
int luxtts_generate(luxtts_ctx* ctx,
                    const char* text,
                    const luxtts_prompt* prompt,
                    luxtts_audio* out,
                    luxtts_status* st) {
  auto* impl = (CtxImpl*)ctx;
  if (out) { out->samples = nullptr; out->num_samples = 0; out->sample_rate = 48000; }
  auto err = [&](int code, const std::string& msg) -> int {
    if (st) { st->code = code; std::snprintf(st->message, sizeof st->message, "%s", msg.c_str()); }
    return 0;
  };
  if (!prompt || prompt->feature_frames <= 0 || prompt->token_len <= 0 || !text || !*text)
    return err(-1, "load a voice first (luxtts_load_voice) and pass non-empty text");

  // onnx_modeling.py L182 (generate_cpu: 1) tokenize the target text)
  std::vector<int> toks = impl->model.tokenizer.texts_to_token_ids(text);
  if (toks.empty()) return err(-2, "text tokenized to zero tokens");

  // sampling.cpp (generate: 2) flow-matching sampling (distill anchor ODE))
  lux::SampleParams p;
  p.num_steps = impl->num_steps;
  p.t_shift = impl->t_shift;
  p.guidance = impl->guidance;
  p.speed = impl->speed;
  // Noise snapshot: seed 0 = fresh randomness per call (the Python pipeline's
  // unseeded torch.randn); nonzero = deterministic, matching
  // torch.manual_seed(seed) + torch.randn (reproducible runs).
  p.seed = impl->seed ? impl->seed : lux::random_seed_u64();
  p.embed = impl->model.embed.data();
  p.embed_dim = impl->model.embed_dim;
  p.pad_id = impl->model.tokenizer.pad_id;

  lux::GeneratedFeatures g = lux::sample_features(
      impl->model.text_encoder, impl->model.fm_decoder,
      toks.data(), (int)toks.size(),
      prompt->tokens, prompt->token_len,
      prompt->features, prompt->feature_frames,
      impl->speed, p);
  if (g.features.empty() || g.frames <= 0)
    return err(-3, "sampling failed (check the model GGUF + prompt features)");

  // onnx_modeling.py L182 (generate_cpu: 3) vocoder -> 48 kHz, clamp(-1,1))
  std::vector<float> wav = impl->model.vocos.decode(g.features.data(), g.frames);
  if (wav.empty()) return err(-4, "vocoder produced no audio");
  for (auto& v : wav) v = std::max(-1.0f, std::min(1.0f, v));
  // onnx_modeling.py L217 (generate_cpu: 4, volume matching): the reference
  // scales the clamped wav DOWN by prompt_rms/target_rms only when the prompt
  // RMS is below target_rms (generate_cpu's 0.1 default) — it never scales up.
  // The port previously FORCED the output RMS to the prompt RMS, which
  // over-amplifies loud prompts into clipping distortion.
  const float vol_target = 0.1f;
  if (prompt->prompt_rms > 0.0f && prompt->prompt_rms < vol_target) {
    const float k = prompt->prompt_rms / vol_target;
    for (auto& v : wav) v *= k;
  }

  if (out) {
    out->samples = (float*)std::malloc(wav.size() * sizeof(float));
    if (!out->samples) return err(-5, "out of memory");
    std::memcpy(out->samples, wav.data(), wav.size() * sizeof(float));
    out->num_samples = (int)wav.size();
    out->sample_rate = 48000;
  }
  if (st) { st->code = 0; st->message[0] = '\0'; }
  return 1;
}

// luxtts_write_wav — write 48 kHz mono audio to a 16-bit PCM WAV.
int luxtts_write_wav(const char* path, const luxtts_audio* audio) {
  if (!path || !*path || !audio || !audio->samples || audio->num_samples <= 0) return 0;
  std::string err;
  return lux::write_wav(path, audio->samples, (size_t)audio->num_samples,
                        (uint32_t)audio->sample_rate, &err) ? 1 : 0;
}

void luxtts_free(luxtts_ctx* ctx) {
  auto* impl = (CtxImpl*)ctx;
  if (!impl) return;
  if (impl->model.gguf_ctx) gguf_free((gguf_context*)impl->model.gguf_ctx);
  if (impl->model.ggml_ctx) ggml_free((ggml_context*)impl->model.ggml_ctx);
  delete impl;
}

void luxtts_free_audio(luxtts_audio* a) {
  if (a && a->samples) std::free(a->samples);
  a->samples = nullptr;
}

void luxtts_free_prompt(luxtts_prompt* p) {
  if (!p) return;
  if (p->features) std::free(p->features);
  p->features = nullptr;
  if (p->tokens) std::free(p->tokens);
  p->tokens = nullptr;
}
