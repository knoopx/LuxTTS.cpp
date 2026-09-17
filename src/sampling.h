// sampling.h — flow-matching sampling loop + text-condition mapping.
//
//
//   zipvoice/models/modules/solver.py  L252 (get_time_steps), L239 (DistillEulerSolver)
//   zipvoice/onnx_modeling.py          L110 (sample)
//   zipvoice/models/zipvoice.py        L290 (forward_text_inference_ratio_duration)
//   zipvoice/utils/infer.py            L262 (rms_norm)
#ifndef LUXTTS_SAMPLING_H
#define LUXTTS_SAMPLING_H

#include <vector>

#include "common.h"
#include "zipformer.h"

namespace lux {

// solver.py L252 (get_time_steps: t = t_shift*t/(1+(t_shift-1)*t) over linspace(0,1,n+1))
std::vector<float> get_time_steps(int num_step, float t_shift);

// solver.py L239 (DistillEulerSolver defaults) + onnx_modeling.py L110 (sample args)
struct SampleParams {
  int   num_steps = 4;
  float t_shift = 0.9f;
  float guidance = 3.0f;
  float speed = 1.3f;
  unsigned long long seed = 42;  // 0 = caller draws fresh randomness (upstream)
  // onnx_export.py L120 (OnnxTextModel.forward L112: fused nn.Embedding lookup L132, embed_dim = text_encoder.in_dim)
  const float* embed = nullptr;
  int embed_dim = 192;
  // onnx_export.py L128 (OnnxTextModel.forward: trailing PAD token appended to cat_tokens)
  int pad_id = 0;
};

// onnx_modeling.py L110 (sample) + zipvoice.py L290 (ratio-duration mapping), excluding the vocoder:
//
//   1) features_len = prompt_frames + ceil(prompt_frames / prompt_tokens
//      * text_tokens / speed)                     (ratio-duration)
//   2) text_condition: embed(text tokens) -> text_encoder -> per-frame token
//      index (avg duration = frames // tokens) -> gather  (T, 100)
//   3) speech_condition: prompt mel features (already *0.1) zero-padded to T
//   4) x = randn(T, 100) (seeded); solver.py L181 (EulerSolver.sample) loop:
//        v = fm_decoder(t_cur, x, text_cond, speech_cond, guidance)
//        x1 = x + (1 - t_cur) * v ;  x0 = x - t_cur * v
//        x = (1 - t_next) * x0 + t_next * x1        (final step: x = x1)
//   5) strip the prompt prefix (columns [prompt_frames:]); scale by 1/0.1
struct GeneratedFeatures {
  std::vector<float> features;
  int  frames;
  float prompt_rms;
};

GeneratedFeatures sample_features(Zipformer& text_encoder,
                                  Zipformer& fm_decoder,
                                  const int* text_tokens, int text_len,
                                  const int* prompt_tokens, int prompt_token_len,
                                  const float* prompt_features, int prompt_frames,
                                  float speed_user,
                                  const SampleParams& p);
// infer.py L262 (rms_norm: rms = sqrt(mean(x^2)); scale up only if below target; returns original rms)
float rms_norm(std::vector<float>& wav, float target_rms);

}

#endif // LUXTTS_SAMPLING_H
