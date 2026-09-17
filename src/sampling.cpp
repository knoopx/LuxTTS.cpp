// sampling.cpp — flow-matching sampling loop (DistillEulerSolver port).
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace lux {

std::vector<float> get_time_steps(int num_step, float t_shift) {
  // solver.py L252 (get_time_steps: t = t_shift*t/(1+(t_shift-1)*t) over linspace(0,1,n+1))
  std::vector<float> ts(num_step + 1);
  for (int i = 0; i <= num_step; ++i) {
    float t = (float)i / num_step;
    ts[i] = t_shift * t / (1.0f + (t_shift - 1.0f) * t);
  }
  return ts;
}

GeneratedFeatures sample_features(Zipformer& text_encoder,
                                  Zipformer& fm_decoder,
                                  const int* text_tokens, int text_len,
                                  const int* prompt_tokens, int prompt_token_len,
                                  const float* prompt_features, int prompt_frames,
                                  float speed_user,
                                  const SampleParams& p) {
  GeneratedFeatures g;
  if (prompt_frames <= 0 || prompt_token_len <= 0 ||
      !(text_tokens || text_len == 0) || !(prompt_tokens && prompt_features))
    return g;

  const int F = text_encoder.cfg().out_dim;
  if (F <= 0) return g;

  // zipvoice.py L309 (forward_text_inference_ratio_duration):
  //   features_lens = prompt_features_lens
  //                 + ceil(prompt_features_lens / prompt_tokens_lens * tokens_lens / speed)
  // The prompt keeps its full P frames at its own rate and ONLY the text part
  // is scaled by speed. (The ONNX export wrapper instead divides the whole
  // cat length by speed — ceil(P/PT*(PT+Len)/speed) — which shortens the
  // generated part by P*(1-1/speed) frames and is the source of the rushed /
  // distorted output for long prompts; speed is always 1.3 in the pipeline.)
  const float speed = speed_user * 1.3f;
  const int cat_len = prompt_token_len + text_len;
  const int T = prompt_frames + (int)std::ceil((float)prompt_frames /
                                  (float)prompt_token_len * (float)text_len / speed);
  if (T <= prompt_frames) return g;

  // zipvoice.py L187 (forward_text_embed): cat_tokens = [prompt, text] + ONE
  // trailing pad (pad_labels appends pad_id); the pad row is masked. The
  // token->frame expansion is forward_text_condition (L228): token_dur =
  // features_lens // cat_len for every real token, the remainder tail maps to
  // the pad row (get_tokens_index appends num_frames - sum(durations)).
  std::vector<float> text_cond((size_t)T * F, 0.0f);
  if (cat_len > 0) {
    const int enc_len = cat_len + 1;
    std::vector<int> cat(enc_len);
    std::copy(prompt_tokens, prompt_tokens + prompt_token_len, cat.begin());
    std::copy(text_tokens, text_tokens + text_len, cat.begin() + prompt_token_len);
    cat[cat_len] = p.pad_id;

    const int ed = p.embed_dim > 0 ? p.embed_dim : text_encoder.cfg().in_dim;
    if (!p.embed || ed != text_encoder.cfg().in_dim) return g;
    if (p.pad_id < 0) return g;
    std::vector<float> embed((size_t)enc_len * (size_t)ed);
    for (int i = 0; i < enc_len; ++i) {
      const int tok = cat[i];
      if (tok < 0) return g;
      std::memcpy(embed.data() + (size_t)i * ed, p.embed + (size_t)tok * ed,
                  (size_t)ed * sizeof(float));
    }
    // zipvoice.py L187 (forward_text_embed): pad_labels appends pad_id and
    // make_pad_mask(tokens_lens, ...) marks ONLY that trailing pad position;
    // the fm decoder is called with no padding mask (zipvoice.py L309 — the
    // mask is all-False for batch 1).
    std::vector<char> te_pad((size_t)enc_len, 0);
    te_pad[enc_len - 1] = 1;
    std::vector<float> enc = text_encoder.forward(embed.data(), enc_len, -1.0f, 0.0f,
                                   (const bool*)te_pad.data());
    if ((int)enc.size() != enc_len * F) return g;
    const int token_dur = T / cat_len;
    const int real_frames = cat_len * token_dur;
    for (int f = 0; f < T; ++f) {
      // zipvoice.py L277 (get_tokens_index): frames before real_frames use
      // row f/token_dur; the remainder tail uses the pad row (cat_len)
      const int row = (token_dur > 0 && f < real_frames) ? f / token_dur : cat_len;
      std::memcpy(text_cond.data() + (size_t)f * F, enc.data() + (size_t)row * F, (size_t)F * sizeof(float));
    }
  }

  // zipvoice.py L290 (speech condition: the prompt features are zeroed beyond
  // prompt_features_lens — the pad below keeps the prompt prefix and zeros the rest)
  std::vector<float> speech((size_t)T * F, 0.0f);
  std::memcpy(speech.data(), prompt_features, (size_t)prompt_frames * F * sizeof(float));

  // solver.py L239 (DistillEulerSolver) + onnx_modeling.py L110 (seeded noise)
  const std::vector<float> ts = get_time_steps(p.num_steps, p.t_shift);
  std::vector<float> x((size_t)T * F);
  fill_noise(x, p.seed);
  std::vector<float> fm_in((size_t)T * 3 * F);
  for (int step = 0; step < p.num_steps; ++step) {
    const float t_cur = ts[step], t_next = ts[step + 1];
    // xt = cat([x, text_cond, speech_cond], dim=2)  (zipvoice.py:163); the
    // fm_decoder in_dim is 3*F ("x|text_cond|speech_cond"), so the ODE step
    // consumes the per-frame concatenation, not x alone.
    for (int i = 0; i < T; ++i) {
      std::memcpy(fm_in.data() + (size_t)i * 3 * F, x.data() + (size_t)i * F,
                  (size_t)F * sizeof(float));
      std::memcpy(fm_in.data() + (size_t)i * 3 * F + F, text_cond.data() + (size_t)i * F,
                  (size_t)F * sizeof(float));
      std::memcpy(fm_in.data() + (size_t)i * 3 * F + 2 * F, speech.data() + (size_t)i * F,
                  (size_t)F * sizeof(float));
    }
    std::vector<float> v = fm_decoder.forward(fm_in.data(), T, t_cur, p.guidance, nullptr);
    if ((int)v.size() != T * F) return g;
    for (size_t i = 0; i < (size_t)T * F; ++i) {
      const float x1 = x[i] + (1.0f - t_cur) * v[i];
      const float x0 = x[i] - t_cur * v[i];
      x[i] = (step == p.num_steps - 1)
                 ? x1
                 : (1.0f - t_next) * x0 + t_next * x1;
    }
  }

  // onnx_modeling.py L110 (sample: strip the prompt prefix; /0.1 undoes the
  // prompt-features scaling, generate_cpu: pred_features.permute(0, 2, 1) / 0.1)
  const int frames = T - prompt_frames;
  g.features = std::vector<float>((size_t)F * frames);
  for (int f = 0; f < frames; ++f)
    for (int c = 0; c < F; ++c)
      g.features[(size_t)c * frames + f] =
          x[(size_t)(prompt_frames + f) * F + c] / 0.1f;
  g.frames = frames;
  return g;
}

float rms_norm(std::vector<float>& wav, float target_rms) {
// infer.py L262 (rms_norm: sqrt(mean(x^2)); scale up only if below target; return original rms)
  if (wav.empty()) return 0.0f;
  double acc = 0.0;
  for (float v : wav) acc += (double)v * v;
  float rms = (float)std::sqrt(acc / (double)wav.size());
  if (rms > 0.0f && rms < target_rms) {
    float s = target_rms / rms;
    for (auto& v : wav) v *= s;
  }
  return rms;
}

}
