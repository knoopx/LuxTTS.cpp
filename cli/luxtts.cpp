// luxtts CLI: text + a voice prompt WAV -> 48 kHz WAV (CPU-only).
#include "luxtts.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static int usage() {
  std::fprintf(stderr,
    "usage: luxtts --model <path> --text \"...\" (--prompt p.wav --prompt-text \"...\" | --voice v.voice.gguf)\n"
    "       [-o out.wav] [--steps 4] [--t-shift 0.9] [--guidance 3.0] [--speed 1.0]\n"
    "       [--rms 0.01] [--duration 5] [--threads N] [--seed N]\n"
    "       <path> is the model GGUF file (e.g. models/luxtts/luxtts-f32.gguf)\n");
  return 2;
}

int main(int argc, char** argv) {
  std::string model_path, text, prompt_wav, prompt_text, voice_gguf, output = "out.wav";
  int steps = 4, threads = 0, duration = 5;
  float t_shift = 0.9f, guidance = 3.0f, speed = 1.0f, rms = 0.01f;
  unsigned long long seed = 0;
  bool have_text = false, have_prompt = false, have_voice = false;
  if (argc < 2) return usage();
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", flag); std::exit(2); }
      return argv[++i];
    };
    if (a == "--text") { text = val("--text"); have_text = true; }
    else if (a == "--prompt") { prompt_wav = val("--prompt"); have_prompt = true; }
    else if (a == "--voice") { voice_gguf = val("--voice"); have_voice = true; }
    else if (a == "--prompt-text") { prompt_text = val("--prompt-text"); }
    else if (a == "-o" || a == "--output") output = val(a.c_str());
    else if (a == "--steps") steps = std::atoi(val("--steps"));
    else if (a == "--t-shift") t_shift = std::atof(val("--t-shift"));
    else if (a == "--guidance") guidance = std::atof(val("--guidance"));
    else if (a == "--speed") speed = std::atof(val("--speed"));
    else if (a == "--rms") rms = std::atof(val("--rms"));
    else if (a == "--duration") duration = std::atoi(val("--duration"));
    else if (a == "--threads") threads = std::atoi(val("--threads"));
    else if (a == "--model") model_path = val("--model");
    else if (a == "--seed") seed = std::strtoull(val("--seed"), nullptr, 10);
    else if (a == "-h" || a == "--help") return usage();
    else { std::fprintf(stderr, "unknown flag: %s\n", a.c_str()); return usage(); }
  }
  if (model_path.empty() || !have_text || (!have_prompt && !have_voice)) return usage();

  luxtts_config cfg;
  cfg.model_path = model_path.c_str(); cfg.threads = threads;
  cfg.num_steps = steps; cfg.t_shift = t_shift; cfg.guidance = guidance;
  cfg.speed = speed; cfg.target_rms = rms; cfg.prompt_seconds = duration;
  cfg.seed = seed;

  luxtts_status st;
  luxtts_ctx* ctx = luxtts_load(&cfg, &st);
  if (!ctx) { std::fprintf(stderr, "luxtts_load: %s\n", st.message); return 1; }

  luxtts_prompt prompt;
  if (have_voice) {
    if (luxtts_load_voice(ctx, voice_gguf.c_str(), &prompt, &st) != 1) {
      std::fprintf(stderr, "luxtts_load_voice: %s\n", st.message);
      luxtts_free(ctx); return 1;
    }
  } else if (luxtts_encode_prompt(ctx, prompt_wav.c_str(), prompt_text.c_str(), rms, &prompt, &st) != 1) {
    std::fprintf(stderr, "luxtts_encode_prompt: %s\n", st.message);
    luxtts_free(ctx); return 1;
  }
  luxtts_audio audio;
  if (luxtts_generate(ctx, text.c_str(), &prompt, &audio, &st) != 1) {
    std::fprintf(stderr, "luxtts_generate: %s\n", st.message);
    luxtts_free_prompt(&prompt); luxtts_free(ctx); return 1;
  }
  if (luxtts_write_wav(output.c_str(), &audio) != 1) {
    std::fprintf(stderr, "luxtts_write_wav: failed to write %s\n", output.c_str());
    luxtts_free_audio(&audio); luxtts_free_prompt(&prompt); luxtts_free(ctx); return 1;
  }
  std::printf("wrote %s: %d samples @ %d Hz (%.2f s)\n", output.c_str(),
              audio.num_samples, audio.sample_rate,
              (double)audio.num_samples / (double)audio.sample_rate);
  luxtts_free_audio(&audio);
  luxtts_free_prompt(&prompt);
  luxtts_free(ctx);
  return 0;
}
