# luxtts.cpp

Self-contained, **CPU-only** C++ port of [LuxTTS](https://github.com/ysharma3501/LuxTTS)
inference: text + a voice prompt → 48 kHz speech. Built on
[ggml](https://github.com/ggml-org/ggml) with weights in a single **GGUF** —
no Python, no onnxruntime, no GPU.

```
./build/cli/luxtts --model models/luxtts/luxtts-f32.gguf \
    --text "Hello, this is luxtts.cpp." \
    --prompt my_voice.wav --prompt-text "the exact words in my_voice.wav" \
    -o out.wav
```

## Build

Requires: CMake ≥ 3.20, a C++17 compiler, internet access (FetchContent).

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Options (CPU-only by design — there are no GPU options):

| flag | default | meaning |
|---|---|---|
| `-DLUX_BUILD_CLI=ON` | ON | build the `luxtts` CLI |
| `-DLUX_BUILD_TESTS=ON` | ON | build parity tests (self-skip without fixtures) |
| `-DLUX_SHARED=OFF` | OFF | build `libluxtts.so` |
| `-DLUX_NATIVE=ON` | ON | compile with `-march=native` (build on the target machine; set OFF for portable binaries) |

## Getting the model (one-time, CPU-only)

The GGUF is produced from the upstream torch artifacts
([YatharthS/LuxTTS](https://huggingface.co/YatharthS/LuxTTS)) by a tool —
the equivalent of magpie-tts.cpp's conversion step:

```sh
# download the repo snapshot (HF hub), then:
pip install torch gguf
python tools/convert_to_gguf.py models/luxtts              # -> luxtts-f32.gguf
python tools/convert_to_gguf.py models/luxtts --quant      # -> luxtts-q8_0.gguf
```

`--model` takes the path to the model GGUF file directly (e.g. `models/luxtts/luxtts-f32.gguf`; the q8_0 variant `luxtts-q8_0.gguf` also works — its zipformer weights run on ggml's native q8_0 matmul kernels, f32 activations quantized on the fly, no load-time dequantization).

The `text_encoder.onnx` / `fm_decoder*.onnx` files in the HF repo are **not
used** by this project (they are frozen graphs of the same models; the port
implements the forward passes directly from the torch definitions).

### Model directory layout

```
models/luxtts/
├── luxtts-f32.gguf      # text_encoder + fm_decoder + embed + vocos (one file)
├── luxtts-q8_0.gguf     # optional quantized variant
└── tokens.txt           # Emilia vocab ("<token>\t<id>" per line)
```

Place the HF repo files (`model.pt`, `vocoder/vocos.bin`, `tokens.txt`,
`config.json`) in this directory for the conversion step; at *runtime* the
CLI only needs `luxtts-*.gguf` + `tokens.txt`.

## Usage

```
luxtts --model <path> --text "..." --prompt p.wav --prompt-text "..."
       [-o out.wav] [--steps 4] [--t-shift 0.9] [--guidance 3.0]
       [--speed 1.0] [--rms 0.01] [--duration 5] [--threads 4] [--seed N]
```

| flag | default | meaning |
|---|---|---|
| `--model <path>` | required | path to the model GGUF file (e.g. `models/luxtts/luxtts-f32.gguf`) |
| `--steps N` | 4 | flow-matching ODE steps (distilled model: 3–4 recommended) |
| `--t-shift F` | 0.9 | time-shift; lower = safer pronunciation, higher = quality |
| `--guidance F` | 3.0 | classifier-free guidance scale |
| `--speed F` | 1.0 | speech rate (internally ×1.3, as in the Python pipeline) |
| `--rms F` | 0.01 | prompt RMS target (0.01 ≈ recommended loudness) |
| `--duration S` | 5 | prompt length cap in seconds |
| `--threads N` | 0 (auto) | CPU threads for ggml; `0` = hardware concurrency capped at 16 (the engine's scaling plateau) |
| `--seed N` | 0 | noise seed: 0 = fresh random noise per run (matches the Python pipeline), nonzero = reproducible |

`--prompt-text` is the transcript of the prompt audio (the Python pipeline
transcribes it with whisper-tiny; this port takes the text directly — there is
no ASR dependency). Output is 48 kHz mono 16-bit WAV.

## Library / C API

`include/luxtts.h` exposes a flat C API loadable via `dlopen`/FFI — the magpie
pattern:

- `luxtts_load` — load the model GGUF + tokenizer (+G2P data)
- `luxtts_load_voice` — load a cloned-voice GGUF as a ready prompt
- `luxtts_encode_prompt` — encode a prompt WAV + transcript at runtime
- `luxtts_generate` — text + prompt -> 48 kHz mono audio
- `luxtts_write_wav` — write 48 kHz mono audio to a 16-bit PCM WAV
- `luxtts_free` / `luxtts_free_audio` / `luxtts_free_prompt` — release resources

Link `libluxtts` (static by default) for C++ use.

## Tests

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLUX_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build
```

The sampling-schedule test is closed-form and always runs. The parity tests
(fbank, tokenizer, vocos, e2e) self-skip (CTest "Not Run") unless
`LUX_REF_FIXTURE` points to a reference-fixture dir:

```sh
LUX_REF_FIXTURE=/path/to/fixtures ctest --test-dir build
```

## Scope

Inference only. Not ported: training, dialog mode, TensorRT, prompt ASR
(whisper), Chinese text normalization (jieba/pypinyin — later phase; CJK
segments are skipped with a warning, matching the Python fallback).
