#!/usr/bin/env python3
"""Convert a LuxTTS cloned-voice .pt -> a voice GGUF for luxtts.cpp.

CPU-only (torch.load map_location='cpu'; no GPU at any point).

Input: a voice .pt produced by the LuxTTS Python pipeline — a python dict
(torch.load weights_only=False) with exactly these keys:
    prompt_tokens         list of lists of ints (take [0]: the single prompt)
    prompt_features_lens  torch int64 tensor (1,)  == number of feature frames T
    prompt_features       torch f32 tensor (1, T, 100) — the encode_prompt
                          output, ALREADY the *0.1-scaled fbank features
                          (do NOT re-scale)
    prompt_rms            0-dim f32 tensor (prompt waveform RMS)

Output: <stem>.voice.gguf next to the input (or -o PATH): a GGUF with
general.architecture="lux-voice" (distinct from the model's "lux") and
tensors (luxtts_load_voice reads these EXACTLY — see include/luxtts.h):
    voice.prompt_tokens         i32 [N]
    voice.prompt_features       f32 [100, T]   (logical dims: feat_dim, frames)
    voice.prompt_features_lens  i32 [1]
    voice.prompt_rms            f32 [1]

Features data order: the flat buffer keeps the .pt order — (T,100) row-major,
100 features contiguous per frame — which is exactly what luxtts_encode_prompt
produces and sample_features consumes. The gguf-python writer maps a numpy
shape (T, 100) onto the ggml dims [100, T] (ne[0]=feat_dim=100, ne[1]=T),
which is what the loader validates (feature dim == 100, frames == lens).

Usage:
    uv run --with torch --with gguf python tools/convert_voice.py <voice.pt> [-o out.voice.gguf]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")

FEATURE_DIM = 100  # VocosFbank mel count (the model's feat_dim)


def convert(voice_pt: Path, out: Path) -> None:
    d = torch.load(voice_pt, map_location="cpu", weights_only=False)
    if not isinstance(d, dict):
        sys.exit(f"{voice_pt}: expected a python dict from torch.load, got {type(d).__name__}")
    want = ("prompt_tokens", "prompt_features", "prompt_features_lens", "prompt_rms")
    missing = [k for k in want if k not in d]
    if missing:
        sys.exit(f"{voice_pt}: missing keys {missing} (have {sorted(d.keys())})")

    tokens = d["prompt_tokens"][0]  # the single prompt's token-id list
    feats = d["prompt_features"].detach().to(torch.float32).squeeze(0).contiguous()
    if feats.dim() != 2 or feats.shape[1] != FEATURE_DIM:
        sys.exit(f"prompt_features: expected (1, T, {FEATURE_DIM}), got {tuple(d['prompt_features'].shape)}")
    T = feats.shape[0]
    lens = int(d["prompt_features_lens"].reshape(-1)[0])  # feature frame count
    rms = float(d["prompt_rms"].reshape(-1)[0])
    if T != lens:
        sys.exit(f"prompt_features_lens ({lens}) != feature frame count ({T})")

    gguf = GGUFWriter(out, arch="lux-voice")
    gguf.add_string("lux.voice.name", voice_pt.stem)
    gguf.add_tensor("voice.prompt_tokens",
                    np.array(list(tokens), dtype=np.int32),
                    raw_dtype=GGMLQuantizationType.I32)
    # numpy shape (T, 100) -> ggml dims [100, T] (ne[0]=feat_dim, ne[1]=frames);
    # the flat bytes stay in .pt order (frame-major, 100 features contiguous).
    gguf.add_tensor("voice.prompt_features",
                    feats.numpy(),
                    raw_dtype=GGMLQuantizationType.F32)
    gguf.add_tensor("voice.prompt_features_lens",
                    np.array([lens], dtype=np.int32),
                    raw_dtype=GGMLQuantizationType.I32)
    gguf.add_tensor("voice.prompt_rms",
                    np.array([rms], dtype=np.float32),
                    raw_dtype=GGMLQuantizationType.F32)

    # gguf writer state machine (convert_to_gguf.py idiom): add KV + tensor
    # info while NO_FILE, then open and flush in order; close() finalizes.
    gguf.open_output_file()
    gguf.write_header_to_file()
    gguf.write_kv_data_to_file()
    gguf.write_tensors_to_file()
    gguf.close()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"voice summary: N={len(tokens)} tokens, T={T} frames, rms={rms:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("voice_pt", type=Path, help="cloned-voice .pt (torch dict)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output voice GGUF (default <stem>.voice.gguf next to the input)")
    args = ap.parse_args()
    out = args.out or args.voice_pt.with_suffix(".voice.gguf")
    convert(args.voice_pt, out)
    print("done")


if __name__ == "__main__":
    main()
