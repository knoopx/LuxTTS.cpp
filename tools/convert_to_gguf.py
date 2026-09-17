#!/usr/bin/env python3
"""Convert LuxTTS torch weights -> a self-contained GGUF for luxtts.cpp.

CPU-only. No GPU is used at any point (torch.load map_location='cpu').

Inputs (a LuxTTS model dir, e.g. the YatharthS/LuxTTS HF snapshot):
    model.pt            torch state dict: text_encoder.*, fm_decoder.*, embed.*
    vocoder/vocos.bin   torch state dict: backbone.*, upsampler.*, head*, ...
    tokens.txt          "<token>\t<id>" vocab (360 tokens)
    config.json         architecture config (the "model" block)

Outputs:
    luxtts-f32.gguf     all tensors in fp32 (ggml-native layout, see below)
    luxtts-q8_0.gguf    same tensors quantized to q8_0 (smaller, faster)

Tensor layout convention (the C++ loader depends on this EXACTLY):
  - The GGUF keeps the torch state-dict keys VERBATIM as tensor names
    (the magpie-tts.cpp convention), so the C++ loader binds weights by name.
    The vocoder keys are prefixed with "vocos." to keep the two dicts disjoint.
  - Weight tensors (ndim >= 2) are stored with their dims REVERED as a
    value-preserving raw-byte reshape (NOT a value transpose):
        2-D linear  (out, in)  -> (in, out)
        3-D conv    (C, 1, K)  -> (K, 1, C)
    Square 2-D weights are shape-unchanged by the raw-byte reshape (a no-op),
    so they keep their TRUE torch W values (NOT W^T) — the C++ lin()
    (ggml_mul_mat(w, x)) reads w as the (out, in) weight and computes W @ x,
    exactly as the non-square weights do. This is the ggml-native orientation:
    a ggml (in, out) matmul weight keeps `in` contiguous, which is exactly the
    torch (out, in) row-major raw bytes. The C++ lin() does a memcpy (no
    transpose) into a ggml (in, out) tensor, and the conv paths read the raw
    bytes accordingly. Verify any regenerated file with tools/verify_gguf.py
    before shipping.
  - 1-D / 0-D tensors (biases, norms, scales) are stored verbatim.

Requires:  pip install torch gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from gguf import GGUFWriter, GGMLQuantizationType
    from gguf import quants
except ImportError:
    sys.exit("pip install gguf")


# Longest observed key ~73 chars; ggml's default name cap is 64 -> raise it.
GGML_MAX_NAME = 128


def load_state(model_dir: Path):
    model_sd = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    for k in ("state_dict", "model"):
        if isinstance(model_sd, dict) and k in model_sd and isinstance(model_sd[k], dict):
            model_sd = model_sd[k]
            break
    vocos_sd = torch.load(model_dir / "vocoder" / "vocos.bin",
                          map_location="cpu", weights_only=False)
    if isinstance(vocos_sd, dict) and "state_dict" in vocos_sd and isinstance(vocos_sd["state_dict"], dict):
        vocos_sd = vocos_sd["state_dict"]
    return model_sd, vocos_sd


def load_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        return json.load(f)["model"]


def write_gguf(model_dir: Path, out_path: Path, quant: bool) -> None:
    model_sd, vocos_sd = load_state(model_dir)
    cfg = load_config(model_dir)

    gguf = GGUFWriter(out_path, arch="lux")
    # Architecture metadata: the C++ loader reads these to size the two
    # TTSZipformer instances and the vocoder (see src/zipformer.h / vocos.h).
    meta = dict(cfg)
    meta["vocab_size"] = cfg.get("vocab_size", 0)  # filled by the caller if known
    for k, v in meta.items():
        if isinstance(v, (list, tuple)):
            gguf.add_string(k, " ".join(str(x) for x in v))
        elif isinstance(v, bool):
            gguf.add_bool(k, v)
        elif isinstance(v, int):
            gguf.add_int32(k, v)
        elif isinstance(v, float):
            gguf.add_float32(k, v)
        else:
            gguf.add_string(k, str(v))

    # Self-contained model: embed the tokenizer vocab so the model dir needs
    # only the GGUF (the C++ loader reads `luxtts.tokens` from metadata).
    tok_path = model_dir / "tokens.txt"
    if tok_path.exists():
        gguf.add_string("luxtts.tokens", tok_path.read_text())
    else:
        print(f"WARNING: {tok_path} not found — the GGUF will NOT be "
              "self-contained (C++ load will fail)")

    def emit(prefix: str, sd: dict):
        is_vocos = prefix.startswith("vocos.")
        for name, t in sd.items():
            t = t.detach().to(torch.float32).contiguous()
            full = f"{prefix}{name}"
            # Quantize only the zipformer 2-D matmul weights (magpie's
            # selective quant pattern): the vocoder is host-side f32
            # (vocos.*) and the small embed.* lookup table must stay f32,
            # and the q8_0 blocks (32 elements) must run along the
            # innermost (in) dim (e.g. fm_decoder.in_proj (100,300) torch
            # shape: in=300, 300 % 32 != 0 -> stays f32).
            #
            # gguf-python's quantize() blocks along the LAST axis of the
            # array it is given, and ggml q8_0 tensors store 32-element
            # blocks along ne0 (the innermost dim). So the q8_0 pass
            # quantizes the PRE-reversed torch (out, in) array (blocks
            # along in) and stores ggml-order dims (in, out); the f32 path
            # does the dim-reversed raw-byte reshape and stores ggml-order
            # dims likewise. The C++ loader reads the raw bytes directly.
            if quant and t.dim() == 2 and min(t.shape) >= 32 \
                    and full.startswith(("text_encoder.", "fm_decoder.")) \
                    and t.shape[-1] % 32 == 0:
                arr = quants.quantize(t.numpy(), GGMLQuantizationType.Q8_0)
                raw_dtype = GGMLQuantizationType.Q8_0
                dims = list(t.shape)                      # torch (out, in)
            elif is_vocos:
                # vocos.* weights: store in the TORCH layout (no value
                # transpose). The C++ vocos loader (src/vocos.cpp) reads the
                # raw bytes directly into the host conv kernels (which index
                # the torch layout: conv1d (C_out,C_in,K), dwconv (C,1,K),
                # conv_transpose1d (C_in,C_out,K)), so no un-reversing is
                # needed. The gguf writer stores the passed torch-order dims
                # reversed (ggml order, ne0 = innermost), matching the torch
                # row-major raw bytes.
                arr = t.numpy()
                raw_dtype = None
                dims = list(t.shape)                       # torch order
            else:
                # zipformer f32 weights. The gguf writer stores `arr` verbatim
                # (row-major tofile) with the passed torch-order dims REVERSED,
                # so the bytes stay in raw torch layout for every ndim. The
                # C++ engine reads:
                #   2-D linear -> RAW torch (out,in) bytes (lin() uses
                #                 ggml_mul_mat(w, x) with w ne=[in,out]; the
                #                 ggml tensor reads the (out,in) bytes as
                #                 (in,out) — a transposed read that the C++
                #                 engine expects; do NOT transpose here);
                #   3-D conv   -> RAW torch (C,1,K) bytes, stored ne=[K,1,C];
                #                 ggml_conv_1d_dw reads channel c's kernel at
                #                 bytes [c*K, c*K+K) = W[c,0,:] — exactly the
                #                 raw torch row for channel c. Do NOT permute
                #                 (a permute double-reverses the values and
                #                 scrambles every depthwise kernel).
                # Stored dims = reversed(passed dims), so pass torch-order dims.
                arr = t.numpy()
                dims = list(t.shape)                  # torch order -> stored reversed
                raw_dtype = None
            # gguf 0.19 records ti.shape and write_ti_data_to_file() writes
            # it REVERSED, so passing the torch-order dims makes the stored
            # dims end up in ggml order (innermost first, ne[0] = in) —
            # exactly what the C++ reader (ggml 0.24 gguf.cpp: ne[j] =
            # dims[j], no reversal) expects and what luxtts-f32.gguf
            # stores. Passing a non-uint8 dtype skips the byte-shape ->
            # logical-shape conversion (which would crash on dims that are
            # not a multiple of the 34-byte q8_0 row).
            gguf.add_tensor_info(full, dims, np.float32,
                                 arr.nbytes, raw_dtype=raw_dtype)
            gguf.tensors[-1][full].tensor = arr

    emit("", model_sd)      # text_encoder.*, fm_decoder.*, embed.*
    emit("vocos.", vocos_sd)

    # gguf 0.19.0 writer state machine: add KV + tensor info while NO_FILE,
    # then open and flush in order (write_tensors_to_file emits the tensor-info
    # section itself before the weights). close() finalizes the file handles.
    gguf.open_output_file()
    gguf.write_header_to_file()
    gguf.write_kv_data_to_file()
    gguf.write_tensors_to_file()
    gguf.close()
    print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path, help="LuxTTS model dir (model.pt + vocoder/)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output .gguf (default <model_dir>/luxtts-f32.gguf)")
    ap.add_argument("--quant", action="store_true",
                    help="also/instead emit a q8_0 GGUF")
    args = ap.parse_args()

    out = args.out or args.model_dir / ("luxtts-q8_0.gguf" if args.quant else "luxtts-f32.gguf")
    write_gguf(args.model_dir, out, quant=args.quant)
    print("done")


if __name__ == "__main__":
    main()
