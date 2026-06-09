#!/usr/bin/env python
"""Convert an F5-TTS-THAI (or any SWivid-layout F5) PyTorch checkpoint into the
safetensors layout f5-tts-mlx's `from_pretrained(convert_weights=True)` expects.

The MLX loader already does the PyTorch→MLX key remap + conv transposes inline at
load time; it just won't *find* the file because Thai repos ship `.pt`, not
`model_v1.safetensors`. So we lift the EMA weights out of the training checkpoint
and write them as fp32 safetensors under their original keys, plus the vocab.

Usage:
    python convert_thai.py [--repo VIZINTZOR/F5-TTS-THAI] [--pt model_1000000.pt]
                           [--out ~/.bwoc/speaker/models/f5-thai]

Point a `bwoc-speaker` `f5_mlx` voice's `model` at the --out dir.
"""
import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

SKIP = {"initted", "step"}  # bookkeeping tensors the MLX loader ignores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="VIZINTZOR/F5-TTS-THAI")
    ap.add_argument("--pt", default="model_1000000.pt")
    ap.add_argument("--vocab", default="vocab.txt")
    ap.add_argument("--out", default="~/.bwoc/speaker/models/f5-thai")
    a = ap.parse_args()

    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    pt_path = hf_hub_download(a.repo, a.pt)
    vocab_path = hf_hub_download(a.repo, a.vocab)

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("ema_model_state_dict") or ckpt.get("model_state_dict") or ckpt

    weights = {}
    for k, v in sd.items():
        if k.replace("ema_model.", "") in SKIP or not torch.is_tensor(v):
            continue
        if "mel_spec." in k:  # loader drops these too
            continue
        weights[k] = v.contiguous().to(torch.float32)

    out_model = out / "model_v1.safetensors"
    save_file(weights, out_model.as_posix())
    shutil.copyfile(vocab_path, out / "vocab.txt")

    print(f"wrote {len(weights)} tensors → {out_model}")
    print(f"model dir: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
