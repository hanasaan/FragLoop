#!/usr/bin/env python
"""Generate shader body from a trained model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import torch

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infshader.model import GPT, ModelConfig
from infshader.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument(
        "--add-bos",
        action="store_true",
        default=True,
        help="prepend <BOS> to the prompt (default: on)",
    )
    parser.add_argument(
        "--no-bos",
        dest="add_bos",
        action="store_false",
        help="do not prepend <BOS> to the prompt",
    )
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--stop-eos",
        action="store_true",
        default=True,
        help="stop when <EOS> is generated (default: on)",
    )
    parser.add_argument(
        "--no-stop-eos",
        dest="stop_eos",
        action="store_false",
        help="do not stop on <EOS>",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def load_model(ckpt_path: str, device: str) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config") or {}
    if "vocab_size" not in cfg_dict:
        raise SystemExit("Checkpoint missing config.vocab_size")
    config = ModelConfig(**cfg_dict)
    model = GPT(config)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    return model


def get_eos_id(tokenizer: Tokenizer) -> Optional[int]:
    ids = tokenizer.encode("", add_bos=False, add_eos=True)
    if not ids:
        return None
    return ids[-1]


def get_special_ids(tokenizer: Tokenizer) -> list[int]:
    ids: list[int] = []
    for tok in ("<BOS>", "<EOS>", "<PAD>", "<CONT>"):
        try:
            ids.append(tokenizer.special_id(tok))
        except Exception:
            continue
    return ids


def trim_after_eos(ids: list[int], eos_id: Optional[int]) -> list[int]:
    if eos_id is None:
        return ids
    try:
        idx = ids.index(eos_id)
    except ValueError:
        return ids
    return ids[:idx]


def strip_special(ids: list[int], remove_ids: set[int]) -> list[int]:
    if not remove_ids:
        return ids
    return [tok for tok in ids if tok not in remove_ids]


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.load(args.tokenizer)
    model = load_model(args.ckpt, args.device)

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()

    input_ids = tokenizer.encode(prompt, add_bos=args.add_bos, add_eos=False)
    idx = torch.tensor([input_ids], dtype=torch.long, device=args.device)

    eos_id = get_eos_id(tokenizer) if args.stop_eos else None
    strip_ids = set(get_special_ids(tokenizer))
    samples = []
    for _ in range(args.num_samples):
        out = model.generate(
            idx,
            max_new_tokens=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p if args.top_p > 0 else None,
            eos_token_id=eos_id,
        )
        ids = trim_after_eos(out[0].tolist(), eos_id)
        ids = strip_special(ids, strip_ids)
        text = tokenizer.decode(ids)
        samples.append(text)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(sample + "\n")
    else:
        for sample in samples:
            print(sample)


if __name__ == "__main__":
    main()
