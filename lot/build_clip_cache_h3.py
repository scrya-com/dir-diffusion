"""A ``train_h3`` cache from clip latents that already exist: ``z.pt`` + ``meta.json`` folders -> latents and text.

    python3 lot/build_clip_cache_h3.py --src /path/to/clips --out /path/to/cache_clips

Walks ``--src`` for folders holding ``z.pt`` (an H3 video-VAE latent ``(1, 24, T, H, W)``) and ``meta.json``
with a ``prompt``. Nothing is re-encoded by the VAE: each latent is written as
``<stem>_<W>x<H>_minimaxh3.safetensors`` and its prompt as ``<stem>_minimaxh3_te.safetensors`` (Qwen3-VL,
one encode per distinct prompt), the layout ``train_h3.list_items`` reads. Skipped: prompts that reference
other inputs (``<Picture 1>``, ``<Subject 1>``, ``<Audio 1>``; they are not text-to-video prompts), prompts under ``--min-words`` words (labels such as ``neutral``), latents
with fewer than ``--min-frames`` frames or a frame count that is not 5n+2, and paths matching ``--exclude``.
Re-running continues where it stopped. ``--src`` is only read.

Not in ``gpu_guard``'s list: the text encoder (about 15 GB) is meant to run on a second card next to a
training job. Pick the card with ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=12, help="latent frames")
    parser.add_argument("--min-words", type=int, default=5, help="shorter prompts are labels, not captions")
    parser.add_argument("--exclude", action="append", default=["_rejected"], help="path substring to skip (repeatable)")
    parser.add_argument("--limit", type=int, default=0, help="stop after N clips (0 = all)")
    args = parser.parse_args()
    from safetensors.torch import save_file

    args.out.mkdir(parents=True, exist_ok=True)
    captions_path = args.out / "captions.json"
    captions = json.loads(captions_path.read_text()) if captions_path.is_file() else {}
    skipped = {"no_prompt": 0, "tagged_prompt": 0, "shape": 0, "unreadable": 0}
    for latent_path in sorted(args.src.rglob("z.pt")):
        relative = str(latent_path.parent.relative_to(args.src))
        if any(part in relative for part in args.exclude):
            continue
        if args.limit and len(captions) >= args.limit:
            break
        stem = hashlib.sha1(relative.encode()).hexdigest()[:16]
        if stem in captions:
            continue
        try:
            prompt = json.loads((latent_path.parent / "meta.json").read_text()).get("prompt")
        except (OSError, ValueError):
            prompt = None
        if not isinstance(prompt, str) or len(prompt.split()) < args.min_words:   # missing, or a label ("neutral")
            skipped["no_prompt"] += 1
            continue
        if "<" in prompt:
            skipped["tagged_prompt"] += 1
            continue
        try:
            latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001  (a foreign file: count it and go on)
            skipped["unreadable"] += 1
            continue
        if (not torch.is_tensor(latent) or latent.ndim != 5 or latent.shape[:2] != (1, 24)
                or latent.shape[2] < args.min_frames or (latent.shape[2] - 2) % 5):
            skipped["shape"] += 1
            continue
        frames, height, width = latent.shape[2:]
        save_file({f"latent_{frames}x{height}x{width}_float16": latent[0].to(torch.float16).contiguous()},
                  str(args.out / f"{stem}_{width * 16:04d}x{height * 16:04d}_minimaxh3.safetensors"),
                  metadata={"source": relative})
        captions[stem] = prompt.strip()
    captions_path.write_text(json.dumps(captions, indent=1) + "\n")
    print(f"LOT_H3 kind=clip_cache_latents out={args.out} clips={len(captions)} "
          f"prompts={len(set(captions.values()))} skipped={json.dumps(skipped)}", flush=True)

    todo = {stem: text for stem, text in captions.items()
            if not (args.out / f"{stem}_minimaxh3_te.safetensors").is_file()}
    if not todo:
        print("LOT_H3 kind=clip_cache_done text=0 ok=1", flush=True)
        return
    import h3_musubi  # noqa: F401  (puts musubi-tuner on sys.path)
    from lot_paths import LOT_H3_TEXT_ENCODER
    from musubi_tuner.minimax_h3.text_encoder import (
        H3Presentation,
        encode_h3_presentation,
        load_h3_processor,
        load_h3_text_encoder,
    )

    processor = load_h3_processor()
    encoder = load_h3_text_encoder(str(LOT_H3_TEXT_ENCODER), device="cuda")
    encoded: dict[str, torch.Tensor] = {}
    for index, (stem, text) in enumerate(todo.items()):
        if text not in encoded:
            with torch.inference_mode():
                hidden, _tags = encode_h3_presentation(processor, encoder, H3Presentation(text=text, processor_text=text))
            encoded[text] = hidden.reshape(-1, hidden.shape[-1]).to(torch.bfloat16).cpu()
        hidden = encoded[text]
        save_file({"hidden_states": hidden, "attention_mask": torch.ones(hidden.shape[0], dtype=torch.bool)},
                  str(args.out / f"{stem}_minimaxh3_te.safetensors"))
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(todo):
            print(f"LOT_H3 kind=clip_cache_text done={index + 1}/{len(todo)} distinct={len(encoded)} "
                  f"rows={hidden.shape[0]}", flush=True)
    print(f"LOT_H3 kind=clip_cache_done text={len(todo)} ok=1", flush=True)


if __name__ == "__main__":
    main()
