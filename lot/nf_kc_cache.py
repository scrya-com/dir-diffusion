"""Known-character prompt cache for data-free DiR Stage 2 (--ref none: the latent only gives the clip shape).

Writes <out>/train/ and <out>/held/: per clip <id>_minimaxh3_te.safetensors (H3 text encoding of the clip's own
prompt) and <id>_0512x0512_minimaxh3.safetensors (zeros, 24x22x32x32, source = kc/<character>), plus held.json.
Held out: Ace Ventura + 15 random good characters (seed 0), every clip of each."""
import json, random, re, sys
from pathlib import Path
import torch
from safetensors.torch import save_file
import h3_musubi  # noqa: F401
from lot_paths import LOT_H3_TEXT_ENCODER
from musubi_tuner.minimax_h3.text_encoder import H3Presentation, encode_h3_presentation, load_h3_processor, load_h3_text_encoder

KC = Path("/run/media/johndpope/2TB/various/h3-center/known-characters")
out = Path(sys.argv[1])
rows = [json.loads(l) for l in (KC / "prompts.jsonl").read_text().splitlines() if l.strip()]
rows = [r for r in rows if r["status"] in ("good", "onthefence") and r.get("prompt")]
good_chars = sorted({r["character"] for r in rows if r["status"] == "good"})
ace = [c for c in good_chars if c.startswith("Ace Ventura")]
rest = [c for c in good_chars if c not in ace]
held_chars = set(ace + random.Random(0).sample(rest, 15))
print("clips", len(rows), "good characters", len(good_chars), "held", sorted(held_chars), flush=True)

processor = load_h3_processor()
encoder = load_h3_text_encoder(str(LOT_H3_TEXT_ENCODER), device="cuda")
held_meta = []
for i, r in enumerate(rows):
    split = "held" if r["character"] in held_chars else "train"
    stem = re.sub(r"\.mp4$", "", Path(r["clip"]).name).strip("_")
    d = out / split
    d.mkdir(parents=True, exist_ok=True)
    te = d / f"{stem}_minimaxh3_te.safetensors"
    if not te.is_file():
        with torch.inference_mode():
            hidden, _ = encode_h3_presentation(processor, encoder, H3Presentation(text=r["prompt"], processor_text=r["prompt"]))
        hidden = hidden.reshape(-1, hidden.shape[-1]).to(torch.bfloat16).cpu()
        save_file({"hidden_states": hidden, "attention_mask": torch.ones(hidden.shape[0], dtype=torch.bool)}, str(te))
    lat = d / f"{stem}_0512x0512_minimaxh3.safetensors"
    if not lat.is_file():
        save_file({"latent_22x32x32_float16": torch.zeros(24, 22, 32, 32, dtype=torch.float16)}, str(lat),
                  metadata={"source": f"kc/{r['character']}"})
    if split == "held":
        held_meta.append({"stem": stem, "character": r["character"], "status": r["status"],
                          "clip": str(KC / r["status"] / Path(r["clip"]).name), "te": str(te)})
    if i % 50 == 0:
        print("encoded", i, len(rows), flush=True)
(out / "held.json").write_text(json.dumps(held_meta, indent=1))
print("KC_DONE train", len(list((out / "train").glob("*_te.safetensors"))), "held", len(held_meta), flush=True)
