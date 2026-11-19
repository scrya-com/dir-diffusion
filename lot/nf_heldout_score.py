"""Face cosine of each held-out render against ITS OWN character's original clip(s), per variant, on the GPU.
Writes <out>/scores.csv and prints a per-variant table. usage: score_heldout.py OUT"""
import collections, csv, json, sys
from pathlib import Path
import numpy as np, imageio.v3 as iio
import onnxruntime
try:
    onnxruntime.preload_dlls()
except Exception:
    pass
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
out = Path(sys.argv[1])
held = json.loads(Path("/media/2TB/lora-data/h3_lot/cache_known_chars/held.json").read_text())
TAIL = 21   # pixel frames of chunk 1 (6 latent frames)


def embeds(path, n=12, start=0):
    frames = iio.imread(path, plugin="pyav")[start:]
    got = []
    for i in np.linspace(0, len(frames) - 1, n).astype(int):
        faces = app.get(frames[i][:, :, ::-1])
        if faces:
            f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            got.append(f.normed_embedding)
    return np.array(got)


refs = {}
for item in held:      # every original clip of a character is its reference
    refs.setdefault(item["character"], []).append(embeds(item["clip"], 16))
ref = {c: (lambda e: e / np.linalg.norm(e))(np.concatenate([e for e in v if len(e)]).mean(0)) for c, v in refs.items()
       if any(len(e) for e in v)}
rows = []
for vdir in sorted(p for p in out.iterdir() if p.is_dir()):
    for item in held:
        path = vdir / f"{item['stem']}.mp4"
        if not path.is_file() or item["character"] not in ref:
            continue
        full, tail = embeds(path), embeds(path, start=TAIL)
        r = ref[item["character"]]
        rows.append({"variant": vdir.name, "character": item["character"], "stem": item["stem"],
                     "full": float((full @ r).mean()) if len(full) else float("nan"),
                     "tail": float((tail @ r).mean()) if len(tail) else float("nan"), "faces": len(full)})
with open(out / "scores.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
by = collections.defaultdict(list)
for r in rows:
    by[r["variant"]].append(r)
for v, rs in by.items():
    full = np.nanmean([r["full"] for r in rs]); tail = np.nanmean([r["tail"] for r in rs])
    ace = [r["full"] for r in rs if r["character"].startswith("Ace")]
    print(f"{v:14s} n={len(rs)} mean full={full:.3f} after-chunk-1={tail:.3f} ace={np.mean(ace):.3f} no-face={sum(r['faces'] == 0 for r in rs)}")
