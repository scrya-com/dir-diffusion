"""Per-chunk scores of the long-stream renders (``nf_longstream_h3.py``), face model on the GPU.

For every OUT/<arm>/<stem>_s<seed>.mp4 and every chunk (pixel ranges from OUT/chunks.json):
  ref_cos     face cosine to the character's own original clip(s) (as nf_heldout_score.py)
  chunk0_cos  face cosine to this video's chunk-0 face (identity held from the start)
  face_area   largest face box / frame area (close-up drift)
  luma, contrast   mean and std of luma, 0-1
  sharpness   variance of the Laplacian of luma (0-255 scale)
  motion      mean absolute luma difference between consecutive frames, 0-255 scale
Writes OUT/chunk_scores.csv and prints per-arm means for the first, middle and last chunks.
usage: nf_longstream_score.py OUT  (onnxruntime-gpu venv; see docs/nf_results/README.md)"""

from __future__ import annotations

import collections
import csv
import json
import re
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import onnxruntime

try:
    onnxruntime.preload_dlls()
except Exception:
    pass
from insightface.app import FaceAnalysis

HELD = Path("/media/2TB/lora-data/h3_lot/cache_known_chars/held.json")
FACE_FRAMES = 6


def luma(frame: np.ndarray) -> np.ndarray:
    return frame[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def laplacian_var(y: np.ndarray) -> float:
    lap = -4 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:]
    return float(lap.var())


def main() -> None:
    out = Path(sys.argv[1])
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    def faces_of(frames, n):
        got, areas = [], []
        for i in np.linspace(0, len(frames) - 1, n).astype(int):
            faces = app.get(frames[i][:, :, ::-1])
            if faces:
                f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                got.append(f.normed_embedding)
                areas.append((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) / (frames.shape[1] * frames.shape[2]))
        return np.array(got), areas

    held = json.loads(HELD.read_text())
    refs = collections.defaultdict(list)
    for item in held:
        refs[item["character"]].append(faces_of(iio.imread(item["clip"], plugin="pyav"), 16)[0])
    ref = {c: (lambda e: e / np.linalg.norm(e))(np.concatenate([e for e in v if len(e)]).mean(0))
           for c, v in refs.items() if any(len(e) for e in v)}
    character = {item["stem"]: item["character"] for item in held}
    chunks = json.loads((out / "chunks.json").read_text())["pixel_chunks"]

    rows = []
    for mp4 in sorted(out.glob("*/*.mp4")):
        stem, seed = re.match(r"(.+)_s(\d+)$", mp4.stem).groups()
        video = iio.imread(mp4, plugin="pyav")
        r = ref.get(character[stem])
        first = None
        for index, (a, b) in enumerate(chunks):
            frames = video[a:b]
            if not len(frames):
                continue
            emb, areas = faces_of(frames, FACE_FRAMES)
            if index == 0 and len(emb):
                first = emb.mean(0) / np.linalg.norm(emb.mean(0))
            y = np.stack([luma(f) for f in frames])
            rows.append({
                "arm": mp4.parent.name, "stem": stem, "character": character[stem], "seed": int(seed), "chunk": index,
                "faces": len(emb),
                "ref_cos": float((emb @ r).mean()) if len(emb) and r is not None else float("nan"),
                "chunk0_cos": float((emb @ first).mean()) if len(emb) and first is not None else float("nan"),
                "face_area": float(np.mean(areas)) if areas else float("nan"),
                "luma": float(y.mean() / 255), "contrast": float(y.std() / 255),
                "sharpness": float(np.mean([laplacian_var(f) for f in y[:: max(1, len(y) // 4)]])),
                "motion": float(np.abs(np.diff(y, axis=0)).mean()) if len(y) > 1 else float("nan"),
            })
        print("scored", mp4.parent.name, mp4.stem, flush=True)

    with open(out / "chunk_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    last = max(r["chunk"] for r in rows)
    keys = ("ref_cos", "chunk0_cos", "face_area", "luma", "contrast", "sharpness", "motion")
    print("arm      chunks   " + "  ".join(f"{k:>10s}" for k in keys) + "  no-face")
    for arm in sorted({r["arm"] for r in rows}):
        for name, pick in (("0", {0}), ("1-3", {1, 2, 3}), (f"{last - 3}-{last}", set(range(last - 3, last + 1)))):
            sel = [r for r in rows if r["arm"] == arm and r["chunk"] in pick]
            print(f"{arm:8s} {name:8s} " + "  ".join(f"{np.nanmean([r[k] for r in sel]):10.3f}" for k in keys)
                  + f"  {sum(r['faces'] == 0 for r in sel)}")


if __name__ == "__main__":
    main()
