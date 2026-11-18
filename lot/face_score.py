"""Face-embedding cosine of each video against reference clips (insightface buffalo_l, CPU).
usage: face_score.py REF1.mp4,REF2.mp4 VID1.mp4 [VID2.mp4 ...]"""
import sys
import numpy as np, imageio.v3 as iio
import onnxruntime
try:
    onnxruntime.preload_dlls()
except Exception:
    pass
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

def embeds(path, n=12):
    frames = iio.imread(path, plugin="pyav")
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    out = []
    for i in idx:
        faces = app.get(frames[i][:, :, ::-1])
        if faces:
            f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            out.append(f.normed_embedding)
    return np.array(out)

refs = np.concatenate([embeds(p, 16) for p in sys.argv[1].split(",")])
ref = refs.mean(0); ref /= np.linalg.norm(ref)
print(f"reference faces={len(refs)} self-consistency={float((refs @ ref).mean()):.3f}")
for path in sys.argv[2:]:
    e = embeds(path)
    if len(e) == 0:
        print(f"{path}: no face"); continue
    s = e @ ref
    print(f"{path}: faces={len(e)} cos mean={s.mean():.3f} min={s.min():.3f} max={s.max():.3f}")
