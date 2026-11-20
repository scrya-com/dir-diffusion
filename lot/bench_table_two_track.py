"""Collect dir_two_track_h3.py meta.json files under a folder into one markdown table (BENCH.md).

Medians over prompts 1.. (prompt 0 carries CUDA and cache warm-up). Columns: total = DiT + upscale + decode."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

LABEL = {"base": "H3, 20 steps", "dir": "DiR, full res", "lr": "DiR, half res, upscaled", "two_track": "DiR two-track"}


def median(values: list, skip: int) -> float | None:
    values = values[skip:] or values
    return round(statistics.median(values), 2) if values else None


def main(root: Path) -> str:
    rows = []
    for meta_path in sorted(root.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        skip = 1 if len(meta["prompts"]) > 1 else 0
        for name in meta["columns"]:
            get = lambda key: median(meta.get(key, {}).get(name, []), skip)
            label = LABEL[name] + (f", switch {meta['switch']}" if name == "two_track" else "")
            if name == "two_track" and (meta.get("switch_upscale", "crop") != "crop" or meta.get("decode", "full") != "full"):
                label += f" (upscale {meta.get('switch_upscale')}, decode {meta.get('decode', 'full')})"
            share = f"{100 * statistics.median(meta['hr_share']):.0f}%" if name == "two_track" and meta["hr_share"] else ""
            rows.append((meta["canvas"], meta["frames"], label, get("seconds"), get("dit_seconds"),
                         get("upscale_seconds"), get("decode_seconds"), share, meta.get("gpu", "")))
    lines = ["| Canvas | Frames | Variant | Total s | DiT s | Upscale s | Decode s | Region | GPU |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows]
    text = "\n".join(lines) + "\n\nMedians over prompts after the first (warm-up). Total = DiT + upscale + decode (VAE).\n"
    (root / "BENCH.md").write_text(text)
    return text


if __name__ == "__main__":
    print(main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/bench_two_track")))
