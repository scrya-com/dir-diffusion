"""Neighbor Forcing held-out character test. For each variant, render every held-out clip's own prompt at
960x544 (NoiseBank seed 0) and write <out>/<variant>/<stem>.mp4.
Variants: dense (DiR-Turbo, 32 frames, no mask: the teacher), zeroshot_cut / zeroshot_fix (DiR-Turbo streamed,
4 chunks = 30 frames, ConvKV memory untrained mean, text cut off / reading chunk 1), s2_<tag> (the Stage 2 run's
checkpoint <tag>, streamed with its trained ConvKV and text_reads_first).
usage: nf_heldout.py OUT RUN_DIR variant [variant ...]"""
import gc, json, sys, time
from pathlib import Path
import imageio, torch
from safetensors import safe_open
from dir_h3 import load_dir_run
from h3_musubi import load_dit, load_vae, pixel_frames
from lot_paths import LOT_H3_CHECKPOINT, LOT_H3_VAE
from nf_h3 import DENSE, ConvKV, NFConfig, NoiseBank, load_nf_dir_head, load_nf_run, nf_dir_parallel_sample
from render_stream_h3 import frames_of

DIR_RUN = "/run/media/johndpope/2TB/lora-data/h3_lot/runs/dir_turbo_clip_iter"
HELD = json.loads(Path("/media/2TB/lora-data/h3_lot/cache_known_chars/held.json").read_text())
W, H, STREAM = 960, 544, 30
out, run_dir, variants = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]


def text_of(path):
    with safe_open(path, "pt") as h:
        hidden, mask = h.get_tensor("hidden_states"), h.get_tensor("attention_mask")
    return hidden[mask].unsqueeze(0).to("cuda", torch.bfloat16)


vae_jobs = []
for variant in variants:
    dit = load_dit(LOT_H3_CHECKPOINT, swap=8)
    if variant.startswith("s2_"):
        tag = "" if variant == "s2_final" else "_" + variant[3:]
        _net, convkv, meta = load_nf_run(run_dir, dit, tag=tag)
        head = load_nf_dir_head(run_dir, meta, tag=tag)
        config, frames, sigmas, mix = NFConfig(memory=True, text_reads_first=True), STREAM, meta["sigmas"], meta.get("mix_state", False)
    else:
        head, _net, meta = load_dir_run(DIR_RUN, dit, tag="_100", ema=True)      # the Stage 2 init
        sigmas, mix = meta["sigmas"], meta.get("mix_state", False)
        convkv = ConvKV.for_dit(dit).cuda().eval()
        config, frames = {"dense": (DENSE, 32), "zeroshot_cut": (NFConfig(memory=True), STREAM),
                          "zeroshot_fix": (NFConfig(memory=True, text_reads_first=True), STREAM)}[variant]
    for item in HELD:
        t0 = time.time()
        with torch.inference_mode():
            x = nf_dir_parallel_sample(dit, head, text_of(item["te"]), frames, lat_h=H // 16, lat_w=W // 16,
                                       config=config, sigmas=sigmas, noise=NoiseBank(0, "cuda"), mix_state=bool(mix),
                                       convkv=None if config is DENSE else convkv)
        vae_jobs.append((variant, item["stem"], x.float().cpu()))
        print(f"sampled {variant} {item['stem']} {item['character']} {time.time() - t0:.0f}s", flush=True)
    del dit, head, convkv, _net; gc.collect(); torch.cuda.empty_cache()

vae = load_vae(LOT_H3_VAE); dtype = next(vae.parameters()).dtype
for variant, stem, x in vae_jobs:
    frames = x.shape[2]
    keep = frames if (frames - 2) % 5 == 0 else frames - (frames - 2) % 5 + 5
    x = torch.cat([x, x[:, :, -1:].expand(-1, -1, keep - frames, -1, -1)], dim=2)
    with torch.inference_mode():
        video = frames_of(vae.decode(x.to("cuda", dtype)))[:pixel_frames(frames)]
    (out / variant).mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out / variant / f"{stem}.mp4", list(video), fps=24, quality=8, macro_block_size=16)
print("HELDOUT_DONE", flush=True)
