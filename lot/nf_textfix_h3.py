"""Text-prefix fix test: DiR-Turbo, Ace prompt, 960x544, NoiseBank seeds; dense vs chunked with the prefix cut
off from the video (old) vs reading chunk 1 (text_reads_first). Writes <out>/<variant>_s<seed>.mp4 and a
_tail.mp4 holding only the frames after chunk 1."""
import gc, sys, time
from pathlib import Path
import imageio, torch
from safetensors import safe_open
from dir_h3 import load_dir_run
from h3_musubi import load_dit, load_vae, pixel_frames
from lot_paths import LOT_H3_CHECKPOINT, LOT_H3_VAE
from nf_h3 import DENSE, ConvKV, NFConfig, NoiseBank, nf_dir_parallel_sample
from render_stream_h3 import frames_of

TE = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).with_name("ace_minimaxh3_te.safetensors")   # the Ace prompt embedding
RUN = "/run/media/johndpope/2TB/lora-data/h3_lot/runs/dir_turbo_clip_iter"
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
W, H = 960, 544
VARIANTS = {   # name: (config, latent frames, memory conv)
    "dense12": (DENSE, 12, False),
    "cut12": (NFConfig(12, 8), 12, False),
    "nf14_cut": (NFConfig(6, 8), 14, False),
    "nf14_fix": (NFConfig(6, 8, text_reads_first=True), 14, False),
    "nf22mem_cut": (NFConfig(memory=True), 22, True),
    "nf22mem_fix": (NFConfig(memory=True, text_reads_first=True), 22, True),
}

with safe_open(str(TE), "pt") as h:
    hidden, mask = h.get_tensor("hidden_states"), h.get_tensor("attention_mask")
dit = load_dit(LOT_H3_CHECKPOINT, swap=8)
text = hidden[mask].unsqueeze(0).to("cuda", torch.bfloat16)
head, network, meta = load_dir_run(RUN, dit)
convkv = ConvKV.for_dit(dit).cuda().eval()          # untrained mean memory, as render_nf_h3 --dir-run --memory
latents = {}
for name, (config, frames, mem) in VARIANTS.items():
    for seed in seeds:
        t0 = time.time()
        with torch.inference_mode():
            x = nf_dir_parallel_sample(dit, head, text, frames, lat_h=H // 16, lat_w=W // 16, config=config,
                                       sigmas=meta["sigmas"], noise=NoiseBank(seed, "cuda"),
                                       mix_state=bool(meta.get("mix_state")), convkv=convkv if mem else None)
        latents[name, seed] = x.float().cpu()
        print(f"sampled {name} s{seed} {time.time() - t0:.0f}s finite={bool(torch.isfinite(x).all())}", flush=True)
del dit, network, head, convkv; gc.collect(); torch.cuda.empty_cache()

vae = load_vae(LOT_H3_VAE); dtype = next(vae.parameters()).dtype
for (name, seed), x in latents.items():
    frames = x.shape[2]
    keep = frames - (frames - 2) % 5
    if keep < frames:   # decode every frame: pad up to the next 5n+2 with the last frame, then drop the padding
        keep = keep + 5
        x = torch.cat([x, x[:, :, -1:].expand(-1, -1, keep - frames, -1, -1)], dim=2)
    with torch.inference_mode():
        video = frames_of(vae.decode(x.to("cuda", dtype)))[:pixel_frames(frames)]
    imageio.mimwrite(out / f"{name}_s{seed}.mp4", list(video), fps=24, quality=8, macro_block_size=16)
    imageio.mimwrite(out / f"{name}_s{seed}_tail.mp4", list(video[pixel_frames(6):]), fps=24, quality=8, macro_block_size=16)
    print("wrote", name, seed, video.shape, flush=True)
print("TEXTFIX_DONE", flush=True)
