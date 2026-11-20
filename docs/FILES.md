# What is in this repository

| Path | What |
|---|---|
| [`lot/dir_h3.py`](../lot/dir_h3.py) | `DirHead` / `DirFakeHead`, the generator step, real and fake x0 predictions, LoRA switching, layout schedules, `dir_sample`, run save/load |
| [`lot/train_dir_h3.py`](../lot/train_dir_h3.py) | DMD trainer: self-rollout, generator and fake updates, the paper's weighting / τ / optimizer options, EMA, `--accum`, `--check` |
| [`lot/render_dir_h3.py`](../lot/render_dir_h3.py) | Dense 20-step vs DiR after 1..R iterations on held-out prompts, VAE decode, grid and timings |
| [`lot/parity_dir_h3.py`](../lot/parity_dir_h3.py) | GPU parity for the DiR forward |
| [`lot/probe_dir_h3.py`](../lot/probe_dir_h3.py) | Does a generator refine, and does it read its estimate: rollout and controlled-feed distances on held-out prompts, picture grid |
| [`lot/build_clip_cache_h3.py`](../lot/build_clip_cache_h3.py) | A training cache from clip latents that already exist (`z.pt` + `meta.json` prompt), text-encoded once per distinct prompt |
| [`lot/bench_dir_h3.py`](../lot/bench_dir_h3.py) | Time and peak memory of DiR sampling against the dense sampler, per canvas |
| [`lot/stream_dir_h3.py`](../lot/stream_dir_h3.py) | Streaming DiR: sliding 5n+2 window, per-frame `r`, FL2VA first-frame hand-off, coarse memory keyframes |
| [`lot/train_stream_dir_h3.py`](../lot/train_stream_dir_h3.py) / [`lot/render_stream_h3.py`](../lot/render_stream_h3.py) / [`lot/parity_stream_h3.py`](../lot/parity_stream_h3.py) | Streaming DMD trainer, mp4 renderer, GPU parity and timing |
| [`lot/nf_h3.py`](../lot/nf_h3.py) / [`lot/train_nf_h3.py`](../lot/train_nf_h3.py) / [`lot/render_nf_h3.py`](../lot/render_nf_h3.py) / [`lot/parity_nf_h3.py`](../lot/parity_nf_h3.py) | Experimental, untrained: Neighbor Forcing and ConvKV Memory (SoulX-LiveAct, [arXiv 2603.11746](https://arxiv.org/abs/2603.11746)) on H3: block-causal same-step context, per-step KV caches, a 6-slot convolutional memory. Paper-to-code table, deviations and what was run: [`docs/SOULX_LIVEACT.md`](SOULX_LIVEACT.md) |
| [`DIR_CHEATSHEET.md`](../DIR_CHEATSHEET.md) | Getting started: samples, cache, training recipe, renders, streaming |
| [`lot/h3_musubi.py`](../lot/h3_musubi.py) | H3 runtime glue; `forward_video` with the `video_extra` and `video_temb_delta` hooks |
| [`lot/`](../lot) (rest) | Level-of-Token Diffusion for H3, unchanged from [scrya-com/lot-diffusion](https://github.com/scrya-com/lot-diffusion); start at [`lot/CHEATSHEET.md`](../lot/CHEATSHEET.md) |
| [`third_party/musubi_tuner/`](../third_party) | The unmodified subset of [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) that runs H3 (Apache-2.0; see [`NOTICE`](../NOTICE)) |
| [`comfyui/`](../comfyui) | LoT ComfyUI workflows (DiR has no ComfyUI node yet) |
