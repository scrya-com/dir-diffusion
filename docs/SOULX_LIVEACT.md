# SoulX-LiveAct on H3: Neighbor Forcing and ConvKV Memory

Paper: **SoulX-LiveAct: Towards Hour-Scale Real-Time Human Animation with Neighbor Forcing and ConvKV Memory**, Dingcheng Zhen, Xu Zheng, Ruixin Zhang, Zhiqi Jiang, Yichao Yan, Ming Tao, Shunshun Yin (Soul AI Lab, HKUST(GZ), Soochow University). [arXiv 2603.11746v2](https://arxiv.org/abs/2603.11746), 18 Mar 2026. Released inference code: [Soul-AILab/SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct) (no training code).

Status: experimental. The code runs and its plumbing is tested (CPU tests, GPU parity, dry passes, short smoke runs; see [What was run](#what-was-run)). No model has been trained to a usable state, and no claim about picture quality is made.

## The paper in one paragraph

An autoregressive video diffusion model generates latent frames in blocks. The paper's question is what a block should be conditioned on. Teacher Forcing uses clean past frames, Diffusion Forcing uses past frames at other noise levels, Self Forcing uses the model's own finished frames. **Neighbor Forcing** uses the past blocks *at the same diffusion step as the block being denoised*: training noises every block of a clip to one shared step and applies a block-causal attention mask (first block 6 latent frames, later blocks 8), so the whole clip is one forward pass with the ordinary flow-matching loss, and at inference the keys and values a block wrote at step `t` are reused unchanged by every later block at step `t`. To keep that cache from growing, **ConvKV Memory** keeps six frame slots of context: two reference frames, two long-term slots and two short-term frames. A long-term slot is a 1D convolution (kernel = stride = 5) over the keys and values of five slots, with its rotary position reset to its slot. Training has two stages: Stage 1 trains the Neighbor Forcing model with the diffusion loss; Stage 2 puts ConvKV into the rollout of a DMD distillation to 3 steps. The base model is Wan2.1 I2V with InfiniteTalk's audio cross-attention; the application is audio-driven talking-human video.

## Files

| File | What |
|---|---|
| [`lot/nf_h3.py`](../lot/nf_h3.py) | The method: block plan, positions, block-causal attention, ConvKV, the one-pass forward, the streaming sampler |
| [`lot/train_nf_h3.py`](../lot/train_nf_h3.py) | Stage 1 and Stage 2 trainer, `--check` dry pass |
| [`lot/parity_nf_h3.py`](../lot/parity_nf_h3.py) | GPU parity on the real H3 |
| [`lot/render_nf_h3.py`](../lot/render_nf_h3.py) | Streams a video block by block and decodes it |
| [`lot/test_lot.py`](../lot/test_lot.py) | `test_nf_*`: six CPU tests on a toy H3 |

Nothing under `third_party/` and no H3 weight is modified. New parameters are the LoRAs (`h3_musubi.make_lora`) and the `ConvKV` module.

## Paper to code

| Paper | What it says | Where it is | How it is checked |
|---|---|---|---|
| Sec. 2.1, fig. 2 left: DiT block with self-attention, text & image cross-attention, audio cross-attention; 3D VAE; flow matching | Wan2.1 I2V + InfiniteTalk audio layers | H3's own DiT (single-stream: text, audio and video rows in one self-attention), H3's VAE and flow head. `nf_h3._run` calls each stock `DiTBlock` with `nf_h3.nf_attention` in place of its `Attention.forward` (an instance attribute for the duration of the call; the class and the weights are untouched) | `test_nf_dense_is_stock_h3`, parity (a): with every row visible it is the stock forward |
| Eq. 1: `x̂^{1:N} = {Ψ_{T:0}(G_θ, c_n, t)}`, `c_n = {x_ref, x_t^{1:n-1}, c_audio, c_text}` | Block `n` is integrated from step `T` to 0 conditioned on the reference latent, the earlier blocks **at step `t`**, audio and text | `nf_h3.nf_stream_sample` (block-outer, step-inner), `NFStream.velocity`. `x_ref` is `first`: H3's FL2VA `first` condition rows. `c_text` is the text rows. `c_audio`: see deviation 1 | `test_nf_stream_equals_parallel`, parity (d)-(f) |
| Eq. 2: `x_t^n = (1 - t) x_0^n + t ε^n` | Every block noised to the same `t` | `train_nf_h3.NFRun.stage1_loss` (one `sigma` for the whole clip) | `test_nf_trainer_updates_on_cpu` rebuilds it by hand |
| Eq. 3: `E_{n,t} ‖(ε^n − x^n) − G_θ(x_t^n, t | x_t^{1:n-1})‖²` | Per-block loss given same-step neighbours | The same loss, all blocks at once (eq. 4) | as eq. 4 |
| Eq. 4: `E_t ‖(ε − x) − G_θ(x_t, t, Mask)‖²`, `t ~ U(0,1)` | All blocks in one masked pass | `NFRun.stage1_loss` -> `nf_h3.nf_forward(config=NFConfig())`; `mse(G, x0 − eps)` (H3's head sign); `--sigma-shift 1` is `U(0,1)` | `test_nf_trainer_updates_on_cpu` |
| Eq. 5: `Mask_{i,j} = 1 if ⌊j/m⌋ ≤ ⌊i/m⌋` | Block-wise causal attention | `nf_h3.nf_attention`: one unmasked attention per block over exactly the rows that block may read (prefix, context, itself). `NFConfig.spans` gives the blocks | `test_nf_causal_is_eq5_mask`: equals one attention under the explicit boolean mask (1e-5); perturbing block `n` leaves blocks `< n` bit-identical. Parity (b) |
| Table 5, sec. 2.2: first block 6, later blocks 8 | Best block sizes | `nf_h3.FIRST_BLOCK`, `NEXT_BLOCK`; `--blocks 6 8` | `test_nf_blocks_and_geometry` |
| Sec. 2.3: KV reuse | Keys/values of a block at step `t` are reused by later blocks at step `t`, append-only | `nf_h3.LayerState` (one per denoising step and DiT block), `NFStream.states[step][layer]` | `test_nf_stream_equals_parallel` |
| Eq. 6: `M_t^{s:e} = (Conv_θ(k_t^{s:e}), Conv_θ(v_t^{s:e}))`, kernel = stride = λ = 5 | Five KV chunks become one | `nf_h3.ConvKVLayer.compress` (`nn.Conv1d`, kernel 5, stride 5, along the frame axis at each spatial position); `ConvKV` holds one per DiT block | `test_nf_convkv_memory` |
| Eq. 7: `RoPE(Conv_θ(k), freq_s)` | Compressed keys take the rotary position of their slot | `nf_h3.Geometry.context_positions`, `window_shift`; keys are cached before the rotation (`LayerState`) and rotated per reader in `nf_attention` | `test_nf_blocks_and_geometry` (slot times, bounded positions) |
| Fig. 3 left, sec. 2.3 "Training of ConvKV Memory": the compressed memory is appended to the keys and values and the mask is modified | Train on a normal clip with the memory visible instead of the frames it replaces | `LayerState.context` inside the one-pass `nf_forward(config=NFConfig(memory=True))`: the third block's key list is `prefix + ref + conv slots + short-term + own` | `test_nf_convkv_memory`: with the convolution zeroed, frames 2-11 cannot reach block 2; with it, they can |
| Fig. 3 right, sec. 2.3 "Inference Pipeline": no compression for the first two blocks; then reference (2 chunks) + long-term (2) + short-term (2) + current block | The context is 6 frame slots | `NFConfig(memory=True)`: `ref = long = short = 2`; `LayerState.context` / `push` | `test_nf_convkv_memory` (exact slot contents over 4 blocks), `test_nf_stream_equals_parallel` (cache rows constant), parity (e), (f) |
| Sec. 2.1, fig. 2 right, Stage 1: Neighbor Forcing training with the diffusion loss | | `train_nf_h3.py --stage 1` | `--check`, smoke run |
| Sec. 2.1, sec. 3, fig. 2 right, Stage 2: ConvKV in the rollout of DMD [29] distillation, 3-step inference; real score and fake score | | `train_nf_h3.py --stage 2`: `NFRun.generator_update` (rollout with ConvKV, DMD direction, real = base H3, fake = base + fake LoRA), `NFRun.fake_update` | `test_nf_trainer_updates_on_cpu`, `--check`, smoke run |
| Sec. 3: 3 steps; the release's `timesteps = [1000, 937.5, 833.3, 0]` | | `nf_h3.few_step_sigmas`: `(1, 0.75, 0.5)` under the schedule shift (Wan's 5 gives the release's numbers; H3's 12 gives 1, 0.973, 0.923) | `test_nf_blocks_and_geometry` |
| Eq. 1: `Ψ`, "e.g. UniPC" | The integrator | `nf_h3.advance`: `renoise` (the release's `generate.py` update: clean estimate, then fresh noise at the next level) or `euler` (H3's sampler) | `test_nf_stream_equals_parallel` (both) |
| Appendix A, propositions 1-2 | Motivation (same-step neighbours stay close); no algorithm | not code | |
| Fig. 1: a causal mask on the pretrained model works zero-shot with same-step context | | `render_nf_h3.py --untrained`, `train_nf_h3.py --stage 1 --check` (prints the base model's loss with and without the mask) | see results |

## Deviations from the paper

Each one is a decision made here. The reviewer should treat these as the places where this port is not the paper.

1. **Audio conditioning without audio cross-attention.** The paper's model is driven by speech through audio cross-attention layers (wav2vec features, layers from InfiniteTalk). H3 has no such layers: its audio is a modality in the same token sequence as the video. The H3 form of `c_audio` here is `--audio driving`: the clip's clean H3 audio latent is fed as the audio rows, at H3's clean-condition time (`AUDIO_CLEAN = 1.0`) instead of the video's noise level, and each block reads the audio of its own time span (`nf_forward(audio=..., audio_t=AUDIO_CLEAN)`, `nf_stream_sample(audio=...)`). The audio latents are `audio_latent.pt` in the clips' source folders (`LOT_H3_CLIPS_SOURCE`); the clip cache itself holds video and text only. The default is still `--audio silent` (silent noise rows at the matching audio sigma, no loss, as in the rest of this repo), because that is what the parity script and the smoke runs used; `--audio driving` has CPU tests and GPU dry passes only (see below). Whether H3 uses clean audio rows the way Wan uses audio cross-attention (lip-sync, the paper's headline result) is untested.
2. **Trained parameters.** Sec. 3 says it for Stage 1 ("In Stage 1, we optimize the audio cross-attention module") and fig. 2 marks both stages with fire and snowflake icons. Stage 1: only the audio cross-attention is trained; self-attention and text & image cross-attention are frozen, so the causal behaviour itself is not trained in Stage 1. Stage 2 (icons only): ConvKV, self-attention and text & image cross-attention are trained, audio cross-attention is frozen; the real score is frozen and the fake score trained. H3 has one attention per block for everything, so the split cannot be reproduced. Here Stage 1 trains a LoRA (rank 16) on `qkv_proj`, `out_proj`, `fc1`, `fc2` of every block; Stage 2 trains that LoRA and ConvKV for the generator and a second LoRA for the fake score. No full fine-tune.
3. **Single-stream prefix.** In Wan, text and the reference image enter by cross-attention and channel concatenation, so the mask of eq. 5 only concerns video tokens. In H3 they are rows of the same self-attention. Here text rows and the FL2VA `first` rows are a *prefix*: every block reads it, and it reads only itself. That is what cross-attention gives Wan (text states do not depend on video) and it is what makes the prefix cacheable. The base H3 lets text rows read video; this port does not, under the causal mask. **Measured cost, and the fix:** cutting the text off from the video loses most of H3's character likeness (face cosine to the original clip: 0.59 streamed against 0.79 dense, 18 held-out clips). `NFConfig(text_reads_first=True)` (`--text-reads-first` in `train_nf_h3.py` / `render_nf_h3.py`) lets the prefix also read block 0, which is still cacheable and makes a one-block clip equal the dense forward; it brings the stream to 0.75 with no training (better on 18/18 clips). Results, scripts and media: `docs/nf_results/`.
4. **Audio rows under the mask.** Each audio frame belongs to the block that contains its time and is visible to that block only. Earlier blocks' audio is not part of any later block's context (the context is video keys/values, as in the paper).
5. **Reference frames.** The paper's "two chunks represent the reference image" are, in the release, the first two latent frames of the first generated block (Wan I2V has put the reference image into them by channel concatenation). Here the same two frames are kept, and in addition the reference image is given the way H3 was trained to take it: one clean latent frame as the FL2VA `first` condition (`--ref first`, the default; `--ref none` removes it).
6. **RoPE reset on H3's time axis.** Eq. 7 says the compressed keys are aligned "to the starting position `s`"; fig. 3 draws them at ℛ2, ℛ3; the release re-indexes the whole 14-slot window every iteration (keys rotated from frame index 0, the current block always at 6..13). H3's rotary time is not a frame index: latent frame `5k` spans one pixel frame and the others four. Here the two reference frames and the two long-term slots take the times of slots 0-3 of the first window (as the release), and the short-term frames and the current block keep their true spacing, translated so the first short-term frame is at slot 4. Positions are bounded for any stream length. Stage 1 (`memory=False`) keeps true positions.
7. **RoPE on values.** Eq. 7 writes `RoPE(Conv_θ(v), freq_s)`. Attention rotates queries and keys only (in H3 and in the release); values are not rotated here.
8. **The convolution's axis.** The paper: "reduces every five chunks of KV states into a single chunk", i.e. along time. The release's `k_compress` runs its `Conv1d` along the flattened token axis of a 5-frame slice, which merges 5 raster-adjacent tokens (and its `kv_mean` option groups the same way). This port follows the paper's text: five frames at the same spatial position.
9. **The convolution's form.** Applied to the keys after `k_norm` and before the rotary rotation, and to the raw values, with no re-normalisation of the compressed keys (as the release; the paper does not say). Depthwise (one filter per channel, 7168 x 5 weights per DiT block), no bias, initialised to `1/5`, i.e. the mean. This is the release's `memory_proj_k`; the paper says only "lightweight 1D convolution". ConvKV cannot be zero-initialised (a zero convolution removes the memory instead of reproducing the base model); the mean is an identity on a still. One convolution serves keys and values, as eq. 6 writes (`Conv_θ` twice) and as the release behaves (`v_compress` calls `memory_proj_k`); `--separate-v` gives values their own.
10. **Recursive long-term memory (a match, listed for its limit).** From the fourth block on the old long-term slots are folded back in: `L0' = conv(L0, L1, S0, S1, C0)`, `L1' = conv(C1..C5)`. This is what fig. 3 (right) draws (the ConvKV arrows at every block take the previous long-term and short-term slots) and what the release does, so it is not a deviation. The limit is in the data: the cached clips have 22 latent frames = 3 blocks, so **training never exercises the fold**; only the samplers and tests do (`test_nf_memory_independent_reference` checks it over 8 blocks against an independent re-derivation).
11. **Block sizes against H3's VAE.** H3's VAE encodes 17-pixel-frame clips of 5 latent frames independently and non-causally. Blocks of 6 and 8 cut through those clips, so a ground-truth context frame carries some information from later pixel frames of its own VAE clip, and decoding needs whole clips (`render_nf_h3.py` decodes the longest `5n + 2` prefix). 6 and 8 are kept because the 2 + 2 + 2 window with ratio 5 allows no other pair (`NFConfig` checks this) and because the cached clips are exactly 6 + 8 + 8 = 22 frames.
12. **Sampler.** `renoise` is the release's few-step update, not UniPC. Stage 1 models can be sampled with H3's Euler schedule (`render_nf_h3.py --steps N`).
13. **Sign.** H3's head predicts `x0 − ε`; the paper's target is `ε − x`.
14. **Stage 2 details the paper does not give.** Chosen here: a random exit step with the earlier steps detached (Self-Forcing's rollout); DMD's normaliser `1 / mean|x0_hat − x0_real|`; `tau` uniform on [0.02, 0.98]; one fake update per generator update; AdamW, lr 1e-4; the real and fake scores are bidirectional and see the same `first` condition as the generator. H3 is guidance-distilled, so there is no CFG on the real score.
15. **Rollout as parallel passes.** Stage 2 does not run the block-by-block loop in training. Under Neighbor Forcing, block `n` at step `i` depends on blocks `< n` at step `i` only, so one masked pass per step computes every block's step; `nf_parallel_sample` is that, and it is tested equal to the block-by-block sampler with the same noise. Gradients flow through the last step's pass, including through the context keys/values (the paper does not say whether the cache is detached).
16. **Scale and data.** Batch 1 (H3's forward takes one sample; `--accum`); 623 Nikki clips of 3 s (about half an hour) instead of 300 hours; no 400-step Stage 2 run. The Nikki clips were themselves generated by H3 (`meta.json`: 20 steps, `res_multistep`), so Stage 1 here regresses onto the base model's own samples of one person, not onto real video. The paper states Stage 2 as 400 steps in sec. 3 and 300 in Table 4.
17. **Not implemented.** The "Emotion and Action Editing Module" (the paper names it and shows fig. 7 but does not describe it; the release switches the text prompt per block, which `nf_stream_sample(text=callable)` supports and `test_nf_stream_equals_parallel` exercises). FP8, sequence parallelism, operator fusion and the 20 FPS system. All evaluation (Sync-C/D, FID, FVD, VBench).

## Data

`train_nf_h3.py` reads an existing H3 clip cache, `/media/2TB/lora-data/h3_lot/cache_atlas_clips` (`LOT_H3_CLIPS_CACHE`): VAE latents `(24, 22, 48, 32)` (73 pixel frames, 512x768) and Qwen3-VL text states, built earlier by `build_clip_cache_h3.py` from `/home/johndpope/Documents/GitHub/h3-atlas/data`. Nothing was re-encoded and nothing is written there. The default `--source set_nk` keeps the Nikki sets (`set_nk_bank`, `set_nk_red`, `set_nk_return` in the files' `source` metadata; `h3-atlas/AGENTS.md` describes `set_nk_bank` as the "Nikki talking-head state bank", and the clips' `meta.json` has `"id": "nikki"`): 623 clips of 22 latent frames with text. `--audio driving` reads `audio_latent.pt` `(1, 32, 2, 122)` from the same source folders (`LOT_H3_CLIPS_SOURCE`). 22 latent frames is exactly three blocks (6 + 8 + 8), which is why the paper's block sizes fit this data without cutting. The raw Nikki folder (`/media/2TB/lora-data/r2-scrya/nikki`: 61 mp4 with sound, expression stills) was inspected and not used: it has no H3 latents, and encoding it would need the VAE and the text encoder on a GPU that is shared.

## What was run

All on 2026-11-18, RTX 5090 (32 GB), pruned int8 ConvRot H3 checkpoint. Commands are run from `lot/`. Commit of each GPU number in this section: parity `ed58072`; the two silent dry passes, the smoke training and the two memory renders `c1c88d4`; the driving-audio dry passes and the no-memory control render the tree committed as `74179cd`. The DiR arm's numbers are in its own section with their commits.

**CPU tests** (`python test_lot.py`, toy 2-layer H3 in fp32, no weights): `ok 52` at `de97766` (`ok 49` before the DiR arm). The first six `test_nf_*`:

| Test | What it proves |
|---|---|
| `test_nf_blocks_and_geometry` | Blocks 6/8; the 2 + 2 + 2 window allows no other sizes; `(1, .75, .5)` under shift 5 is the release's 1000 / 937.5 / 833.3; audio frames tile the clip; positions and timestep rows equal H3's `build_position_grid` / `build_timestep_rows`; the reset window is at slots 0-5 and never moves |
| `test_nf_dense_is_stock_h3` | With every row visible, `nf_forward` equals the stock `dit(...)` (T2VA and FL2VA-first), max abs diff < 1e-6 |
| `test_nf_causal_is_eq5_mask` | The per-block attention equals one attention under the explicit eq. 5 boolean mask (< 1e-5); perturbing block `n` or a later block's audio leaves earlier blocks bit-identical |
| `test_nf_convkv_memory` | Conv1d kernel = stride = 5, mean init, identity on a still, one conv for K and V; exact slot contents over 4 blocks including the recursive fold; in a 1-layer model frames 2-11 reach block 2 only through the convolution |
| `test_nf_stream_equals_parallel` | Block-by-block inference with per-step caches equals the one-pass forward (< 1e-5) and the step-outer sampler (< 1e-4), with and without ConvKV, with and without `first`, 4 blocks, both update rules; the memory cache stays at prefix + 14 frames; a prompt switch at block 2 changes blocks 2+ only |
| `test_nf_trainer_updates_on_cpu` | Stage 1 loss equals eq. 2 + eq. 4 written by hand; the Stage 2 generator update gives non-zero gradients to the generator LoRA and to ConvKV and none to the fake LoRA; the fake update the reverse |

**GPU parity** (`python parity_nf_h3.py`, 22 latent frames at 16x16 latent, `swap=32`):

```
LOT_H3 kind=nf_parity frames=22 latent_hw=16x16 blocks=[(0, 6), (6, 14), (14, 22)] rel_rms_floor=0 a_t2va=0/1 a_first=0/1 b_mask_moved=0.221942 b_leak=0 b_changed=1.02368/1 c_lora0=0/1 d=6.64156e-07 d_first=6.4704e-07 e=6.59505e-07 e_first=6.42699e-07 e_rough=6.58529e-07 same_early=0 control_memory=0.130264 control_conv=0.095602 stream_ok=1 f_cache_rows=[388, 900, 900, 900] f_stream_ms=35754/1 peak_mb=18464.1 dtype=bf16 ok=1
```

This line is from the final forward path (commit `ed58072`, each stock `DiTBlock` called as a module). The run before that change gave the same numbers.

Reading: with the mask off the forward is the stock H3 to the bit (`a_t2va`, `a_first` = 0), which is the only sense in which "step 0 equals the base model" holds; the causal mask alone moves the output by 22% rel. RMS (`b_mask_moved`) and nothing flows backwards (`b_leak` = 0); a zero-initialised LoRA changes nothing (`c_lora0` = 0); block-by-block inference equals the one-pass forward to 6.6e-7 with the Stage 1 context (`d`) and with ConvKV (`e`), against controls of 0.13 (ConvKV context instead of the Stage 1 context) and 0.10 (a perturbed convolution); a 4-block stream keeps 900 cached key rows per step and DiT block from block 1 on (`f`). The first run of this script failed case `a_t2va` at 0.024: the script built its silent audio with a sigma computed from an un-rounded `t`, one float32 ulp away from the model's; fixed in the script, not in the model code.

**Dry passes** (`python train_nf_h3.py --stage N --check --max-latent-hw 32 32 --swap 4`, one Nikki clip, 22 frames):

```
LOT_H3 kind=nf_check stage=1 frames=22 latent_hw=(32, 32) blocks=[(0, 6), (6, 14), (14, 22)] loss_dense=0.04841 loss_causal=0.08004 loss_grad=0.08004 gen_grad=7.233 peak_mb=25196 ok=1
LOT_H3 kind=nf_check stage=2 frames=22 latent_hw=(32, 32) sigmas=1.0000,0.9730,0.9231,0.0000 loss=0.002352 fake_loss=0.06278 gap=0.003489 match_rel=0 exit_step=2 gen_grad=0.3572 conv_grad=0.014 fake_grad=30.31 peak_mb=25724 ok=1
```

With `--audio driving` (the clip's clean audio latent as conditioning; this clip is a silent "listening" take, so the numbers barely move):

```
LOT_H3 kind=nf_check stage=1 audio=driving frames=22 latent_hw=(32, 32) blocks=[(0, 6), (6, 14), (14, 22)] loss_dense=0.04852 loss_causal=0.07991 loss_grad=0.07991 gen_grad=7.711 peak_mb=25196 ok=1
LOT_H3 kind=nf_check stage=2 audio=driving frames=22 latent_hw=(32, 32) sigmas=1.0000,0.9730,0.9231,0.0000 loss=0.0027361 fake_loss=0.05948 gap=0.003586 match_rel=0 exit_step=2 gen_grad=0.858 conv_grad=0.03077 fake_grad=11.55 peak_mb=25724 ok=1
```

`loss_dense` / `loss_causal` is the paper's fig. 1 setting in one number: the untouched base model on a real clip at sigma 0.6 has flow loss 0.048 with every frame visible and 0.080 under the block-causal mask with same-step neighbours. The first two attempts at these dry passes failed in `backward()` and were fixed in the code: the blocks have to be called as modules (the block-swap offloader moves them during backward from module hooks), and the backward has to run while the same LoRAs are switched on as in the forward (activation checkpointing re-runs the blocks).

**Smoke training**, 512x512 crops (`--max-latent-hw 32 32`), `--audio silent`, 621 Nikki clips for training, 2 held out. Stage 1 used `--lr 2e-4` (the default is 1e-4). These show that the trainers run and that the loss moves. They are not trained models.

```
LOT_H3_TRAIN=1 python train_nf_h3.py --stage 1 --steps 30 --lr 2e-4 --max-latent-hw 32 32 --holdout 2 --swap 4 --out runs/nf_smoke_stage1
LOT_H3 kind=nf_eval step=0 heldout_loss=0.09993
LOT_H3 kind=nf_eval step=30 heldout_loss=0.06608
LOT_H3 kind=nf_train_done stage=1 steps=30 out=runs/nf_smoke_stage1 peak_mb=25506 ok=1

LOT_H3_TRAIN=1 python train_nf_h3.py --stage 2 --init runs/nf_smoke_stage1 --steps 5 --max-latent-hw 32 32 --holdout 2 --swap 4 --out runs/nf_smoke_stage2
LOT_H3 kind=nf_train step=1 loss=0 tau=0.202 exit_step=1 match=0 match_rel=0 gap=0 x0_abs=0.90305 real_gap=0.059005 gen_grad=0 conv_grad=0 fake_loss=0.088525 fake_tau=0.43805 seconds=22.633
LOT_H3 kind=nf_train step=2 loss=0.0051615 tau=0.54513 exit_step=2 match=0 match_rel=0 gap=0.006637 x0_abs=0.88354 real_gap=0.088172 gen_grad=4.1803 conv_grad=0.071966 fake_loss=0.038335 fake_tau=0.65281 seconds=39.32
LOT_H3 kind=nf_train step=5 loss=0.0018433 tau=0.8516 exit_step=0 match=0 match_rel=0 gap=0.0077045 x0_abs=1.1586 real_gap=0.17387 gen_grad=10.684 conv_grad=0.054944 fake_loss=0.060939 fake_tau=0.60145 seconds=87.464
LOT_H3 kind=nf_train_done stage=2 steps=5 out=runs/nf_smoke_stage2 peak_mb=26174 ok=1
```

Stage 1: the held-out eq. 4 loss (2 clips x 3 fixed noise levels, fixed noise) went from 0.0999 to 0.0661 in 30 updates at 5.4 s per update. Stage 2: about 16 s per update; the first generator update is exactly zero (`gap=0`) because the fake LoRA starts equal to the base model, and from the second on the generator LoRA and ConvKV receive gradient. The Stage 2 DMD loss value is not a quality measure. Peak memory is 25-26 GB at this crop with `--swap 4`; the full 48x32 latent needs more blocks streamed (not tried).

**Renders** (one held-out clip each; text and first frame from the clip):

```
python render_nf_h3.py --untrained --blocks 4 --size 256 256 --steps 12 --out runs/nf_untrained_render
LOT_H3 kind=render_nf prompt=0 latent_frames=30 steps=12 sampler=euler memory=1 block_ms=10663,10741,11822,11731 abs_mean=0.859 real_abs_mean=0.951 finite=1 peak_mb=20399
LOT_H3 kind=render_nf_done out=runs/nf_untrained_render decoded=1 ok=1

python render_nf_h3.py --run runs/nf_smoke_stage2 --blocks 3 --size 512 512 --out runs/nf_smoke_render
LOT_H3 kind=render_nf prompt=0 latent_frames=22 steps=3 sampler=renoise memory=1 block_ms=9791,11564,10577 abs_mean=0.944 real_abs_mean=0.959 finite=1 peak_mb=20810
LOT_H3 kind=render_nf_done out=runs/nf_smoke_render decoded=1 ok=1
```

```
python render_nf_h3.py --untrained --no-memory --blocks 4 --size 256 256 --steps 12 --out runs/nf_untrained_render_nomem
LOT_H3 kind=render_nf prompt=0 latent_frames=30 steps=12 sampler=euler memory=0 audio=silent block_ms=9276,11827,19954,30416 abs_mean=0.858 real_abs_mean=0.951 finite=1 peak_mb=20392
LOT_H3 kind=render_nf_done out=runs/nf_untrained_render_nomem decoded=1 ok=1
```

One sample each, looked at as a contact sheet, so these are observations and not results.

- Untrained base H3, 12 Euler steps, 4 blocks, same noise with and without memory. The first two blocks are the same in both and are a coherent talking-head clip of one person: on this one sample the paper's fig. 1 claim (a causal mask with same-step neighbours works on the pretrained model without training) holds for 14 latent frames. After that the person drifts in both: with the mean-initialised ConvKV the face changes at the third block, without memory the third block stays closer and the fourth turns away. So the drift is not only the untrained memory; these four-block clips are also longer than anything the cache contains. No conclusion beyond that.
- Time per block: with memory 10.7, 10.7, 11.8, 11.7 s (constant); without, 9.3, 11.8, 20.0, 30.4 s (the context grows). These include moving the caches between host and GPU memory and are not a speed claim.
- The 35-update smoke model at 3 steps: a clean, nearly static 3-second clip with the same person through all three blocks.
- The same render with `--cache-device cuda` ran out of memory (three step caches of about 5 GB each next to a resident model); `cpu` is the default.

Outputs were written under `lot/runs/` (gitignored, not committed; about 450 MB, mostly the three smoke LoRAs). All NF run outputs (checkpoints, renders, logs; 25 GB) have since moved to `/media/2TB/lora-data/h3_lot/runs/nf_worktree/` on the training machine.

## What was not run

- No training beyond the smoke runs above. No Stage 1 run to convergence, no 400-step Stage 2.
- No render of anything beyond the 35-update smoke model, so nothing here shows that the method produces good video on H3.
- `--audio driving` was not trained and not rendered (dry passes only).
- Training at the full 48x32 latent (the smoke runs used 32x32 crops).
- The recursive long-term fold (deviation 10) is exercised by tests and the sampler only, never by a training step.
- No long (minutes) stream, no speed claim.

## Things in the paper that stayed unclear

- Whether the DMD rollout detaches the cache, and every Stage 2 hyperparameter (learning rate, batch, fake/generator update ratio, DMD noise levels). The paper has no hyperparameter appendix: appendix A is the smoothness argument and appendix B is related work.
- What Stage 2 trains is given only by the icons of fig. 2 (deviation 2); Stage 1's trained part is also stated in sec. 3.
- Whether the training-time memory (fig. 3 left) shifts the query positions of the block that reads it. The text only says the memory is appended and the mask modified; this port makes training equal to the inference window (deviation 6).
- Eq. 7's position `s` against fig. 3's ℛ2, ℛ3 against the release's slot indices.
- Eq. 5 with unequal block sizes (6 then 8): taken as "block index of `j` ≤ block index of `i`".
- Whether the convolution is along time (text) or along the token axis (release).
- 400 or 300 Stage 2 steps.
