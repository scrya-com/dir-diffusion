# ComfyUI-DiR-H3

ComfyUI nodes for **DiR** (data-space iteration, arXiv 2610.12102) on **MiniMax H3**: a few-step generator that refines its own clean estimate from one noise sample. The published run uses 4 forward passes and no CFG. Optional, experimental **DWD** decodes the result to pixels without the VAE.

DiR runs inside ComfyUI's own H3 model (`comfy/ldm/minimax/model.py`). It uses the stock UNET, text encoder and VAE loaders, and ComfyUI's model management, previews and progress bar. No ComfyUI file is edited.

## Install

1. Link or copy this folder into `ComfyUI/custom_nodes/`:

   ```bash
   ln -s /path/to/dir-diffusion/comfyui/ComfyUI-DiR-H3 /path/to/ComfyUI/custom_nodes/
   ```

   The folder is self-contained. The code it needs from `lot/` is vendored in `vendor/`.
2. Get the base model: `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (Comfy-Org/MiniMax-H3), the pruned int8 DiT the run was trained on. You also need the stock text encoder and video VAE.
3. Get the DiR run. The loader's default source, `johndpope/dir-diffusion-h3`, downloads it from Hugging Face on first use. A local run folder or `models/dir_h3/<name>` also works.

## Nodes

| Node | What it does | `lot/` counterpart |
|---|---|---|
| **DiR H3 Loader** | Reads `config.json`, `lora_gen.safetensors` and `dir_head.pt` from a run folder or HF repo (`tag`, `ema` pick other saves). Puts the generator LoRA on MODEL and outputs `DIR_H3` (the `DirHead` plus the run's sigmas and state update). | `dir_h3.load_dir_run` |
| **DiR H3 Sampler** | MODEL, DIR_H3, positive conditioning, an H3 AV latent (sets size and length), seed → LATENT. Runs the R iterations. | `dir_h3.dir_sample` / `generator_step` |
| **DiR H3 DWD Loader** (experimental) | Paths to the final-iteration LoRA, the pixel head and the refiner. Local paths or `hf://owner/repo/file`. Blank `refiner_path` skips the refiner. | `refine_dwd_h3.cache` (LoRA), `train_dwd_h3.load_head`, `refine_dwd_h3.Refiner` |
| **DiR H3 DWD Sampler** (experimental) | Same inputs as the sampler plus `DWD_H3` → IMAGE (head + refiner, no VAE) and LATENT (the plain DiR latent, to VAE-decode for comparison). | `dwd_h3.DWDHead.decode`, `refine_dwd_h3.refine_clip` |

**Loader `style_lora` / `style_strength`:** an optional MiniMax H3 LoRA or LoKr, such as an isometric style adapter. It runs in bypass beside the generator LoRA, summed per layer in one injection, so DWD's re-run sees it too. Use this input rather than a Load LoRA (Bypass) node upstream: ComfyUI keeps one bypass injection per model, so the generator LoRA would replace an upstream one, and the loader refuses that model. Results: `docs/ISO.md`.

**Loader `lora_mode`:**
- `bypass (as trained)` is the only mode: the LoRA runs beside the int8 weights through ComfyUI's bypass LoRA, as musubi-tuner trained it (bit-identical to musubi's `up(down(x)) * alpha / rank` arithmetic in bf16). A merge mode was removed: merging re-quantizes the int8 ConvRot weights, and this generator does not tolerate that (see GPU parity below).

**Sampler inputs:**
- `noise_device = gpu` draws the same `z` (and silent-audio noise) as `dir_sample` for that seed. `cpu` gives device-independent noise.
- `sigmas` blank uses the run's schedule (`1, 0.947, 0.857, 0.667`). You can also give one value per iteration, `1` (the paper's pinned form), `grid[:SHIFT]` or `turbo[:SHIFT]`.
- `mix_state` sets the state update between iterations: `from run` (Euler for the published run), `off`, or `on` (re-noise the estimate with the same `z`).

## How the hooks map onto ComfyUI's H3 forward

The sampler adds a `DIFFUSION_MODEL` wrapper to a clone of the model. The wrapper is active only during DiR calls. Inside each call it does the following, mirroring `lot/h3_musubi._lot_forward`:

- **`video_extra`.** The block-0 patch (`patches_replace["dit"]`) rewrites the target-video rows. They become `video_patch_proj(rows(state)) + DirHead.x0_proj(rows(previous estimate))`, from the fp32 state, in `_lot_forward`'s cast order.
- **`video_temb_delta`.** Every block patch appends the timestep row `t_emb[video] + r_embed[r]` and points only the target-video modulation segment at it. Text and audio keep their rows. A pre-hook on `final_layer` does the same for the video head.
- **Silent audio.** The audio rows are rewritten from `sigma_a · eps`, with one `eps` per run from `Generator(seed + 1)`, as `h3_musubi._silence` does. `sigma_a` is fixed to H3's 12 / 3 shifts. A `ModelSamplingMiniMaxH3` (SigmaShift) node upstream does not change it.
- **fp32 estimate.** A forward hook on `final_layer` keeps the fp32 velocity, so the estimate is `state + σ · out` in fp32, as in `generator_step`.
- **Chaining.** Existing `double_block` patches from other packs are chained. A pack that replaces H3's whole forward (for example LoT Apply) makes the sampler stop with an error instead of silently ignoring DiR.

**DWD.** At the final iteration, the patch of the first DWD block (block 42) saves that block's input. After the forward:
1. Blocks 42–49 run again from the saved input, with the DWD LoRA on as forward hooks. `fc2` is forced through its module call, as ComfyUI's bypass LoRA does.
2. The final layer's video modulation gives the tokens `h` that `video_out` would read.
3. `DWDHead.decode(h, pack_rows(state))` and `refine_clip(refiner, pixels, latent)` produce the IMAGE.

Blocks 0–41 carry no DWD LoRA, so this equals a full final iteration with the LoRA on, at 8/50 of the cost. The base pass gives the LATENT.

The refiner's latent is that base estimate: `refine_dwd_h3.py` trained on `collect`'s latent, which had no DWD LoRA. The pixel head (3.15 GB) is memory-mapped and moves to the GPU one class (630 MB) at a time.

## Limits

- Text-to-video, which is what DiR was trained on, and Ref2VA references (MiniMax H3 Reference to Video). First/last-frame conditioning, denoise masks and PDD heads are refused.
- Ref2VA: keep the fl2va DiT. It reads the reference rows (RTX 5090, `hub.png`, 512x768x39, seed 7: identity held with both the VAE and DWD). On `minimax_h3_ref2va_pruned_int8_convrot` the DiR latent is fine through the VAE but DWD decodes to a patch grid: its LoRA and head were trained on fl2va features. Audio stays silent (DiR's silent-audio rows); reference audio is read but not generated.
- Batch size 1. Width and height must be multiples of 32.
- The run was trained at 512×768, 39 frames (12 latent frames).
- DWD is experimental. Its weights are local research files, not yet published, and its fp16 / bf16 head accuracy is unchecked.
- The DWD re-run calls blocks 42–49 directly, so other packs' block patches do not apply to that re-run.

## Example workflows

These are in `../` (dir-diffusion's `comfyui/`), in API format (`*_api.json`) and UI format:

- `dir_h3_video_api.json` / `dir_h3_video.json`: DiR + VAE Decode.
- `dir_dwd_h3_video_api.json` / `dir_dwd_h3_video.json`: DiR + DWD (IMAGE without the VAE), plus a VAE Decode of the same latent for A/B.
- `dir_dwd_h3_ref2va_api.json` / `dir_dwd_h3_ref2va.json`: the same with Ref2VA conditioning (one reference image as `<Picture 1>`).
- `dir_dwd_h3_iso_api.json` / `dir_dwd_h3_iso.json`: DiR + DWD with the isometric style LoKr on the loader's `style_lora`.

`tools/api_to_ui.py` regenerates the UI files from the API files.

## Tests

```bash
# CPU only (no GPU, no server): imports, schemas, both workflows, vendored == lot/, and a tiny random H3
# built as ComfyUI's model and as musubi's with the same weights: the sampler path == lot/dir_h3.dir_sample
CUDA_VISIBLE_DEVICES= /path/to/comfyui/python tests/test_cpu.py [--real-files]

# GPU parity against lot/dir_h3.dir_sample (same seed and text), three processes; see the script's docstring
CUDA_VISIBLE_DEVICES=0 python3 tests/parity_gpu.py all --out /tmp/dir_parity \
    --comfy-python /path/to/comfyui/python --musubi-python /path/to/lot/python
```

GPU parity is judged against musubi's own noise. With the generator LoRA on, a bf16-level change to musubi's forward
(`LOT_H3_SPEED_PATCHES=0`) moves one sigma-1 forward by 0.125 rel_rms (0.017 without the LoRA) and the 4-iteration
result by 0.27, more than the DiR head's whole effect (0.19). ComfyUI vs musubi (RTX 5090, 512x768x39, seed 7):
teacher-forced 0.127 / 0.099 / 0.064 / 0.048 against musubi-vs-musubi 0.125 / 0.084 / 0.060 / 0.042 (worst ratio
1.18, pass <= 1.5), end to end final 0.136 against 0.266. The hook and LoRA arithmetic is checked exactly (fp32,
tiny H3) by `test_cpu.py`.

## Licence

Apache-2.0 (`LICENSE`), as dir-diffusion. `vendor/` holds code copied from dir-diffusion's `lot/` (commit db1f6f3).

MiniMax H3 weights, and the DiR / DWD weights trained from them, are under the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). That licence excludes use in the EU, the UK, the Republic of Korea and the USA.
