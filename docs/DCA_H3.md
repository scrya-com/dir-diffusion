# DeepCrossAttention on H3: a retrofit prototype

**Status: research prototype on `exp/dca-h3`. Nothing is trained. Verdict: not worth a fine-tune now** (reasons at the end).

**DeepCrossAttention (DCA)** is by **Mike Heddes, Adel Javanmard, Kyriakos Axiotis, Gang Fu, MohammadHossein Bateni and Vahab Mirrokni** (Google Research, UC Irvine, USC; ICML 2025; [arXiv 2502.06785](https://arxiv.org/abs/2502.06785), [blog](https://research.google/blog/supercharging-transformer-residual-connections/)). The idea and the equations in "From the paper" are theirs. The port to H3, the AdaLN placement and every estimate below are ours and are not affiliated with the authors.

The paper reports language-model and ImageNet-classification results only. It has nothing on diffusion, video, or dropping blocks from a pretrained model.

## Files

| File | What |
|---|---|
| [`lot/dca_h3.py`](../lot/dca_h3.py) | `Grn`, `DcaH3`, `dca_block`, `run_blocks_dca`, `install(dit, dca)` |
| [`lot/test_dca_h3.py`](../lot/test_dca_h3.py) | 7 CPU tests on a tiny H3 built from musubi-tuner's own classes |
| [`lot/dca_gpu_check_h3.py`](../lot/dca_gpu_check_h3.py) | Real-DiT check: identity, time, memory (`--resident`, `--mix-only`) |
| [`docs/dca_results/`](dca_results/) | Raw JSON of the GPU runs below |

```bash
CUDA_VISIBLE_DEVICES="" python3 lot/test_dca_h3.py                      # last line "ok 7"
cd lot && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 -u dca_gpu_check_h3.py \
    --sizes 512x288x12,768x512x12 --swap 16 --resident 30               # clean per-block times
```

```python
from dca_h3 import DcaH3, install
dca = DcaH3.for_dit(dit, k=2)          # 5.40 M parameters, all zero: the DiT is unchanged
install(dit, dca)                      # forward_video / pih3_sample now run through DCA; install(dit, None) undoes it
dca.skip = frozenset({44, 46, 48})     # blocks that are not run (depth reduction)
```

## From the paper

A residual network keeps a stream `g_t = x + f_0 + … + f_{t-1}`. The paper keeps the stack of terms instead, `G_t = [x, f_0, …, f_{t-1}]`, and feeds each layer a learned mix of it (GRN-v3):

```
g_t = (G_t ⊙ (b_t + w̄_t)) 1          w̄_t = 1 · relu(w_tᵀ G_t)
```

- `b_t` has one weight per feature and per stack entry (`d × t`), initialised to ones.
- `w_t` is one `d`-vector per mix, initialised to zeros. It gives each stack entry one input-dependent scalar per token.
- **DCA** uses three such mixes per block, for the attention's queries, keys and values. Each goes through its own norm. The block's output is `f_t` only; the input is not added to it.
- **First and last-k** (`k-DCA`): the stack is `[x, the sum of the intermediate outputs, the last k outputs]`.
- With `b = 1, w = 0` every mix is the plain stream, so DCA can be added to a pretrained model.

The paper's numbers: a 30-layer DCA beats a 42-layer transformer on LM1B perplexity (each trained from scratch); the transformer's final perplexity is reached in about a third of the training time; `k = 2` has 48% lower inference latency than `k = 24`; retrofitting a 6-layer pretrained model gains 0.19 perplexity in 60k steps.

## As implemented on H3

H3's block is pre-norm with AdaLN and gated updates:

```
h'  = h  + gate_msa ⊙ Attn(mod_msa(norm1(h)))
h'' = h' + gate_mlp ⊙ MLP (mod_mlp(norm2(h')))
```

The stack entry of block `t` is its **gated** update, `f_t = gate_msa ⊙ attn + gate_mlp ⊙ mlp`: what the block adds to the stream. The stack has `k + 2` slots:

| Slot | Content |
|---|---|
| 0 | `x`, the embedded text + audio + video rows |
| 1 | `mid`, the sum of the updates older than the last `k` |
| 2 … k+1 | `f_{t-1}, …, f_{t-k}` |

Each mix is written as the stream plus a correction, with `β = b − 1` stored:

```
mix_m = h + Σ_s (β_{m,s} + relu(w_mᵀ G_s)) ⊙ G_s        m ∈ {Q, K, V, R}
```

This equals the paper's `Σ_s (b + w̄) ⊙ G_s` because the slots sum to `h` (tested in float64). It is written this way because zero parameters then give `h` itself, bit for bit. A sum of the stack entries in bf16 does not reproduce the running sum.

The block with DCA:

```
attn = gate_msa ⊙ Attn(q = mod_msa(norm1(mix_Q)), k = mod_msa(norm1(mix_K)), v = mod_msa(norm1(mix_V)))
mlp  = gate_mlp ⊙ MLP(mod_mlp(norm2(mix_R + attn)))
f_t  = attn + mlp            h ← h + attn + mlp
```

One more mix after the last block is what the final layer reads. Parameters: `4 · d · (k + 3)` per block plus `d · (k + 3)` for the output mix. At `d = 5376, k = 2` that is **5,402,880, or 0.027%** of the DiT's 20.11 B parameters.

**Hook.** `install` wraps `h3_musubi.run_blocks` and stores the DCA on that one `dit`; musubi-tuner is not edited. Block streaming and activation checkpointing work as in `run_blocks`. The DWD re-run tail (`tap=`) raises.

**Q, K, V from three inputs.** `qkv_proj` is one fused Linear. `project_qkv` slices its weight rows (plain and ConvRot int8 Linears), or, for a wrapped Linear such as a LoRA, runs the whole projection on the three inputs and keeps the diagonal thirds (3× the projection's work).

**Dropped blocks.** A block in `dca.skip` is not run and leaves no stack entry. Weights belong to slots, not to block numbers, so the last-k window slides over the blocks that ran, and a slot with nothing in it carries no weight. GRN weights are additive (each starts at 1); there is no sum-to-one constraint to restore.

### Guessed or changed, not from the paper

| Choice | Why | Risk |
|---|---|---|
| The MLP input is `norm2(mix_R + attn)` with a fourth mix R | The paper's figure shows a skip from the stack into "Add & Norm" before the feed-forward; the text does not say which mix it is | A different reading (reuse one of Q/K/V, or the plain sum) is as likely |
| The stack holds gated updates; `norm1` and the MSA modulation are applied to Q, K and V separately | The gates are part of what the stream receives; the paper puts a norm after each GRN | Untested alternative: ungated updates |
| ReLU with gradient 1 at exactly 0 (`relu1`) | With `w = 0` every score is 0, where torch's ReLU has gradient 0, so `w` would never leave the paper's initialisation (tested) | The paper does not say how its `w` starts to move |
| One output mix before the final layer | The paper's `y = (G_{T+1} ⊙ (b + w̄)) 1` | None known |
| The DCA weights do not see the timestep | Simplest; the paper has no conditioning | A diffusion model may want per-timestep mixes |

**A finding that matters for training.** H3's stream is large: RMS 2.1 at the stack's input and **1,670** after 30 blocks, with entries up to 3.7e4 (measured, 768×512×39). The score `wᵀG_s` scales with that. Random weights of standard deviation 1e-6 already change the output by 82% of its largest value; 1e-4 changes it 1,767×. The paper's `relu(wᵀG)` would need a normalised input (for example an RMS norm on `G_s` before `w`) or a very small learning rate on `w`. This is not in the prototype.

## CPU tests

`CUDA_VISIBLE_DEVICES="" python3 lot/test_dca_h3.py`, an 8-block, 64-wide H3 with random weights, through `forward_video`:

| Test | Result |
|---|---|
| Identity at init, zero-parameter shortcut (fused attention), `k` = 0, 1, 2, 4, 8 | Bit-identical velocity and hidden states ✅ |
| Identity at init, general path (mixes computed, Q/K/V projected separately) | Bit-identical in fp32 and bf16 with sliced weights; 2.2e-7 relative with the stacked projection. Asserted tolerance: 1e-5 relative ✅ |
| The mix equals the paper's GRN-v3 | Agrees to 1e-12 in float64 ✅ |
| Gradients | Every `w` and every `β` of an existing slot gets a nonzero gradient (`w`: 0.35 to 13.2, `β`: 0.028 to 0.33); empty slots get none; 0 of 114 DiT parameters get one; checkpointed gradients match ✅ |
| History memory, `k` = 0, 1, 2, 4 | Measured bytes = `(k + 2) · tokens · d · bytes` exactly (33,792 at `k = 2`, 33 tokens) ✅ |
| Dropped blocks | Zero parameters + skip = the plain stack without those blocks, bit-identical; weights on empty slots and on dropped blocks change nothing; random weights + skip stay finite ✅ |
| Hook | Sliced and stacked QKV agree to 1e-5; a second model is untouched ✅ |

## GPU measurements (RTX PRO 4000 Blackwell, 24 GB)

Real int8 ConvRot DiT, `load_dit(swap=16)`, `k = 2`, random text states (128 tokens), one forward at `t = 0.5`, medians of 5 warm runs. The RTX 5090 was not used.

**Identity at init on the real DiT** (full 50-block forward): bit-identical at both sizes, for the shortcut, the general path with sliced int8 weights, and the stacked projection ✅. Two plain forwards are also bit-identical to each other.

**Full forward with block streaming.** The 16 streamed blocks dominate (7.6 s at either size), and their host-to-card copies hide added GPU work. These times are a floor on the overhead, not a measure of it.

| Size (latent frames) | Tokens | Plain | DCA zero | DCA general, zero weights | DCA random weights | Peak plain → DCA | History, measured = formula |
|---|---|---|---|---|---|---|---|
| 512×288×12 | 1,986 | 7.65 s | 7.78 s | 7.91 s | 7.88 s | 14.98 → 15.32 GB | 0.080 GB |
| 768×512×12 (39 frames) | 4,866 | 7.62 s | 7.70 s | 8.00 s | 8.08 s | 15.64 → 16.39 GB | 0.195 GB |

768×512×39 fits the 24 GB card with 16 blocks streamed. Dropping blocks saved nothing here (40 blocks: 7.81 s), because a dropped block is still streamed in this prototype.

**Per-block times without streaming** (`--resident 30`: 30 blocks that stay on the card, run as a stack of their own on the real stack inputs). This is the overhead measurement.

| Per block, ms | 1,986 tokens | 4,866 tokens |
|---|---|---|
| Plain | 13.65 | 45.78 |
| DCA, zero weights, shortcut (stack bookkeeping only) | 14.05 (+2.9%) | 48.04 (+4.9%) |
| DCA, only the R mix on | 17.42 (+27.6%) | 54.38 (+18.8%) |
| DCA general path, zero weights | 17.65 (+29.3%) | 57.58 (+25.8%) |
| **DCA, random nonzero weights** | **19.86 (+45.5%)** | **62.44 (+36.4%)** |
| DCA, stacked QKV projection | 24.42 (+78.9%) | 77.98 (+70.3%) |

The four mixes alone (`--mix-only`): 4.22 ms at 2,795 tokens, 9.08 ms at 4,801, 74.2 ms at 37,631. That is linear in tokens, about 1.9 ms per 1,000.

**Memory.** The extra peak is 0.75 GB at 4,866 tokens, 15 stack-entry sizes (an entry is `tokens · 5376 · 2` bytes = 0.049 GB). Four of those are the history; the rest are the four mixes, the three normed attention inputs and temporaries.

## Back-of-envelope speed estimates

Everything in this section is an estimate. Each assumption is numbered; A3, A5 and A6 are guesses.

**Baselines** (measured, RTX 5090, dense attention, [`PIH3.md`](PIH3.md)):

| Case | Total | DiT | Per block per forward |
|---|---|---|---|
| pi-H3 2-step, 768×512×39 | 1.55 s | 1.47 s | 14.7 ms |
| pi-H3 4-step, 768×512×39 | 3.06 s | 2.98 s | 14.9 ms |
| DiR 4 iterations, 768×512×39 | 5.42 s with the VAE | 3.64 s | 18.2 ms |
| pi-H3 2-step, 1344×768×124 | 30.95 s | 30.32 s | 303 ms |

**Assumptions**

- **A1.** DiT time is `blocks × per-block time`; embedding and the final layer are ignored.
- **A2.** Tokens: `T · (H/32) · (W/32)` video rows, `2 · audio frames` audio rows, 128 text rows. 768×512×39 is 4,608 + 130 + 128 = **4,866**. 1344×768×124 is 37 · 24 · 42 = 37,296 + 414 + 128 = **37,838**.
- **A3 (guess).** The DCA overhead scales from the PRO 4000 to the 5090 like the plain block does: 45.78 / 14.7 = **3.1×**. The mixes are bound by memory bandwidth, where the cards differ by about 2.7× (1,792 against 672 GB/s; the second figure is from memory, not checked), so this may flatter DCA by about 15%.
- **A4.** The overhead is linear in tokens (measured: 6.2 ms at 1,986, 16.7 ms at 4,866, ratio 2.7 for 2.45× the tokens). Attention is quadratic, so the overhead's share falls with size.
- **A5 (guess).** A fused kernel could do the mixes in about a quarter of the time. Basis: the prototype makes about 92 passes over a `tokens × 5376` bf16 tensor per block (per mix and slot: build the coefficient, then `addcmul`), where one kernel needs 9 (read 5 tensors, write 4). The separate norms and QKV projections do not shrink. Not built, not measured.
- **A6 (guess, the big one).** After fine-tuning, a DCA H3 with 40 or 35 blocks matches the 50-block model. The paper's 30-against-42 layers (29% fewer) is for language models trained from scratch at each depth. There is **no evidence it transfers** to a pretrained video DiT, to dropping blocks after the fact, or to a 2-step distilled generator.

**DCA's own overhead per block** (random-weights row, A3, A4):

| | 768×512×39 | 1344×768×124 |
|---|---|---|
| Stack reads and mixes: traffic per block as implemented | 92 × 52 MB = 4.8 GB | 92 × 407 MB = 37 GB |
| Mixes alone, 5090 equivalent | 2.9 ms | 23 ms |
| Separate norms + QKV + bookkeeping, 5090 equivalent | 2.4 ms | 19 ms |
| **Total per block** | **5.3 ms = +36%** | **42 ms = +14%** |
| Per forward, 50 blocks | +0.27 s | +2.1 s |
| With a fused kernel (A5) | 1.3 ms = +9% | 10 ms = +3% |

**Depth reduction** (the only route to a speed-up; A6). Total = DiT + the measured decode (0.08 s; 0.62 s at 124 frames). Percentages are against today's total.

| Case | Overhead | 50 blocks | 40 blocks | 35 blocks | Break-even |
|---|---|---|---|---|---|
| 2-step 768×512×39 (1.55 s) | as implemented | 2.08 s (+35%) | 1.68 s (+9%) | 1.48 s (−4%) | 37 blocks |
| | fused kernel (A5) | 1.68 s (+9%) | 1.36 s (−12%) | 1.20 s (−22%) | 46 |
| | none (bound) | 1.55 s | 1.26 s (−19%) | 1.11 s (−28%) | 50 |
| 4-step 768×512×39 (3.06 s) | as implemented | 4.13 s (+35%) | 3.32 s (+8%) | 2.91 s (−5%) | 37 |
| | fused kernel (A5) | 3.33 s (+9%) | 2.68 s (−12%) | 2.35 s (−23%) | 46 |
| | none (bound) | 3.06 s | 2.46 s (−19%) | 2.17 s (−29%) | 50 |
| 2-step 1344×768×124 (30.95 s) | as implemented | 35.11 s (+13%) | 28.21 s (−9%) | 24.77 s (−20%) | 44 |
| | fused kernel (A5) | 31.99 s (+3%) | 25.72 s (−17%) | 22.58 s (−27%) | 48 |
| | none (bound) | 30.95 s | 24.89 s (−20%) | 21.85 s (−29%) | 50 |

DiR, 4 iterations (DiT 3.64 s): 4.96 s at 50 blocks, 3.97 s at 40 (+9%), 3.48 s at 35 (−5%) as implemented.

At 768×512 the prototype only breaks even below 37 blocks, which is the paper's best case. The "none" row is what dropping blocks buys **without** DCA: a plain block-drop fine-tune has the same ceiling and no overhead. DCA earns its place only if it lets more blocks go than a plain fine-tune would.

**Memory at 1344×768×124 against a 32 GB card** (31.8 GiB on the RTX 5090; the 2-step run peaks at 27.0 GB today):

| | GB |
|---|---|
| One stack entry, 37,838 × 5376 × 2 bytes | 0.38 |
| History at `k = 2` (4 entries) / `k = 4` / `k = 8` | 1.5 / 2.3 / 3.8 |
| Extra peak as implemented (15 entries, measured at 4,866 tokens and scaled by tokens) | 5.8 |
| 50 blocks + DCA as implemented | 32.8: **does not fit** |
| 50 blocks + a lean DCA (history + one mix at a time, about 6 entries; guess) | 29.3: fits with 2.5 GB to spare |
| 40 / 35 blocks + DCA as implemented (0.42 GB per dropped block) | 28.6 / 26.5: fits |

## Retrofit cost

Everything trained on the 50-block stream would read a different stream.

| Piece | Needs refitting? |
|---|---|
| **pi-H3 adapter** (rank-64 LoRA on blocks 0–45, full weights for 46–49, AdaLN in all 50, final layer) | Yes, and it is the hard one. It is the authors' 2/4-step distillation. Dropping blocks removes some of its weights; keeping quality means re-distilling with their recipe (LakonLab), which is not set up here and needs the 40 GB bf16 base this machine has no room for |
| **pi-H3 DWD head** (reads the block stack's output on the video rows) | Yes: its input is the stream after the last block, which the output mix and the dropped blocks change. The authors' head, so again their recipe, or ours |
| **DiR LoRA** (rank 128 on every block) and **our DWD / Stage-1 decoder** | Yes: retrain both ([`DIR.md`](DIR.md), [`DWD.md`](DWD.md)) |
| **Alpha head** (reads decoder taps) | Probably: the taps move if the DWD head is refitted |
| **cam_static dial** | Re-validate (15/15 today); refit if the direction moved |
| ConvRot int8 weights | No: DCA adds fp32 parameters next to them |

**GPU-hours for a DCA + block-drop fine-tune (guess).** Basis: one training step is a teacher forward plus a checkpointed student forward and backward, about 5 forwards. At 768×512×39 a forward is 0.74 s on the 5090 with every block on the card; training has to stream blocks, which costs 2–3× here, so about 8–10 s per step at batch 1. The paper's retrofit ran 60k steps at batch 2,048; a first signal here might take 5k–10k steps: **15–30 RTX 5090 hours** for one depth, one `k`. A usable 2-step model needs the distillation redone on top, the DWD head refitted, and several depths tried: **150–300 hours** would not be surprising. None of this is measured.

## Verdict

**Not worth a real fine-tune now.**

1. **No expected speed-up at the size that matters.** As measured, DCA costs +36% per block at 768×512×39. It needs fewer than 37 blocks to break even, and 35 blocks (the paper's best ratio) gives −4%. A kernel that does not exist yet would make that −22%.
2. **The depth claim does not cover this case.** It is for language models trained from scratch. The paper's only retrofit result is a 6-layer model gaining 0.19 perplexity; it never removes layers from a pretrained model.
3. **A plain block-drop fine-tune is the better first experiment.** It has the same ceiling (−19% at 40 blocks, −28% at 35), no overhead, no new kernel, and it is the control DCA would have to beat anyway.
4. **The retrofit breaks the pi-H3 adapter and its DWD head**, which are the authors' distillation and cannot be redone on this machine today.
5. **Memory:** 50 blocks + DCA as implemented does not fit 1344×768×124 in 32 GB. It fits once blocks are dropped.
6. **The H3 stream's scale** (RMS 1,670) makes the paper's input-dependent weight ill-conditioned without a change to the method.

What would change this: a cheap, training-free probe showing that H3 tolerates dropping blocks at all (zero-weight `dca.skip` is exactly that probe, and costs minutes), followed by a plain block-drop fine-tune that stalls at a depth where DCA could plausibly help.
