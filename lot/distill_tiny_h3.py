"""Self-distillation of the published DiR generator onto H3-Tiny (H3 with blocks dropped, ``h3_musubi.H3_TINY_DROP``).

    # GPU dry pass: teacher / student rollouts against the probe's saved ones, a few steps, memory peak
    python3 lot/distill_tiny_h3.py --check --run RUN --out OUT --probe-dir /media/2TB/lora-data/h3_lot/tiny
    # the run (resumes from OUT/last.pt when it exists)
    LOT_H3_TRAIN=1 python3 lot/distill_tiny_h3.py --run RUN --out OUT --probe-dir ... --hours 7.25
    # pass check after probe_dir_h3.py --name tiny_ft (CPU)
    python3 lot/distill_tiny_h3.py --report --out OUT --probe-dir ... --name tiny_ft

Why distillation and not the DiR trainer again: the target is "behave like the full model", and the full model is
right here. One model copy plays both parts. The TEACHER is all 50 blocks with the published LoRA and DiR head
(frozen). The STUDENT is the same weights with the blocks skipped, the published LoRA, a NEW zero-initialised LoRA
(``delta``, rank ``--rank``, fp32) and a trainable copy of the DiR head. At step 0 the student IS the untuned
H3-Tiny, and the teacher never moves (nothing it uses is trained). The DMD trainer needs a fake model, 4 fake updates per
generator update and about 72 s per step, and it pulls toward base H3, not toward the published generator.

Loss, per sample (``Distill.sample``): the teacher's 4-iteration rollout from one ``z`` gives the inputs
``(x_r, x0_hat[r])`` and the targets ``x0_hat[r+1]`` of every iteration. The student redoes each iteration from the
teacher's inputs (MSE to the teacher's output, the last iteration weighted double). With probability ``p_on``
(0 for the first ``--on-policy-start`` of the budget, then ``--on-policy``) the inputs are the STUDENT's own rollout
instead, and the targets the teacher's answer from those inputs (DAgger): it learns to correct its own drift.

Order (activation checkpointing): every no-grad forward (teacher, then student) runs first; the grad forwards and
their backwards run last, in student mode, because the backward recomputes through the blocks' forward as it is then.

Saves in ``--out``: ``last.pt`` (delta, head, optimizer, step, clock; every ``--save-minutes``), ``best/`` and
``final/`` (run dirs that ``probe_dir_h3.py --run`` / ``bench_accel.sh RUN=`` load: the published rank-128 LoRA and
the delta concatenated along the rank, an exact merge), ``log.jsonl``. ``best`` is picked by the held-out eval
(every ``--eval-minutes``): the final estimate's mean |student - teacher| on held-out prompts that are NOT the
probe's three. The probe's items (held-out item 0..2 of the stills cache) are scored too, for the log only.

Metric note: ``probe_dir_h3.py``'s ``own_to_dense`` is a DISTANCE (mean |x0_hat - dense20|): lower is closer.
Untuned tiny is 0.910, the full model 0.929, so "own_to_dense >= 0.925" would reward moving away from dense20.
The pass check (``--report``) uses the distance to the full model's own final estimate instead (``pass_check``).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

TAG = "LOT_H3 kind=tiny_ft"


# ----------------------------------------------------------------------------- pure helpers

def merge_lora_states(pub: dict, delta: dict, dtype=torch.bfloat16) -> dict:
    """Two stacked LoRAs (outputs added) as ONE LoRA of rank r1 + r2: down rows and up columns concatenated, each
    part's scale (alpha / rank) folded into its down rows, alpha = the new rank (scale 1). Exact up to ``dtype``."""
    suffix = ".lora_down.weight"
    names = sorted(k[:-len(suffix)] for k in pub if k.endswith(suffix))
    if names != sorted(k[:-len(suffix)] for k in delta if k.endswith(suffix)):
        raise ValueError("the two LoRAs wrap different modules")
    merged = {}
    for name in names:
        downs, ups = [], []
        for state in (pub, delta):
            down, up = state[name + suffix].float(), state[name + ".lora_up.weight"].float()
            rank = down.shape[0]
            alpha = float(state[name + ".alpha"]) if name + ".alpha" in state else float(rank)
            downs.append(down * (alpha / rank))
            ups.append(up)
        down, up = torch.cat(downs, 0), torch.cat(ups, 1)
        merged[name + suffix] = down.to(dtype).contiguous()
        merged[name + ".lora_up.weight"] = up.to(dtype).contiguous()
        merged[name + ".alpha"] = torch.tensor(float(down.shape[0]), dtype=dtype)
    return merged


def parse_sizes(spec: str) -> list[tuple[int, int]]:
    """``"32x48,48x32"`` -> latent ``(H, W)`` crop caps (even, so whole 2x2 patches)."""
    sizes = []
    for part in spec.split(","):
        h, _, w = part.strip().lower().partition("x")
        h, w = int(h), int(w)
        if h < 2 or w < 2 or h % 2 or w % 2:
            raise ValueError(f"size {part!r}: need even latent H x W")
        sizes.append((h, w))
    if not sizes:
        raise ValueError("no sizes")
    return sizes


def iteration_weights(iters: int, last: float) -> list[float]:
    """Loss weight per iteration, summing to 1, the last one ``last`` times the others."""
    raw = [1.0] * (iters - 1) + [float(last)]
    return [w / sum(raw) for w in raw]


def lr_factor(step: int, frac: float, warmup: int, floor: float = 0.1) -> float:
    """Linear warm-up over ``warmup`` steps, then a cosine over the time budget (``frac`` = elapsed / budget) to ``floor``."""
    warm = min(1.0, step / warmup) if warmup > 0 else 1.0
    frac = min(max(frac, 0.0), 1.0)
    return warm * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * frac)))


def on_policy_prob(frac: float, start: float, prob: float) -> float:
    return 0.0 if frac < start else float(prob)


def mad(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean())


def pass_check(pub: dict, tiny: dict, ft: dict, pub_json: dict, tiny_json: dict, ft_json: dict, evals: list,
               name: str = "tiny_ft", iters: int = 4, ratio: float = 0.5, dense_band: float = 0.02) -> dict:
    """The pass check. ``pub`` / ``tiny`` / ``ft``: ``probe_dir_h3.py`` ``.pt`` dicts (keys ``(i, "<name>_r<r>")``);
    ``*_json`` their ``.json``; ``evals``: this script's eval records (``log.jsonl`` lines with ``"eval"``).

    Pass when all three hold:
    1. ``to_full``: mean over the probe items of |final estimate - the full model's final estimate| is at most
       ``ratio`` x the untuned tiny's (recovers the full model's output, the thing distillation optimises);
    2. ``own_to_dense``: the mean is within ``dense_band`` of the full model's (both directions: it is a distance,
       and being further from dense20 is not better);
    3. held-out (not the probe items): the best eval's ``select`` is below the step-0 eval's, for every cache.
    """
    items = sorted({i for i, _ in pub})
    final = f"_r{iters}"
    base = [mad(tiny[(i, "tiny" + final)], pub[(i, "pub" + final)]) for i in items]
    mine = [mad(ft[(i, name + final)], pub[(i, "pub" + final)]) for i in items]
    dense = {k: sum(row["own_to_dense"][-1] for row in j["items"]) / len(j["items"])
             for k, j in (("full", pub_json), ("tiny", tiny_json), ("ft", ft_json))}
    first = next((e for e in evals if e.get("step") == 0), None)
    best = min(evals, key=lambda e: e["select"]) if evals else None
    held = {}
    if first and best:
        held = {c: {"step0": first["by_cache"][c], "best": best["by_cache"][c]} for c in first["by_cache"]}
    checks = {
        "to_full": sum(mine) / len(mine) <= ratio * sum(base) / len(base),
        "own_to_dense": abs(dense["ft"] - dense["full"]) <= dense_band,
        "heldout": bool(held) and all(v["best"] < v["step0"] for v in held.values()),
    }
    return {"to_full_tiny": [round(x, 4) for x in base], "to_full_ft": [round(x, 4) for x in mine],
            "to_full_tiny_mean": round(sum(base) / len(base), 4), "to_full_ft_mean": round(sum(mine) / len(mine), 4),
            "own_to_dense_mean": {k: round(v, 4) for k, v in dense.items()},
            "user_literal_own_to_dense_ge_0.925": dense["ft"] >= 0.925,
            "heldout": held, "best_step": None if best is None else best["step"],
            "checks": checks, "pass": all(checks.values())}


# ----------------------------------------------------------------------------- the two models in one

class Distill:
    """Teacher and student on one DiT. ``step`` is one DiR iteration ``(head, state, estimate, r, text, seed) ->
    x0_hat[r+1]`` (default: ``dir_h3.generator_step``); tests pass a stand-in."""

    def __init__(self, dit, drop, teacher_head, student_head, delta_net, sigmas, step=None, device="cuda"):
        self.dit, self.drop = dit, tuple(drop)
        self.teacher_head, self.student_head = teacher_head, student_head
        self.delta_net, self.sigmas, self.device = delta_net, list(sigmas), device
        self.iters = len(self.sigmas)
        self._step = step or self._generator_step

    def _generator_step(self, head, state, estimate, r, text, seed):
        from dir_h3 import generator_step

        audio = torch.Generator(self.device).manual_seed(seed)
        return generator_step(self.dit, head, state, estimate, r, text, sigma=self.sigmas[r], audio_generator=audio)

    @contextlib.contextmanager
    def mode(self, student: bool):
        from h3_musubi import blocks_dropped

        saved = [(m, m.enabled) for m in self.delta_net.unet_loras]
        for m, _ in saved:
            m.enabled = student
        try:
            with blocks_dropped(self.dit, self.drop, student):
                yield self.student_head if student else self.teacher_head
        finally:
            for m, was in saved:
                m.enabled = was

    def rollout(self, head, z, text, seed):
        """``(inputs, outputs)``: iteration r's ``(state, estimate)`` and its ``x0_hat[r+1]``, r = 0..R-1."""
        from dir_h3 import next_state

        inputs, outputs = [], []
        state, estimate = z, torch.zeros_like(z)
        for r in range(self.iters):
            inputs.append((state, estimate))
            estimate = self._step(head, state, estimate, r, text, seed).float()
            outputs.append(estimate)
            if r + 1 < self.iters:
                state = next_state(state, estimate, self.sigmas[r], self.sigmas[r + 1])
        return inputs, outputs

    def sample(self, z, text, seed, *, on_policy: bool, weights, loss_scale: float = 1.0, backward: bool = True) -> dict:
        with torch.no_grad():
            with self.mode(False) as head:
                t_in, t_out = self.rollout(head, z, text, seed)
            gap = None
            if on_policy:
                with self.mode(True) as head:
                    s_in, s_out = self.rollout(head, z, text, seed)
                gap = mad(s_out[-1], t_out[-1])
                with self.mode(False) as head:      # iteration 0's input is (z, 0) for both: reuse the teacher's answer
                    targets = [t_out[0]] + [self._step(head, *s_in[r], r, text, seed).float() for r in range(1, self.iters)]
                inputs = s_in
            else:
                inputs, targets = t_in, t_out
        losses = []
        with self.mode(True) as head, torch.enable_grad():
            for r in range(self.iters):
                out = self._step(head, *inputs[r], r, text, seed).float()
                loss = F.mse_loss(out, targets[r])
                if backward:
                    (loss * weights[r] * loss_scale).backward()
                losses.append(float(loss.detach()))
        return {"loss_r": losses, "loss": sum(w * l for w, l in zip(weights, losses)), "on_policy": int(on_policy),
                "rollout_gap": gap}

    @torch.no_grad()
    def final(self, student: bool, z, text, seed) -> list[torch.Tensor]:
        with self.mode(student) as head:
            return self.rollout(head, z, text, seed)[1]


# ----------------------------------------------------------------------------- GPU side

def eval_items(caches, probe_cache, holdout, probe_items):
    """``(probe, select)``: lists of ``(cache name, index, item)``. Probe = the first ``probe_items`` held-out items
    of ``probe_cache`` (what ``probe_dir_h3.py --items N`` scores); select = every other held-out item."""
    import train_h3

    probe, select = [], []
    for cache in caches:
        _train, held = train_h3.list_items(cache, holdout)
        for i, item in enumerate(held):
            (probe if (Path(cache) == Path(probe_cache) and i < probe_items) else select).append((Path(cache).name, i, item))
    return probe, select


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, help="the published DiR run dir (teacher; lora_gen, dir_head, config.json)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, action="append", default=None,
                        help="H3 caches (default: $ACCEL_CACHES, colon-separated, else the stills cache)")
    parser.add_argument("--probe-cache", type=Path, default=None, help="default: the stills cache (probe_dir_h3's)")
    parser.add_argument("--probe-dir", type=Path, default=None, help="probe_dir_h3 outputs: pub.pt, tiny.pt, ref.pt (+ json)")
    parser.add_argument("--probe-items", type=int, default=3)
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--drop", default="h3tiny", help="blocks the student skips: h3tiny or indices")
    parser.add_argument("--rank", type=int, default=64, help="the delta LoRA's rank")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accum", type=int, default=2, help="samples per update (each trains all R iterations)")
    parser.add_argument("--last-weight", type=float, default=2.0)
    parser.add_argument("--on-policy", type=float, default=0.5)
    parser.add_argument("--on-policy-start", type=float, default=0.25, help="fraction of the budget before any on-policy sample")
    parser.add_argument("--sizes", default="32x48,48x32,32x32,24x48,40x40,32x40", help="latent H x W crop caps, one drawn per sample")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--hours", type=float, default=7.25, help="training wall clock (summed over resumes)")
    parser.add_argument("--save-minutes", type=float, default=30)
    parser.add_argument("--eval-minutes", type=float, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="rollouts vs the probe's, --check-steps updates, memory; writes nothing")
    parser.add_argument("--check-steps", type=int, default=3)
    parser.add_argument("--report", action="store_true", help="the pass check (CPU), after probe_dir_h3 --name NAME")
    parser.add_argument("--name", default="tiny_ft")
    args = parser.parse_args()
    if args.report:
        return report(args)
    if not args.check and os.environ.get("LOT_H3_TRAIN") != "1":
        raise SystemExit("training is gated: set LOT_H3_TRAIN=1 (or --check)")
    if os.environ.get("LOT_H3_DROP_BLOCKS"):
        raise SystemExit("unset LOT_H3_DROP_BLOCKS: the teacher needs every block (the student's skip is --drop)")
    train(args)


def train(args) -> None:
    import train_dir_h3
    import train_h3
    from dir_h3 import load_dir_run, parse_sigmas
    from h3_musubi import H3_TINY_DROP, load_dit, make_lora
    from lot_paths import LOT_H3_STILLS_CACHE

    caches = args.cache or [Path(p) for p in os.environ.get("ACCEL_CACHES", "").split(":") if p] or [LOT_H3_STILLS_CACHE]
    probe_cache = args.probe_cache = args.probe_cache or LOT_H3_STILLS_CACHE
    drop = H3_TINY_DROP if args.drop == "h3tiny" else tuple(sorted({int(x) for x in args.drop.replace(",", " ").split()}))
    sizes = parse_sizes(args.sizes)
    train_h3.CROP_MULTIPLE = 2
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    train_items = []
    for cache in caches:
        train_items += train_h3.list_items(cache, args.holdout)[0]
    probe, select = eval_items(caches, probe_cache, args.holdout, args.probe_items)

    dit = load_dit(device=device, swap=0, train=True)
    teacher_head, pub_net, config = load_dir_run(args.run, dit, device=device)
    sigmas = parse_sigmas(config.get("sigmas") or "1", int(config["iters"]))
    if config.get("mix_state"):
        raise SystemExit("mix_state runs are not supported here")
    delta = make_lora(dit, args.rank)                 # wraps the published LoRA; .enabled switches it alone
    delta.requires_grad_(True)
    delta.to(device=device, dtype=torch.float32)
    student_head = copy.deepcopy(teacher_head).train()
    student_head.requires_grad_(True)
    teacher_head.requires_grad_(False)
    delta_params = [p for m in delta.unet_loras for p in m.parameters() if p.requires_grad]
    head_params = list(student_head.parameters())
    opt = torch.optim.AdamW([{"params": delta_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}],
                            betas=(0.9, 0.99), weight_decay=0.0)
    base_lrs = [args.lr, args.head_lr]
    distill = Distill(dit, drop, teacher_head, student_head, delta, sigmas, device=device)
    weights = iteration_weights(distill.iters, args.last_weight)
    trainable = sum(p.numel() for p in delta_params) + sum(p.numel() for p in head_params)
    print(f"{TAG} event=start items={len(train_items)} probe={len(probe)} select={len(select)} caches={len(caches)} "
          f"drop={','.join(map(str, drop))} rank={args.rank} trainable={trainable} sizes={args.sizes} frames={args.frames} "
          f"sigmas={[round(s, 4) for s in sigmas]} allocated_mb={torch.cuda.memory_allocated() / 2**20:.0f}", flush=True)

    def load(entry, index, frames=args.frames, cap=(32, 48)):
        train_h3.MAX_LATENT_HW = cap
        return train_dir_h3.load_item(entry, random.Random(1000 + index), device, frames)

    if args.check:
        return check(args, distill, probe, weights, load, opt, delta_params + head_params, sizes, train_items, rng)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = dict(config)
    meta.update({"rank": int(config["rank"]) + args.rank, "distill": {
        "from": str(args.run), "drop": list(drop), "delta_rank": args.rank, "lr": args.lr, "head_lr": args.head_lr,
        "accum": args.accum, "last_weight": args.last_weight, "on_policy": args.on_policy,
        "on_policy_start": args.on_policy_start, "sizes": args.sizes, "frames": args.frames, "hours": args.hours,
        "caches": [str(c) for c in caches], "holdout": args.holdout}})

    # Teacher finals for the eval items, once (the teacher never changes).
    eval_set = []
    probe_keys = {(n, k) for n, k, _ in probe}
    for name, i, entry in probe + select:
        x0, text = load(entry, i)
        z = torch.randn(x0.shape, device=device, generator=torch.Generator(device).manual_seed(i))
        eval_set.append({"cache": name, "i": i, "probe": (name, i) in probe_keys,
                         "z": z.cpu(), "text": text.cpu(), "teacher": distill.final(False, z, text, 7 * i)[-1].cpu()})
    dense = {}
    if args.probe_dir is not None and (args.probe_dir / "ref.pt").is_file():
        ref = torch.load(args.probe_dir / "ref.pt")
        dense = {i: ref[(i, "dense20")] for i in range(args.probe_items) if (i, "dense20") in ref}

    log = open(args.out / "log.jsonl", "a")
    state = {"step": 0, "elapsed": 0.0, "best": None, "evals": []}
    last = args.out / "last.pt"
    if last.is_file():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        delta.load_state_dict(saved["delta"])
        student_head.load_state_dict(saved["head"])
        opt.load_state_dict(saved["opt"])
        state.update({k: saved[k] for k in ("step", "elapsed", "best", "evals")})
        rng.setstate(saved["py_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        print(f"{TAG} event=resume step={state['step']} elapsed_h={state['elapsed'] / 3600:.2f}", flush=True)

    def save_last():
        tmp = last.with_suffix(".tmp")
        torch.save({"delta": delta.state_dict(), "head": student_head.state_dict(), "opt": opt.state_dict(),
                    **{k: state[k] for k in ("step", "elapsed", "best", "evals")}, "py_rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, tmp)
        os.replace(tmp, last)
        print(f"{TAG} event=saved_last step={state['step']}", flush=True)

    def save_run(name):
        from safetensors.torch import load_file, save_file

        run_dir = args.out / name
        run_dir.mkdir(exist_ok=True)
        pub = load_file(str(args.run / "lora_gen.safetensors"))
        merged = merge_lora_states(pub, {k: v.detach().cpu() for k, v in delta.state_dict().items()})
        del pub
        save_file(merged, str(run_dir / "lora_gen.safetensors.tmp"), {"lot_h3": "dir", "kind": "gen", "rank": str(meta["rank"])})
        os.replace(run_dir / "lora_gen.safetensors.tmp", run_dir / "lora_gen.safetensors")
        torch.save(student_head.state_dict(), run_dir / "dir_head.pt")
        (run_dir / "config.json").write_text(json.dumps({**meta, "step": state["step"]}, indent=1))
        print(f"{TAG} event=saved_run name={name} step={state['step']} out={run_dir}", flush=True)

    @torch.no_grad()
    def evaluate():
        by_cache, probe_rows = {}, []
        for row in eval_set:
            z, text = row["z"].to(device), row["text"].to(device)
            outs = distill.final(True, z, text, 7 * row["i"])
            gap = mad(outs[-1], row["teacher"])
            if row["probe"]:
                probe_rows.append({"i": row["i"], "to_full": round(gap, 4),
                                   "to_dense": round(mad(outs[-1], dense[row["i"]]), 4) if row["i"] in dense else None})
            else:
                by_cache.setdefault(row["cache"], []).append(gap)
        by_cache = {c: round(sum(v) / len(v), 4) for c, v in by_cache.items()}
        select_mean = round(sum(by_cache.values()) / len(by_cache), 4)    # each cache counts once
        record = {"step": state["step"], "elapsed_h": round(state["elapsed"] / 3600, 3), "select": select_mean,
                  "by_cache": by_cache, "probe": probe_rows}
        state["evals"].append(record)
        log.write(json.dumps({"eval": True, **record}) + "\n")
        log.flush()
        improved = state["best"] is None or select_mean < state["best"]
        print(f"{TAG} event=eval step={state['step']} select={select_mean} by_cache={json.dumps(by_cache)} "
              f"probe_to_full={[p['to_full'] for p in probe_rows]} probe_to_dense={[p['to_dense'] for p in probe_rows]} "
              f"best={int(improved)}", flush=True)
        if improved and state["step"] > 0:
            state["best"] = select_mean
            save_run("best")
        elif state["best"] is None:
            state["best"] = select_mean

    if state["step"] == 0:
        evaluate()
    budget = args.hours * 3600
    params = delta_params + head_params
    mark = time.perf_counter()
    since_save = since_eval = 0.0
    while state["elapsed"] < budget:
        state["step"] += 1
        frac = state["elapsed"] / budget
        for group, lr in zip(opt.param_groups, base_lrs):
            group["lr"] = lr * lr_factor(state["step"], frac, args.warmup)
        p_on = on_policy_prob(frac, args.on_policy_start, args.on_policy)
        opt.zero_grad(set_to_none=True)
        stats = []
        for _ in range(args.accum):
            entry = rng.choice(train_items)
            train_h3.MAX_LATENT_HW = rng.choice(sizes)
            x0, text = train_dir_h3.load_item(entry, rng, device, args.frames)
            z = torch.randn_like(x0)
            seed = rng.randrange(2**31 - 1)
            s = distill.sample(z, text, seed, on_policy=rng.random() < p_on, weights=weights, loss_scale=1.0 / args.accum)
            if not math.isfinite(s["loss"]):
                raise RuntimeError(f"step {state['step']}: loss {s['loss']}")
            s["shape"] = list(x0.shape[2:])
            stats.append(s)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
        opt.step()
        now = time.perf_counter()
        state["elapsed"] += now - mark
        since_save += now - mark
        since_eval += now - mark
        mark = now
        loss = sum(s["loss"] for s in stats) / len(stats)
        loss_r = [round(sum(s["loss_r"][r] for s in stats) / len(stats), 5) for r in range(distill.iters)]
        gaps = [s["rollout_gap"] for s in stats if s["rollout_gap"] is not None]
        record = {"step": state["step"], "loss": loss, "loss_r": loss_r, "grad_norm": grad_norm,
                  "lr": opt.param_groups[0]["lr"], "p_on": p_on, "on": sum(s["on_policy"] for s in stats),
                  "rollout_gap": sum(gaps) / len(gaps) if gaps else None, "shapes": [s["shape"] for s in stats],
                  "elapsed_h": round(state["elapsed"] / 3600, 4)}
        log.write(json.dumps(record) + "\n")
        if state["step"] % 10 == 0:
            log.flush()
            print(f"{TAG} step={state['step']} loss={loss:.5f} loss_r={loss_r} grad_norm={grad_norm:.3g} "
                  f"lr={record['lr']:.3g} p_on={p_on} gap={record['rollout_gap']} elapsed_h={record['elapsed_h']} "
                  f"s_per_step={state['elapsed'] / state['step']:.2f} peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f}",
                  flush=True)
        if since_eval >= args.eval_minutes * 60:
            evaluate()
            since_eval = 0.0
            mark = time.perf_counter()          # eval time is not training time
        if since_save >= args.save_minutes * 60:
            save_last()
            since_save = 0.0
            mark = time.perf_counter()
    evaluate()
    save_last()
    save_run("final")
    if not (args.out / "best").is_dir():
        save_run("best")
    log.close()
    print(f"{TAG} event=done steps={state['step']} hours={state['elapsed'] / 3600:.2f} best={state['best']} ok=1", flush=True)


def rollout_gate(teacher_vs_pub: float, student_vs_tiny: float, pub_vs_tiny: float, fraction: float = 0.25) -> bool:
    """``--check``: the toggle works when the teacher reproduces the full model's saved rollout and the student the untuned
    tiny one, each far closer than the two models are to each other. Relative, not bit-exact: the saved rollouts come
    from another process (eval-mode load, its own int8 autotune choices), and small differences compound over iterations."""
    return teacher_vs_pub < fraction * pub_vs_tiny and student_vs_tiny < fraction * pub_vs_tiny


def check(args, distill, probe, weights, load, opt, params, sizes, train_items, rng) -> None:
    """Rollouts on probe item 0 against ``probe_dir_h3.py``'s (teacher = pub.pt, untuned student = tiny.pt) at r = 1
    (one forward from (z, 0)) and r = R, the save path's LoRA merge on the real files, then ``--check-steps`` updates
    (half on-policy; the first on the largest crop of a stills item, so the memory peak is the worst case), the time
    and the memory peak. Writes nothing."""
    import train_dir_h3
    import train_h3
    from safetensors.torch import load_file

    device = distill.device
    ok = True
    if args.probe_dir is not None and (args.probe_dir / "pub.pt").is_file():
        pub, tiny = torch.load(args.probe_dir / "pub.pt"), torch.load(args.probe_dir / "tiny.pt")
        _name, i, entry = probe[0]
        x0, text = load(entry, i)
        z = torch.randn(x0.shape, device=device, generator=torch.Generator(device).manual_seed(i))
        teacher = distill.final(False, z, text, 7 * i)
        student = distill.final(True, z, text, 7 * i)
        for r in (1, distill.iters):
            t_pub = mad(teacher[r - 1], pub[(i, f"pub_r{r}")])
            s_tiny = mad(student[r - 1], tiny[(i, f"tiny_r{r}")])
            base = mad(tiny[(i, f"tiny_r{r}")], pub[(i, f"pub_r{r}")])
            good = rollout_gate(t_pub, s_tiny, base)
            ok = ok and good
            print(f"{TAG} event=check_rollouts r={r} teacher_vs_pub={t_pub:.4f} student_vs_tiny={s_tiny:.4f} "
                  f"student_vs_teacher={mad(student[r - 1], teacher[r - 1]):.4f} pub_vs_tiny={base:.4f} ok={int(good)}",
                  flush=True)
    pub_lora = load_file(str(args.run / "lora_gen.safetensors"))
    delta_state = {k: v.detach().cpu() for k, v in distill.delta_net.state_dict().items()}
    merged = merge_lora_states(pub_lora, delta_state)
    rank = next(v.shape[0] for k, v in merged.items() if k.endswith(".lora_down.weight"))
    good = set(merged) == set(pub_lora) and rank == args.rank + next(
        v.shape[0] for k, v in pub_lora.items() if k.endswith(".lora_down.weight"))
    ok = ok and good
    print(f"{TAG} event=check_merge keys={len(merged)} rank={rank} ok={int(good)}", flush=True)
    del pub_lora, delta_state, merged
    stills = [e for e in train_items if Path(e[0]).parent == Path(args.probe_cache)] or train_items
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(args.check_steps):
        opt.zero_grad(set_to_none=True)
        losses, shapes = [], []
        for j in range(args.accum):
            first = step == 0 and j == 0
            train_h3.MAX_LATENT_HW = max(sizes, key=lambda s: s[0] * s[1]) if first else sizes[(step * args.accum + j) % len(sizes)]
            x0, text = train_dir_h3.load_item(rng.choice(stills if first else train_items), rng, device, args.frames)
            s = distill.sample(torch.randn_like(x0), text, rng.randrange(2**31 - 1), on_policy=bool(j % 2), weights=weights,
                               loss_scale=1.0 / args.accum)
            losses.append(s["loss"])
            shapes.append("x".join(map(str, x0.shape[2:])))
            ok = ok and math.isfinite(s["loss"])
        grads = sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
        ok = ok and grads > 0
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        print(f"{TAG} event=check_step step={step} shapes={shapes} loss={sum(losses) / len(losses):.5f} grad_sum={grads:.4g} "
              f"seconds={time.perf_counter() - started:.1f} peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f}", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    ok = ok and reserved < 31000
    print(f"{TAG} event=check_done s_per_step={(time.perf_counter() - started) / args.check_steps:.1f} "
          f"peak_mb={peak:.0f} reserved_mb={reserved:.0f} rank={args.rank} ok={int(ok)}", flush=True)
    if not ok:
        raise SystemExit(1)


def report(args) -> None:
    d = args.probe_dir
    load_pt = lambda n: torch.load(d / f"{n}.pt", map_location="cpu")  # noqa: E731
    load_js = lambda n: json.loads((d / f"{n}.json").read_text())  # noqa: E731
    evals = []
    log = args.out / "log.jsonl"
    if log.is_file():
        evals = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{"eval"')]
    result = pass_check(load_pt("pub"), load_pt("tiny"), load_pt(args.name), load_js("pub"), load_js("tiny"),
                        load_js(args.name), evals, name=args.name)
    (args.out / "pass.json").write_text(json.dumps(result, indent=1))
    print(f"{TAG} event=pass to_full_tiny={result['to_full_tiny_mean']} to_full_ft={result['to_full_ft_mean']} "
          f"own_to_dense={json.dumps(result['own_to_dense_mean'])} heldout={json.dumps(result['heldout'])} "
          f"checks={json.dumps(result['checks'])} pass={int(result['pass'])}", flush=True)


if __name__ == "__main__":
    main()
