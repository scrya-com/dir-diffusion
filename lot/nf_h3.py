"""Neighbor Forcing and ConvKV Memory for H3 (SoulX-LiveAct, arXiv 2603.11746). Experimental.

Paper: "SoulX-LiveAct: Towards Hour-Scale Real-Time Human Animation with Neighbor Forcing and ConvKV
Memory" (Zhen, Zheng, Zhang, Jiang, Yan, Tao, Yin). ``docs/SOULX_LIVEACT.md`` maps every equation and
figure to this file and lists every deviation.

What the paper does, and where it is here:

* **Neighbor Forcing** (sec. 2.2, eq. 2-5). The latent frames are cut into blocks (first 6, then 8,
  Table 5). Every block is noised to the SAME diffusion step (eq. 2) and block ``n`` attends to blocks
  ``<= n`` only (eq. 5), so the context of a block is its neighbours at its own step, never clean or
  differently-noised frames. ``nf_forward`` is that one masked pass over a whole clip (eq. 4).
* **ConvKV Memory** (sec. 2.3, eq. 6-7, fig. 3). From the third block on the context is six frame
  slots: 2 reference frames, 2 long-term slots, 2 short-term frames. A long-term slot is a 1D
  convolution (kernel = stride = 5) over the keys and values of five frame slots; its rotary position
  is reset to its slot. ``ConvKV`` holds the convolutions (one per DiT block), ``LayerState`` the fold.
* **Inference** (eq. 1, fig. 3 right). Block by block, each block denoised from noise, with one
  key/value cache per denoising step: block ``n`` at step ``i`` reads what blocks ``< n`` wrote at
  step ``i``. ``NFStream`` / ``nf_stream_sample``. ``nf_parallel_sample`` is the same sampler run
  step-outer over a clip of known length (what Stage 2 trains through); the two are equal
  (``test_lot.py`` on the CPU, ``parity_nf_h3.py`` on the GPU).

H3 is not Wan, so three things have no counterpart in the paper and are decided here:

* H3 is single-stream: text, audio and video rows share one self-attention. Text rows (and the FL2VA
  ``first`` rows, the reference image ``x_ref`` of eq. 1) form a *prefix* every block can read and that
  reads only itself, which is what cross-attention to text gives Wan and what makes the prefix
  cacheable. ``NFConfig(text_reads_first=True)`` instead lets the prefix also read block 0 (as dense H3's
  text rows read the video): the cached prefix then carries block 0, still cacheable, and a one-block
  clip equals the dense forward. Each audio row belongs to the block that contains its time and is visible inside that
  block only. By default audio rows are silent noise, as everywhere in this repo; with driving audio
  (``audio=..., audio_t=AUDIO_CLEAN``) they are the clip's clean H3 audio latent, which is this port's
  ``c_audio``: H3 has no audio cross-attention, its audio is rows of the same attention.
* H3's rotary time is not a frame index (latent frame ``5k`` spans 1 pixel frame, the others 4). The
  RoPE reset keeps the short-term frames and the current block at their true spacing, translated so
  the first short-term frame sits at slot 4 of the first window, and puts the reference frames and the
  long-term slots at the times of slots 0-3 of that window. Positions stay bounded for any length.
* H3's head predicts ``x0 - eps`` (the paper's target is ``eps - x0``).

Nothing in ``third_party/`` is modified: the attention below calls each block's own ``qkv_proj``,
norms and ``out_proj`` (so LoRAs apply) from inside the stock ``DiTBlock``, and with ``causal=False`` it is
the stock forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from h3_musubi import (AUDIO_CHANNELS, VIDEO_SHIFT, _sync_swap, audio_frames, audio_sigma, pixel_frames,
                       shift_sigma)

FIRST_BLOCK, NEXT_BLOCK = 6, 8      # Table 5: memory-block 6, current-block 8
RATIO = 5                           # lambda, eq. 6
REF_FRAMES, LONG_SLOTS, SHORT_FRAMES = 2, 2, 2    # sec. 2.3, "Inference Pipeline"
BASE_SIGMAS = (1.0, 0.75, 0.5)      # 3 steps; the release's 1000, 937.5, 833.3 are these under Wan's shift 5
VISUAL_CLEAN = 0.999                # H3's clean coefficient for a visual condition
AUDIO_CLEAN = 1.0                   # H3's clean coefficient for an audio condition: the time of driving audio rows


def _packing():
    from musubi_tuner.minimax_h3 import packing

    return packing


# ------------------------------------------------------------------------------- config

@dataclass(frozen=True)
class NFConfig:
    """Block sizes and what a block may read.

    ``causal=False`` is the base model (one block, everything visible). ``memory=False`` is Stage 1:
    block ``n`` reads every earlier block. ``memory=True`` is Stage 2 / inference: from block 2 on the
    context is ``ref`` reference frames, ``long`` ConvKV slots and ``short`` short-term frames.
    """
    first_block: int = FIRST_BLOCK
    next_block: int = NEXT_BLOCK
    causal: bool = True
    memory: bool = False
    ratio: int = RATIO
    ref: int = REF_FRAMES
    long: int = LONG_SLOTS
    short: int = SHORT_FRAMES
    text_reads_first: bool = False      # prefix rows also read block 0 (dense H3's text reads the video; deviation 3)

    def __post_init__(self) -> None:
        if self.first_block < 1 or self.next_block < 1:
            raise ValueError("block sizes must be positive")
        if self.memory:
            if not self.causal:
                raise ValueError("memory needs causal blocks")
            # The window after the reference frames must hold ratio * long + short frame slots at every
            # iteration: (first - ref) + next the first time, long + short + next afterwards.
            if self.first_block != self.context_slots:
                raise ValueError(f"memory needs first_block == ref + long + short = {self.context_slots}, "
                                 f"got {self.first_block}")
            if self.long + self.short + self.next_block != self.ratio * self.long + self.short:
                raise ValueError(f"memory needs next_block == long * (ratio - 1) = {self.long * (self.ratio - 1)}, "
                                 f"got {self.next_block}")

    @property
    def context_slots(self) -> int:
        return self.ref + self.long + self.short

    def span(self, index: int) -> tuple[int, int]:
        """Latent frames ``[start, stop)`` of block ``index``."""
        if index == 0:
            return 0, self.first_block
        start = self.first_block + (index - 1) * self.next_block
        return start, start + self.next_block

    def spans(self, latent_t: int) -> list[tuple[int, int]]:
        """The blocks of a ``latent_t``-frame clip (one block when not causal)."""
        if not self.causal:
            return [(0, latent_t)]
        rest = latent_t - self.first_block
        if rest < 0 or rest % self.next_block:
            raise ValueError(f"{latent_t} latent frames are not {self.first_block} + k * {self.next_block}")
        return [self.span(index) for index in range(1 + rest // self.next_block)]

    def frames(self, blocks: int) -> int:
        return self.first_block + (blocks - 1) * self.next_block


DENSE = NFConfig(causal=False)


def few_step_sigmas(base=BASE_SIGMAS, shift: float = VIDEO_SHIFT) -> list[float]:
    """``len(base) + 1`` sigmas ending at 0: ``base`` under the model's schedule shift."""
    return [float(shift_sigma(torch.tensor(float(s), dtype=torch.float64), shift)) for s in base] + [0.0]


# ----------------------------------------------------------------------------- geometry

@dataclass(frozen=True)
class Geometry:
    """Rotary positions of every row kind, for a clip of any length (no ``5n + 2`` restriction).

    Equal to ``packing.build_position_grid`` on H3's own T2VA and FL2VA-first layouts
    (``test_nf_geometry_matches_stock``).
    """
    text_length: int
    lat_h: int
    lat_w: int
    first: bool = False

    @property
    def frame_rows(self) -> int:
        return (self.lat_h // 2) * (self.lat_w // 2)

    @property
    def prefix_rows(self) -> int:
        return self.text_length + (self.frame_rows if self.first else 0)

    @staticmethod
    def frame_time(frame: int) -> float:
        """Rotary time of latent frame ``frame`` from the start of the video (``5/3`` per pixel frame),
        summed as ``packing._video_grid`` sums it so the positions are H3's to the bit."""
        if frame <= 0:
            return 0.0
        spans = torch.tensor(_packing()._video_t_spans(frame), dtype=torch.float64)
        return float(spans.cumsum(0)[-1])

    @staticmethod
    def audio_span(start: int, stop: int, total: int | None = None) -> tuple[int, int]:
        """Audio frames ``[a0, a1)`` whose time lies inside latent frames ``[start, stop)``.

        Rotary time of a pixel frame is 5/3 and of an audio frame 1, so the first audio frame at or
        after ``p`` pixel frames is ``ceil(5 p / 3)``. The last block of a clip of ``total`` latent
        frames takes what is left of H3's audio length for that clip.
        """
        a0 = (5 * pixel_frames(start) + 2) // 3
        a1 = (5 * pixel_frames(stop) + 2) // 3
        if total is not None:
            a0, a1 = min(a0, audio_frames(total)), (audio_frames(total) if stop >= total else min(a1, audio_frames(total)))
        return a0, a1

    def _grid(self):
        packing = _packing()
        frame, width = packing._frame_grid(packing.H3VideoGeometry(1, self.lat_h, self.lat_w))
        return frame, float(width[0]), float(width[-1])

    def text_positions(self) -> torch.Tensor:
        positions = torch.zeros(self.text_length, 3, dtype=torch.float64)
        positions[:, 0] = torch.arange(self.text_length, dtype=torch.float64)
        return positions

    def frame_positions(self, times) -> torch.Tensor:
        """One frame of rows at each rotary time in ``times`` (seconds-like units after the text)."""
        frame, _low, _high = self._grid()
        times = torch.as_tensor(times, dtype=torch.float64).reshape(-1)
        grid = torch.empty(times.numel(), frame.shape[0], 3, dtype=torch.float64)
        grid[..., 0] = self.text_length + times[:, None]
        grid[..., 1:] = frame[None]
        return grid.reshape(-1, 3)

    def video_positions(self, start: int, stop: int, shift: float = 0.0) -> torch.Tensor:
        return self.frame_positions([self.frame_time(f) - shift for f in range(start, stop)])

    def audio_positions(self, a0: int, a1: int, shift: float = 0.0) -> torch.Tensor:
        """Rows in ``pack_audio_rows`` order for audio frames ``[a0, a1)``: low channel, then high."""
        _frame, low, high = self._grid()
        count = a1 - a0
        grid = torch.zeros(2 * count, 3, dtype=torch.float64)
        grid[:, 0] = (self.text_length + torch.arange(a0, a1, dtype=torch.float64) - shift).repeat(2)
        grid[:count, 2] = low
        grid[count:, 2] = high
        return grid

    def prefix_positions(self) -> torch.Tensor:
        parts = [self.text_positions()]
        if self.first:
            parts.append(self.frame_positions([0.0]))
        return torch.cat(parts)

    # --- the RoPE reset (eq. 7) ---
    def window_shift(self, config: NFConfig, index: int) -> float:
        """Time subtracted from the short-term frames and block ``index`` in the memory phase, so the
        first short-term frame lands on slot ``ref + long`` of the first window. 0 before that."""
        if not config.memory or index < 2:
            return 0.0
        start, _stop = config.span(index)
        return self.frame_time(start - config.short) - self.frame_time(config.ref + config.long)

    def context_positions(self, config: NFConfig, index: int) -> torch.Tensor | None:
        """Positions of block ``index``'s video context, in the order ``LayerState.context`` returns it."""
        start, _stop = config.span(index)
        if index == 0 or not config.causal:
            return None
        if not config.memory or index < 2:
            return self.video_positions(0, start)
        shift = self.window_shift(config, index)
        return torch.cat([self.video_positions(0, config.ref + config.long),              # reference + long-term slots
                          self.video_positions(start - config.short, start, shift)])      # short-term frames


def timestep_table(t_video: float, t_audio: float, first: bool):
    """``packing.build_timestep_rows`` for a row subset: the unique timesteps and each row kind's AdaLN code.

    Returns ``(unique, codes, video_index, audio_index)``; ``codes`` maps ``text | first | audio | video``
    to ``3 * timestep_index + token_tag`` (text tags are 1 in this repo, video 0, audio 2).
    """
    values = {float(t_video), float(t_audio)}
    if first:
        values.add(max(float(t_video), VISUAL_CLEAN))
    unique = sorted(values)
    index = {value: position for position, value in enumerate(unique)}
    codes = {"text": 3 * index[float(t_video)] + 1, "audio": 3 * index[float(t_audio)] + 2,
             "video": 3 * index[float(t_video)]}
    if first:
        codes["first"] = 3 * index[max(float(t_video), VISUAL_CLEAN)]
    return torch.tensor(unique, dtype=torch.float32), codes, index[float(t_video)], index[float(t_audio)]


# -------------------------------------------------------------------------------- ConvKV

class ConvKVLayer(nn.Module):
    """Eq. 6 for one DiT block: a 1D convolution over time, ``kernel = stride = ratio``.

    Depthwise (one filter per key/value channel) and bias-free, initialised to the mean (``1 / ratio``),
    as the released ``WanSelfAttention.memory_proj_k``. ``shared`` uses one convolution for keys and
    values: eq. 6 writes a single ``Conv_theta`` for both, and the release's ``v_compress`` calls
    ``memory_proj_k``. ``shared=False`` gives the values their own ``memory_proj_v``.
    """

    def __init__(self, channels: int, ratio: int = RATIO, shared: bool = True) -> None:
        super().__init__()
        self.ratio = ratio
        self.memory_proj_k = nn.Conv1d(channels, channels, kernel_size=ratio, stride=ratio, groups=channels, bias=False)
        self.memory_proj_v = None if shared else nn.Conv1d(channels, channels, kernel_size=ratio, stride=ratio,
                                                           groups=channels, bias=False)
        for conv in (self.memory_proj_k, self.memory_proj_v):
            if conv is not None:
                nn.init.constant_(conv.weight, 1.0 / ratio)

    def compress(self, states: torch.Tensor, frame_rows: int, which: str) -> torch.Tensor:
        """``(1, F * frame_rows, heads, dim)`` frame-major keys or values -> ``(1, F / ratio * frame_rows, heads, dim)``.

        The convolution runs along the frame axis at each spatial position: frames ``[5j, 5j + 5)`` of a
        position become slot ``j`` of that position ("every five chunks of KV states into a single chunk").
        """
        conv = self.memory_proj_v if (which == "v" and self.memory_proj_v is not None) else self.memory_proj_k
        batch, rows, heads, dim = states.shape
        frames = rows // frame_rows
        if rows != frames * frame_rows or frames % self.ratio:
            raise ValueError(f"ConvKV needs a multiple of {self.ratio} frames of {frame_rows} rows, got {rows} rows")
        series = states.reshape(batch, frames, frame_rows, heads * dim).permute(0, 2, 3, 1)      # (B, P, C, F)
        series = series.reshape(batch * frame_rows, heads * dim, frames)
        out = conv(series.to(conv.weight.dtype))                                                # (B * P, C, F / ratio)
        out = out.reshape(batch, frame_rows, heads * dim, frames // self.ratio).permute(0, 3, 1, 2)
        return out.reshape(batch, frames // self.ratio * frame_rows, heads, dim).to(states.dtype)


class ConvKV(nn.Module):
    """One ``ConvKVLayer`` per DiT block (the paper's "plug-in" module; the base weights are untouched)."""

    def __init__(self, layers: int, channels: int, ratio: int = RATIO, shared: bool = True) -> None:
        super().__init__()
        self.ratio, self.shared = ratio, shared
        self.layers = nn.ModuleList(ConvKVLayer(channels, ratio, shared) for _ in range(layers))

    @classmethod
    def for_dit(cls, dit, ratio: int = RATIO, shared: bool = True) -> "ConvKV":
        return cls(len(dit.blocks), dit.config.attention_inner_dim, ratio, shared)


# ------------------------------------------------------------------------------ KV state

@dataclass
class LayerState:
    """One DiT block's keys and values at ONE diffusion step: the prefix, and the video frames a later
    block may read. Keys are stored after ``k_norm`` and before the rotary rotation, as the release
    caches them, so the same key can be rotated to a different position by each reader.

    Without memory ``frames`` grows by one block per ``push``. With memory it is the fixed window of
    fig. 3: ``ref`` holds the first ``config.ref`` frames, ``window`` the ``ratio * long + short``
    frame slots after them, folded by ``context`` exactly as the release's ``update_cache``:
    ``[L0 L1 S0 S1 C0..C7] -> L0' = conv(L0 L1 S0 S1 C0), L1' = conv(C1..C5), S' = (C6, C7)``.
    """
    config: NFConfig
    frame_rows: int
    store: torch.device | None = None           # where a stream keeps its tensors (None: where they are)
    prefix: tuple | None = None
    ref: tuple | None = None
    window: tuple | None = None
    pending: tuple | None = None
    frames: int = 0                             # video frames pushed so far (bookkeeping, tests)

    def _keep(self, k: torch.Tensor, v: torch.Tensor) -> tuple:
        return (k, v) if self.store is None else (k.detach().to(self.store), v.detach().to(self.store))

    def set_prefix(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.prefix = self._keep(k, v)

    def context(self, index: int, conv: ConvKVLayer | None, device=None) -> tuple | None:
        """Keys and values (un-rotated) of block ``index``'s video context, or ``None`` for block 0."""
        if index == 0 or self.window is None:
            return None
        k, v = (self.window if device is None else tuple(t.to(device) for t in self.window))
        if not self.config.memory:
            return k, v
        ref_k, ref_v = (self.ref if device is None else tuple(t.to(device) for t in self.ref))
        if index == 1:
            return torch.cat([ref_k, k], dim=1), torch.cat([ref_v, v], dim=1)
        if conv is None:
            raise ValueError("memory=True needs a ConvKV module from block 2 on")
        cut = self.config.ratio * self.config.long * self.frame_rows
        folded_k = torch.cat([conv.compress(k[:, :cut], self.frame_rows, "k"), k[:, cut:]], dim=1)
        folded_v = torch.cat([conv.compress(v[:, :cut], self.frame_rows, "v"), v[:, cut:]], dim=1)
        self.pending = (folded_k, folded_v)
        return torch.cat([ref_k, folded_k], dim=1), torch.cat([ref_v, folded_v], dim=1)

    def push(self, index: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Record block ``index``'s own video keys and values (after ``context(index)``)."""
        self.frames += k.shape[1] // self.frame_rows
        if not self.config.memory:
            self.window = self._keep(k, v) if self.window is None else self._keep(
                torch.cat([self.window[0].to(k.device), k], dim=1), torch.cat([self.window[1].to(v.device), v], dim=1))
            return
        if index == 0:
            cut = self.config.ref * self.frame_rows
            self.ref, self.window = self._keep(k[:, :cut], v[:, :cut]), self._keep(k[:, cut:], v[:, cut:])
            return
        base = self.window if index == 1 else self.pending
        self.window = self._keep(torch.cat([base[0].to(k.device), k], dim=1), torch.cat([base[1].to(v.device), v], dim=1))
        self.pending = None


# ----------------------------------------------------------------------------- attention

@dataclass
class BlockCall:
    """One causal block inside a forward: which rows are its queries and how they are rotated."""
    index: int
    rows: torch.Tensor                  # indices into the forward's rows: its audio rows, then its video rows
    video_rows: int                     # the trailing rows that are video
    rotation: torch.Tensor              # for its own queries and keys
    context_rotation: torch.Tensor | None


@dataclass
class Call:
    """What one forward computes. ``dense`` is the stock attention over every row."""
    prefix_rows: int                    # leading rows that are the prefix (0: the prefix comes from the state)
    prefix_rotation: torch.Tensor | None
    blocks: list = field(default_factory=list)
    dense_rotation: torch.Tensor | None = None
    prefix_reads_block: bool = False    # the prefix's queries also read the first block of ``blocks``


def _rope(states: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    from musubi_tuner.minimax_h3 import model as h3_model

    return h3_model._apply_rope_split_half(states, rotation)       # looked up late: speed_patches() replaces it


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``(1, Lq, heads, dim)`` queries over ``(1, Lk, heads, dim)`` keys/values -> ``(1, Lq, heads * dim)``."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.to(v).transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)


def nf_attention(attn, hidden: torch.Tensor, call: Call, state: LayerState, conv: ConvKVLayer | None) -> torch.Tensor:
    """``Attention.forward`` with the block-causal reads of eq. 5 and the ConvKV context of eq. 6-7.

    Row ``i`` reads row ``j`` when ``j`` is in the prefix, or ``i`` and ``j`` are in the same block, or
    ``j`` is a video row of the context of ``i``'s block. Prefix rows read the prefix only. Instead of
    one masked attention this runs one unmasked attention per block over exactly the rows it may read,
    which is the same function and is what lets the memory keys carry their own rotary positions.
    """
    batch, rows, _ = hidden.shape
    query, key, value = attn.qkv_proj(hidden).split(attn.inner_dim, dim=-1)
    query = attn.q_norm(query.reshape(batch, rows, attn.num_heads, attn.head_dim))
    key = attn.k_norm(key.reshape(batch, rows, attn.num_heads, attn.head_dim))
    value = value.reshape(batch, rows, attn.num_heads, attn.head_dim)
    if call.dense_rotation is not None:        # the stock forward, operation for operation
        from musubi_tuner.modules.attention import AttentionParams, attention

        query = _rope(query, call.dense_rotation).to(value)
        key = _rope(key, call.dense_rotation).to(value)
        params = AttentionParams.create_attention_params(attn.attn_mode, attn.split_attn)
        return attn.out_proj(attention([query, key, value], attn_params=params))

    out = torch.zeros(batch, rows, attn.inner_dim, dtype=value.dtype, device=value.device)
    count = call.prefix_rows
    if count:
        prefix_k, prefix_v = key[:, :count], value[:, :count]
        state.set_prefix(prefix_k, prefix_v)
        rotated = _rope(prefix_k, call.prefix_rotation)
        keys, values = rotated, prefix_v
        if call.prefix_reads_block:         # the prefix also reads the block computed with it (block 0, or
            block = call.blocks[0]          # the block at which a stream switches prompt); its cached keys and
            keys = torch.cat([rotated, _rope(key[:, block.rows], block.rotation)], dim=1)    # values carry that read
            values = torch.cat([prefix_v, value[:, block.rows]], dim=1)
        out[:, :count] = _attend(_rope(query[:, :count], call.prefix_rotation).to(value), keys.to(value), values)
    else:
        prefix_k, prefix_v = (t.to(value.device) for t in state.prefix)
        rotated = _rope(prefix_k, call.prefix_rotation)
    for block in call.blocks:
        own_k, own_v = key[:, block.rows], value[:, block.rows]
        keys, values = [rotated], [prefix_v]
        context = state.context(block.index, conv, value.device)
        if context is not None:
            keys.append(_rope(context[0].to(own_k.dtype), block.context_rotation))
            values.append(context[1])
        keys.append(_rope(own_k, block.rotation))
        values.append(own_v)
        out[:, block.rows] = _attend(_rope(query[:, block.rows], block.rotation).to(value),
                                     torch.cat(keys, dim=1), torch.cat([t.to(value) for t in values], dim=1))
        state.push(block.index, own_k[:, -block.video_rows:], own_v[:, -block.video_rows:])
    return attn.out_proj(out)


def _run(dit, hidden, temb, segments, call: Call, states, convkv: ConvKV | None, config: NFConfig, frame_rows: int):
    """The block stack (``h3_musubi.run_blocks``: block streaming, activation checkpointing), each block
    being the stock ``DiTBlock`` called as a module with ``nf_attention`` in place of its attention.

    The swap only lasts for the call (an instance attribute on ``block.attn``, removed on the way out),
    and it is made inside the function that activation checkpointing re-runs during backward. The block
    is called as a module, not unrolled here, because the block-swap offloader moves blocks during
    backward from hooks on the block modules. ``states`` is ``None`` for a parallel pass (a fresh
    ``LayerState`` per call, also when checkpointing recomputes) or one ``LayerState`` per block.
    """
    from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper

    device = dit.video_patch_proj.weight.device
    for index, block in enumerate(dit.blocks):
        if dit.blocks_to_swap:
            dit.offloader.wait_for_block(index)
        conv = None if convkv is None else convkv.layers[index]
        state = None if states is None else states[index]

        def forward(hidden_in, temb_in, block=block, conv=conv, state=state):
            attn = block.attn

            def attend(normalized, _rotation=None):
                return nf_attention(attn, normalized, call, LayerState(config, frame_rows) if state is None else state, conv)

            attn.forward = attend
            try:
                return block(hidden_in, temb_in, segments, None)
            finally:
                del attn.forward

        if dit.gradient_checkpointing and dit.training and states is None:
            if dit.activation_cpu_offloading:
                forward = create_cpu_offloading_wrapper(forward, device)
            hidden = torch.utils.checkpoint.checkpoint(forward, hidden, temb, use_reentrant=False)
        else:
            hidden = forward(hidden, temb)
        if dit.blocks_to_swap:
            dit.offloader.submit_move_blocks_forward(dit.blocks, index)
    return hidden.to(device)


def _rotation(dit, positions: torch.Tensor | None, dtype) -> torch.Tensor | None:
    if positions is None:
        return None
    device = dit.video_patch_proj.weight.device
    return dit._rotation_table(positions.unsqueeze(0).to(device), dtype)


def _embed(dit, text, first, audio, video):
    """Embedded rows ``(prefix or None, audio, video)`` as the stock forward embeds them (the ``first``
    frame and the target share one ``video_patch_proj`` call)."""
    packing = _packing()
    device = dit.video_patch_proj.weight.device
    visual = [] if first is None else [packing.pack_video_rows(first.to(device))]
    visual.append(packing.pack_video_rows(video.to(device)))
    projected = dit.video_patch_proj(torch.cat(visual, dim=1).to(torch.float32)).to(dit.dtype)
    audio_rows = dit.audio_patch_proj(torch.cat([packing.pack_audio_rows(audio.to(device))], dim=1).to(torch.float32))
    audio_rows = audio_rows.to(dit.dtype)
    if text is None:
        return None, audio_rows, projected
    parts = [dit.token_refiner(dit.condition_proj(text.to(device, dit.dtype)))]
    cut = 0
    if first is not None:
        cut = visual[0].shape[1]
        parts.append(projected[:, :cut])
    return torch.cat(parts, dim=1), audio_rows, projected[:, cut:]


def _video_out(dit, hidden_video, temb, video_index: int, frames: int, lat_h: int, lat_w: int) -> torch.Tensor:
    packing = _packing()
    final = dit.final_layer
    shift, scale = final.adaln_proj(temb)
    states = final.norm(hidden_video) * (1.0 + scale[video_index]) + shift[video_index]
    velocity = final.video_out(states.to(torch.float32))
    return packing._unpack_video_rows(velocity.to(torch.float32), packing.H3VideoGeometry(frames, lat_h, lat_w))


def _segments(codes: list[tuple[int, int]]):
    """``[(row_count, adaln_code), ...]`` in row order -> the blocks' modulation segments."""
    indices = torch.cat([torch.full((count,), code, dtype=torch.int64) for count, code in codes if count])
    return _packing()._build_modulation_segments(indices)


def _delta_row(temb: torch.Tensor, codes: dict, video_index: int, delta: torch.Tensor | None):
    """DiR's iteration embedding (``h3_musubi._lot_forward``'s ``video_temb_delta``): a NEW timestep row
    ``temb[video] + delta`` used by the target-video rows and the final layer's video head only. Text,
    reference and audio rows keep their rows."""
    if delta is None:
        return temb, codes, video_index
    new_index = temb.shape[0]
    row = temb[video_index].float() + delta.to(device=temb.device, dtype=torch.float32)
    temb = torch.cat([temb, row.to(temb.dtype).unsqueeze(0)], dim=0)
    return temb, {**codes, "video": 3 * new_index}, new_index


def _add_extra(video_rows: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
    """DiR's previous-estimate embedding, added to the embedded target-video rows (``video_extra``)."""
    if extra is None:
        return video_rows
    if tuple(extra.shape) != tuple(video_rows.shape):
        raise ValueError(f"video_extra {tuple(extra.shape)} != video rows {tuple(video_rows.shape)}")
    return video_rows + extra.to(device=video_rows.device, dtype=video_rows.dtype)


def silence(count: int, sigma_a: float, device, generator=None) -> torch.Tensor:
    """``(1, 32, 2, count)`` silent audio noised to ``sigma_a`` (``h3_musubi._silence`` for any count)."""
    return sigma_a * torch.randn(1, AUDIO_CHANNELS, 2, count, device=device, generator=generator)


# ------------------------------------------------------------------------ parallel forward

def nf_forward(dit, latent: torch.Tensor, t, text: torch.Tensor, *, config: NFConfig = DENSE,
               convkv: ConvKV | None = None, first: torch.Tensor | None = None, audio: torch.Tensor | None = None,
               audio_t: float | None = None, audio_generator: torch.Generator | None = None,
               video_extra: torch.Tensor | None = None, video_temb_delta: torch.Tensor | None = None) -> torch.Tensor:
    """H3's velocity (``x0 - eps``) for a whole clip ``(1, 24, T, H, W)`` at cleanness ``t`` (eq. 4).

    Every frame of ``latent`` is at the same step; ``config`` says what each block reads. With
    ``config=DENSE`` this is the stock ``dit(...)`` call (T2VA, or FL2VA-first with ``first``).
    ``first`` ``(1, 24, 1, H, W)`` is the reference frame ``x_ref``, fed as H3's FL2VA ``first``
    condition. ``audio`` ``(1, 32, 2, A)`` defaults to silent noise at the matching audio sigma.
    Driving audio (``c_audio`` of eq. 1): pass the clean H3 audio latent as ``audio`` with
    ``audio_t=AUDIO_CLEAN``; the audio rows then carry the clean-condition timestep instead of the
    video's noise level, and each block reads the audio of its own time span.
    ``video_extra`` ``(1, T * rows, hidden)`` and ``video_temb_delta`` ``(temb_dim,)`` are the two hooks of
    ``h3_musubi.forward_video`` (DiR's previous estimate and iteration embedding); with ``config=DENSE``
    and no ``first`` this is ``forward_video`` with the same hooks.
    """
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(f"latent must be (1, 24, T, H, W), got {tuple(latent.shape)}")
    if text.ndim == 2:
        text = text.unsqueeze(0)
    _sync_swap(dit)
    t_video = float(torch.as_tensor(t).reshape(-1)[0])
    sigma_a = float(audio_sigma(1.0 - t_video))
    latent_t, lat_h, lat_w = latent.shape[2:]
    device = dit.video_patch_proj.weight.device
    geometry = Geometry(text.shape[1], lat_h, lat_w, first is not None)
    total_audio = audio_frames(latent_t)
    if audio is None:
        audio = silence(total_audio, sigma_a, device, audio_generator)
    if audio.shape[-1] != total_audio:
        raise ValueError(f"audio has {audio.shape[-1]} frames, expected {total_audio}")
    prefix, audio_rows, video_rows = _embed(dit, text, first, audio, latent)
    hidden = torch.cat([prefix, audio_rows, _add_extra(video_rows, video_extra)], dim=1)
    unique, codes, video_index, _audio_index = timestep_table(
        t_video, 1.0 - sigma_a if audio_t is None else float(audio_t), first is not None)
    temb, codes, video_index = _delta_row(dit._timestep_embeddings(unique, device), codes, video_index, video_temb_delta)
    rows_per_frame = geometry.frame_rows
    segments = _segments([(geometry.text_length, codes["text"]), (rows_per_frame if first is not None else 0, codes.get("first", 0)),
                          (2 * total_audio, codes["audio"]), (latent_t * rows_per_frame, codes["video"])])

    count = geometry.prefix_rows
    audio_base, video_base = count, count + 2 * total_audio
    dtype = hidden.dtype
    if not config.causal:
        positions = torch.cat([geometry.prefix_positions(), geometry.audio_positions(0, total_audio),
                               geometry.video_positions(0, latent_t)])
        call = Call(0, None, dense_rotation=_rotation(dit, positions, dtype))
    else:
        call = Call(count, _rotation(dit, geometry.prefix_positions(), dtype), prefix_reads_block=config.text_reads_first)
        for index, (start, stop) in enumerate(config.spans(latent_t)):
            a0, a1 = geometry.audio_span(start, stop, latent_t)
            shift = geometry.window_shift(config, index)
            rows = torch.cat([audio_base + torch.arange(a0, a1), audio_base + total_audio + torch.arange(a0, a1),
                              video_base + torch.arange(start * rows_per_frame, stop * rows_per_frame)]).to(device)
            own = torch.cat([geometry.audio_positions(a0, a1, shift), geometry.video_positions(start, stop, shift)])
            call.blocks.append(BlockCall(index, rows, (stop - start) * rows_per_frame, _rotation(dit, own, dtype),
                                         _rotation(dit, geometry.context_positions(config, index), dtype)))
    hidden = _run(dit, hidden, temb, segments, call, None, convkv, config, rows_per_frame)
    return _video_out(dit, hidden[:, video_base:], temb, video_index, latent_t, lat_h, lat_w).to(latent.dtype)


# ------------------------------------------------------------------------------ streaming

class NFStream:
    """Block-by-block inference with one key/value cache per denoising step (eq. 1, fig. 3 right).

    ``velocity(index, step, x, t, text, audio)`` is block ``index`` at step ``step``: only that block's
    rows run through the DiT, reading the prefix and the context that earlier blocks wrote AT THE SAME
    STEP. Blocks must be visited in order at each step (block-outer or step-outer). With
    ``config.memory`` the cache is the fixed window of fig. 3, so its size does not grow.

    The prefix (text, reference frame) is recomputed whenever a different ``text`` tensor is passed,
    which is how the release switches emotion/action prompts mid-stream.
    """

    def __init__(self, dit, *, lat_h: int, lat_w: int, steps: int, config: NFConfig, convkv: ConvKV | None = None,
                 first: torch.Tensor | None = None, cache_device=None, total_frames: int | None = None) -> None:
        if not config.causal:
            raise ValueError("a stream needs causal blocks")
        self.dit, self.config, self.convkv, self.first = dit, config, convkv, first
        self.lat_h, self.lat_w, self.total_frames = lat_h, lat_w, total_frames
        store = None if cache_device is None else torch.device(cache_device)
        rows = (lat_h // 2) * (lat_w // 2)
        self.states = [[LayerState(config, rows, store=store if store is not None else dit.video_patch_proj.weight.device)
                        for _ in dit.blocks] for _ in range(steps)]
        self._text = [None] * steps

    def audio_span(self, index: int) -> tuple[int, int]:
        return Geometry.audio_span(*self.config.span(index), self.total_frames)

    def cache_rows(self, step: int = 0, layer: int = 0) -> int:
        """Key rows held for one (step, layer): prefix + reference + window."""
        state = self.states[step][layer]
        return sum(0 if part is None else part[0].shape[1] for part in (state.prefix, state.ref, state.window))

    @torch.no_grad()
    def velocity(self, index: int, step: int, x: torch.Tensor, t, text: torch.Tensor, audio: torch.Tensor,
                 audio_t: float | None = None, video_extra: torch.Tensor | None = None,
                 video_temb_delta: torch.Tensor | None = None) -> torch.Tensor:
        dit, config = self.dit, self.config
        if text.ndim == 2:
            text = text.unsqueeze(0)
        _sync_swap(dit)
        t_video = float(torch.as_tensor(t).reshape(-1)[0])
        sigma_a = float(audio_sigma(1.0 - t_video))
        start, stop = config.span(index)
        if x.shape[2] != stop - start:
            raise ValueError(f"block {index} has {stop - start} latent frames, got {x.shape[2]}")
        a0, a1 = self.audio_span(index)
        if audio.shape[-1] != a1 - a0:
            raise ValueError(f"block {index} has {a1 - a0} audio frames, got {audio.shape[-1]}")
        device = dit.video_patch_proj.weight.device
        geometry = Geometry(text.shape[1], self.lat_h, self.lat_w, self.first is not None)
        states = self.states[step]
        fresh = self._text[step] is not text
        self._text[step] = text
        prefix, audio_rows, video_rows = _embed(dit, text if fresh else None, self.first if fresh else None, audio, x)
        hidden = torch.cat(([prefix] if fresh else []) + [audio_rows, _add_extra(video_rows, video_extra)], dim=1)
        unique, codes, video_index, _audio_index = timestep_table(
            t_video, 1.0 - sigma_a if audio_t is None else float(audio_t), self.first is not None)
        temb, codes, video_index = _delta_row(dit._timestep_embeddings(unique, device), codes, video_index,
                                              video_temb_delta)
        rows_per_frame = geometry.frame_rows
        count = geometry.prefix_rows if fresh else 0
        layout = [(2 * (a1 - a0), codes["audio"]), ((stop - start) * rows_per_frame, codes["video"])]
        if fresh:
            layout = [(geometry.text_length, codes["text"]),
                      (rows_per_frame if self.first is not None else 0, codes.get("first", 0))] + layout
        segments = _segments(layout)
        shift = geometry.window_shift(config, index)
        own = torch.cat([geometry.audio_positions(a0, a1, shift), geometry.video_positions(start, stop, shift)])
        dtype = hidden.dtype
        call = Call(count, _rotation(dit, geometry.prefix_positions(), dtype),
                    prefix_reads_block=config.text_reads_first and bool(count))
        call.blocks.append(BlockCall(index, torch.arange(count, hidden.shape[1], device=device),
                                     (stop - start) * rows_per_frame, _rotation(dit, own, dtype),
                                     _rotation(dit, geometry.context_positions(config, index), dtype)))
        hidden = _run(dit, hidden, temb, segments, call, states, self.convkv, config, rows_per_frame)
        video = hidden[:, hidden.shape[1] - (stop - start) * rows_per_frame:]
        return _video_out(dit, video, temb, video_index, stop - start, self.lat_h, self.lat_w).to(x.dtype)


# ------------------------------------------------------------------------------- sampling

class NoiseBank:
    """Noise keyed by (block, step), so block-outer and step-outer sampling draw the same numbers."""

    def __init__(self, seed: int, device) -> None:
        self.seed, self.device = int(seed), device

    def _generator(self, block: int, step: int, kind: int) -> torch.Generator:
        key = (self.seed * 1_000_003 + block * 4_099 + step * 17 + kind) % (2**63 - 1)
        return torch.Generator(self.device).manual_seed(key)

    def video(self, block: int, step: int, frames: int, lat_h: int, lat_w: int) -> torch.Tensor:
        return torch.randn(1, 24, frames, lat_h, lat_w, device=self.device, generator=self._generator(block, step, 0))

    def audio(self, block: int, step: int, count: int) -> torch.Tensor:
        return torch.randn(1, AUDIO_CHANNELS, 2, count, device=self.device, generator=self._generator(block, step, 1))


def advance(x: torch.Tensor, velocity: torch.Tensor, sigma: float, sigma_next: float, fresh, sampler: str) -> torch.Tensor:
    """One sampler step. ``renoise`` is the release's few-step update (``generate.py``): the clean
    estimate re-noised with fresh noise to the next level. ``euler`` is H3's own deterministic step."""
    x0 = x.float() + sigma * velocity.float()
    if sigma_next <= 0.0:
        return x0.to(x.dtype)
    if sampler == "renoise":
        return ((1.0 - sigma_next) * x0 + sigma_next * fresh().float()).to(x.dtype)
    if sampler == "euler":
        ratio = sigma_next / sigma
        return (ratio * x.float() + (1.0 - ratio) * x0).to(x.dtype)
    raise ValueError(f"unknown sampler {sampler!r}")


@torch.no_grad()
def nf_stream_sample(dit, text, blocks: int, *, lat_h: int, lat_w: int, config: NFConfig, sigmas, noise: NoiseBank,
                     convkv: ConvKV | None = None, first: torch.Tensor | None = None, sampler: str = "renoise",
                     cache_device=None, total_frames: int | None = None, on_block=None,
                     audio: torch.Tensor | None = None) -> torch.Tensor:
    """Generate ``blocks`` blocks one after another; returns ``(1, 24, T, lat_h, lat_w)``.

    ``text`` is one tensor, or a function ``block index -> tensor`` (prompt switching). ``sigmas`` has
    one more entry than steps and ends at 0. ``on_block(index, latent)`` is called per finished block.
    ``audio`` ``(1, 32, 2, A)`` is clean driving audio covering every block (default: silent noise).
    """
    steps = len(sigmas) - 1
    # A stream of a known number of blocks is a clip of known length: its last block takes what is left of
    # H3's audio length for that clip (for 2 or 5 blocks that is one audio frame fewer than its time span).
    stream = NFStream(dit, lat_h=lat_h, lat_w=lat_w, steps=steps, config=config, convkv=convkv, first=first,
                      cache_device=cache_device, total_frames=total_frames or config.frames(blocks))
    out = []
    for index in range(blocks):
        start, stop = config.span(index)
        a0, a1 = stream.audio_span(index)
        prompt = text(index) if callable(text) else text
        x = noise.video(index, 0, stop - start, lat_h, lat_w)
        for step in range(steps):
            sigma, sigma_next = float(sigmas[step]), float(sigmas[step + 1])
            if audio is None:
                rows, audio_t = float(audio_sigma(sigma)) * noise.audio(index, step, a1 - a0), None
            else:
                if audio.shape[-1] < a1:
                    raise ValueError(f"driving audio has {audio.shape[-1]} frames, block {index} needs {a1}")
                rows, audio_t = audio[..., a0:a1], AUDIO_CLEAN
            velocity = stream.velocity(index, step, x, 1.0 - sigma, prompt, rows, audio_t)
            x = advance(x, velocity, sigma, sigma_next,
                        lambda: noise.video(index, step + 1, stop - start, lat_h, lat_w), sampler)
        out.append(x)
        if on_block is not None:
            on_block(index, x)
    return torch.cat(out, dim=2)


def parallel_noise(noise: NoiseBank, config: NFConfig, latent_t: int, step: int, lat_h: int, lat_w: int):
    """``(video, audio)`` unit noise for a whole clip at ``step``: the per-block draws of ``NoiseBank``."""
    video, audio = [], []
    for index, (start, stop) in enumerate(config.spans(latent_t)):
        a0, a1 = Geometry.audio_span(start, stop, latent_t)
        video.append(noise.video(index, step, stop - start, lat_h, lat_w))
        audio.append(noise.audio(index, step, a1 - a0))
    return torch.cat(video, dim=2), torch.cat(audio, dim=3)


def nf_parallel_sample(dit, text, latent_t: int, *, lat_h: int, lat_w: int, config: NFConfig, sigmas, noise: NoiseBank,
                       convkv: ConvKV | None = None, first: torch.Tensor | None = None, sampler: str = "renoise",
                       stop_step: int | None = None, audio: torch.Tensor | None = None):
    """The sampler of ``nf_stream_sample`` run step-outer over a clip of known length.

    Under Neighbor Forcing block ``n`` at step ``i`` depends on blocks ``< n`` at step ``i`` only, so
    one masked pass per step produces every block's step: the autoregressive rollout of a clip costs
    ``steps`` forwards. Returns the clean latent, or with ``stop_step=s`` the state entering step ``s``
    and that step's audio ``(x_s, audio_s)`` (Stage 2 re-runs step ``s`` with gradients). ``audio`` is
    clean driving audio for the clip (default: silent noise). No ``torch.no_grad`` here: the caller chooses.
    """
    steps = len(sigmas) - 1
    x, _audio = parallel_noise(noise, config, latent_t, 0, lat_h, lat_w)
    for step in range(steps):
        sigma, sigma_next = float(sigmas[step]), float(sigmas[step + 1])
        rows = audio if audio is not None else (
            float(audio_sigma(sigma)) * parallel_noise(noise, config, latent_t, step, lat_h, lat_w)[1])
        if stop_step is not None and step == stop_step:
            return x, rows
        velocity = nf_forward(dit, x, 1.0 - sigma, text, config=config, convkv=convkv, first=first, audio=rows,
                              audio_t=None if audio is None else AUDIO_CLEAN)
        x = advance(x, velocity, sigma, sigma_next,
                    lambda: parallel_noise(noise, config, latent_t, step + 1, lat_h, lat_w)[0], sampler)
    return x


# ------------------------------------------------- DiR under Neighbor Forcing (not in the paper)

def dir_step(dit, head, state: torch.Tensor, estimate: torch.Tensor, r: int, text, sigma: float, *,
             config: NFConfig, convkv: ConvKV | None = None, first: torch.Tensor | None = None,
             audio: torch.Tensor | None = None, audio_generator=None) -> torch.Tensor:
    """One DiR iteration ``x0_hat[r+1] = G(x0_hat[r], z, r)`` on a whole clip under ``config``:
    ``dir_h3.generator_step`` (dense) with the block-causal reads. Every block is at iteration ``r``."""
    from dir_h3 import estimate_extra

    out = nf_forward(dit, state, 1.0 - sigma, text, config=config, convkv=convkv, first=first, audio=audio,
                     audio_generator=audio_generator, video_extra=estimate_extra(head, estimate, None),
                     video_temb_delta=head.r_embed[r])
    return state.float() + sigma * out.float()


def _dir_audio(noise: NoiseBank, config: NFConfig, latent_t: int, sigma: float, lat_h: int, lat_w: int) -> torch.Tensor:
    """DiR draws the silent-audio noise once per sample and rescales it per iteration."""
    return float(audio_sigma(sigma)) * parallel_noise(noise, config, latent_t, 0, lat_h, lat_w)[1]


def nf_dir_parallel_sample(dit, head, text, latent_t: int, *, lat_h: int, lat_w: int, config: NFConfig, sigmas,
                           noise: NoiseBank, mix_state: bool = False, convkv: ConvKV | None = None,
                           first: torch.Tensor | None = None, stop_iter: int | None = None):
    """``R = len(sigmas)`` DiR iterations of a whole clip, one masked pass each (``dir_h3.dir_sample`` with
    the causal reads). Returns the final estimate, or with ``stop_iter=r`` what enters iteration ``r``:
    ``(state, estimate, audio)``. ``mix_state``: the DiR run's state law (see ``dir_h3.next_state``)."""
    from dir_h3 import next_state

    z = parallel_noise(noise, config, latent_t, 0, lat_h, lat_w)[0]
    state, estimate = z, torch.zeros_like(z)
    for r, sigma in enumerate(float(s) for s in sigmas):
        audio = _dir_audio(noise, config, latent_t, sigma, lat_h, lat_w)
        if stop_iter is not None and r == stop_iter:
            return state, estimate, audio
        estimate = dir_step(dit, head, state, estimate, r, text, sigma, config=config, convkv=convkv, first=first,
                            audio=audio)
        if r + 1 < len(sigmas):
            state = next_state(state, estimate, sigma, float(sigmas[r + 1]), z if mix_state else None)
    return estimate


@torch.no_grad()
def nf_dir_stream_sample(dit, head, text, blocks: int, *, lat_h: int, lat_w: int, config: NFConfig, sigmas,
                         noise: NoiseBank, mix_state: bool = False, convkv: ConvKV | None = None,
                         first: torch.Tensor | None = None, cache_device=None, total_frames: int | None = None,
                         on_block=None) -> torch.Tensor:
    """A streamed DiR generator: each block runs its ``R`` iterations from its own noise, and at iteration
    ``r`` reads the keys and values earlier blocks wrote at iteration ``r`` (Neighbor Forcing's same-step
    rule with "step" = DiR iteration; one cache per iteration), with the ConvKV window from block 2 on."""
    from dir_h3 import estimate_extra, next_state

    stream = NFStream(dit, lat_h=lat_h, lat_w=lat_w, steps=len(sigmas), config=config, convkv=convkv, first=first,
                      cache_device=cache_device, total_frames=total_frames or config.frames(blocks))
    out = []
    for index in range(blocks):
        start, stop = config.span(index)
        a0, a1 = stream.audio_span(index)
        prompt = text(index) if callable(text) else text
        z = noise.video(index, 0, stop - start, lat_h, lat_w)
        unit = noise.audio(index, 0, a1 - a0)
        state, estimate = z, torch.zeros_like(z)
        for r, sigma in enumerate(float(s) for s in sigmas):
            velocity = stream.velocity(index, r, state, 1.0 - sigma, prompt, float(audio_sigma(sigma)) * unit,
                                       video_extra=estimate_extra(head, estimate, None), video_temb_delta=head.r_embed[r])
            estimate = state.float() + sigma * velocity.float()
            if r + 1 < len(sigmas):
                state = next_state(state, estimate, sigma, float(sigmas[r + 1]), z if mix_state else None)
        out.append(estimate)
        if on_block is not None:
            on_block(index, estimate)
    return torch.cat(out, dim=2)


# --------------------------------------------------------------------------------- saving

def save_nf_run(run_dir, tag: str, *, gen_network, fake_network, convkv: ConvKV | None, meta: dict,
                head=None, fake_head=None) -> None:
    """``lora_gen{tag}.safetensors``, ``lora_fake{tag}.safetensors`` (Stage 2), ``convkv{tag}.pt`` and ``config.json``;
    for a DiR run also ``dir_head{tag}.pt`` and ``dir_fake_head{tag}.pt``."""
    import json
    from pathlib import Path

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if gen_network is not None:
        gen_network.save_weights(str(run_dir / f"lora_gen{tag}.safetensors"), torch.bfloat16,
                                 {"lot_h3": "nf", "kind": "gen", "rank": str(meta["rank"])})
    if fake_network is not None:
        fake_network.save_weights(str(run_dir / f"lora_fake{tag}.safetensors"), torch.bfloat16,
                                  {"lot_h3": "nf", "kind": "fake", "rank": str(meta.get("fake_rank", meta["rank"]))})
    if convkv is not None:
        torch.save(convkv.state_dict(), run_dir / f"convkv{tag}.pt")
    if head is not None:
        torch.save(head.state_dict(), run_dir / f"dir_head{tag}.pt")
    if fake_head is not None:
        torch.save(fake_head.state_dict(), run_dir / f"dir_fake_head{tag}.pt")
    (run_dir / "config.json").write_text(json.dumps(meta, indent=1))


def load_nf_run(run_dir, dit, *, tag: str = "", device="cuda"):
    """Apply a run's generator LoRA to ``dit`` and return ``(network, convkv or None, meta)``."""
    import json
    from pathlib import Path

    from safetensors import safe_open

    from h3_musubi import make_lora

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "config.json").read_text())
    lora_file = run_dir / f"lora_gen{tag}.safetensors"
    with safe_open(str(lora_file), "pt") as handle:
        key = next(k for k in handle.keys() if k.endswith("lora_down.weight"))
        rank = int(handle.get_slice(key).get_shape()[0])
    network = make_lora(dit, rank)
    info = network.load_weights(str(lora_file))
    if info.missing_keys:
        raise RuntimeError(f"{lora_file}: missing {len(info.missing_keys)} LoRA keys")
    network.to(device=device, dtype=torch.bfloat16)
    convkv = None
    conv_file = run_dir / f"convkv{tag}.pt"
    if conv_file.is_file():
        convkv = ConvKV.for_dit(dit, int(meta.get("ratio", RATIO)), bool(meta.get("shared_kv", True)))
        convkv.load_state_dict(torch.load(conv_file, map_location="cpu", weights_only=True))
        convkv.to(device).eval()
    return network, convkv, meta


def load_nf_dir_head(run_dir, meta: dict, *, tag: str = "", device="cuda"):
    """The ``DirHead`` of a DiR-under-Neighbor-Forcing run (``meta["dir"]``), in eval mode."""
    from pathlib import Path

    from dir_h3 import DirHead

    head = DirHead(int(meta["hidden"]), int(meta["temb_dim"]), int(meta["iters"]))
    head.load_state_dict(torch.load(Path(run_dir) / f"dir_head{tag}.pt", map_location="cpu", weights_only=True))
    return head.to(device).eval().requires_grad_(False)
