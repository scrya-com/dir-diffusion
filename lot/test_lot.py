#!/usr/bin/env python3
"""CPU checks for the Level-of-Token adapter. No weights and no GPU.

    python3 lot/test_lot.py

Tests that compare against musubi-tuner's H3 code (positions, final layer,
patch order) run when ``MUSUBI_SRC`` points at a checkout and are skipped otherwise.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import LotVisualAdapter  # noqa: E402
from flow import (  # noqa: E402
    _asymflow,
    apply_extent_scales,
    asymmetric_target,
    clean_from_velocity,
    compress_patches,
    euler_step,
    fit_procrustes,
    gather_extent,
    lot_clean_loss,
    lot_h3_clean_loss,
    mean_basis,
    recover_dense_velocity,
    sample_noisy,
    scatter_extent,
)
from infer import integrate, sigma_grid  # noqa: E402
from h3_positions import packed_positions, sample_axis, spatial_axis, video_positions  # noqa: E402
from h3 import (  # noqa: E402
    H3_EXTENTS,
    clip_layout,
    H3_HIDDEN,
    H3_PATCH,
    H3_TOKEN_DIM,
    gate_frame_layout,
    grid_layout,
    make_h3_adapter,
    over_wan_frame_budget,
    patchify,
    unpatchify,
)
from h3_splice import LotSplice  # noqa: E402
from layout import (  # noqa: E402
    TokenRect,
    blur_radius,
    dense_layout,
    layout_from_blur,
    layout_from_detail,
    layout_from_rects,
    layout_from_regions,
    layout_from_vrs,
    shape_features,
)
from sanity import (  # noqa: E402
    block_macs,
    check_h3_bank_rejects_coarse_square,
    check_short_sequence_full_canvas,
)


def _musubi():
    """musubi-tuner's H3 package, or None when no checkout is available."""
    import h3_musubi  # noqa: F401  (puts MUSUBI_SRC on sys.path)

    try:
        import musubi_tuner.minimax_h3.model as model
        import musubi_tuner.minimax_h3.packing as packing
    except ImportError:
        return None
    return model, packing


def test_shape_and_centers():
    features = shape_features((1, 2, 4))
    assert torch.allclose(features, torch.tensor([1.0, 2.0, 3.0, -1.0]))
    timed = shape_features((2, 2, 2), include_time=True)
    assert timed.shape == (5,) and float(timed[-1]) == 1.0
    rect = TokenRect(3, 4, 8, 1, 2, 4)
    assert rect.center() == (3.0, 4.5, 9.5)
    layout = dense_layout(1, 1, 1)
    assert torch.equal(layout.centers(), torch.zeros(1, 3))
    assert layout.compression == 1.0


def test_partition_rules():
    detail = torch.zeros(4, 4)
    detail[0, 0] = 1
    layout = layout_from_detail(detail, {4: 0.5, 2: 0.5}, root=4)
    assert layout.count == 7
    detail_t = torch.zeros(2, 4, 4)
    detail_t[1, 0, 0] = 1
    timed = layout_from_detail(detail_t, {4: 0.5, 2: 0.5}, root=4)
    assert [rect.eh for rect in timed.rects if rect.t == 0] == [4]
    assert sum(rect.t == 1 for rect in timed.rects) == 7
    assert layout.rects[0] == TokenRect(0, 0, 0, 1, 1, 1)
    assert any(rect.eh == 2 for rect in layout.rects)
    try:
        layout_from_detail(detail, {4: 0.8, 2: 0.2}, root=4)
    except ValueError:
        pass
    else:
        raise AssertionError("decreasing thresholds should fail")
    try:
        layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2), TokenRect(0, 0, 0, 1, 1, 1)])
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping rects should fail")


def test_regions_prefer_finer_level():
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[:2, :2] = True
    layout = layout_from_regions(8, 8, background_level=0, regions=[(mask, 3)], root=8)
    covering = [rect for rect in layout.rects if rect.u == 0 and rect.v == 0]
    assert len(covering) == 1 and covering[0].extent == (1, 1, 1)
    assert any(rect.eh == 4 for rect in layout.rects)
    assert layout.dense_count == 64


def test_blur_and_vrs():
    depth = torch.full((8, 8), 10.0)
    depth[0, 0] = 1.0
    radius = blur_radius(depth, focal_depth=1.0, r_target=4.0)
    assert float(radius[0, 0]) < 1e-4
    assert float(radius[7, 7]) > 3.0
    layout = layout_from_blur(radius, {8: 3.0, 4: 1.0, 2: 0.2}, root=8)
    origin = next(rect for rect in layout.rects if rect.u == 0 and rect.v == 0)
    assert origin.eh == 1 and origin.ew == 1
    assert any(rect.eh == 4 for rect in layout.rects)

    flat = layout_from_vrs(torch.ones(8, 8), sensitivity=0.05, ambient=0.05, root=8)
    assert flat.count == 1 and flat.rects[0].extent == (1, 8, 8)
    spiked = torch.zeros(8, 8)
    spiked[0, 0] = 1
    busy = layout_from_vrs(spiked, sensitivity=0.01, ambient=0.01, root=8)
    assert busy.count > 1


def test_procrustes_and_mean_basis():
    basis = mean_basis(4, (1, 2, 2))
    assert basis.shape == (16, 4)
    assert torch.allclose(basis.T @ basis, torch.eye(4), atol=1e-6)
    assert torch.allclose(mean_basis(5, (1, 1, 1)), torch.eye(5))

    torch.manual_seed(0)
    dense_dim, token_dim, rows = 12, 4, 64
    guess = torch.linalg.qr(torch.randn(dense_dim, token_dim, dtype=torch.float64), mode="reduced").Q
    reference = torch.randn(rows, token_dim, dtype=torch.float64)
    dense = 2.0 * reference @ guess.T
    fitted, scale = fit_procrustes(dense.float(), reference.float())
    assert torch.allclose(fitted.double(), guess, atol=1e-4)
    assert torch.allclose(scale, torch.tensor(2.0), atol=1e-4)
    assert torch.allclose(fitted.T @ fitted, torch.eye(token_dim), atol=1e-4)


def test_velocity_recovery_matches_equation_9():
    try:
        _calibration, _mixin, path = _asymflow()
        assert path.name == "common.py" and "LakonLab" in str(path)
    except FileNotFoundError:
        pass   # no LakonLab checkout: recover_dense_velocity uses the plain-torch eq. 9 below
    basis = mean_basis(4, (1, 1, 3))
    y0 = torch.randn(2, 12)
    eps = torch.randn(2, 12)
    sigma = torch.tensor([0.2, 0.8])
    y_t = (1 - sigma)[:, None] * y0 + sigma[:, None] * eps
    predicted = asymmetric_target(y0, eps, basis)
    recovered = recover_dense_velocity(predicted, y_t, basis, sigma)
    assert torch.allclose(recovered, eps - y0, atol=1e-5)
    y0_hat = clean_from_velocity(y_t, recovered, sigma)
    assert float(lot_clean_loss(y0_hat, y0, sigma)) < 1e-8
    assert torch.allclose(euler_step(eps, eps - y0, 1.0, 0.0), y0, atol=1e-6)


def test_unit_layout_reproduces_pretrained_head():
    torch.manual_seed(1)
    batch, token_dim, hidden = 2, 4, 8
    adapter = LotVisualAdapter(token_dim, hidden, [(1, 1, 1), (1, 2, 2)])
    weight_in = torch.randn(hidden, token_dim)
    bias_in = torch.randn(hidden)
    weight_out = torch.randn(token_dim, hidden)
    bias_out = torch.randn(token_dim)
    adapter.init_from_pretrained(weight_in, weight_out, bias_in, bias_out)
    tokens = torch.randn(batch, 1, 2, 2, token_dim)
    layout = dense_layout(1, 2, 2)
    velocity = adapter(tokens, 0.4, layout, lambda hidden_states, _coords, _sigma: hidden_states)
    flat = tokens.reshape(batch, 4, token_dim)
    manual = (flat @ weight_in.T + bias_in) @ weight_out.T + bias_out
    assert torch.allclose(velocity.reshape(batch, 4, token_dim), manual, atol=1e-5)

    coarse = layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2)])
    embedded, centers = adapter.embed(tokens, coarse)
    packed = tokens.reshape(batch, 1, token_dim * 4)
    compressed = compress_patches(packed, adapter.bank.basis((1, 2, 2)))
    shared = compressed @ weight_in.T + bias_in
    assert torch.allclose(embedded, shared, atol=1e-5)
    assert torch.allclose(centers, torch.tensor([[0.0, 0.5, 0.5]]))


def test_extent_scale_and_backward():
    torch.manual_seed(2)
    adapter = LotVisualAdapter(4, 8, [(1, 1, 1), (1, 2, 2)])
    guess = torch.linalg.qr(torch.randn(16, 4), mode="reduced").Q
    reference = torch.randn(32, 4)
    dense = 2.0 * reference @ guess.T
    adapter.bank.fit_((1, 2, 2), dense, reference)
    assert torch.allclose(adapter.bank.scale((1, 2, 2)), torch.tensor(2.0), atol=1e-4)
    clean = torch.randn(1, 1, 2, 2, 4)
    layout = layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2)])
    scaled = adapter.bank.scale_clean(clean, layout)
    assert torch.allclose(adapter.bank.unscale(scaled, layout), clean, atol=1e-5)

    y0 = torch.randn(2, 1, 4, 4, 4)
    detail = torch.zeros(4, 4)
    detail[0, 0] = 1
    mixed = layout_from_detail(detail, {4: 0.5, 2: 0.5}, root=4)
    y_t, _noise = sample_noisy(y0, 0.5)
    velocity = adapter(y_t, 0.5, mixed, lambda states, _coords, _sigma: states)
    loss = lot_clean_loss(clean_from_velocity(y_t, velocity, 0.5), y0, 0.5)
    loss.backward()
    assert adapter.out_proj["1x1x1"].weight.grad.abs().sum() > 0
    assert adapter.out_proj["1x2x2"].weight.grad.abs().sum() > 0
    assert adapter.bank.basis((1, 2, 2)).grad is None


def test_h3_patch_geometry():
    assert H3_TOKEN_DIM == 96
    assert H3_PATCH == (1, 2, 2)
    assert H3_HIDDEN == 5376
    assert len(H3_EXTENTS) == 9
    assert over_wan_frame_budget(3073, 1)
    assert not over_wan_frame_budget(3072, 1)
    latent = torch.zeros(1, 2, 1, 2, 2)
    latent[0, 1, 0, 0, 1] = 7
    tokens = patchify(latent, patch=(1, 2, 2))
    assert tokens.shape[-1] == 8
    assert float(tokens[0, 0, 0, 0, 5]) == 7
    video = torch.randn(1, 24, 2, 4, 6)
    roundtrip = unpatchify(patchify(video))
    assert torch.equal(roundtrip, video)
    adapter = make_h3_adapter(hidden_size=32)
    assert adapter.token_dim == 96
    assert adapter.bank.basis((1, 4, 4)).shape == (16 * 96, 96)


def test_fit_extent_rebuilds_heads():
    torch.manual_seed(3)
    adapter = LotVisualAdapter(4, 8, [(1, 1, 1), (1, 2, 2)])
    weight_in = torch.randn(8, 4)
    weight_out = torch.randn(4, 8)
    bias_out = torch.randn(4)
    adapter.init_from_pretrained(weight_in, weight_out, None, bias_out)
    before = adapter.out_proj["1x2x2"].weight.detach().clone()
    guess = torch.linalg.qr(torch.randn(16, 4), mode="reduced").Q
    reference = torch.randn(32, 4)
    dense = 2.0 * reference @ guess.T
    adapter.fit_extent((1, 2, 2), dense, reference)
    expected = guess.float() @ weight_out
    assert torch.allclose(adapter.out_proj["1x2x2"].weight, expected, atol=1e-4)
    assert not torch.allclose(adapter.out_proj["1x2x2"].weight, before, atol=1e-4)
    assert torch.allclose(adapter.out_proj["1x1x1"].weight, weight_out, atol=1e-5)
    lifted = adapter.bank.basis((1, 2, 2)) @ bias_out
    assert torch.allclose(adapter.out_proj["1x2x2"].bias, lifted, atol=1e-4)


def test_h3_positions_match_base_grid():
    axis = torch.tensor([0.0, 10.0, 30.0], dtype=torch.float64)
    assert float(sample_axis(axis, 0)) == 0.0
    assert float(sample_axis(axis, 1.5)) == 20.0
    layout = dense_layout(2, 2, 3)
    positions = video_positions(layout, latent_height=4, latent_width=6, origin=5.0)
    area = math.sqrt(4 * 6)
    height = spatial_axis(4, 2, area)
    width = spatial_axis(6, 2, area)
    assert torch.allclose(positions[0, 1], height[0])
    assert torch.allclose(positions[1, 2], width[1])
    coarse = layout_from_rects(2, 2, 3, [
        TokenRect(0, 0, 0, 1, 2, 2),
        TokenRect(0, 0, 2, 1, 2, 1),
        TokenRect(1, 0, 0, 1, 2, 2),
        TokenRect(1, 0, 2, 1, 2, 1),
    ])
    coarse_pos = video_positions(coarse, 4, 6, origin=5.0)
    assert torch.allclose(coarse_pos[0, 1], 0.5 * (height[0] + height[1]))
    packed, video_start = packed_positions(layout, 4, 6, text_len=3, num_audio_latents=2)
    assert video_start == 3 + 4
    assert torch.allclose(packed[video_start:], video_positions(layout, 4, 6, origin=3.0))
    assert float(packed[0, 0]) == 0.0 and float(packed[2, 0]) == 2.0
    assert float(packed[3, 0]) == 3.0 and float(packed[4, 0]) == 4.0
    assert float(packed[5, 0]) == 3.0 and float(packed[5, 2]) != float(packed[3, 2])

    found = _musubi()
    if found is None:
        print("  (musubi-tuner not found: skipped the comparison with its position grid)")
        return
    _model, packing = found
    from h3_musubi import audio_frames

    clip = dense_layout(2, 2, 3)
    for task, keyframes, conditions, roles in (("t2va", None, (), None), ("fl2va", [0], ((1, 4, 6),), ("first",))):
        ours, _start = packed_positions(clip, 4, 6, text_len=3, num_audio_latents=audio_frames(2),
                                        keyframes=keyframes)
        reference = packing.build_position_grid(packing.build_h3_layout(
            task=task, text_length=3, target_video=(2, 4, 6), target_audio_frames=audio_frames(2),
            visual_conditions=conditions, condition_roles=roles))[0]
        assert torch.allclose(ours, reference), task


def test_vectorized_gather_and_scale():
    torch.manual_seed(0)
    layout = layout_from_rects(1, 4, 4, [
        TokenRect(0, 0, 0, 1, 2, 2),
        TokenRect(0, 0, 2, 1, 2, 2),
        TokenRect(0, 2, 0, 1, 1, 2),
        TokenRect(0, 2, 2, 1, 1, 2),
        TokenRect(0, 3, 0, 1, 1, 4),
    ])
    tokens = torch.randn(2, 1, 4, 4, 3, requires_grad=True)
    gathered = []
    for _extent, indices in layout.groups():
        rects = [layout.rects[index] for index in indices]
        gathered.append(gather_extent(tokens, rects))
        slow = torch.stack([
            tokens[:, rect.t:rect.t + rect.et, rect.u:rect.u + rect.eh, rect.v:rect.v + rect.ew]
            .reshape(tokens.shape[0], -1)
            for rect in rects
        ], dim=1)
        assert torch.equal(gathered[-1], slow)
    canvas = tokens.new_zeros(tokens.shape)
    for (_extent, indices), values in zip(layout.groups(), gathered):
        scatter_extent(canvas, values, [layout.rects[index] for index in indices])
    assert torch.equal(canvas, tokens)
    scales = {extent: torch.tensor(2.0 + index) for index, (extent, _rects) in enumerate(layout.groups())}
    scaled = apply_extent_scales(tokens.detach(), layout, scales, invert=False)
    restored = apply_extent_scales(scaled, layout, scales, invert=True)
    assert torch.allclose(restored, tokens.detach())
    assert not torch.allclose(scaled, tokens.detach())


def test_gate_band_and_sigma():
    layout = gate_frame_layout()
    assert layout.count == 462
    assert layout.dense_count == 1008
    assert abs(layout.compression - 1008 / 462) < 1e-9
    assert {rect.extent for rect in layout.rects} == {(1, 1, 1), (1, 2, 2), (1, 4, 2)}
    coarse = layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2)])
    adapter = LotVisualAdapter(4, 8, [(1, 1, 1), (1, 2, 2)])
    tokens = torch.randn(1, 1, 2, 2, 4)
    states = torch.randn(1, 1, 8)
    at_sigma = adapter.velocity_from_states(states, tokens, 0.4, coarse)
    swapped = adapter.velocity_from_states(states, tokens, 0.6, coarse)
    assert not torch.allclose(at_sigma, swapped)


def test_final_layer_sees_modulated_states():
    found = _musubi()
    if found is None:
        print("  (musubi-tuner not found: skipped)")
        return
    FinalLayer = found[0].FinalLayer

    torch.manual_seed(1)
    hidden, token_dim, t_dim = 8, 4, 4
    layer = FinalLayer(hidden, t_dim, token_dim, 2, norm_eps=1e-6, dtype=torch.float32)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_()
    adapter = LotVisualAdapter(token_dim, hidden, [(1, 1, 1)])
    adapter.init_from_pretrained(torch.randn(hidden, token_dim), layer.video_out.weight.detach(), None, layer.video_out.bias.detach())
    states = torch.randn(3, hidden)
    t_emb = torch.randn(1, t_dim)
    splice = LotSplice(adapter, dense_layout(1, 1, 1))
    modulated = splice.modulate(layer, states, t_emb, 0)
    assert torch.allclose(adapter.out_proj["1x1x1"](modulated), layer.video_out(modulated))
    assert not torch.allclose(adapter.out_proj["1x1x1"](states), layer.video_out(modulated))


def test_splice_shortens():
    layout = layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2)])
    adapter = LotVisualAdapter(4, 8, [(1, 2, 2)])
    splice = LotSplice(adapter, layout)
    rows = torch.randn(4, 4)
    tokens = splice.video_tokens(rows, 1, 4, 4)
    assert splice.embed_rows(tokens).shape == (1, 8)
    dense_pos = torch.zeros(2 + 4, 3, dtype=torch.float64)
    dense_pos[-4:, 0] = 5
    replaced = splice.replace_video_positions(dense_pos, 1, 4, 4)
    assert replaced.shape == (3, 3)
    assert float(replaced[-1, 0]) == 5.0
    projected = splice.project_rows(torch.randn(1, 8), tokens, 0.4)
    assert projected.shape == (4, 4)


def test_sanity_short_sequence_writes_nothing():
    def refuse_save(*_args, **_kwargs):
        raise AssertionError("sanity check tried to write a checkpoint")

    original = torch.save
    torch.save = refuse_save
    try:
        numbers = check_short_sequence_full_canvas()
        check_h3_bank_rejects_coarse_square()
    finally:
        torch.save = original
    assert numbers["tokens"] == 6 and numbers["dense"] == 64
    assert numbers["macs"] == block_macs(6, 32, 2, 2)
    assert numbers["macs"] < numbers["dense_macs"] / 2


def test_euler_inference_recovers_clean():
    layout = dense_layout(1, 2, 2)
    clean = torch.randn(2, 1, 2, 2, 3)
    noise = torch.randn_like(clean)

    def predict(_state, _t, _layout):
        return noise - clean

    sampled = integrate(predict, noise, layout, sigma_grid(4))
    assert torch.allclose(sampled, clean, atol=1e-5)
    assert sigma_grid(4).shape == (5,)
    assert float(sigma_grid(1)[0]) == 1.0 and float(sigma_grid(1)[-1]) == 0.0


def test_h3_head_sign_recovery():
    """H3's head is ``x0 - P eps``. Eq. 9 needs the flip; a 1x1 basis cannot show it."""
    torch.manual_seed(5)
    adapter = LotVisualAdapter(4, 16, [(1, 2, 2)])
    with torch.no_grad():
        adapter.out_proj["1x2x2"].weight.copy_(torch.eye(16))
        adapter.out_proj["1x2x2"].bias.zero_()
    layout = layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2)])
    basis = adapter.bank.basis((1, 2, 2))
    y0 = torch.randn(1, 1, 2, 2, 4)
    eps = torch.randn_like(y0)
    sigma = 0.3
    y_t = (1 - sigma) * y0 + sigma * eps
    head = -asymmetric_target(y0.reshape(1, 16), eps.reshape(1, 16), basis)
    states = head.reshape(1, 1, 16)
    h3_velocity = adapter.velocity_from_states(states, y_t, sigma, layout, x0_minus_eps=True)
    assert torch.allclose(h3_velocity, y0 - eps, atol=1e-5)
    wrong = adapter.velocity_from_states(states, y_t, sigma, layout)
    assert not torch.allclose(-wrong, y0 - eps, atol=1e-2)
    # One H3 step to sigma 0 (x0 = x_t + sigma * v, H3's scheduler) lands on the clean latent.
    assert torch.allclose(y_t + sigma * h3_velocity, y0, atol=1e-5)


def test_splice_y_space_contract():
    found = _musubi()

    def patchify_video(latent, _patch):
        """H3's packed row order: musubi's ``pack_video_rows`` when available, else ours."""
        if found is None:
            return patchify(latent).reshape(-1, 96)
        return found[1].pack_video_rows(latent)[0]

    torch.manual_seed(6)
    adapter = make_h3_adapter(hidden_size=8)
    guess = torch.linalg.qr(torch.randn(384, 96), mode="reduced").Q
    reference = torch.randn(400, 96)
    adapter.bank.fit_((1, 2, 2), 2.0 * reference @ guess.T, reference)
    layout = layout_from_rects(1, 2, 4, [TokenRect(0, 0, 0, 1, 2, 2), *(
        TokenRect(0, u, v, 1, 1, 1) for u in range(2) for v in range(2, 4))])
    x0 = torch.randn(1, 24, 1, 4, 8)
    rows = patchify_video(x0, (1, 2, 2))
    assert torch.equal(rows.reshape(1, 1, 2, 4, 96), patchify(x0))

    x_space = LotSplice(adapter, layout)
    assert not x_space.unit_scales()
    try:
        x_space.video_tokens(rows, 1, 4, 8)
    except ValueError as exc:
        assert "y_t" in str(exc)
    else:
        raise AssertionError("fitted scales accepted an x-space splice")

    splice = LotSplice(adapter, layout, y_space=True)
    y0 = splice.to_y(x0)
    assert torch.allclose(patchify(y0), adapter.bank.scale_clean(patchify(x0), layout))
    assert torch.allclose(y0[..., :4, :4], x0[..., :4, :4] / 2.0, atol=1e-4)
    assert torch.equal(y0[..., 4:], x0[..., 4:])
    assert torch.allclose(splice.to_x(y0), x0, atol=1e-5)
    assert splice.video_tokens(patchify_video(y0, (1, 2, 2)), 1, 4, 8).shape == (1, 1, 2, 4, 96)
    assert LotSplice(adapter, dense_layout(1, 2, 4)).unit_scales()


def test_procrustes_h3_fit_and_guards():
    from procrustes_h3 import check_pair, fit_bank, load_pairs, parse_extent

    try:
        load_pairs(Path("/nonexistent/lot_pairs"))
    except SystemExit as exc:
        assert "does not invent" in str(exc)
    else:
        raise AssertionError("a missing pair directory did not exit")
    assert parse_extent("1x4x2") == (1, 4, 2)
    for bad in ("1x4", "a_b_c"):
        try:
            parse_extent(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} parsed")
    try:
        check_pair((1, 2, 2), torch.randn(50, 384), torch.randn(50, 96))
    except ValueError as exc:
        assert "rank" in str(exc)
    else:
        raise AssertionError("too few rows were accepted")

    torch.manual_seed(7)
    adapter = make_h3_adapter(hidden_size=8)
    try:
        fit_bank(adapter, {})
    except RuntimeError:
        pass
    else:
        raise AssertionError("fit before init_from_pretrained")
    adapter.init_from_pretrained(torch.randn(8, 96), torch.randn(96, 8), torch.randn(8), torch.randn(96))
    pairs = {}
    for extent, true_scale in (((1, 2, 2), 2.0), ((1, 4, 2), 0.5)):
        dim = 96 * extent[1] * extent[2]
        lift = torch.linalg.qr(torch.randn(dim, 96), mode="reduced").Q
        reference = torch.randn(300, 96)
        pairs[extent] = check_pair(extent, true_scale * reference @ lift.T, reference)
    reports = {tuple(r["extent"]): r for r in fit_bank(adapter, pairs)}
    assert abs(reports[(1, 2, 2)]["scale"] - 2.0) < 1e-3
    assert abs(reports[(1, 4, 2)]["scale"] - 0.5) < 1e-3
    assert all(r["ortho_err"] < 1e-4 for r in reports.values())
    assert float(adapter.bank.scale((1, 1, 1))) == 1.0


def test_phase4_smoke_canvas():
    layout = clip_layout(7, 12, 20)
    assert layout.count == 7 * (4 * 20 + 2 * 10 + 10)
    assert {rect.extent for rect in layout.rects} == {(1, 1, 1), (1, 2, 2), (1, 4, 2)}
    assert clip_layout(37).count == 17094
    try:
        clip_layout(1, 10, 20)
    except ValueError:
        pass
    else:
        raise AssertionError("a 10-row grid was banded")


def test_pair_rows_align_with_coarse_grid():
    """Row i of ``dense`` must cover the pixels of reference token i (row-major)."""
    from make_pairs_h3 import block_rects

    tokens = torch.randn(1, 1, 8, 4, 3)
    for extent in ((1, 2, 2), (1, 4, 2), (1, 2, 4), (1, 1, 2)):
        _et, eh, ew = extent
        if 4 % ew:
            continue
        dense = gather_extent(tokens, block_rects(8, 4, extent))[0]
        pooled = tokens[0, 0].reshape(8 // eh, eh, 4 // ew, ew, 3).mean(dim=(1, 3)).reshape(-1, 3)
        site_mean = dense.reshape(dense.shape[0], eh * ew, 3).mean(dim=1)
        assert torch.allclose(site_mean, pooled, atol=1e-6), extent


def test_h3_loss_is_velocity_mse_in_y_space():
    torch.manual_seed(8)
    y0 = torch.randn(1, 24, 1, 4, 4)
    eps = torch.randn_like(y0)
    for sigma in (0.3, 0.92):
        y_t = (1 - sigma) * y0 + sigma * eps
        out = torch.randn_like(y0)
        velocity_mse = (out - (y0 - eps)).square().mean()
        assert torch.allclose(lot_h3_clean_loss(out, y_t, y0, sigma), velocity_mse, atol=1e-5)
        assert float(lot_h3_clean_loss(y0 - eps, y_t, y0, sigma)) < 1e-10
    # Below the floor the weight stops growing.
    small = 0.01
    y_t = (1 - small) * y0 + small * eps
    out = torch.randn_like(y0)
    assert float(lot_h3_clean_loss(out, y_t, y0, small)) < float((out - (y0 - eps)).square().mean())


def test_train_layouts_tile_and_mix():
    import random as _random
    from train_h3 import _tile, sample_layout

    torch.manual_seed(9)
    latent = torch.randn(1, 24, 1, 32, 56)                 # 16x28 tokens, 4x7 super-cells
    latent[..., :8, :8] *= 6.0                              # one detailed corner
    kinds = {}
    for seed in range(40):
        layout, kind = sample_layout(latent, _random.Random(seed))
        kinds.setdefault(kind, layout)
        assert layout.dense_count == 16 * 28 and layout.count <= layout.dense_count
        assert {rect.extent for rect in layout.rects} <= set(H3_EXTENTS)
    assert {"dense", "uniform", "mosaic", "box", "focus", "texture"} <= set(kinds) | {"mosaic"}
    for name in ("dense", "uniform", "mosaic", "box", "focus", "texture"):
        layout, kind = sample_layout(latent, _random.Random(1), force=name)
        assert kind == name and layout.dense_count == 16 * 28
        assert {rect.extent for rect in layout.rects} <= set(H3_EXTENTS)
    clip = torch.randn(1, 24, 7, 32, 56)
    moving, _ = sample_layout(clip, _random.Random(3), force="box")
    per_frame = [frozenset((r.u, r.v) for r in moving.rects if r.t == t and r.extent == (1, 1, 1)) for t in range(7)]
    assert len(set(per_frame)) > 1                          # the box drifts across latent frames
    from train_h3 import _legacy_mosaic, parse_mix
    assert sample_layout(latent, _random.Random(5), force="eval_mosaic")[0] == _legacy_mosaic(latent, _random.Random(5), 0.0)[0]
    assert parse_mix("box:0.5,focus:0.5") == {"box": 0.5, "focus": 0.5}
    mosaic = sample_layout(latent, _random.Random(11), force="mosaic")[0]
    kinds["mosaic"] = mosaic
    mosaic = kinds["mosaic"]
    assert len({rect.extent for rect in mosaic.rects}) > 1
    corner = {rect.extent for rect in mosaic.rects if rect.u < 4 and rect.v < 4}
    assert corner == {(1, 1, 1)}                            # the detailed cell stays fine
    for extent in H3_EXTENTS:
        assert _tile(1, 8, 8, lambda *_: extent).count == 64 // (extent[1] * extent[2])


def test_train_shift_moves_mass_to_low_sigma():
    from h3_musubi import sample_sigma
    from train_h3 import parse_shift

    def sample_sigmas(count, device, shift, generator):
        return sample_sigma(shift, device, generator, count)

    assert parse_shift("12") == 12.0 and parse_shift("sigmoid") == "sigmoid"
    assert parse_shift("lognorm:3") == "lognorm:3"
    gen = torch.Generator().manual_seed(0)
    h3 = sample_sigmas(20000, "cpu", shift=parse_shift("12"), generator=gen)
    low = sample_sigmas(20000, "cpu", shift=parse_shift("3"), generator=gen)
    assert float((h3 < 0.3).float().mean()) < 0.06            # H3's own: ~3.5% below 0.3
    assert float((low < 0.3).float().mean()) > 0.10            # shift 3: several times more
    assert float(low.median()) < float(h3.median())


def test_torch_eq9_matches_lakonlab():
    from flow import _broadcast_sigma, asymflow_velocity_torch

    torch.manual_seed(10)
    basis = torch.linalg.qr(torch.randn(16, 4), mode="reduced").Q
    u_a = torch.randn(3, 5, 16)
    x_t = torch.randn(3, 5, 16)
    for sigma in (0.05, torch.tensor([0.2, 0.5, 0.9])):
        lakon = recover_dense_velocity(u_a, x_t, basis, sigma)
        plain = asymflow_velocity_torch(u_a, x_t, basis, _broadcast_sigma(sigma, u_a))
        assert torch.allclose(lakon, plain, atol=1e-5)


def test_grid_layouts_any_shape():
    from h3 import GRID_LAYOUTS, grid_layout

    for height, width in ((36, 24), (19, 30), (24, 42), (12, 20)):
        for name in GRID_LAYOUTS:
            layout = grid_layout(name, 1, height, width)
            assert layout.dense_count == height * width
            if name == "dense":
                assert layout.count == layout.dense_count
            else:
                assert layout.count < layout.dense_count
    center = grid_layout("center", 1, 36, 24)
    middle = [r.extent for r in center.rects if 16 <= r.u < 20 and 8 <= r.v < 16]
    corner = [r.extent for r in center.rects if r.u < 4 and r.v < 4]
    assert set(middle) == {(1, 1, 1)} and set(corner) == {(1, 4, 4)}
    bands = grid_layout("bands", 1, 36, 24)
    assert {r.extent for r in bands.rects if r.u < 12} == {(1, 1, 1)}
    assert {r.extent for r in bands.rects if r.u >= 24} == {(1, 4, 2)}


def test_layout_sources():
    from layout_sources import (boxes_to_mask, compression, depth_layout, latent_frame_spans,
                                layout_from_desired_any, mask_layout, texture_layout, to_latent_frames)

    assert latent_frame_spans(7)[-1][1] == 22 and latent_frame_spans(1) == [(0, 1)]
    # mask: box in the middle of a 768x512 still -> fine inside, coarse outside
    mask = boxes_to_mask([(0.375, 0.375, 0.625, 0.625)], 768, 512)
    lay = mask_layout(mask, 1, 24, 16, dilate=0)
    inside = {r.extent for r in lay.rects if 10 <= r.u < 14 and 7 <= r.v < 9}
    corner = {r.extent for r in lay.rects if r.u < 4 and r.v < 4}
    assert inside == {(1, 1, 1)} and corner == {(1, 4, 4)}
    assert lay.dense_count == 24 * 16 and 1.0 < compression(lay) < 16
    # a moving box over 22 pixel frames -> the layout changes per latent frame
    frames = torch.zeros(22, 768, 512)
    for f in range(22):
        x = 0.1 + 0.6 * f / 21
        frames[f] = boxes_to_mask([(x, 0.4, x + 0.2, 0.6)], 768, 512)[0]
    moving = mask_layout(frames, 7, 24, 16, dilate=0)
    first = {(r.u, r.v) for r in moving.rects if r.t == 0 and r.extent == (1, 1, 1)}
    last = {(r.u, r.v) for r in moving.rects if r.t == 6 and r.extent == (1, 1, 1)}
    assert first and last and first != last
    # depth: in-focus band fine, far plane coarse
    depth = torch.linspace(0, 1, 768)[:, None].expand(768, 512)
    dl = depth_layout(depth, 1, 24, 16, focus=1.0, band=0.15)
    assert {r.extent for r in dl.rects if r.u >= 22} == {(1, 1, 1)}
    assert (1, 4, 4) in {r.extent for r in dl.rects if r.u < 8}
    # texture: noisy half fine, flat half coarse
    image = torch.zeros(1, 768, 512, 3)
    image[:, :, :256] = torch.rand(1, 768, 256, 3)
    tl = texture_layout(image, 1, 24, 16, fine_q=0.5, mid_q=0.3)
    assert {r.extent for r in tl.rects if r.v < 6} == {(1, 1, 1)}
    assert (1, 4, 4) in {r.extent for r in tl.rects if r.v >= 10}
    # odd grids: edges stay 1x1 and every token is covered exactly once
    odd = layout_from_desired_any(torch.full((2, 19, 30), 4))
    assert odd.dense_count == 2 * 19 * 30
    assert to_latent_frames(torch.rand(5, 3, 3), 2).shape == (2, 3, 3)


def test_dir_pool_rows_and_head():
    from dir_h3 import DirHead, estimate_extra, pool_rows

    rows = torch.arange(1 * 8 * 6 * 3, dtype=torch.float32).reshape(1, 48, 3)
    dense = dense_layout(1, 8, 6)
    assert pool_rows(rows, dense) is rows            # dense raster layout: identity
    # a 2x2 rect averages its four dense tokens; rect order follows layout.rects
    rects = [TokenRect(0, 0, 0, 1, 2, 2)] + [TokenRect(0, u, v, 1, 1, 1)
                                              for u in range(8) for v in range(6) if not (u < 2 and v < 2)]
    layout = layout_from_rects(1, 8, 6, rects)
    pooled = pool_rows(rows, layout)
    assert pooled.shape == (1, layout.count, 3)
    grid = rows.reshape(1, 8, 6, 3)
    assert torch.allclose(pooled[0, 0], grid[0, :2, :2].reshape(4, 3).mean(0))
    assert torch.equal(pooled[0, 1], grid[0, 0, 2])  # next rect is the 1x1 at (0, 2)
    assert torch.equal(pooled[0, -1], grid[0, 7, 5])
    # bands layout: every rect equals the mean of the dense tokens it covers
    bands = grid_layout("bands", 1, 12, 8)
    dense_rows = torch.randn(1, 96, 5)
    out = pool_rows(dense_rows, bands)
    for index in (0, bands.count // 2, bands.count - 1):
        rect = bands.rects[index]
        want = torch.stack([dense_rows[0, rect.u * 8 + rect.v + du * 8 + dv]
                            for du in range(rect.eh) for dv in range(rect.ew)]).mean(0)
        assert torch.allclose(out[0, index], want, atol=1e-6)
    # untrained head: the new projection and r-embedding are exactly zero
    head = DirHead(hidden=16, temb_dim=8, iterations=4)
    estimate = torch.randn(1, 24, 1, 16, 12)
    assert float(estimate_extra(head, estimate, None).abs().max()) == 0.0
    coarse = grid_layout("uniform4", 1, 8, 6)
    assert estimate_extra(head, estimate, coarse).shape == (1, coarse.count, 16)
    assert head.r_embed.shape == (4, 8) and float(head.r_embed.abs().max()) == 0.0


def test_dir_layout_schedule():
    from dir_h3 import parse_layout_schedule

    plan = parse_layout_schedule("bands,bands,dense,dense", 4, 12, 8, 1)
    assert plan[2] is None and plan[3] is None
    assert plan[0] is plan[1] and plan[0].count < plan[0].dense_count == 96
    assert parse_layout_schedule("dense", 4, 12, 8, 1) == [None] * 4      # one name repeats
    assert all(entry is not None for entry in parse_layout_schedule("center", 3, 12, 8, 2))
    for bad in ("dense,dense", "bands,nope,dense,dense"):
        try:
            parse_layout_schedule(bad, 4, 12, 8, 1)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_dir_video_modulation_rewrite():
    from lot_paths import MUSUBI_SRC

    if MUSUBI_SRC not in sys.path:
        sys.path.insert(0, MUSUBI_SRC)
    try:
        from musubi_tuner.minimax_h3.packing import build_h3_layout, build_timestep_rows
    except ImportError as error:
        print(f"SKIPPED test_dir_video_modulation_rewrite: musubi_tuner not importable ({error})", flush=True)
        return
    from h3_musubi import audio_frames, audio_sigma, video_modulation_rewrite

    layout = build_h3_layout(task="t2va", text_length=4, target_video=(1, 8, 8),
                             target_audio_frames=audio_frames(1), one_frame=True)
    tags = torch.ones(1, 4, dtype=torch.int64)
    for t_video in (0.0, 0.6):                       # t = 0: audio shares the video's timestep row
        rows = build_timestep_rows(layout, text_token_tags=tags, model_t_video=t_video,
                                   model_t_audio=1.0 - float(audio_sigma(1.0 - t_video)))
        video = layout.target_video_segment.row_slice
        original = rows.block_adaln_indices[0]
        new = len(rows.unique_timesteps)
        indices, segments = video_modulation_rewrite(original, video, new)
        outside = torch.ones_like(indices, dtype=torch.bool)
        outside[video] = False
        assert torch.equal(indices[outside], original[outside])          # text and audio untouched
        assert bool((indices[video] == 3 * new).all())                   # video: new row, tag 0
        assert segments[-1] == (video.start, video.stop, 3 * new)
        assert segments[0][0] == 0 and segments[-1][1] == layout.row_count
        assert all(a[1] == b[0] for a, b in zip(segments, segments[1:]))
        assert int(original[video][0]) == 3 * rows.video_timestep_index


def test_dir_ema_warmup():
    """EMA warm-up: early updates follow the weights; the decay ramps to its cap; the average forgets init."""
    from train_dir_h3 import Ema

    w = torch.nn.Parameter(torch.zeros(3))
    ema = Ema([w], 0.999)
    assert abs(ema.current_decay() - 0.1) < 1e-9          # n = 0: (1 + 0) / (10 + 0)
    with torch.no_grad():
        w.fill_(1.0)
    for _ in range(600):
        ema.update()
    # Plain 0.999 from step 0 would still be 0.999^600 = 55% at the init value 0.
    assert float(ema.shadow[0].min()) > 0.999, ema.shadow[0]
    assert ema.updates == 600 and ema.current_decay() < 0.999
    assert Ema([w], 0.999, updates=10**6).current_decay() == 0.999


def test_dir_use_networks_restores():
    from dir_h3 import use_networks

    class Module:
        enabled = True

    class Net:
        def __init__(self):
            self.unet_loras = [Module(), Module()]

    gen, fake = Net(), Net()
    with use_networks(gen, fake, gen=False, fake=True):
        assert not gen.unet_loras[0].enabled and fake.unet_loras[1].enabled
        with use_networks(gen, None, gen=True, fake=False):
            assert gen.unet_loras[1].enabled
        assert not gen.unet_loras[1].enabled
    assert all(m.enabled for m in gen.unet_loras + fake.unet_loras)


# ------------------------------------------------------------------------- streaming DiR

def test_dir_paper_init():
    """--paper-init: Xavier x0_proj with a zero, frozen bias; r embeddings ~ N(0, 0.02); default stays zero."""
    from dir_h3 import DirFakeHead, DirHead

    zero = DirHead(512, 8, 4)
    assert float(zero.x0_proj.weight.abs().max()) == 0.0 and float(zero.r_embed.abs().max()) == 0.0
    head = DirHead(512, 8, 4).paper_init(torch.Generator().manual_seed(0))
    bound = (6.0 / (96 + 512)) ** 0.5
    w = head.x0_proj.weight
    assert float(w.abs().max()) <= bound and float(w.std()) > 0.5 * bound / 3 ** 0.5
    assert float(head.x0_proj.bias.abs().max()) == 0.0 and not head.x0_proj.bias.requires_grad
    assert 0.01 < float(head.r_embed.std()) < 0.04
    fake = DirFakeHead(8, 4).paper_init(torch.Generator().manual_seed(1))
    assert 0.01 < float(fake.fake_r_embed.std()) < 0.04
    head.load_state_dict(zero.state_dict())          # old zero-init checkpoints still load


def test_dir_sigma_schedule_is_euler():
    """Zero head: all-ones sigmas keep the paper's pinned DiR (state = z); a schedule below 1 is
    H3's deterministic Euler sampler on that grid from the same z (Turbo's grid primes DiR)."""
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_dir_sigma_schedule_is_euler: musubi_tuner not importable", flush=True)
        return
    from dir_h3 import DirHead, dir_sample, generator_step, parse_sigmas, probe_temb_dim
    from h3_musubi import forward_video, shift_sigma

    head = DirHead(dit.config.hidden_size, probe_temb_dim(dit), 4)
    text = torch.randn(1, 5, 16)
    with torch.no_grad():
        _, pinned = dir_sample(dit, head, text, height=64, width=96, seed=3, device="cpu")
        z = torch.randn(1, 24, 1, 4, 6, generator=torch.Generator("cpu").manual_seed(3))
        estimate = torch.zeros_like(z)
        for r in range(4):
            estimate = generator_step(dit, head, z, estimate, r, text,
                                      audio_generator=torch.Generator("cpu").manual_seed(4))
            assert float((pinned[r] - estimate).abs().max()) < 1e-5, r

        sigmas = parse_sigmas("turbo", 4)
        q = torch.linspace(1.0, 0.0, 5, dtype=torch.float64)
        assert all(abs(a - float(shift_sigma(b, 6.0))) < 1e-12 for a, b in zip(sigmas, q))
        assert parse_sigmas("grid:6", 4) == sigmas and parse_sigmas("grid", 4)[1] == float(shift_sigma(q[1], 12.0))
        _, iterated = dir_sample(dit, head, text, height=64, width=96, seed=3, sigmas=sigmas, device="cpu")
        x = z.clone()
        for r, sigma in enumerate(sigmas + [0.0]):
            if r == 4:
                break
            out = forward_video(dit, x, 1.0 - sigma, text, audio_generator=torch.Generator("cpu").manual_seed(4)).float()
            x0 = x + sigma * out
            assert float((iterated[r] - x0).abs().max()) < 1e-4, r
            ratio = (sigmas + [0.0])[r + 1] / sigma
            x = ratio * x + (1.0 - ratio) * x0
        assert float((iterated[3] - pinned[3]).abs().max()) > 1e-3     # the schedule changes the result
        # --mix-state: iteration r sees sigma_r * z + (1 - sigma_r) * x0_hat[r] at flow time sigma_r
        _, mixed = dir_sample(dit, head, text, height=64, width=96, seed=3, sigmas=sigmas, mix_state=True, device="cpu")
        estimate = torch.zeros_like(z)
        for r, sigma in enumerate(sigmas):
            x = sigma * z + (1.0 - sigma) * estimate
            out = forward_video(dit, x, 1.0 - sigma, text, audio_generator=torch.Generator("cpu").manual_seed(4)).float()
            estimate = x + sigma * out
            assert float((mixed[r] - estimate).abs().max()) < 1e-4, r
        assert float((mixed[0] - iterated[0]).abs().max()) < 1e-6 and float((mixed[3] - iterated[3]).abs().max()) > 1e-4
        _, pinned_mix = dir_sample(dit, head, text, height=64, width=96, seed=3, mix_state=True, device="cpu")
        assert float((pinned_mix[3] - pinned[3]).abs().max()) < 1e-6    # all-ones sigmas: the state is z either way
    for bad in ("0.5,0.4,0.3,0.2", "1,0.5", "1,0.5,0.7,0.2"):
        try:
            parse_sigmas(bad, 4)
        except ValueError:
            continue
        raise AssertionError(bad)


def _tiny_h3(num_layers: int = 2):
    """A 2-layer, 64-wide MiniMax-H3 on the CPU in fp32 (random weights), or None without musubi."""
    if _musubi() is None:
        return None
    from musubi_tuner.minimax_h3.model import MiniMaxH3Config, MiniMaxH3Model

    config = MiniMaxH3Config(hidden_size=64, num_layers=num_layers, num_attention_heads=2, attention_head_dim=32,
                             ffn_hidden_size=128, text_dim=16, timestep_input_dim=16, time_embed_hidden_size=32,
                             time_embed_dim=16, rope_inv_freq_len=4)
    torch.manual_seed(0)
    dit = MiniMaxH3Model(config, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for param in dit.parameters():
            param.normal_(0.0, 0.3)       # large enough that the AdaLN gates let rows talk
    return dit.eval()


def _stream_head(dit, iterations=2):
    from dir_h3 import probe_temb_dim
    from stream_dir_h3 import StreamHead

    return StreamHead(dit.config.hidden_size, probe_temb_dim(dit), iterations)


def test_stream_layout_and_positions_match_stock():
    if _musubi() is None:
        print("SKIPPED test_stream_layout_and_positions_match_stock: musubi_tuner not importable", flush=True)
        return
    from musubi_tuner.minimax_h3.packing import build_position_grid
    from stream_dir_h3 import MemoryEntry, memory_rows, pooled_centres, stream_layout, stream_positions

    for first in (False, True):
        layout = stream_layout(5, 7, 4, 6, first=first, memory=0)
        assert torch.equal(stream_positions(layout), build_position_grid(layout)[0])
    # pool 1 centres are the dense frame grid; pool 2 centres are block means
    dense = stream_layout(5, 7, 8, 12, first=True, memory=0)
    first_rows = dense.segment("first").row_slice
    grid = build_position_grid(dense)[0][first_rows, 1:]
    assert torch.equal(pooled_centres(8, 12, 1), grid)
    assert torch.allclose(pooled_centres(8, 12, 2)[0], grid.reshape(4, 6, 2)[:2, :2].reshape(4, 2).mean(0))
    # memory goes after the text, the window keeps every relative time and moves by the oldest age
    entries = [MemoryEntry(torch.zeros(3, 96), torch.zeros(3, 2), time) for time in (0.0, 10.0)]
    tokens, positions = memory_rows(entries, 30.0, "real")
    assert tokens.shape == (6, 96) and positions[:3, 0].eq(-30.0).all() and positions[3:, 0].eq(-20.0).all()
    layout = stream_layout(5, 7, 4, 6, first=True, memory=6)
    assert [s.role for s in layout.segments] == ["text", "memory", "first", "target_audio", "target_video"]
    assert layout.segments[1].start == 5 and layout.segments[1].stop == 11
    got = stream_positions(layout, positions)
    stock = build_position_grid(stream_layout(5, 7, 4, 6, first=True, memory=0))[0]
    tail = slice(5 + 6, None)
    assert torch.allclose(got[tail, 0] - 30.0, stock[5:, 0], atol=1e-9) and torch.equal(got[tail, 1:], stock[5:, 1:])
    assert torch.equal(got[5:8, 0], torch.full((3,), 5.0, dtype=torch.float64))   # oldest memory right after text
    _tokens, packed = memory_rows(entries, 1000.0, "packed")
    assert packed[0, 0] < packed[3, 0] < 0 and float(packed[3, 0]) > -50                # ages capped by count


def test_stream_frame_modulation_rewrite():
    if _musubi() is None:
        print("SKIPPED test_stream_frame_modulation_rewrite: musubi_tuner not importable", flush=True)
        return
    from musubi_tuner.minimax_h3.packing import build_timestep_rows
    from stream_dir_h3 import frame_modulation_rewrite, stream_layout

    layout = stream_layout(4, 7, 4, 6, first=True, memory=3)
    rows = build_timestep_rows(layout, text_token_tags=torch.ones(1, 4, dtype=torch.int64), model_t_video=0.0,
                               model_t_audio=0.0)
    video = layout.target_video_segment.row_slice
    frame_index = [9, 9, 7, 7, 7, 7, 8]
    indices, segments = frame_modulation_rewrite(rows.block_adaln_indices[0], video, 6, frame_index)
    outside = torch.ones_like(indices, dtype=torch.bool)
    outside[video] = False
    assert torch.equal(indices[outside], rows.block_adaln_indices[0][outside])
    assert indices[video].tolist() == [3 * k for k in frame_index for _ in range(6)]
    assert segments[-3:] == ((video.start, video.start + 12, 27), (video.start + 12, video.start + 36, 21),
                             (video.start + 36, video.stop, 24))
    # memory and first are clean conditions: their timestep differs from the target's
    memory = layout.segment("memory").row_slice
    assert rows.row_timesteps[0, memory].eq(0.999).all() and rows.row_timesteps[0, video].eq(0.0).all()


def test_stream_forward_matches_stock_h3():
    """Zero head, one shared r, no memory: the stream forward is H3's own T2VA / FL2VA-first forward."""
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_stream_forward_matches_stock_h3: musubi_tuner not importable", flush=True)
        return
    from h3_musubi import _silence, audio_frames
    from stream_dir_h3 import stream_forward, stream_layout

    head = _stream_head(dit)
    z, text = torch.randn(1, 24, 7, 4, 6), torch.randn(1, 5, 16)
    estimate = torch.randn_like(z)          # x0_proj is zero: the estimate cannot matter yet
    tags = torch.ones(1, 5, dtype=torch.int64)
    for first in (None, torch.randn(1, 24, 1, 4, 6)):
        layout = stream_layout(5, 7, 4, 6, first=first is not None, memory=0)
        audio = _silence(7, 1.0, "cpu", torch.Generator().manual_seed(3))
        with torch.no_grad():
            stock = dit(video_latents=z, audio_latents=audio, text_hidden_states=text, text_token_tags=tags,
                        layout=layout, model_t_video=0.0, model_t_audio=0.0,
                        visual_condition_latents=[] if first is None else [first]).video
            for r in (0, 2):
                got = stream_forward(dit, head, z, estimate, torch.full((7,), r), text, first=first,
                                     audio_generator=torch.Generator().manual_seed(3))
                assert float((got - (z + stock)).abs().max()) < 1e-5, (first is None, r)
    assert audio.shape[-1] == audio_frames(7)


def test_stream_per_frame_r_and_memory_reach_the_output():
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_stream_per_frame_r_and_memory_reach_the_output: musubi_tuner not importable", flush=True)
        return
    from stream_dir_h3 import MemoryEntry, coarse_tokens, memory_rows, pooled_centres, stream_forward

    head = _stream_head(dit)
    z, text = torch.randn(1, 24, 7, 4, 6), torch.randn(1, 5, 16)
    r = torch.tensor([2, 2, 0, 0, 0, 0, 0])

    def run(memory=None):
        with torch.no_grad():
            return stream_forward(dit, head, z, torch.zeros_like(z), r, text, first=torch.zeros(1, 24, 1, 4, 6),
                                  memory=memory, audio_generator=torch.Generator().manual_seed(1))

    # zero-init r rows still get separate gradients (grouping is by r, not by the delta's value)
    head.r_embed.requires_grad_(True)
    stream_forward(dit, head, z, torch.zeros_like(z), r, text, first=torch.zeros(1, 24, 1, 4, 6),
                   audio_generator=torch.Generator().manual_seed(1)).square().mean().backward()
    norms = head.r_embed.grad.norm(dim=1)
    assert norms[0] > 0 and norms[2] > 0 and norms[1] == 0, norms
    head.r_embed.requires_grad_(False)
    base = run()
    with torch.no_grad():
        head.r_embed[1].normal_()            # no frame is at r = 1: nothing moves
    assert torch.equal(run(), base)
    with torch.no_grad():
        head.r_embed[2].normal_()            # frames 0-1 are done (r = R = 2): the output moves
    moved = run()
    assert float((moved - base).abs().max()) > 1e-3
    # memory rows attend into the window; coarse tokens average the patch tokens of a block
    frame = torch.randn(1, 24, 1, 4, 6)
    tokens = coarse_tokens(frame, 2)
    assert tokens.shape == (2, 96)
    assert torch.allclose(tokens[0], patchify(frame)[0, 0, :2, :2].reshape(4, 96).mean(0))
    assert coarse_tokens(frame, 4).shape == (1, 96)                     # ceil: edge block averages its 2x3
    assert torch.allclose(coarse_tokens(frame, 4)[0], patchify(frame)[0, 0].reshape(6, 96).mean(0))
    memory = memory_rows([MemoryEntry(tokens, pooled_centres(4, 6, 2), -40.0)], 0.0)
    with_memory = run(memory)
    other = memory_rows([MemoryEntry(-tokens, pooled_centres(4, 6, 2), -40.0)], 0.0)
    scale = float(moved.abs().max())
    assert float((with_memory - moved).abs().max()) > 1e-4 * scale        # memory rows reach the window
    assert float((run(other) - with_memory).abs().max()) > 1e-4 * scale   # through their content


def test_stream_slide_is_exact_and_bounded():
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_stream_slide_is_exact_and_bounded: musubi_tuner not importable", flush=True)
        return
    from stream_dir_h3 import SLIDE, advance, clip_time, ready, rows_per_window, slide, start_state, stream_sample

    head = _stream_head(dit, iterations=2)
    text = torch.randn(1, 5, 16)
    first = torch.randn(1, 24, 1, 4, 6)
    state = start_state((7, 4, 6), 2, first=first, generator=torch.Generator().manual_seed(0), device="cpu")
    assert state.r.tolist() == [2, 0, 0, 0, 0, 0, 0] and torch.equal(state.estimate[:, :, :1], first)
    for _ in range(2):
        assert not ready(state, 2)
        advance(dit, head, state, text)
    assert ready(state, 2) and torch.equal(state.estimate[:, :, :1], first)   # the condition never changes
    old = state.estimate.clone()
    old_z = state.z.clone()
    emitted = slide(state, 2, pool=2, capacity=2, generator=torch.Generator().manual_seed(1))
    assert torch.equal(emitted, old[:, :, :SLIDE])
    assert torch.equal(state.estimate[:, :, :2], old[:, :, SLIDE:]) and torch.equal(state.z[:, :, :2], old_z[:, :, SLIDE:])
    assert state.estimate[:, :, 2:].abs().max() == 0 and state.r.tolist() == [2, 2, 0, 0, 0, 0, 0]
    assert state.has_first and abs(state.time - clip_time()) < 1e-9 and len(state.memory) == 1
    try:
        slide(state, 2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("slide before the window is done")
    # a full stream: one clip per window, bounded memory, emitted clips chain through the condition frame
    clips = []
    latents = stream_sample(dit, head, text, shape=(7, 4, 6), clips=4, first=first, pool=2, capacity=2,
                            device="cpu", on_clip=lambda i, c: clips.append(c))
    assert latents.shape == (1, 24, 20, 4, 6) and torch.equal(latents[:, :, :1], first)
    assert torch.equal(torch.cat(clips, 2), latents)
    with_tail = stream_sample(dit, head, text, shape=(7, 4, 6), clips=4, first=first, pool=2, capacity=2, device="cpu",
                              tail=True)
    assert with_tail.shape[2] == 22 and torch.equal(with_tail[:, :, :20], latents)   # 5n+2: decodes unpadded
    assert rows_per_window((7, 4, 6), memory=2, pool=2, text_length=5) == 5 + 2 * 2 + 6 + 2 * 37 + 7 * 6


def test_stream_head_warm_start():
    from dir_h3 import DirHead
    from stream_dir_h3 import StreamHead

    still = DirHead(hidden=16, temb_dim=8, iterations=4)
    with torch.no_grad():
        for param in still.parameters():
            param.normal_()
    stream = StreamHead(16, 8, 4)
    stream.load_dir_head(still.state_dict())
    assert torch.equal(stream.r_embed[:4], still.r_embed) and stream.r_embed[4].abs().max() == 0
    assert torch.equal(stream.x0_proj.weight, still.x0_proj.weight)
    assert stream.mem_proj.weight.abs().max() == 0 and stream.r_embed.shape == (5, 8)


def test_stream_trainer_updates_on_cpu():
    """One window-0 and one rolled DMD generator update and a fake update on the toy H3: grads reach
    only their own side, the regenerated sample matches, and done frames get no DMD signal."""
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_stream_trainer_updates_on_cpu: musubi_tuner not importable", flush=True)
        return
    import argparse
    import random

    from stream_dir_h3 import StreamFakeHead
    from train_stream_dir_h3 import StreamRun

    head = _stream_head(dit, iterations=2)
    fake_head = StreamFakeHead(head.r_embed.shape[1], 2)
    with torch.no_grad():
        fake_head.fake_r_embed.normal_(0.0, 0.3)          # fake != real, so the DMD gradient is non-zero
    args = argparse.Namespace(frames=7, rolled=0.0, pool=2, capacity=4, memory_time="real",
                              tau_shift="beta:2.5", dmd_weight="paper")
    run = StreamRun(dit, head, fake_head, None, None, args, "cpu")
    still, text = torch.randn(1, 24, 1, 4, 6), torch.randn(1, 5, 16)
    rng = random.Random(0)
    for rolled in (False, True):
        head.zero_grad(set_to_none=True)
        fake_head.zero_grad(set_to_none=True)
        loss, sample, stats = run.generator_update(still, text, rng, rolled=rolled)
        assert math.isfinite(stats["loss"]) and stats["match"] < 1e-5, stats
        assert stats["active"] == (6 if not rolled else 5) and stats["rolled"] == int(rolled)
        assert head.x0_proj.weight.grad is not None and float(head.r_embed.grad.abs().sum()) > 0
        assert fake_head.fake_r_embed.grad is None
        state, r_frames, x0_hat, _text = sample
        done = r_frames >= 2
        assert torch.equal(x0_hat[:, :, done], state.estimate[:, :, done])       # context frames untouched
        if rolled:
            assert len(state.memory) == 1 and head.mem_proj.weight.grad is not None
            assert float(head.mem_proj.weight.grad.abs().sum()) > 0
        head.zero_grad(set_to_none=True)
        floss, _tau = run.fake_update(sample)
        assert math.isfinite(float(floss.detach())) and float(fake_head.fake_r_embed.grad.abs().sum()) > 0
        assert head.x0_proj.weight.grad is None


def test_dir_trainer_draws_on_cpu():
    """The still DMD generator update on the toy H3: one draw reproduces the single-draw direction, K draws
    average K independent (tau, eps) directions, and grads reach only the generator's head."""
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_dir_trainer_draws_on_cpu: musubi_tuner not importable", flush=True)
        return
    import argparse

    from dir_h3 import DirFakeHead, DirHead, probe_temb_dim
    from train_dir_h3 import Run

    temb = probe_temb_dim(dit)
    head = DirHead(dit.config.hidden_size, temb, 2).paper_init(torch.Generator().manual_seed(1))
    fake_head = DirFakeHead(temb, 2)
    with torch.no_grad():
        fake_head.fake_r_embed.normal_(0.0, 0.3)          # fake != real, so the DMD gradient is non-zero
    args = argparse.Namespace(iterations=2, sigmas="1", seed=0, schedule="dense", tau_shift="beta:2.5",
                              dmd_weight="paper", priming_weight=0.0, priming_refs=1, ot_weight=0.0, dmd_draws=1)
    run = Run(dit, head, fake_head, None, None, None, args, "cpu")
    z, text = torch.randn(1, 24, 1, 4, 6), torch.randn(1, 5, 16)

    torch.manual_seed(5)
    _loss, x0_hat, one = run.generator_update(z, text, 1)
    assert one["draw_cos"] is None and one["match"] < 1e-5 and math.isfinite(one["loss"]), one
    grad_one = head.x0_proj.weight.grad.clone()
    assert float(grad_one.abs().sum()) > 0 and fake_head.fake_r_embed.grad is None
    # the helper alone, from the same RNG point, gives the direction that update used
    torch.manual_seed(5)
    seed = int(torch.randint(0, 2**31 - 1, (1,)))
    direction, tau, _w, _gap, _fr = run.dmd_direction(x0_hat, 1, text, seed)
    assert tau == one["tau"] and abs(float(direction.pow(2).mean().sqrt()) - one["grad_rms"]) < 1e-6

    head.zero_grad(set_to_none=True)
    args.dmd_draws = 3
    torch.manual_seed(5)
    _loss, x0_hat3, many = run.generator_update(z, text, 1)
    assert torch.equal(x0_hat3, x0_hat) and -1.0 <= many["draw_cos"] <= 1.0 and many["match"] < 1e-5
    torch.manual_seed(5)
    seed = int(torch.randint(0, 2**31 - 1, (1,)))
    mean = sum(run.dmd_direction(x0_hat, 1, text, seed)[0] for _ in range(3)) / 3
    assert abs(float(mean.pow(2).mean().sqrt()) - many["grad_rms"]) < 1e-6
    assert float(head.x0_proj.weight.grad.abs().sum()) > 0 and fake_head.fake_r_embed.grad is None


# ------------------------------------------------- Neighbor Forcing + ConvKV (arXiv 2603.11746)

def _nf_stock(dit, z, t, text, first, audio):
    """The stock ``dit(...)`` call on H3's own T2VA / FL2VA-first layout."""
    from h3_musubi import audio_frames, audio_sigma
    from musubi_tuner.minimax_h3.packing import build_h3_layout

    frames, height, width = z.shape[2:]
    kwargs = dict(text_length=text.shape[1], target_video=(frames, height, width), target_audio_frames=audio_frames(frames))
    if first is None:
        layout = build_h3_layout(task="t2va", **kwargs)
    else:
        layout = build_h3_layout(task="fl2va", visual_conditions=[(1, height, width)], condition_roles=("first",), **kwargs)
    tags = torch.ones(1, text.shape[1], dtype=torch.int64)
    out = dit(video_latents=z, audio_latents=audio, text_hidden_states=text, text_token_tags=tags, layout=layout,
              model_t_video=t, model_t_audio=1.0 - float(audio_sigma(1.0 - t)),
              visual_condition_latents=[] if first is None else [first])
    return out.video, layout


def test_nf_blocks_and_geometry():
    """Block sizes (Table 5), the 2 + 2 + 2 context, audio spans, and positions / timestep rows equal to H3's own."""
    if _musubi() is None:
        print("SKIPPED test_nf_blocks_and_geometry: musubi_tuner not importable", flush=True)
        return
    from h3_musubi import audio_frames, pixel_frames
    from musubi_tuner.minimax_h3.packing import (FRAME_RESCALE, build_h3_layout, build_position_grid,
                                                 build_timestep_rows)
    from nf_h3 import DENSE, Geometry, NFConfig, few_step_sigmas, timestep_table

    config = NFConfig()
    assert (config.first_block, config.next_block) == (6, 8)
    assert config.spans(22) == [(0, 6), (6, 14), (14, 22)] and config.frames(3) == 22 and DENSE.spans(22) == [(0, 22)]
    for bad in (21, 5):
        try:
            config.spans(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    memory = NFConfig(memory=True)
    assert memory.context_slots == 6 and (memory.ref, memory.long, memory.short, memory.ratio) == (2, 2, 2, 5)
    for kwargs in (dict(first_block=7), dict(next_block=10), dict(causal=False)):
        try:
            NFConfig(memory=True, **kwargs)       # 6 / 8 is the only pair the 2 + 2 + 2 window with ratio 5 allows
        except ValueError:
            continue
        raise AssertionError(kwargs)
    # the release's 1000, 937.5, 833.3 are (1, .75, .5) under Wan's shift 5; H3's shift is 12
    wan = few_step_sigmas(shift=5.0)
    assert [round(1000 * s, 1) for s in wan] == [1000.0, 937.5, 833.3, 0.0]
    assert [round(s, 4) for s in few_step_sigmas()] == [1.0, 0.973, 0.9231, 0.0]
    # audio frames are split by time and tile the clip's audio
    for total in (22, 30, 38):
        spans = [Geometry.audio_span(a, b, total) for a, b in config.spans(total)]
        assert spans[0][0] == 0 and spans[-1][1] == audio_frames(total)
        assert all(left[1] == right[0] for left, right in zip(spans[:-1], spans[1:]))
        for (start, _stop), (a0, _a1) in zip(config.spans(total), spans):
            assert a0 - 1 < FRAME_RESCALE * pixel_frames(start) <= a0     # first audio frame at or after the block
    assert Geometry.audio_span(6, 14) == Geometry.audio_span(6, 14, 22)   # a stream needs no clip length
    for first in (False, True):
        geometry = Geometry(5, 4, 6, first)
        kwargs = dict(text_length=5, target_video=(22, 4, 6), target_audio_frames=audio_frames(22))
        layout = (build_h3_layout(task="fl2va", visual_conditions=[(1, 4, 6)], condition_roles=("first",), **kwargs)
                  if first else build_h3_layout(task="t2va", **kwargs))
        mine = torch.cat([geometry.prefix_positions(), geometry.audio_positions(0, audio_frames(22)),
                          geometry.video_positions(0, 22)])
        assert torch.equal(mine, build_position_grid(layout)[0])
        for t in (0.0, 0.4, 0.9995):
            t_audio = 0.37 if t else 0.0
            rows = build_timestep_rows(layout, text_token_tags=torch.ones(1, 5, dtype=torch.int64), model_t_video=t,
                                       model_t_audio=t_audio)
            unique, codes, video_index, audio_index = timestep_table(t, t_audio, first)
            assert torch.equal(unique, rows.unique_timesteps)
            assert (video_index, audio_index) == (rows.video_timestep_index, rows.audio_timestep_index)
            for role, name in (("text", "text"), ("target_audio", "audio"), ("target_video", "video")) + (
                    (("first", "first"),) if first else ()):
                assert rows.block_adaln_indices[0][layout.segment(role).row_slice].eq(codes[name]).all(), (role, t)
    # RoPE reset: from block 2 on, the first short-term frame sits at slot 4 of the first window, the
    # reference and long-term slots at slots 0-3, and the window never moves again.
    geometry = Geometry(5, 4, 6)
    slot = [geometry.frame_time(k) for k in range(6)]
    assert geometry.window_shift(memory, 0) == geometry.window_shift(memory, 1) == 0.0
    assert geometry.context_positions(memory, 0) is None
    assert torch.equal(geometry.context_positions(memory, 1), geometry.video_positions(0, 6))
    for index in (2, 3, 7, 40):
        start, stop = memory.span(index)
        times = geometry.context_positions(memory, index)[::geometry.frame_rows, 0] - 5
        assert torch.allclose(times[:5], torch.tensor(slot[:5], dtype=torch.float64), atol=1e-9)
        own = geometry.video_positions(start, stop, geometry.window_shift(memory, index))[::geometry.frame_rows, 0] - 5
        real = torch.tensor([geometry.frame_time(f) for f in range(start - 2, stop)], dtype=torch.float64)
        got = torch.cat([times[4:], own])
        assert torch.allclose(got - got[0], real - real[0], atol=1e-9)      # true H3 spacing kept
        assert float(own.max()) < 80.0                                     # bounded for any stream length
    assert torch.equal(geometry.context_positions(NFConfig(), 3), geometry.video_positions(0, 22))


def test_nf_dense_is_stock_h3():
    """``nf_forward`` with every row visible is the stock forward, T2VA and FL2VA-first (the base model)."""
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_nf_dense_is_stock_h3: musubi_tuner not importable", flush=True)
        return
    from h3_musubi import _silence
    from nf_h3 import nf_forward

    z, text = torch.randn(1, 24, 22, 4, 6), torch.randn(1, 5, 16)
    audio = _silence(22, 0.7, "cpu", torch.Generator().manual_seed(3))
    for first in (None, torch.randn(1, 24, 1, 4, 6)):
        with torch.no_grad():
            stock, _layout = _nf_stock(dit, z, 0.4, text, first, audio)
            got = nf_forward(dit, z, 0.4, text, first=first, audio=audio)
        assert float((got - stock).abs().max()) < 1e-6, first is None


def test_nf_causal_is_eq5_mask():
    """The per-block attention equals one attention under the explicit eq. 5 mask, and nothing leaks backwards."""
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_nf_causal_is_eq5_mask: musubi_tuner not importable", flush=True)
        return
    import torch.nn.functional as F
    from h3_musubi import _silence, audio_frames
    from musubi_tuner.minimax_h3 import model as h3_model
    from nf_h3 import Geometry, NFConfig, nf_forward

    config = NFConfig()
    z, text = torch.randn(1, 24, 22, 4, 6), torch.randn(1, 5, 16)
    audio = _silence(22, 0.7, "cpu", torch.Generator().manual_seed(3))
    total = audio_frames(22)
    for first in (None, torch.randn(1, 24, 1, 4, 6)):
        geometry = Geometry(5, 4, 6, first is not None)
        # block of every row in stock order text | first | audio (low, high) | video; -1 is the prefix
        audio_block = torch.empty(total, dtype=torch.int64)
        video_block = torch.empty(22, dtype=torch.int64)
        for index, (start, stop) in enumerate(config.spans(22)):
            a0, a1 = Geometry.audio_span(start, stop, 22)
            audio_block[a0:a1] = index
            video_block[start:stop] = index                 # eq. 5 with a first block of 6: floor((f + 2) / 8)
            assert all((f + 2) // 8 == index for f in range(start, stop))
        block = torch.cat([torch.full((geometry.prefix_rows,), -1), audio_block, audio_block,
                           video_block.repeat_interleave(geometry.frame_rows)])
        is_video = torch.zeros_like(block, dtype=torch.bool)
        is_video[geometry.prefix_rows + 2 * total:] = True
        i, j = block[:, None], block[None, :]
        mask = (j == -1) | ((i >= 0) & ((j == i) | ((j < i) & (j >= 0) & is_video[None, :])))

        def masked(attn):
            def forward(hidden, rotation=None):
                batch, rows, _ = hidden.shape
                q, k, v = attn.qkv_proj(hidden).split(attn.inner_dim, dim=-1)
                q = attn.q_norm(q.reshape(batch, rows, attn.num_heads, attn.head_dim))
                k = attn.k_norm(k.reshape(batch, rows, attn.num_heads, attn.head_dim))
                v = v.reshape(batch, rows, attn.num_heads, attn.head_dim)
                q, k = h3_model._apply_rope_split_half(q, rotation), h3_model._apply_rope_split_half(k, rotation)
                out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
                return attn.out_proj(out.transpose(1, 2).reshape(batch, rows, -1))
            return forward

        with torch.no_grad():
            got = nf_forward(dit, z, 0.4, text, config=config, first=first, audio=audio)
            dense = nf_forward(dit, z, 0.4, text, first=first, audio=audio)
            for layer in dit.blocks:
                layer.attn.forward = masked(layer.attn)
            try:
                reference, _layout = _nf_stock(dit, z, 0.4, text, first, audio)
            finally:
                for layer in dit.blocks:
                    del layer.attn.forward
            assert float((got - reference).abs().max()) < 1e-5, first is None
            assert float((got - dense).abs().max()) > 1e-2                 # the mask matters
            # perturb block n: every earlier block is unchanged to the bit, the block itself moves
            for index, (start, stop) in enumerate(config.spans(22)):
                moved = z.clone()
                moved[:, :, start:stop] += torch.randn(1, 24, stop - start, 4, 6)
                out = nf_forward(dit, moved, 0.4, text, config=config, first=first, audio=audio)
                assert torch.equal(out[:, :, :start], got[:, :, :start]), index
                assert float((out[:, :, start:stop] - got[:, :, start:stop]).abs().max()) > 1e-3
            # a later block's audio does not reach an earlier block either
            a0, _a1 = Geometry.audio_span(14, 22, 22)
            loud = audio.clone()
            loud[..., a0:] += 1.0
            out = nf_forward(dit, z, 0.4, text, config=config, first=first, audio=loud)
            assert torch.equal(out[:, :, :14], got[:, :, :14]) and not torch.equal(out[:, :, 14:], got[:, :, 14:])
            # the prefix reads only itself: text changes everything, but video cannot change what text does
            other = nf_forward(dit, z, 0.4, torch.randn(1, 5, 16), config=config, first=first, audio=audio)
            assert float((other[:, :, :6] - got[:, :, :6]).abs().max()) > 1e-3


def test_nf_convkv_memory():
    """Eq. 6-7: kernel = stride = 5 over time, mean init, one conv for K and V; the 2 + 2 + 2 context of fig. 3."""
    from nf_h3 import ConvKV, ConvKVLayer, LayerState, NFConfig

    layer = ConvKVLayer(8, ratio=5)
    conv = layer.memory_proj_k
    assert conv.kernel_size == (5,) and conv.stride == (5,) and conv.bias is None and layer.memory_proj_v is None
    assert torch.allclose(conv.weight, torch.full_like(conv.weight, 0.2))
    assert ConvKVLayer(8, shared=False).memory_proj_v is not None
    states = torch.randn(1, 10 * 3, 2, 4)                    # 10 frames x 3 positions, 2 heads x 4
    out = layer.compress(states, 3, "k")
    frames = states.reshape(1, 10, 3, 2, 4)
    assert out.shape == (1, 2 * 3, 2, 4)
    assert torch.allclose(out.reshape(1, 2, 3, 2, 4), frames.reshape(1, 2, 5, 3, 2, 4).mean(2), atol=1e-6)   # same position, 5 frames
    constant = frames[:, :1].expand(-1, 10, -1, -1, -1).reshape(1, 30, 2, 4)
    assert torch.allclose(layer.compress(constant, 3, "v"), constant[:, :6], atol=1e-6)       # identity on a still
    with torch.no_grad():
        conv.weight[3, 0, 4] = 2.0                           # one channel, last tap: the conv is real, not a mean
    assert not torch.allclose(layer.compress(states, 3, "k"), out)
    assert torch.equal(layer.compress(states, 3, "k"), layer.compress(states, 3, "v"))        # shared theta
    try:
        layer.compress(states[:, :9 * 3], 3, "k")
    except ValueError:
        pass
    else:
        raise AssertionError("9 frames accepted")

    # the fold, with frame f's key filled with the value f: block sizes 6, 8, 8, 8
    config = NFConfig(memory=True)
    state = LayerState(config, frame_rows=3)
    mean = ConvKVLayer(8)

    def kv(start, stop):
        k = torch.arange(start, stop, dtype=torch.float32).repeat_interleave(3).reshape(1, -1, 1, 1).expand(-1, -1, 2, 4)
        return k.clone(), -k.clone()

    def frames_of(context):
        return context[0][0, ::3, 0, 0].tolist()

    assert state.context(0, mean) is None
    state.push(0, *kv(0, 6))
    assert frames_of(state.context(1, mean)) == [0, 1, 2, 3, 4, 5]
    state.push(1, *kv(6, 14))
    context = state.context(2, mean)                         # ref 0, 1 | conv(2..6), conv(7..11) | short 12, 13
    assert frames_of(context) == [0, 1, 4, 9, 12, 13] and torch.equal(context[1], -context[0])
    state.push(2, *kv(14, 22))
    # the release's recursion: L0' = conv(L0, L1, S0, S1, C0), L1' = conv(C1..C5), S' = (C6, C7)
    expected = [0, 1, (4 + 9 + 12 + 13 + 14) / 5, 17, 20, 21]
    assert torch.allclose(torch.tensor(frames_of(state.context(3, mean))), torch.tensor(expected))
    state.push(3, *kv(22, 30))
    assert state.window[0].shape[1] == 12 * 3 and state.ref[0].shape[1] == 2 * 3 and state.frames == 30
    plain = LayerState(NFConfig(), frame_rows=3)             # Stage 1: every earlier frame, uncompressed
    for index, (start, stop) in enumerate(NFConfig().spans(22)):
        context = plain.context(index, None)
        assert (context is None) if index == 0 else frames_of(context) == list(range(start))
        plain.push(index, *kv(start, stop))
    try:
        LayerState(config, 3, window=kv(0, 12), ref=kv(0, 2)).context(2, None)
    except ValueError:
        pass
    else:
        raise AssertionError("memory without a ConvKV module accepted")

    # through the model (one layer, so a row's key depends on its own frame only): a frame in the
    # compressed range reaches block 2 through the convolution and through nothing else
    dit = _tiny_h3(num_layers=1)
    if dit is None:
        print("SKIPPED test_nf_convkv_memory (model part): musubi_tuner not importable", flush=True)
        return
    from h3_musubi import _silence
    from nf_h3 import nf_forward

    convkv = ConvKV.for_dit(dit)
    assert len(convkv.layers) == 1 and convkv.layers[0].memory_proj_k.weight.shape == (64, 1, 5)
    z, text = torch.randn(1, 24, 22, 4, 6), torch.randn(1, 5, 16)
    audio = _silence(22, 0.7, "cpu", torch.Generator().manual_seed(3))
    with torch.no_grad():
        base = nf_forward(dit, z, 0.4, text, config=config, convkv=convkv, audio=audio)
        stage1 = nf_forward(dit, z, 0.4, text, config=NFConfig(), audio=audio)
        assert torch.equal(base[:, :, :14], stage1[:, :, :14])            # blocks 0 and 1 are not compressed
        assert float((base[:, :, 14:] - stage1[:, :, 14:]).abs().max()) > 1e-3
        for frame, visible in ((0, True), (3, True), (7, True), (11, True), (12, True)):
            moved = z.clone()
            moved[:, :, frame] += torch.randn(24, 4, 6)
            out = nf_forward(dit, moved, 0.4, text, config=config, convkv=convkv, audio=audio)
            assert (float((out[:, :, 14:] - base[:, :, 14:]).abs().max()) > 1e-4) == visible, frame
        zero = ConvKV.for_dit(dit)
        for param in zero.parameters():
            param.zero_()
        base0 = nf_forward(dit, z, 0.4, text, config=config, convkv=zero, audio=audio)
        for frame, visible in ((0, True), (1, True), (3, False), (7, False), (11, False), (12, True), (13, True)):
            moved = z.clone()
            moved[:, :, frame] += torch.randn(24, 4, 6)
            out = nf_forward(dit, moved, 0.4, text, config=config, convkv=zero, audio=audio)
            assert (float((out[:, :, 14:] - base0[:, :, 14:]).abs().max()) > 1e-4) == visible, frame
    try:
        nf_forward(dit, z, 0.4, text, config=config, audio=audio)
    except ValueError:
        pass
    else:
        raise AssertionError("memory forward without ConvKV accepted")


def test_nf_stream_equals_parallel():
    """Fig. 3 right: block-by-block inference with per-step KV caches equals the one-pass masked forward and the
    step-outer sampler, with and without ConvKV (4 blocks: the recursive fold runs), and the memory cache is bounded."""
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_nf_stream_equals_parallel: musubi_tuner not importable", flush=True)
        return
    from h3_musubi import audio_sigma
    from nf_h3 import (ConvKV, Geometry, NFConfig, NFStream, NoiseBank, few_step_sigmas, nf_forward,
                       nf_parallel_sample, nf_stream_sample, parallel_noise)

    text = torch.randn(1, 5, 16)
    convkv = ConvKV.for_dit(dit)
    with torch.no_grad():
        for layer in convkv.layers:
            layer.memory_proj_k.weight.normal_(0.2, 0.2)     # a trained-looking conv, not the mean
    sigmas = few_step_sigmas()
    frames = NFConfig().frames(4)                            # 30 latent frames
    for memory in (False, True):
        config = NFConfig(memory=memory)
        for first in (None, torch.randn(1, 24, 1, 4, 6)):
            noise = NoiseBank(11, "cpu")
            kwargs = dict(lat_h=4, lat_w=6, config=config, sigmas=sigmas, noise=noise, first=first,
                          convkv=convkv if memory else None)
            # one step: the parallel pass against the stream, block by block
            sigma = sigmas[1]
            x, unit = parallel_noise(noise, config, frames, 1, 4, 6)
            audio = float(audio_sigma(sigma)) * unit
            with torch.no_grad():
                whole = nf_forward(dit, x, 1.0 - sigma, text, config=config, convkv=kwargs["convkv"], first=first, audio=audio)
            stream = NFStream(dit, lat_h=4, lat_w=6, steps=3, config=config, convkv=kwargs["convkv"], first=first,
                              total_frames=frames)
            sizes = []
            for index, (start, stop) in enumerate(config.spans(frames)):
                a0, a1 = Geometry.audio_span(start, stop, frames)
                got = stream.velocity(index, 1, x[:, :, start:stop], 1.0 - sigma, text, audio[..., a0:a1])
                assert float((got - whole[:, :, start:stop]).abs().max()) < 1e-5, (memory, first is None, index)
                sizes.append(stream.cache_rows(1))
            prefix = 5 + (6 if first is not None else 0)
            if memory:      # prefix + 2 reference frames + the 12-slot window, however long the stream
                assert sizes[1:] == [prefix + 14 * 6] * 3, sizes
            else:           # Stage 1 keeps everything
                assert sizes == [prefix + stop * 6 for _start, stop in config.spans(frames)], sizes
            # the whole sampler, both update rules
            for sampler in ("renoise", "euler"):
                with torch.no_grad():
                    parallel = nf_parallel_sample(dit, text, frames, sampler=sampler, **kwargs)
                streamed = nf_stream_sample(dit, text, 4, sampler=sampler, total_frames=frames, **kwargs)
                assert streamed.shape == (1, 24, frames, 4, 6) and bool(torch.isfinite(streamed).all())
                assert float((streamed - parallel).abs().max()) < 1e-4, (memory, first is None, sampler)
            state, audio_s = nf_parallel_sample(dit, text, frames, stop_step=0, **kwargs)
            assert torch.equal(state, parallel_noise(noise, config, frames, 0, 4, 6)[0]) and audio_s.shape[-1] > 0
    # driving audio (clean audio latent rows at the clean-condition time): same equality, and block n hears
    # only its own time span, so audio after block 1 cannot change blocks 0 and 1
    from h3_musubi import audio_frames
    config = NFConfig(memory=True)
    driving = torch.randn(1, 32, 2, audio_frames(frames))
    kwargs = dict(lat_h=4, lat_w=6, config=config, sigmas=sigmas, noise=NoiseBank(5, "cpu"), convkv=convkv, audio=driving)
    with torch.no_grad():
        parallel = nf_parallel_sample(dit, text, frames, **kwargs)
    streamed = nf_stream_sample(dit, text, 4, total_frames=frames, **kwargs)
    assert float((streamed - parallel).abs().max()) < 1e-4
    silent = nf_stream_sample(dit, text, 4, total_frames=frames, **{**kwargs, "audio": None})
    assert float((streamed - silent).abs().max()) > 1e-3
    later = driving.clone()
    later[..., Geometry.audio_span(14, 22, frames)[0]:] += 1.0
    moved = nf_stream_sample(dit, text, 4, total_frames=frames, **{**kwargs, "audio": later})
    assert torch.equal(moved[:, :, :14], streamed[:, :, :14]) and not torch.equal(moved[:, :, 14:], streamed[:, :, 14:])
    try:
        nf_stream_sample(dit, text, 4, total_frames=frames, **{**kwargs, "audio": driving[..., :50]})
    except ValueError:
        pass
    else:
        raise AssertionError("short driving audio accepted")
    # prompt switching (the release's emotion/action editing): a new prompt from block 2 on changes only blocks 2+
    config = NFConfig(memory=True)
    other = torch.randn(1, 5, 16)
    kwargs = dict(lat_h=4, lat_w=6, config=config, sigmas=sigmas, noise=NoiseBank(11, "cpu"), convkv=convkv)
    same = nf_stream_sample(dit, text, 3, **kwargs)
    edited = nf_stream_sample(dit, lambda index: text if index < 2 else other, 3, **kwargs)
    assert torch.equal(edited[:, :, :14], same[:, :, :14]) and float((edited[:, :, 14:] - same[:, :, 14:]).abs().max()) > 1e-3


def test_nf_trainer_updates_on_cpu():
    """Stage 1 (eq. 4) and Stage 2 (DMD through the ConvKV rollout) updates on the toy H3 with real LoRAs:
    each side's gradient reaches only its own parameters, and ConvKV is trained by the generator update."""
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_nf_trainer_updates_on_cpu: musubi_tuner not importable", flush=True)
        return
    import argparse

    from h3_musubi import make_lora
    from nf_h3 import DENSE, ConvKV, NFConfig, nf_forward
    from train_nf_h3 import NFRun, grad_sum, zero_grads

    gen_net, fake_net = make_lora(dit, 2), make_lora(dit, 2)
    for net in (gen_net, fake_net):
        net.requires_grad_(True)
    gen_params = [p for p in gen_net.parameters()]
    fake_params = [p for p in fake_net.parameters()]
    x0, text = torch.randn(1, 24, 22, 4, 6), torch.randn(1, 5, 16)
    args = argparse.Namespace(stage=1, blocks=[6, 8], ref="first", sigmas="1,0.75,0.5", sampler_shift=12.0,
                              sampler="renoise", sigma_shift=1.0, tau_shift=1.0, audio="silent")
    run = NFRun(dit, gen_net, None, None, args, "cpu")
    eps = torch.randn_like(x0)
    loss, stats = run.stage1_loss(x0, text, sigma=0.6, eps=eps, audio_seed=3, backward=True)
    # eq. 2 and eq. 4 by hand; the LoRA starts at zero, so this is the base model under the causal mask
    x_t = 0.4 * x0 + 0.6 * eps
    with torch.no_grad():
        by_hand = nf_forward(dit, x_t, 0.4, text, config=NFConfig(), first=x0[:, :, :1],
                             audio_generator=torch.Generator().manual_seed(3))
    assert abs(float(loss) - float((by_hand - (x0 - eps)).pow(2).mean())) < 1e-5 and stats["sigma"] == 0.6
    assert grad_sum(gen_params) > 0
    dense, _ = run.stage1_loss(x0, text, sigma=0.6, eps=eps, audio_seed=3, config=DENSE)
    assert abs(float(dense) - float(loss)) > 1e-4
    assert all(0.0 < run.draw_sigma() < 1.0 and 0.02 <= run.draw_tau() <= 0.98 for _ in range(50))

    args.stage = 2
    convkv = ConvKV.for_dit(dit)
    with torch.no_grad():
        for module in fake_net.unet_loras:
            module.lora_up.weight.normal_(0.0, 0.05)        # fake != real, so the DMD direction is non-zero
    run = NFRun(dit, gen_net, fake_net, convkv, args, "cpu")
    assert run.memory.memory and [round(s, 3) for s in run.sigmas] == [1.0, 0.973, 0.923, 0.0]
    from h3_musubi import audio_frames
    driving = torch.randn(1, 32, 2, audio_frames(22))          # a clean audio latent as conditioning (c_audio)
    with torch.no_grad():
        silent, _ = run.stage1_loss(x0, text, sigma=0.6, eps=eps, audio_seed=3, config=NFConfig())
        driven, _ = run.stage1_loss(x0, text, sigma=0.6, eps=eps, audio=driving, config=NFConfig())
    assert math.isfinite(float(driven)) and abs(float(driven) - float(silent)) > 1e-5
    for exit_step, audio in ((0, None), (2, None), (1, driving)):
        zero_grads(gen_params + fake_params + list(convkv.parameters()))
        loss, sample, stats = run.generator_update(x0, text, exit_step=exit_step, audio=audio)
        assert math.isfinite(stats["loss"]) and stats["match"] < 1e-5 and stats["gap"] > 0, stats
        assert stats["exit_step"] == exit_step and sample[0].shape == x0.shape
        assert grad_sum(gen_params) > 0 and grad_sum(list(convkv.parameters())) > 0
        assert grad_sum(fake_params) == 0.0
        zero_grads(gen_params + list(convkv.parameters()))
        floss, tau = run.fake_update(sample)
        assert math.isfinite(float(floss)) and grad_sum(fake_params) > 0 and 0.02 <= tau <= 0.98
        assert grad_sum(gen_params) == 0.0 and grad_sum(list(convkv.parameters())) == 0.0


def test_nf_text_reads_first():
    """``text_reads_first``: one causal block is the dense forward; with more blocks the stream still equals the
    parallel pass (memory on and off), nothing leaks backwards, and the switch changes the output."""
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_nf_text_reads_first: musubi_tuner not importable", flush=True)
        return
    from h3_musubi import _silence, audio_sigma
    from nf_h3 import ConvKV, Geometry, NFConfig, NFStream, NoiseBank, few_step_sigmas, nf_forward, parallel_noise

    text = torch.randn(1, 5, 16)
    with torch.no_grad():
        for frames in (7, 12):
            z = torch.randn(1, 24, frames, 4, 6)
            audio = _silence(frames, 0.7, "cpu", torch.Generator().manual_seed(3))
            for first in (None, torch.randn(1, 24, 1, 4, 6)):
                dense = nf_forward(dit, z, 0.4, text, first=first, audio=audio)
                one = nf_forward(dit, z, 0.4, text, config=NFConfig(frames, 8, text_reads_first=True), first=first, audio=audio)
                cut = nf_forward(dit, z, 0.4, text, config=NFConfig(frames, 8), first=first, audio=audio)
                assert float((one - dense).abs().max()) < 1e-5, (frames, first is None)
                assert float((cut - dense).abs().max()) > 1e-3          # the isolated prefix is a different model
        convkv = ConvKV.for_dit(dit)
        for layer in convkv.layers:
            layer.memory_proj_k.weight.normal_(0.2, 0.2)
        frames, sigma = NFConfig().frames(4), few_step_sigmas()[1]
        for memory in (False, True):
            config = NFConfig(memory=memory, text_reads_first=True)
            conv = convkv if memory else None
            noise = NoiseBank(11, "cpu")
            x, unit = parallel_noise(noise, config, frames, 1, 4, 6)
            audio = float(audio_sigma(sigma)) * unit
            whole = nf_forward(dit, x, 1.0 - sigma, text, config=config, convkv=conv, audio=audio)
            off = nf_forward(dit, x, 1.0 - sigma, text, config=NFConfig(memory=memory), convkv=conv, audio=audio)
            assert float((whole - off).abs().max()) > 1e-3
            stream = NFStream(dit, lat_h=4, lat_w=6, steps=3, config=config, convkv=conv, total_frames=frames)
            for index, (start, stop) in enumerate(config.spans(frames)):
                a0, a1 = Geometry.audio_span(start, stop, frames)
                got = stream.velocity(index, 1, x[:, :, start:stop], 1.0 - sigma, text, audio[..., a0:a1])
                assert float((got - whole[:, :, start:stop]).abs().max()) < 1e-5, (memory, index)
            moved = x.clone()
            moved[:, :, 6:] += torch.randn_like(moved[:, :, 6:])
            out = nf_forward(dit, moved, 1.0 - sigma, text, config=config, convkv=conv, audio=audio)
            assert torch.equal(out[:, :, :6], whole[:, :, :6])          # later blocks never reach block 0 or the text


def test_nf_memory_independent_reference():
    """The reviewer's independent re-derivation of the ConvKV forward (stock layout and position grid, own
    index arithmetic, einsum instead of Conv1d, the release's recursive fold) against ``nf_forward``, 8 blocks."""
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_nf_memory_independent_reference: musubi_tuner not importable", flush=True)
        return
    import torch.nn.functional as F
    from h3_musubi import _silence, audio_frames, pixel_frames
    from musubi_tuner.minimax_h3 import model as h3_model
    from musubi_tuner.minimax_h3.packing import build_position_grid
    from nf_h3 import ConvKV, NFConfig, nf_forward

    torch.manual_seed(1)
    T, H, W, FR = 62, 4, 6, 6          # 6 + 8 * 7 = 5 * 12 + 2: the stock layout accepts it
    z, text = torch.randn(1, 24, T, H, W), torch.randn(1, 5, 16)
    audio = _silence(T, 0.7, "cpu", torch.Generator().manual_seed(3))
    A = audio_frames(T)
    convkv = ConvKV.for_dit(dit)
    with torch.no_grad():
        for layer in convkv.layers:
            layer.memory_proj_k.weight.normal_(0.2, 0.2)
    spans = [(0, 6)] + [(6 + 8 * i, 14 + 8 * i) for i in range(7)]
    for first in (None, torch.randn(1, 24, 1, H, W)):
        P = 5 + (FR if first is not None else 0)
        vb = P + 2 * A
        with torch.no_grad():
            _out, layout = _nf_stock(dit, z, 0.4, text, first, audio)
        pos = build_position_grid(layout)[0].double()
        ftime = [float(pos[vb + f * FR, 0] - pos[vb, 0]) for f in range(T)]

        def rot(x, p):
            return h3_model._apply_rope_split_half(x, dit._rotation_table(p.unsqueeze(0), x.dtype))

        def att(q, k, v):
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            return o.transpose(1, 2).reshape(1, q.shape[1], -1)

        def make(attn, w):
            shape = (attn.num_heads, attn.head_dim)

            def conv5(x):      # (5 frames * FR rows, heads, dim) frame-major -> (FR, heads, dim)
                return torch.einsum("fpc,cf->pc", x.reshape(5, FR, -1), w[:, 0, :]).reshape(FR, *shape)

            def vid(f0, f1):
                return slice(vb + f0 * FR, vb + f1 * FR)

            def frame_pos(f, shift=0.0):
                return torch.cat([pos[vid(f, f + 1), :1] - shift, pos[vid(f, f + 1), 1:]], dim=1)

            def forward(hidden, rotation=None):
                n = hidden.shape[1]
                q, k, v = attn.qkv_proj(hidden).split(attn.inner_dim, dim=-1)
                q = attn.q_norm(q.reshape(1, n, *shape))
                k = attn.k_norm(k.reshape(1, n, *shape))
                v = v.reshape(1, n, *shape)
                out = torch.zeros(1, n, attn.inner_dim)
                pk = rot(k[:, :P], pos[:P])
                out[:, :P] = att(rot(q[:, :P], pos[:P]), pk, v[:, :P])
                longs = None
                for b, (s, e) in enumerate(spans):
                    a0 = -(-5 * pixel_frames(s) // 3)
                    a1 = A if e >= T else -(-5 * pixel_frames(e) // 3)
                    rows = torch.cat([torch.arange(P + a0, P + a1), torch.arange(P + A + a0, P + A + a1),
                                      torch.arange(vb + s * FR, vb + e * FR)])
                    shift = 0.0 if b < 2 else ftime[s - 2] - ftime[4]
                    own = pos[rows].clone()
                    own[:, 0] -= shift
                    keys, vals = [pk], [v[:, :P]]
                    if b == 1:
                        keys.append(rot(k[:, vid(0, 6)], pos[vid(0, 6)]))
                        vals.append(v[:, vid(0, 6)])
                    if b >= 2:
                        if b == 2:      # raw frames 2..11 -> two slots
                            src_k, src_v = k[0, vid(2, 12)], v[0, vid(2, 12)]
                        else:           # [L0 L1 S0 S1 C0..C5] of the previous window
                            ps = spans[b - 1][0]
                            src_k = torch.cat([longs[0], longs[2], k[0, vid(ps - 2, ps + 6)]])
                            src_v = torch.cat([longs[1], longs[3], v[0, vid(ps - 2, ps + 6)]])
                        longs = (conv5(src_k[:5 * FR]), conv5(src_v[:5 * FR]), conv5(src_k[5 * FR:]), conv5(src_v[5 * FR:]))
                        ck = torch.cat([k[0, vid(0, 2)], longs[0], longs[2], k[0, vid(s - 2, s)]])[None]
                        cv = torch.cat([v[0, vid(0, 2)], longs[1], longs[3], v[0, vid(s - 2, s)]])[None]
                        cp = torch.cat([frame_pos(0), frame_pos(1), frame_pos(2), frame_pos(3),
                                        frame_pos(s - 2, shift), frame_pos(s - 1, shift)])
                        keys.append(rot(ck, cp))
                        vals.append(cv)
                    keys.append(rot(k[:, rows], own))
                    vals.append(v[:, rows])
                    out[:, rows] = att(rot(q[:, rows], own), torch.cat(keys, 1), torch.cat(vals, 1))
                return attn.out_proj(out)
            return forward

        config = NFConfig(memory=True)
        with torch.no_grad():
            got = nf_forward(dit, z, 0.4, text, config=config, convkv=convkv, first=first, audio=audio)
            for layer, conv in zip(dit.blocks, convkv.layers):
                layer.attn.forward = make(layer.attn, conv.memory_proj_k.weight)
            try:
                reference, _layout = _nf_stock(dit, z, 0.4, text, first, audio)
            finally:
                for layer in dit.blocks:
                    del layer.attn.forward
            assert float((got - reference).abs().max()) < 1e-4 * max(1.0, float(got.abs().mean())), first is None
            for s, e in spans[2::3]:        # no future leak in memory mode
                moved = z.clone()
                moved[:, :, s:e] += torch.randn(1, 24, e - s, H, W)
                out = nf_forward(dit, moved, 0.4, text, config=config, convkv=convkv, first=first, audio=audio)
                assert torch.equal(out[:, :, :s], got[:, :, :s])


def _dir_head(dit, iterations=4, seed=0):
    from dir_h3 import DirHead, probe_temb_dim

    return DirHead(dit.config.hidden_size, probe_temb_dim(dit), iterations).paper_init(torch.Generator().manual_seed(seed))


def test_nf_dir_hooks_and_samplers():
    """DiR under Neighbor Forcing (not in the paper): the hooked forward with every row visible is
    ``dir_h3``'s generator / fake forward; the streamed DiR sampler (one cache per iteration) equals the one-pass
    sampler with and without ConvKV and for both state laws; later blocks never reach earlier ones."""
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_nf_dir_hooks_and_samplers: musubi_tuner not importable", flush=True)
        return
    from dir_h3 import DirFakeHead, estimate_extra, fake_x0, generator_step, probe_temb_dim
    from h3_musubi import audio_frames
    from nf_h3 import (DENSE, ConvKV, Geometry, NFConfig, NoiseBank, dir_step, nf_dir_parallel_sample,
                       nf_dir_stream_sample, nf_forward, nf_stream_sample, NFStream)

    head = _dir_head(dit)
    text = torch.randn(1, 5, 16)
    z, estimate = torch.randn(1, 24, 22, 4, 6), torch.randn(1, 24, 22, 4, 6)
    sigmas = [1.0, 0.947, 0.857, 0.667]
    with torch.no_grad():
        # (a) dense + hooks == dir_h3
        for r, sigma in enumerate(sigmas):
            stock = generator_step(dit, head, z, estimate, r, text, sigma=sigma, audio_generator=torch.Generator().manual_seed(3))
            mine = dir_step(dit, head, z, estimate, r, text, sigma, config=DENSE, audio_generator=torch.Generator().manual_seed(3))
            assert float((mine - stock).abs().max()) < 1e-6, r
        fake_head = DirFakeHead(probe_temb_dim(dit), 4).paper_init(torch.Generator().manual_seed(1))
        stock = fake_x0(dit, fake_head, z, 0.6, 3, text, audio_generator=torch.Generator().manual_seed(3))
        mine = z + 0.6 * nf_forward(dit, z, 0.4, text, audio_generator=torch.Generator().manual_seed(3),
                                    video_temb_delta=fake_head.fake_r_embed[3])
        assert float((mine - stock).abs().max()) < 1e-6
        # the hooks do something, and only to the video rows' path
        plain = nf_forward(dit, z, 0.4, text, config=NFConfig(), audio_generator=torch.Generator().manual_seed(3))
        hooked = nf_forward(dit, z, 0.4, text, config=NFConfig(), audio_generator=torch.Generator().manual_seed(3),
                            video_extra=estimate_extra(head, estimate, None), video_temb_delta=head.r_embed[1])
        assert float((hooked - plain).abs().max()) > 1e-3
        # (c) with the hooks on, a later block's state and estimate cannot reach an earlier block
        config = NFConfig(memory=True)
        convkv = ConvKV.for_dit(dit)
        for layer in convkv.layers:
            layer.memory_proj_k.weight.normal_(0.2, 0.2)
        audio = torch.randn(1, 32, 2, audio_frames(22)) * 0.1
        base = dir_step(dit, head, z, estimate, 2, text, sigmas[2], config=config, convkv=convkv, audio=audio)
        for start, stop in config.spans(22)[1:]:
            z2, e2 = z.clone(), estimate.clone()
            z2[:, :, start:stop] += torch.randn(1, 24, stop - start, 4, 6)
            e2[:, :, start:stop] += torch.randn(1, 24, stop - start, 4, 6)
            out = dir_step(dit, head, z2, e2, 2, text, sigmas[2], config=config, convkv=convkv, audio=audio)
            assert torch.equal(out[:, :, :start], base[:, :, :start])
            assert float((out[:, :, start:stop] - base[:, :, start:stop]).abs().max()) > 1e-3
        e2 = estimate.clone()
        e2[:, :, 14:] += 1.0            # the estimate alone, through x0_proj
        out = dir_step(dit, head, z, e2, 2, text, sigmas[2], config=config, convkv=convkv, audio=audio)
        assert torch.equal(out[:, :, :14], base[:, :, :14]) and not torch.equal(out[:, :, 14:], base[:, :, 14:])
    # (b) stream == one-pass, 4 blocks
    frames = NFConfig().frames(4)
    for memory in (False, True):
        config = NFConfig(memory=memory)
        for mix_state in (False, True):
            kwargs = dict(lat_h=4, lat_w=6, config=config, sigmas=sigmas, noise=NoiseBank(3, "cpu"), mix_state=mix_state,
                          convkv=convkv if memory else None)
            with torch.no_grad():
                parallel = nf_dir_parallel_sample(dit, head, text, frames, **kwargs)
                state, estimate_in, _audio = nf_dir_parallel_sample(dit, head, text, frames, stop_iter=2, **kwargs)
            streamed = nf_dir_stream_sample(dit, head, text, 4, **kwargs)
            assert streamed.shape == (1, 24, frames, 4, 6) and bool(torch.isfinite(streamed).all())
            assert float((streamed - parallel).abs().max()) < 1e-4, (memory, mix_state)
            assert state.shape == estimate_in.shape == streamed.shape and float(estimate_in.abs().max()) > 0
        other = nf_dir_stream_sample(dit, head, text, 4, **{**kwargs, "mix_state": False})
        assert float((other - streamed).abs().max()) > 1e-4      # the state law matters
    # a stream of N blocks is a clip of known length: its audio is H3's for that length (2 and 5 blocks used to
    # get one audio frame too many)
    for blocks in (2, 3, 5):
        total = NFConfig().frames(blocks)
        stream = NFStream(dit, lat_h=4, lat_w=6, steps=1, config=NFConfig(), total_frames=total)
        assert stream.audio_span(blocks - 1)[1] == audio_frames(total)
        out = nf_stream_sample(dit, text, blocks, lat_h=4, lat_w=6, config=NFConfig(), sigmas=[1.0, 0.5, 0.0],
                               noise=NoiseBank(0, "cpu"))
        assert out.shape[2] == total
    assert Geometry.audio_span(6, 14)[1] == audio_frames(14) + 1


def test_nf_dir_trainer_on_cpu():
    """The DiR-generator Stage 2 update on the toy H3: gradient reaches the generator LoRA, the DirHead and
    ConvKV and not the fake side; the fake update the reverse; and the DMD target points toward the real model
    (this fails if the sign is flipped, which the loss value alone cannot show)."""
    dit = _tiny_h3()
    if dit is None:
        print("SKIPPED test_nf_dir_trainer_on_cpu: musubi_tuner not importable", flush=True)
        return
    import argparse

    import train_nf_h3
    from dir_h3 import DirFakeHead, probe_temb_dim
    from h3_musubi import make_lora
    from nf_h3 import ConvKV
    from train_nf_h3 import NFRun, dmd_target, grad_sum, zero_grads

    # the sign, on numbers: real says "up", fake says "down" -> the target is above the sample
    sample, real, fake = torch.zeros(4), torch.ones(4), -torch.ones(4)
    assert bool((dmd_target(sample, real, fake) > sample).all()) and bool((dmd_target(sample, fake, real) < sample).all())
    assert torch.allclose(dmd_target(sample, real, fake), torch.full((4,), 2.0))      # (real - fake) / mean|x - real|

    gen_net, fake_net = make_lora(dit, 2), make_lora(dit, 2)
    for net in (gen_net, fake_net):
        net.requires_grad_(True)
    with torch.no_grad():
        for module in gen_net.unet_loras + fake_net.unet_loras:
            module.lora_up.weight.normal_(0.0, 0.05)        # a trained generator and a fake that differs from the base
    head = _dir_head(dit)
    fake_head = DirFakeHead(probe_temb_dim(dit), 4).paper_init(torch.Generator().manual_seed(1))
    convkv = ConvKV.for_dit(dit)
    lora_params, fake_params = list(gen_net.parameters()), list(fake_net.parameters()) + list(fake_head.parameters())
    head_params, conv_params = [p for p in head.parameters() if p.requires_grad], list(convkv.parameters())
    x0, text = torch.randn(1, 24, 22, 4, 6), torch.randn(1, 5, 16)
    recorded = []
    original = train_nf_h3.dmd_target

    def recording(x0_hat, real, fake):
        target = original(x0_hat, real, fake)
        recorded.append((x0_hat, real, fake, target))
        return target

    train_nf_h3.dmd_target = recording
    try:
        for fake_source, mix_state, ref in (("final", False, "none"), ("iteration", True, "first")):
            args = argparse.Namespace(stage=2, blocks=[6, 8], ref=ref, sigmas="1,0.75,0.5", sampler_shift=12.0,
                                      sampler="renoise", sigma_shift=1.0, tau_shift="beta:2.5", audio="silent")
            meta = {"sigmas": [1.0, 0.947, 0.857, 0.667], "mix_state": mix_state, "fake_source": fake_source,
                    "tau_max": 0.9}
            args.dmd_draws = 2 if mix_state else 1
            run = NFRun(dit, gen_net, fake_net, convkv, args, "cpu", head=head, fake_head=fake_head, dir_meta=meta)
            assert all(0.02 <= run.draw_tau() <= 0.98 for _ in range(20))
            assert all(run.draw_tau(0.9) <= 0.9 for _ in range(50)) and run.tau_max == 0.9
            for r in (0, 3):
                zero_grads(lora_params + fake_params + head_params + conv_params)
                loss, sample, stats = run.dir_generator_update(x0, text, r)
                assert math.isfinite(stats["loss"]) and stats["match"] < 1e-5 and stats["gap"] > 0 and stats["r"] == r, stats
                assert grad_sum(lora_params) > 0 and grad_sum(conv_params) > 0 and grad_sum(fake_params) == 0.0
                assert grad_sum(head_params) > 0
                x0_hat, real, fake, target = recorded[-1]
                assert float(((target - x0_hat) * (real - fake)).sum()) > 0          # toward real, away from fake
                if args.dmd_draws == 1:
                    assert abs(stats["loss"] - 0.5 * float((x0_hat - target).pow(2).mean())) < 1e-6
                zero_grads(lora_params + head_params + conv_params)
                floss, tau = run.dir_fake_update(sample, fresh=1 if r else 0)
                assert math.isfinite(float(floss)) and grad_sum(fake_params) > 0 and 0.02 <= tau <= 0.98
                assert grad_sum(lora_params) == 0.0 and grad_sum(head_params) == 0.0 and grad_sum(conv_params) == 0.0
            result = run.dir_eval([], 22)
            assert result["real_gap"] == 0 and len(result["real_gap_blocks"]) == 3
        # the paper-form Stage 2 update goes through the same target
        count = len(recorded)
        args.ref = "first"
        plain = NFRun(dit, gen_net, fake_net, convkv, args, "cpu")
        plain.generator_update(x0, text, exit_step=2)
        x0_hat, real, fake, target = recorded[-1]
        assert len(recorded) == count + 1 and float(((target - x0_hat) * (real - fake)).sum()) > 0
    finally:
        train_nf_h3.dmd_target = original


def test_base_lokr_hook():
    """base_lokr: the hook adds strength * x @ kron(w1, w2).T to the layer's output and removes cleanly."""
    import tempfile

    from safetensors.torch import save_file

    from base_lokr import apply_base_lokr, kron_apply

    g = torch.Generator().manual_seed(0)
    x = torch.randn(5, 3, 24, generator=g)
    w1, w2 = torch.randn(4, 3, generator=g), torch.randn(5, 8, generator=g)
    assert torch.allclose(kron_apply(x, w1, w2), x @ torch.kron(w1, w2).T, atol=1e-4)
    root = torch.nn.Module()
    root.blocks = torch.nn.ModuleList([torch.nn.Module()])
    root.blocks[0].fc = torch.nn.Linear(24, 20)
    plain = root.blocks[0].fc(x)
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/iso.safetensors"
        save_file({"diffusion_model.blocks.0.fc.lokr_w1": w1, "diffusion_model.blocks.0.fc.lokr_w2": w2,
                   "diffusion_model.blocks.0.fc.alpha": torch.tensor(1.0)}, path)
        handle = apply_base_lokr(root, path + ":0.6", dtype=torch.float32)
        assert torch.allclose(root.blocks[0].fc(x), plain + 0.6 * x @ torch.kron(w1, w2).T, atol=1e-4)
        handle.remove()
        assert torch.equal(root.blocks[0].fc(x), plain)
        save_file({"diffusion_model.blocks.9.fc.lokr_w1": w1, "diffusion_model.blocks.9.fc.lokr_w2": w2}, path)
        try:
            apply_base_lokr(root, path)
            raise AssertionError("a LoKr layer with no module was accepted")
        except ValueError:
            pass


def main():
    tests = [
        test_shape_and_centers,
        test_partition_rules,
        test_regions_prefer_finer_level,
        test_blur_and_vrs,
        test_procrustes_and_mean_basis,
        test_velocity_recovery_matches_equation_9,
        test_unit_layout_reproduces_pretrained_head,
        test_extent_scale_and_backward,
        test_h3_patch_geometry,
        test_fit_extent_rebuilds_heads,
        test_h3_positions_match_base_grid,
        test_sanity_short_sequence_writes_nothing,
        test_vectorized_gather_and_scale,
        test_gate_band_and_sigma,
        test_final_layer_sees_modulated_states,
        test_splice_shortens,
        test_euler_inference_recovers_clean,
        test_h3_head_sign_recovery,
        test_splice_y_space_contract,
        test_procrustes_h3_fit_and_guards,
        test_phase4_smoke_canvas,
        test_pair_rows_align_with_coarse_grid,
        test_h3_loss_is_velocity_mse_in_y_space,
        test_train_layouts_tile_and_mix,
        test_train_shift_moves_mass_to_low_sigma,
        test_torch_eq9_matches_lakonlab,
        test_grid_layouts_any_shape,
        test_layout_sources,
        test_dir_pool_rows_and_head,
        test_dir_layout_schedule,
        test_dir_video_modulation_rewrite,
        test_dir_use_networks_restores,
        test_dir_ema_warmup,
        test_dir_paper_init,
        test_dir_sigma_schedule_is_euler,
        test_dir_trainer_draws_on_cpu,
        test_stream_layout_and_positions_match_stock,
        test_stream_frame_modulation_rewrite,
        test_stream_forward_matches_stock_h3,
        test_stream_per_frame_r_and_memory_reach_the_output,
        test_stream_slide_is_exact_and_bounded,
        test_stream_head_warm_start,
        test_stream_trainer_updates_on_cpu,
        test_nf_blocks_and_geometry,
        test_nf_dense_is_stock_h3,
        test_nf_causal_is_eq5_mask,
        test_nf_convkv_memory,
        test_nf_stream_equals_parallel,
        test_nf_text_reads_first,
        test_nf_trainer_updates_on_cpu,
        test_nf_memory_independent_reference,
        test_nf_dir_hooks_and_samplers,
        test_nf_dir_trainer_on_cpu,
        test_base_lokr_hook,
    ]
    for test in tests:
        test()
        print(test.__name__)
    print(f"ok {len(tests)}")


if __name__ == "__main__":
    main()
