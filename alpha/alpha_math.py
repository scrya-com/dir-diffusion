# Adapted from AlphaAnything (https://github.com/WLuLi/AlphaAnything, Apache-2.0):
#   alphaanything/modules/alpha_math.py and alphaanything/data/rgba.py (layerdiffuse_pad_rgb).
# Copyright the AlphaAnything authors. Changes: trimmed to the functions used here; the padding works on
# (N, C, H, W) batches of any N; the OpenCV-area resize is replaced with torch's antialiased area resize.
# Licensed under the Apache License, Version 2.0. See alpha/NOTICE.
"""Endpoint-Aware alpha loss, its score <-> alpha mapping, the sigma weight, and LayerDiffuse colour padding."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _mean_per_sample(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1).mean(dim=1)


def _zero_safe_sqrt(value: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(value.dtype).tiny
    return torch.where(value > 0.0, torch.sqrt(value.clamp_min(tiny)), torch.zeros_like(value))


def alpha_to_ea_coordinate(alpha: torch.Tensor) -> torch.Tensor:
    """alpha in [0, 1] -> u = (2 / pi) asin(sqrt(alpha)) in [0, 1]."""
    alpha = alpha.float().clamp(0.0, 1.0)
    return torch.atan2(torch.sqrt(alpha), torch.sqrt(1.0 - alpha)) * (2.0 / math.pi)


def ea_coordinate_from_score(score: torch.Tensor) -> torch.Tensor:
    return ((score.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def ea_score_to_alpha(score: torch.Tensor) -> torch.Tensor:
    """Raw head score -> alpha = sin^2(pi u / 2)."""
    return torch.sin(ea_coordinate_from_score(score) * (math.pi / 2.0)).square()


def ea_pointwise_gap(score: torch.Tensor, target_coordinate: torch.Tensor) -> torch.Tensor:
    """Squared gap inside [-1, 1]; linear (Fenchel-Young) outside so the gradient never vanishes."""
    score = score.float()
    target_coordinate = target_coordinate.float().clamp(0.0, 1.0)
    active_gap = (ea_coordinate_from_score(score) - target_coordinate).square()
    lower_gap = target_coordinate * (target_coordinate - 1.0 - score)
    upper_gap = (1.0 - target_coordinate) * (score - target_coordinate)
    return torch.where(score <= -1.0, lower_gap, torch.where(score >= 1.0, upper_gap, active_gap))


def ea_loss_per_sample(score: torch.Tensor, target_alpha: torch.Tensor):
    """Endpoint-Aware alpha loss per sample: (total, transparent root, opaque root)."""
    target_alpha = target_alpha.float().clamp(0.0, 1.0)
    gap = ea_pointwise_gap(score, alpha_to_ea_coordinate(target_alpha))
    transparent_root = _zero_safe_sqrt(_mean_per_sample((1.0 - target_alpha) * gap))
    opaque_root = _zero_safe_sqrt(_mean_per_sample(target_alpha * gap))
    return transparent_root + opaque_root, transparent_root, opaque_root


def sigma_image_loss_weight(sigmas: torch.Tensor, reference_sigma: float = 0.5, epsilon: float = 1e-4) -> torch.Tensor:
    """w(sigma) = min(1, reference^2 / (sigma^2 + eps)): image-space losses count less at high noise."""
    s = sigmas.float().reshape(sigmas.shape[0], -1)[:, 0]
    return torch.minimum(torch.ones_like(s), reference_sigma ** 2 / (s.square() + epsilon))


def layerdiffuse_pad_rgb(rgb01: torch.Tensor, alpha01: torch.Tensor, downsample_factor: float = 1.2) -> torch.Tensor:
    """Fill colour under transparent pixels by pulling foreground colour outward through a premultiplied pyramid
    (LayerDiffuse). ``rgb01`` (N, 3, H, W), ``alpha01`` (N, 1, H, W), both in [0, 1]."""
    out_dtype = rgb01.dtype
    alpha = alpha01.float().clamp(0.0, 1.0)
    premul = rgb01.float().clamp(0.0, 1.0) * alpha
    pyramid = [(premul, alpha)]
    while min(premul.shape[-2:]) > 1:
        h, w = premul.shape[-2:]
        size = (max(1, int(h / downsample_factor)), max(1, int(w / downsample_factor)))
        both = F.interpolate(torch.cat((premul, alpha), 1), size=size, mode="area")
        premul, alpha = both[:, :3], both[:, 3:4]
        pyramid.append((premul, alpha))
    coarse_color, coarse_alpha = pyramid[-1]
    fg = coarse_color.sum((-2, -1), keepdim=True) / coarse_alpha.sum((-2, -1), keepdim=True).clamp_min(1e-8)
    for layer_color, layer_alpha in reversed(pyramid):
        fg = F.interpolate(fg, size=layer_color.shape[-2:], mode="bilinear", align_corners=False)
        fg = layer_color + fg * (1.0 - layer_alpha)
    return fg.clamp(0.0, 1.0).to(out_dtype)
