import torch


def flat_to_2d(flat: torch.Tensor, width: int) -> torch.Tensor:
    heights = torch.div(flat, width, rounding_mode="floor")
    widths = flat.remainder(width)
    return torch.stack([heights, widths], -1)


def patch_coords(centers: torch.Tensor, p_side: int, width: int) -> torch.Tensor:
    b, n, _ = centers.shape
    padding = p_side // 2
    padded_w = width + 2 * padding
    steps = torch.arange(p_side, dtype=centers.dtype, device=centers.device)
    dy = steps.view(-1, 1).expand(p_side, p_side)
    dx = steps.view(1, -1).expand(p_side, p_side)

    absolute_coords = centers.view(b, n, 1, 1, 2) + torch.stack([dy, dx], -1).view(1, 1, p_side, p_side, 2)
    return (absolute_coords[..., 0] * padded_w + absolute_coords[..., 1]).view(b, n * p_side * p_side)


def make_pictures(flat_coords: torch.Tensor, patches: torch.Tensor, scores: torch.Tensor, p_side: int, hwc):
    height, width, channels = hwc
    b, npp = flat_coords.shape
    padding = p_side // 2
    padded_h = height + 2 * padding
    padded_w = width + 2 * padding
    padded_hw = padded_h * padded_w

    flat_scores = scores.reshape(b, npp)
    pixel_max = torch.full(
        (b, padded_hw), float("-inf"), dtype=flat_scores.dtype, device=patches.device
    ).scatter_reduce_(1, flat_coords, flat_scores, reduce="amax", include_self=True)
    exp_scores = (flat_scores - pixel_max.gather(1, flat_coords)).exp()

    weighted_values = patches.reshape(b, npp, channels) * exp_scores.unsqueeze(-1)

    numerator = torch.zeros(b, padded_hw, channels, dtype=flat_scores.dtype, device=patches.device).scatter_add_(
        1, flat_coords.unsqueeze(-1).expand(b, npp, channels), weighted_values
    )
    denominator = torch.zeros(b, padded_hw, dtype=flat_scores.dtype, device=patches.device).scatter_add_(
        1, flat_coords, exp_scores
    )

    mask_has_data = (
        torch.zeros(b, padded_hw, dtype=torch.int64, device=patches.device)
        .scatter_(1, flat_coords, 1)
        .eq(1)
    )

    safe_denominator = torch.where(mask_has_data, denominator, torch.ones_like(denominator))
    prediction = (numerator / safe_denominator.view(b, padded_hw, 1)).view(b, padded_h, padded_w, channels)[
        :, padding : padding + height, padding : padding + width
    ]
    variances = (
        torch.exp(-pixel_max).unsqueeze(-1) / denominator.view(b, padded_hw, 1)
    ).view(b, padded_h, padded_w, 1)[:, padding : padding + height, padding : padding + width]
    mask = mask_has_data.view(b, padded_h, padded_w)[:, padding : padding + height, padding : padding + width]
    return prediction, variances, mask
