import numpy as np
import torch
from PIL import Image

from .output import flat_to_2d, make_pictures, patch_coords


def merge_images(images, horizontally: bool) -> Image.Image:
    w, h = images[0].size
    if horizontally:
        canvas = Image.new("RGBA", (w * len(images), h))
        for i, img in enumerate(images):
            canvas.paste(img, (i * w, 0))
    else:
        canvas = Image.new("RGBA", (w, h * len(images)))
        for i, img in enumerate(images):
            canvas.paste(img, (0, i * h))
    return canvas


def tensor_to_rgba(image: torch.Tensor, active: torch.Tensor) -> Image.Image:
    h, w, c = image.shape
    image_flat = image.reshape(h * w, c).to(torch.uint8).cpu().numpy()
    active_flat = active.reshape(-1).to(torch.bool).cpu().numpy()

    rgba = np.zeros((h * w, 4), dtype=np.uint8)
    if c == 3:
        rgba[active_flat, 0:3] = image_flat[active_flat]
    else:
        rgba[active_flat, 0:3] = image_flat[active_flat, 0:1]
    rgba[active_flat, 3] = 255
    return Image.fromarray(rgba.reshape(h, w, 4), "RGBA")


def visualize_generation(visualization_classes, batcher, m):
    with torch.no_grad():
        batch = batcher.next()
        if batch is None:
            raise RuntimeError("visualization batcher is exhausted")
        sx, target = batch
        b, h, w, c = target.shape
        sx = sx[:, : sum(visualization_classes)]
        n = sx.shape[1]

        gathered = target.view(b, h * w, c).gather(1, sx.unsqueeze(-1).expand(b, n, c))
        context = (
            torch.zeros(b, h * w, c, device=target.device)
            .scatter(1, sx.unsqueeze(-1).expand(b, n, c), gathered)
            .view(b, h, w, c)
        )
        context_mask = torch.zeros(b, h * w, device=target.device).scatter(1, sx, 1).view(b, h, w)

        loss, raw_pred = m.forward_with_loss(sx, target, False)
        mu, logvar = m.mu_logvar(raw_pred, c)
        prediction, variances, generated_pixels_mask = make_pictures(
            patch_coords(flat_to_2d(sx, w), m.patch_side, w),
            mu,
            -logvar,
            m.patch_side,
            (h, w, c),
        )

        def batch_to_images(batch_t, active_mask):
            return [
                tensor_to_rgba(batch_t[i] * 255.0, active_mask[i]) for i in range(b)
            ]

        target_imgs = batch_to_images(target, generated_pixels_mask)
        context_imgs = batch_to_images(context, context_mask)
        pred_imgs = batch_to_images(prediction, generated_pixels_mask)
        var_imgs = batch_to_images(variances, generated_pixels_mask)
        grid = merge_images(
            [
                merge_images(target_imgs, True),
                merge_images(context_imgs, True),
                merge_images(pred_imgs, True),
                merge_images(var_imgs, True),
            ],
            False,
        )
        return float(loss.item()), grid


def visualize_source_points(images: torch.Tensor, sample_idx: int, source_points: int) -> Image.Image:
    single_img = images[sample_idx]
    h, w, c = single_img.shape
    class_ix = torch.randint(100000, (h * w,)).argsort(-1)
    source_ix = class_ix[:source_points]
    mask = (
        torch.zeros(h * w, c)
        .scatter(0, source_ix.unsqueeze(-1).expand(-1, c), 1)
        .eq(1)
        .view(h, w, c)
    )
    return tensor_to_rgba(single_img * 255.0, mask)
