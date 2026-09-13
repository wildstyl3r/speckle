use image::{ImageBuffer, RgbaImage, imageops};
use tch::{IndexOp, Kind, Tensor};

use crate::{
    batcher::Batcher,
    model,
    utils::{
        flat_to_2d,
        output::{make_pictures, patch_coords},
    },
};

pub fn merge_images(images: &[RgbaImage], horizontally: bool) -> RgbaImage {
    let (w, h) = images[0].dimensions();
    if horizontally {
        let combined_width = w * images.len() as u32;
        let mut canvas = ImageBuffer::new(combined_width, h);

        for (i, img) in images.iter().enumerate() {
            let x_offset = (i as u32 * w) as i64;
            let y_offset = 0;

            imageops::overlay(&mut canvas, img, x_offset, y_offset);
        }

        canvas
    } else {
        let combined_height = h * images.len() as u32;
        let mut canvas = ImageBuffer::new(w, combined_height);

        for (i, img) in images.iter().enumerate() {
            let x_offset = 0;
            let y_offset = (i as u32 * h) as i64;

            imageops::overlay(&mut canvas, img, x_offset, y_offset);
        }

        canvas
    }
}

pub fn tensor_to_argb(image: &Tensor, active: &Tensor) -> RgbaImage {
    let (h, w, c) = image.size3().unwrap();

    let num_pixels = (h * w) as usize;

    let image_flat = image.view([h * w, c]).to_kind(Kind::Uint8);

    let image_cpu = image_flat.to_device(tch::Device::Cpu);

    let bytes: Vec<u8> = Vec::try_from(&image_cpu.view([-1]).contiguous()).unwrap();
    let active_flat: Vec<bool> = Vec::try_from(&active.reshape([-1]).contiguous()).unwrap();

    let mut rgba_pixels = Vec::with_capacity(num_pixels * 4);

    for i in 0..num_pixels {
        let (r, g, b) = if c == 3 {
            let idx = i * 3;
            (bytes[idx], bytes[idx + 1], bytes[idx + 2])
        } else {
            let val = bytes[i];
            (val, val, val)
        };
        if active_flat[i] {
            rgba_pixels.push(r);
            rgba_pixels.push(g);
            rgba_pixels.push(b);
            rgba_pixels.push(255);
        } else {
            rgba_pixels.push(0);
            rgba_pixels.push(0);
            rgba_pixels.push(0);
            rgba_pixels.push(0);
        }
    }

    image::ImageBuffer::from_raw(w as u32, h as u32, rgba_pixels.clone()).unwrap()
}

pub fn visualize_generation(
    visualization_classes: &[i64],
    batcher: &mut Batcher,
    m: &mut model::Model,
) -> (f64, RgbaImage) {
    tch::no_grad(|| {
        let (sx, target) = batcher.next().unwrap();
        let (b, h, w, c) = target.size4().unwrap();
        let sx = sx.i((.., ..visualization_classes.iter().sum()));
        let (_, n) = sx.size2().unwrap();
        let context = Tensor::zeros([b, h * w, c], (target.kind(), target.device()))
            .scatter_(
                -2,
                &sx.unsqueeze(-1).expand([b, n, c], false),
                &target.view([b, h * w, c]).gather(
                    -2,
                    &sx.unsqueeze(-1).expand([b, n, c], false),
                    false,
                ),
            )
            .view([b, h, w, c]);
        let context_mask = Tensor::zeros([b, h * w], (target.kind(), target.device()))
            .scatter_value_(-1, &sx.expand([b, n], false), 1)
            .view([b, h, w]);
        let (loss, raw_pred) = m.forward_with_loss(&sx, &target, false);
        let (mu, logvar) = m.mu_logvar(&raw_pred, c);
        let (prediction, variances, generated_pixels_mask) = make_pictures(
            &patch_coords(&flat_to_2d(&sx, w), m.patch_side, w),
            &mu,
            &-logvar,
            m.patch_side,
            (h, w, c),
        );

        let batch_to_images = |batch: &Tensor, active_mask: &Tensor| -> Vec<_> {
            batch
                .split(1, 0)
                .into_iter()
                .enumerate()
                .map(|(i, t)| tensor_to_argb(&(t.squeeze_dim(0) * 255.), &active_mask.i(i as i64)))
                .collect()
        };
        let target_imgs = batch_to_images(&target, &generated_pixels_mask);
        let context_imgs = batch_to_images(&context, &context_mask);
        let pred_imgs = batch_to_images(&prediction, &generated_pixels_mask);
        let var_imgs = batch_to_images(&variances, &generated_pixels_mask);
        (
            loss.double_value(&[]),
            merge_images(
                &[
                    merge_images(&target_imgs, true),
                    merge_images(&context_imgs, true),
                    merge_images(&pred_imgs, true),
                    merge_images(&var_imgs, true),
                ],
                false,
            ),
        )
    })
}
