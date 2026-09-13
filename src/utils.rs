use crate::{batcher::Batcher, cli::TrainConfig, model};
use tch::{IndexOp, TchError, Tensor};

pub mod image;
pub mod info;
pub mod output;

pub fn train_val_split(data: &Tensor, train_share: f32) -> Result<(Tensor, Tensor), TchError> {
    let len = data.size()[0];
    let data = data.index_select(0, &Tensor::randperm(len, (tch::Kind::Int64, data.device())));
    let n = (train_share * len as f32) as i64;
    Ok((data.i(0..n), data.i(n..len - 1)))
}

pub struct LossStatEntry {
    pub prediction: f64,
    // pub prediction_coords: f64,
}

pub fn loss_to_double(
    config: &TrainConfig,
    batcher: &mut Batcher,
    m: &model::Model,
    class_sizes: &[i64],
) -> LossStatEntry {
    tch::no_grad(|| {
        let run = config.eval_iters as usize;
        let mut pred_img = Vec::with_capacity(run);
        for _ in 0..config.eval_iters {
            if let Some((sx, target)) = batcher.next() {
                let sx = sx.i((.., ..class_sizes.iter().sum()));
                let (loss, _) = m.forward_with_loss(&sx, &target, false);
                pred_img.push(loss);
            } else {
                break;
            }
        }
        LossStatEntry {
            prediction: Tensor::stack(&pred_img, 0)
                .mean(tch::Kind::Float)
                .double_value(&[]),
        }
    })
}

pub fn estimate_loss(
    config: &TrainConfig,
    train_batcher: &mut Batcher,
    validation_batcher: &mut Batcher,
    m: &model::Model,
    class_sizes: &[i64],
) -> (LossStatEntry, LossStatEntry) {
    (
        loss_to_double(config, train_batcher, m, class_sizes),
        loss_to_double(config, validation_batcher, m, class_sizes),
    )
}

pub fn flat_to_2d(flat: &Tensor, width: i64) -> Tensor {
    let (_b, _n) = flat.size2().unwrap();
    let heights = flat.floor_divide_scalar(width).to_kind(flat.kind());
    let widths = flat.remainder(width);
    Tensor::stack(&[heights, widths], -1)
}
