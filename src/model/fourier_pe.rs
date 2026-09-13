use std::f64::consts::TAU;

use tch::{
    Tensor,
    nn::{self, Module},
};

#[derive(Debug)]
// following https://arxiv.org/abs/2103.15813 (PixelTransformer) and https://arxiv.org/abs/2006.10739 (Fourier Features Positional Encoding)
pub struct PositionEncoder {
    projection: Tensor,
}

impl PositionEncoder {
    pub fn new(path: nn::Path, in_dim: i64, out_dim: i64) -> Self {
        Self {
            projection: path
                .randn("proj", &[in_dim, out_dim / 2], 0., 2.)
                .requires_grad_(false),
        }
    }
}

impl Module for PositionEncoder {
    fn forward(&self, pos: &Tensor) -> Tensor {
        let stub = (pos * TAU).matmul(&self.projection);
        Tensor::cat(&[stub.cos(), stub.sin()], -1)
    }
}
