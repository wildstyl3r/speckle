use std::borrow::Borrow;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tch::{
    Tensor,
    nn::{self, LayerNorm, Module, Path},
};

#[derive(Debug)]
pub enum Norm {
    Layer(LayerNorm),
    Rms(RMSNorm),
}
impl Module for Norm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self {
            Norm::Layer(layer_norm) => layer_norm.forward(xs),
            Norm::Rms(rmsnorm) => rmsnorm.forward(xs),
        }
    }
}

#[derive(ValueEnum, Debug, Serialize, Deserialize, Clone)]
pub enum NormOptions {
    LayerNorm,
    RMSNorm,
}

#[derive(Debug, Clone, Copy)]
pub struct RMSNormConfig {
    pub eps: f64,
    pub elementwise_affine: bool,
    pub ws_init: nn::Init,
}

impl Default for RMSNormConfig {
    fn default() -> Self {
        RMSNormConfig {
            eps: 1e-5,
            elementwise_affine: true,
            ws_init: nn::Init::Const(1.),
        }
    }
}

#[derive(Debug)]
pub struct RMSNorm {
    config: RMSNormConfig,
    pub ws: Option<Tensor>,
    pub normalized_shape: Vec<i64>,
}

pub fn rms_norm<'a, T: Borrow<nn::Path<'a>>>(
    vs: T,
    normalized_shape: Vec<i64>,
    config: RMSNormConfig,
) -> RMSNorm {
    let vs = vs.borrow();

    let ws = if config.elementwise_affine {
        let ws = vs.var("weight", normalized_shape.as_slice(), config.ws_init);
        Some(ws)
    } else {
        None
    };

    RMSNorm {
        config,
        ws,
        normalized_shape,
    }
}

impl Module for RMSNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::rms_norm(
            xs,
            self.normalized_shape.as_slice(),
            self.ws.as_ref(),
            self.config.eps,
        )
    }
}

pub fn norm(path: Path, norm: &NormOptions, emb_dim: i64) -> Norm {
    match norm {
        NormOptions::LayerNorm => {
            Norm::Layer(nn::layer_norm(path, vec![emb_dim], Default::default()))
        }
        NormOptions::RMSNorm => Norm::Rms(rms_norm(path, vec![emb_dim], Default::default())),
    }
}
