use crate::model::{
    mhsa::{AttentionConfig, SelfAttention, self_attention},
    norm::{self, Norm},
    swiglu::{SwiGLU, swiglu},
};

use clap::Args;
use serde::{Deserialize, Serialize};
use tch::{
    Tensor,
    nn::{self, Module, ModuleT},
};

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct BlockConfig {
    #[arg(long, value_enum, default_value_t = norm::NormOptions::LayerNorm)]
    pub norm: norm::NormOptions,

    #[arg(long, default_value_t = 128)]
    pub swiglu_dim: i64,

    #[command(flatten)]
    pub attention: AttentionConfig,
}

pub fn block(
    path: nn::Path,
    config: &BlockConfig,
    emb_dim: i64,
    dropout: f64,
) -> super::Result<Block> {
    Ok(Block {
        attention_norm: norm::norm(&path / "sa_norm", &config.norm, emb_dim),
        self_attention: self_attention(
            &path / "self_attention",
            emb_dim,
            &config.attention,
            dropout,
        )?,
        attention_residual_scale: path.var("attention_residual_scale", &[1], nn::Init::Const(1.)),
        storage_norm: norm::norm(&path / "st_norm", &config.norm, emb_dim),
        storage: swiglu(&path / "storage", emb_dim, config.swiglu_dim, dropout),
    })
}

#[derive(Debug)]
pub struct Block {
    attention_norm: Norm,
    pub self_attention: SelfAttention,
    attention_residual_scale: Tensor,
    storage_norm: Norm,
    pub storage: SwiGLU,
}

impl Block {
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = self
            .self_attention
            .forward_t(&self.attention_norm.forward(xs), train)
            * &self.attention_residual_scale
            + xs;
        self.storage
            .forward_t(&self.storage_norm.forward(&xs), train)
            + xs
    }
}
