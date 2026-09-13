use tch::{
    Tensor,
    nn::{self, Module},
};

use crate::model::Result;

use clap::Args;
use serde::{Deserialize, Serialize};

pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, dropout: f64, train: bool) -> Tensor {
    q.matmul(&k.transpose(-1, -2))
        .softmax(-1, None)
        .dropout(dropout, train)
        .matmul(v)
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct AttentionConfig {
    #[arg(long, default_value_t = 4)]
    pub num_heads: i64,

    #[arg(long)]
    pub qk_norm: bool,
}

#[derive(Debug)]
pub struct SelfAttention {
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    output: nn::Linear,

    qk_norm_scale: Option<Tensor>,
    head_dim: i64,
    dropout: f64,
    vec_scale: f64,
}
impl SelfAttention {
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let (b, t, c) = xs.size3().unwrap();

        let q = {
            let q = (self.query.forward(xs) * self.vec_scale)
                .view([b, t, c / self.head_dim, self.head_dim])
                .transpose(-2, -3);
            if self.qk_norm_scale.as_ref().is_some() {
                q.rms_norm([self.head_dim], None::<Tensor>, 1e-5)
            } else {
                q
            }
        };
        let k = {
            let k = (self.key.forward(xs) * self.vec_scale)
                .view([b, t, c / self.head_dim, self.head_dim])
                .transpose(-2, -3);

            if let Some(ref_k_scale) = self.qk_norm_scale.as_ref() {
                k.rms_norm([self.head_dim], None::<Tensor>, 1e-5) * ref_k_scale
            } else {
                k
            }
        };

        let v = self
            .value
            .forward(xs)
            .view([b, t, c / self.head_dim, self.head_dim])
            .transpose(-2, -3);

        self.output
            .forward(
                &attention(&q, &k, &v, self.dropout, train)
                    .transpose(-2, -3)
                    .reshape([b, t, c]),
            )
            .dropout(self.dropout, train)
    }
}

pub fn self_attention(
    path: nn::Path,
    emb_dim: i64,
    config: &AttentionConfig,
    dropout: f64,
) -> Result<SelfAttention> {
    Ok(SelfAttention {
        query: nn::linear(
            &path / "w_q",
            emb_dim,
            emb_dim,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        ),
        key: nn::linear(
            &path / "w_k",
            emb_dim,
            emb_dim,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        ),
        value: nn::linear(
            &path / "w_v",
            emb_dim,
            emb_dim,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        ),
        output: nn::linear(
            &path / "w_o",
            emb_dim,
            emb_dim,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        ),
        head_dim: emb_dim / config.num_heads,
        dropout,
        vec_scale: ((emb_dim / config.num_heads) as f64).powf(-0.25),
        qk_norm_scale: if config.qk_norm {
            Some(path.var(
                "k_norm_scale",
                &[1, config.num_heads, 1, 1],
                nn::Init::Const(1.),
            ))
        } else {
            None
        },
    })
}
