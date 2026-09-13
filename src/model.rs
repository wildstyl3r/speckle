pub mod block;
pub mod fourier_pe;
pub mod mhsa;
pub mod norm;
pub mod swiglu;

use crate::{
    model::{block::Block, fourier_pe::PositionEncoder},
    utils::{flat_to_2d, output::patch_coords},
};
use thiserror::Error;

use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use tch::{
    IndexOp, Kind, Tensor,
    nn::{self, LinearConfig, Module},
};
use winit::error::EventLoopError;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("event loop error {0}")]
    EventLoop(#[from] EventLoopError),
    #[error("io error {0}")]
    IoError(#[from] std::io::Error),
    #[error("networking error {0}")]
    NetworkError(#[from] reqwest::Error),
    #[error("parquet error {0}")]
    ParquetError(#[from] parquet::errors::ParquetError),
    #[error("arrow error {0}")]
    ArrowError(#[from] arrow::error::ArrowError),
    #[error("image error {0}")]
    ImageErrpr(#[from] image::error::ImageError),
}

pub type Result<T, E = ModelError> = core::result::Result<T, E>;

#[derive(ValueEnum, Debug, Serialize, Deserialize, Clone)]
pub enum ResidualOptions {
    Plain,
    Attentive,
}

#[derive(Debug)]
pub enum Residual {
    Plain,
    Attentive(Tensor),
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    #[arg(long, default_value_t = 0.2)]
    pub dropout: f64,

    #[arg(long, default_value_t = 0.0)]
    pub id_loss_scale: f64,

    #[arg(long, default_value_t = 3)]
    pub image_channels: i64,

    #[serde(skip)]
    #[arg(skip)]
    pub max_image_side: i64,

    #[arg(long, default_value_t = 4)]
    pub n_blocks: i64,

    #[command(flatten)]
    pub block: block::BlockConfig,

    #[arg(long, default_value_t = 3)]
    pub patch_side: i64,

    #[arg(long, default_value_t = 64)]
    pub emb_dim: i64,

    #[arg(long, default_value_t = 1)]
    pub batch_size: i64,

    #[arg(long, default_value_t = 10)]
    pub eval_batch_size: i64,

    #[arg(long, value_enum, default_value_t = ResidualOptions::Plain)]
    pub residual: ResidualOptions,
}

#[derive(Debug)]
pub struct Model {
    pos_encoder: PositionEncoder,
    pixel_encoder: nn::Linear,
    blocks: Vec<Block>, //nn::SequentialT,
    residual: Residual,
    final_norm: norm::Norm,
    pixel_modeling_head: nn::Linear,
    pub patch_side: i64,
}

impl Model {
    pub fn new(path: nn::Path, config: &mut ModelConfig) -> Result<Self> {
        let residual = match config.residual {
            ResidualOptions::Attentive => Residual::Attentive(path.var(
                "block_ar_queries",
                &[config.n_blocks, config.emb_dim],
                nn::Init::Const(0.),
            )),
            ResidualOptions::Plain => Residual::Plain,
        };

        Ok(Model {
            pos_encoder: PositionEncoder::new(&path / "pos_enc", 2, config.emb_dim / 2),
            pixel_encoder: nn::linear(
                &path / "embedding",
                config.image_channels,
                config.emb_dim / 2,
                LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            blocks: (0..config.n_blocks)
                .map(|i| {
                    block::block(
                        &path / ("b".to_owned() + &i.to_string()),
                        &config.block,
                        config.emb_dim,
                        config.dropout,
                    )
                })
                .collect::<Result<Vec<Block>, ModelError>>()?,
            final_norm: norm::norm(&path / "ln_f", &config.block.norm, config.emb_dim),
            pixel_modeling_head: nn::linear(
                &path / "unembedding",
                config.emb_dim,
                config.patch_side * config.patch_side * (1 + config.image_channels),
                LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            residual,
            patch_side: config.patch_side,
        })
    }

    pub fn mu_logvar(&self, raw_pred: &Tensor, c: i64) -> (Tensor, Tensor) {
        let (b, n, _raw_c) = raw_pred.size3().unwrap();
        //[b,n,p*p]
        let logvariance = raw_pred.narrow(-1, 0, self.patch_side * self.patch_side);
        //[b,n,p*p,c]
        let predictions = raw_pred
            .narrow(
                -1,
                self.patch_side * self.patch_side,
                self.patch_side * self.patch_side * c,
            )
            .view([b, n, self.patch_side * self.patch_side, c]);
        (predictions, logvariance)
    }

    pub fn forward_with_loss(&self, sx: &Tensor, target: &Tensor, train: bool) -> (Tensor, Tensor) {
        let (b, h, w, c) = target.size4().unwrap();
        let (_, n) = sx.size2().unwrap();
        let lin = Tensor::linspace(-1.0, 1.0, h, (Kind::Float, tch::Device::Cpu));
        let x_grid = lin.unsqueeze(0).expand([h, w], false);
        let y_grid = lin.unsqueeze(1).expand([h, w], false);
        let pos: Tensor = Tensor::stack(&[y_grid, x_grid], -1)
            .view([h * w, 2])
            .index_select(0, &sx.reshape([-1]))
            .view([b, n, 2]);
        let raw_predictions = self.forward_t(
            &target.view([b, h * w, c]).gather(
                -2,
                &sx.unsqueeze(-1).expand([b, n, c], false),
                false,
            ),
            &pos,
            train,
        );
        let (mu, logvar) = self.mu_logvar(&raw_predictions, c);
        let padding = self.patch_side / 2;
        let padded_h = h + 2 * padding;
        let padded_w = w + 2 * padding;
        let mask = Tensor::zeros([b, padded_h, padded_w], (target.kind(), target.device()));
        mask.slice(1, 1, h + 1, 1)
            .slice(2, 1, w + 1, 1)
            .copy_(&Tensor::ones([b, h, w], (target.kind(), target.device())));
        let padded_target =
            Tensor::zeros([b, padded_h, padded_w, c], (target.kind(), target.device()));
        padded_target
            .slice(1, 1, h + 1, 1)
            .slice(2, 1, w + 1, 1)
            .copy_(target);

        //[b,n*p*p]
        let flat_coords = patch_coords(&flat_to_2d(sx, w), self.patch_side, w);
        //[b,n,p*p,c]
        let patch_targets = padded_target
            .view([b, padded_h * padded_w, c])
            .gather(
                -2,
                &flat_coords
                    .unsqueeze(-1)
                    .expand([b, n * self.patch_side * self.patch_side, c], false),
                false,
            )
            .view([b, n, self.patch_side * self.patch_side, c]);
        //[b,n,p*p]
        let patch_mask = mask
            .view([b, padded_h * padded_w])
            .gather(-1, &flat_coords, false)
            .view([b, n, self.patch_side * self.patch_side]);

        let loss = (patch_mask.unsqueeze(-1)
            * ((patch_targets - mu).square() * (-logvar.unsqueeze(-1)).exp()
                + logvar.unsqueeze(-1)))
        .mean(tch::Kind::Float);
        (loss, raw_predictions)
    }

    fn forward_t(&self, context: &Tensor, positions: &Tensor, train: bool) -> Tensor {
        //[b,context,c]
        let context = {
            //pseudo-inverse for logistic sigmoid
            let k = context * 2 - 1;
            (&k + k.pow_tensor_scalar(5)) * 2
        };
        let xs: Tensor = self.pixel_encoder.forward(&context);
        let ps: Tensor = self.pos_encoder.forward(positions);
        let mut xs = Tensor::cat(&[ps, xs], -1);

        match &self.residual {
            Residual::Plain => {
                for block in &self.blocks {
                    xs = block.forward_t(&xs, train);
                }

                self.pixel_modeling_head
                    .forward(&self.final_norm.forward(&xs))
            }
            Residual::Attentive(block_queries) => {
                //[b,t,1,c]
                let mut prefix_hs = xs.unsqueeze(-2);
                let c = xs.size()[xs.size().len() - 1];

                for (i, block) in self.blocks.iter().enumerate() {
                    let norm_keys = &prefix_hs.rms_norm(&[c][..], None::<Tensor>, 1e-7);
                    //[b,t,1,h] @ [b,t,h,c] -> [b,t,c]
                    let hs = norm_keys
                        .matmul(&block_queries.i((i as i64, ..)).view([-1]))
                        .softmax(-1, None)
                        .unsqueeze(-2)
                        .matmul(&prefix_hs)
                        .squeeze_dim(-2);
                    xs = block.forward_t(&hs, train);
                    if i + 1 < self.blocks.len() {
                        prefix_hs = Tensor::cat(&[prefix_hs, xs.unsqueeze(-2)], -2);
                    }
                }

                self.pixel_modeling_head
                    .forward(&self.final_norm.forward(&xs))
            }
        }
    }
}
