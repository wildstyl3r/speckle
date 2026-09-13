use std::f64::consts::PI;

use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};

#[derive(ValueEnum, Debug, Serialize, Deserialize, Clone)]
pub enum DecayMode {
    Cosine,
    Linear,
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct LrScheduleConfig {
    #[arg(long, default_value_t = 10001)]
    pub max_iters: i64,

    #[arg(long, default_value_t = 3e-3)]
    pub max_lr: f64,

    #[arg(long, default_value_t = 0.10)]
    pub min_lr_share: f64,

    #[arg(long, value_enum,default_value_t = DecayMode::Cosine)]
    pub lr_decay: DecayMode,

    #[arg(long, default_value_t = 1500)]
    pub warmup_iters: i64,

    #[arg(long, default_value_t = 1000)]
    pub max_coast_iters: i64,
}

impl LrScheduleConfig {
    pub(crate) fn get_lr(&self, step: i64) -> f64 {
        if step < self.warmup_iters {
            self.max_lr * ((step as f64) / (self.warmup_iters as f64))
        } else if step < self.warmup_iters + self.max_coast_iters {
            self.max_lr
        } else {
            let decay_steps = (self.max_iters - self.warmup_iters - self.max_coast_iters) as f64;
            let step = step as f64 - decay_steps;
            match self.lr_decay {
                DecayMode::Cosine => {
                    self.max_lr
                        * (self.min_lr_share
                            + (1. - self.min_lr_share)
                                * 0.5
                                * (1. + f64::cos(PI * step / decay_steps)))
                }
                DecayMode::Linear => {
                    self.max_lr
                        * (self.min_lr_share + (1. - self.min_lr_share) * (1. - step / decay_steps))
                }
            }
        }
    }
}
