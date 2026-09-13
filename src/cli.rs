use std::{path::PathBuf, result};

use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use toml::de;

use crate::{dataset::DatasetConfig, lr_schedule::LrScheduleConfig, model::ModelConfig};

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("toml deserialization error: {0}")]
    Serde(#[from] de::Error),
}

pub type Result<T> = result::Result<T, ConfigError>;

#[derive(Parser, Debug, Serialize, Deserialize)]
#[command(author, version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub mode: Mode,
    #[arg(long, default_value_t = 1337)]
    pub seed: i64,
}

#[derive(Subcommand, Debug, Serialize, Deserialize)]
pub enum Mode {
    Train {
        #[command(subcommand)]
        config: ConfigSource,
        #[arg(long)]
        tag: Option<String>,
    },
    Eval {
        checkpoint: std::path::PathBuf,
    },
    Vis {
        #[command(flatten)]
        dataset: DatasetConfig,

        #[arg(long, default_value_t = 192)]
        source_points: i64,
    },
}

#[derive(Subcommand, Debug, Serialize, Deserialize)]
pub enum ConfigSource {
    File { path: std::path::PathBuf },
    Cli(TrainConfig),
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    #[command(flatten)]
    pub lr_schedule: LrScheduleConfig,

    #[arg(long, default_value_t = 500)]
    pub eval_interval: i64,

    #[arg(long, default_value_t = 200)]
    pub eval_iters: i64,

    #[arg(long, default_value_t = 4)]
    pub accumulation_steps: i64,

    #[arg(long, default_value_t = 0.5)]
    pub starting_density: f64,
    #[arg(long, default_value_t = 0.1)]
    pub ending_density: f64,

    #[command(flatten)]
    pub model: ModelConfig,

    #[command(flatten)]
    pub dataset: DatasetConfig,
}

impl TrainConfig {
    pub fn load(path: PathBuf) -> Result<Self> {
        Ok(toml::from_str(&std::fs::read_to_string(path)?)?)
    }
}
