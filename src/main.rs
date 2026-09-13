use std::f64::consts::PI;
use std::fs;
use std::io::Write;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use tch::IndexOp;
use tch::{Tensor, nn, nn::OptimizerConfig};

mod batcher;
mod cli;
mod dataset;
mod lr_schedule;
mod model;
mod muon;
mod utils;
mod visualizer;

use crate::batcher::Batcher;
use crate::cli::{Cli, ConfigSource, Mode, TrainConfig};
use crate::muon::muon;
use crate::utils::{
    estimate_loss,
    image::{tensor_to_argb, visualize_generation},
    info::{param_count, write_summary},
    train_val_split,
};

#[derive(serde::Serialize)]
struct LossRecord {
    step: i64,
    train_loss_pred: f64,
    val_loss_pred: f64,
    time_from_start: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let git_hash = env!("GIT_HASH");

    match cli.mode {
        Mode::Train { config, tag } => {
            let mut config = match config {
                ConfigSource::File { path } => TrainConfig::load(path)?,
                ConfigSource::Cli(config) => config,
            };

            tch::manual_seed(cli.seed);

            let images = dataset::employ(config.dataset.dataset, true)?;
            let (h, w, c) = config.dataset.dataset.hwc();

            let (train, val) = train_val_split(&images, config.dataset.train_share)?;

            let mut train_batcher = Batcher::new(train.shallow_clone(), config.model.batch_size);
            let mut val_batcher = Batcher::new(val.shallow_clone(), config.model.batch_size);
            let visualization_classes = &[256];
            let mut train_vis_batcher = Batcher::new(train, config.model.eval_batch_size);
            let mut val_vis_batcher = Batcher::new(val, config.model.eval_batch_size);

            let model_creation_start = Instant::now();
            let device = tch::Device::Cpu;
            let vs: nn::VarStore = tch::nn::VarStore::new(device);
            config.model.max_image_side = std::cmp::max(h, w);
            config.model.image_channels = c;
            let mut model = model::Model::new(vs.root(), &mut config.model)?;
            println!(
                "model created in {:?}",
                model_creation_start - Instant::now()
            );

            let (mut adam_p, mut muon_p) = (Vec::new(), Vec::new());
            for (name, w) in vs.variables() {
                if w.requires_grad() {
                    if w.size().len() == 1 || name.contains("embedding") || name.contains("bias") {
                        adam_p.push(w);
                    } else {
                        muon_p.push(w);
                    }
                }
            }

            let mut adamw = nn::AdamW::default().beta2(0.95).build_copt(0.)?;
            for w in adam_p {
                adamw.add_parameters(&w, 0)?;
            }
            let mut muon = muon(muon_p, 0., 0.95, 0., true, true, 1e-7);

            let log_dir = std::path::Path::new("checkpoints").join(format!(
                "run_{}_{}{}",
                chrono::Local::now().format("%Y%m%d_%H%M"),
                git_hash,
                match tag {
                    Some(tag) => format!("_{}", tag),
                    None => String::new(),
                },
            ));
            fs::create_dir_all(&log_dir)?;

            let mut scheme = fs::File::create(log_dir.join("scheme.txt"))?;

            let (total_params, trainable_params) = param_count(&vs);
            scheme.write_fmt(format_args!(
                "total parameters: {}\ntrainable parameters: {}\n",
                total_params, trainable_params
            ))?;
            write_summary(&vs, &scheme)?;

            println!(
                "total parameters: {}\ntrainable parameters: {}",
                total_params, trainable_params
            );

            fs::write(
                log_dir.join("config.toml"),
                toml::to_string_pretty(&config)?,
            )?;
            let mut wtr = csv::Writer::from_path(log_dir.join("losses.csv"))?;
            let start = Instant::now();
            adamw.zero_grad()?;
            muon.zero_grad();

            let val_step_size = [192];
            let total_steps = config.lr_schedule.max_iters * config.accumulation_steps;
            for step in 0..total_steps {
                if (step / config.accumulation_steps) % config.eval_interval == 0
                    && step % config.accumulation_steps == 0
                {
                    let eval_start = Instant::now();
                    let losses = estimate_loss(
                        &config,
                        &mut train_batcher,
                        &mut val_batcher,
                        &model,
                        &val_step_size,
                    );
                    if (step / config.accumulation_steps) % config.eval_interval == 0
                        && step % config.accumulation_steps == 0
                    {
                        let (train_pred_loss, train_img) = visualize_generation(
                            visualization_classes,
                            &mut train_vis_batcher,
                            &mut model,
                        );
                        train_img.save(log_dir.join(format!(
                            "train_step{}_pred{:.4}.png",
                            step / config.accumulation_steps,
                            train_pred_loss,
                        )))?;
                        let (val_pred_loss, val_img) = visualize_generation(
                            visualization_classes,
                            &mut val_vis_batcher,
                            &mut model,
                        );
                        val_img.save(log_dir.join(format!(
                            "val_step{}_pred{:.4}.png",
                            step / config.accumulation_steps,
                            val_pred_loss,
                        )))?;
                    }
                    let eval_end = Instant::now();
                    let dt = eval_end - start;
                    println!(
                        "step: {}, train loss: [p_i: {:.4}], val loss: [p_i: {:.4}], time from start {:?}, {} eval iterations done in {:?}",
                        step / config.accumulation_steps,
                        losses.0.prediction,
                        losses.1.prediction,
                        dt,
                        config.eval_iters,
                        eval_end - eval_start
                    );
                    wtr.serialize(LossRecord {
                        step: step / config.accumulation_steps,
                        train_loss_pred: losses.0.prediction,
                        val_loss_pred: losses.1.prediction,
                        time_from_start: format!("{:?}", dt),
                    })?;
                    wtr.flush()?;
                }

                if let Some((sx, target)) = train_batcher.next() {
                    let sx = sx.i((
                        ..,
                        ..((config.ending_density
                            + (config.starting_density - config.ending_density)
                                * 0.5
                                * (1. + f64::cos(PI * step as f64 / total_steps as f64)))
                            * (h * w) as f64) as i64,
                    ));
                    let (loss, _) = model.forward_with_loss(&sx, &target, true);
                    let normalized_loss = loss / (config.accumulation_steps as f64);
                    normalized_loss.backward();
                    if step % config.accumulation_steps == 0 {
                        let lr = config.lr_schedule.get_lr(step / config.accumulation_steps);
                        adamw.set_learning_rate(lr)?;
                        muon.set_lr(7. * lr);
                        adamw.step()?;
                        muon.step();
                        adamw.zero_grad()?;
                        muon.zero_grad();
                    }
                } else {
                    break;
                }
            }
            wtr.flush()?;
            println!("elapsed time: {:?}", start.elapsed());

            vs.save(log_dir.join("model.safetensors"))?;
        }
        Mode::Eval { checkpoint } => {
            let device = tch::Device::Cpu;
            let mut vs: nn::VarStore = tch::nn::VarStore::new(device);
            let mut config = TrainConfig::load(checkpoint.join("config.toml"))?.model;
            let _model = model::Model::new(vs.root(), &mut config)?;
            vs.load(checkpoint.join("model.safetensors"))?;
        }
        Mode::Vis {
            dataset,
            source_points,
        } => {
            let (h, w, c) = dataset.dataset.hwc();
            let images = dataset::employ(dataset.dataset, true)?;
            let sample_idx = cli.seed;
            let single_img = images.get(sample_idx);
            {
                let class_ix =
                    Tensor::randint(100000, [h * w], (tch::Kind::Int, single_img.device()))
                        .argsort(-1, false);
                let input = single_img * 255.;
                let source_ix = class_ix.i(..source_points);
                visualizer::display_image_buffer(tensor_to_argb(
                    &input,
                    &Tensor::zeros_like(&input)
                        .view([h * w, c])
                        .scatter_value_(0, &source_ix.unsqueeze(-1).repeat([1, c]), 1)
                        .eq(1)
                        .view([h, w, c]),
                ))
                .unwrap();
            };
        }
    };
    Ok(())
}
