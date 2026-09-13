use arrow_array::{Array, BinaryArray, Int64Array, StructArray};
use bytes::Bytes;
use clap::{Args, ValueEnum};
use image::DynamicImage;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use reqwest::blocking::get;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fs::{File, create_dir_all};
use std::io::{Read, Write};
use std::path::Path;
use tch::{Kind, Tensor};

use crate::model::ModelError;

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct DatasetConfig {
    #[arg(long, default_value_t = 0.9)]
    pub train_share: f32,

    #[arg(long, value_enum,default_value_t = Dataset::Mnist)]
    pub dataset: Dataset,
}

#[derive(Clone, Copy, ValueEnum, Serialize, Deserialize, Debug)]
pub enum Dataset {
    SplitLine,
    SmoothedSplitLine,
    Mnist,
    Cifar10,
}

impl Dataset {
    pub fn image_column_name(&self) -> &'static str {
        match self {
            Dataset::Mnist => "image",
            Dataset::Cifar10 => "img",
            _ => "",
        }
    }

    pub fn filename(&self) -> &'static str {
        match self {
            Dataset::Mnist => "mnist.parquet",
            Dataset::Cifar10 => "cifar10.parquet",
            _ => "",
        }
    }

    pub fn link(&self, train: bool) -> &'static str {
        if train {
            match self {
                Dataset::Mnist => {
                    "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet"
                }
                Dataset::Cifar10 => {
                    "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet"
                }
                _ => "",
            }
        } else {
            match self {
                Dataset::Mnist => {
                    "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet"
                }
                Dataset::Cifar10 => {
                    "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"
                }
                _ => "",
            }
        }
    }

    pub fn vectorizer(&self, di: DynamicImage) -> Vec<f32> {
        match self.hwc().2 {
            1 => di
                .to_luma8()
                .into_raw()
                .into_iter()
                .map(|u| u as f32 / 255.)
                .collect(),
            3 => di
                .to_rgb8()
                .into_raw()
                .into_iter()
                .map(|u| u as f32 / 255.)
                .collect(),
            _ => unimplemented!(),
        }
    }

    pub fn hwc(&self) -> (i64, i64, i64) {
        match self {
            Dataset::Mnist => (28, 28, 1),
            Dataset::Cifar10 => (32, 32, 3),
            Dataset::SmoothedSplitLine | Dataset::SplitLine => (64, 64, 1),
        }
    }
}

pub fn employ(dataset: Dataset, train: bool) -> Result<Tensor, ModelError> {
    match dataset {
        Dataset::SplitLine | Dataset::SmoothedSplitLine => generate(dataset, train),
        Dataset::Mnist | Dataset::Cifar10 => load_or_download(dataset, train),
    }
}

fn generate(dataset: Dataset, train: bool) -> Result<Tensor, ModelError> {
    match dataset {
        Dataset::SplitLine => {
            let lin = Tensor::linspace(-1.0, 1.0, 64, (Kind::Float, tch::Device::Cpu));
            let x_grid = lin
                .unsqueeze(0)
                .expand([64, 64], false)
                .unsqueeze(0)
                .unsqueeze(-1);
            let y_grid = lin
                .unsqueeze(1)
                .expand([64, 64], false)
                .unsqueeze(0)
                .unsqueeze(-1);
            let offset =
                Tensor::rand([100000, 1, 1, 1], (tch::Kind::Float, tch::Device::Cpu)) - 0.5;
            let tilt = Tensor::rand([100000, 1, 1, 1], (tch::Kind::Float, tch::Device::Cpu)) * PI;
            let a = tilt.cos();
            let b = tilt.sin();
            let distances = a * x_grid + b * y_grid + offset;
            Ok(distances.ge(0).to_kind(tch::Kind::Float))
        }
        Dataset::SmoothedSplitLine => todo!(),
        _ => unimplemented!(),
    }
}

fn load_or_download(dataset: Dataset, train: bool) -> Result<Tensor, ModelError> {
    let cache_dir = "data/".to_owned() + if train { "train" } else { "test" };
    create_dir_all(&cache_dir)?;
    let file_path = Path::new(&cache_dir).join(dataset.filename());

    let raw_bytes: Bytes = if file_path.exists() {
        println!("Found cached dataset at: {}", file_path.display());
        let mut file = File::open(&file_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Bytes::from(buffer)
    } else {
        println!("Downloading dataset from Hugging Face...");
        let response_bytes = get(dataset.link(train))?.bytes()?;
        let mut file = File::create(&file_path)?;
        file.write_all(&response_bytes)?;
        response_bytes
    };

    let parquet_reader = ParquetRecordBatchReaderBuilder::try_new(raw_bytes)?.build()?;
    let mut all_images: Vec<f32> = Vec::new();
    let mut all_labels: Vec<i64> = Vec::new();
    let mut total_samples = 0;

    for batch_result in parquet_reader {
        let batch = batch_result?;
        total_samples += batch.num_rows();

        let image_struct_col = batch
            .column_by_name(dataset.image_column_name())
            .expect("No 'image' column");
        let label_col = batch.column_by_name("label").expect("No 'label' column");
        let image_struct = image_struct_col
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let label_array = label_col.as_any().downcast_ref::<Int64Array>().unwrap();
        let bytes_col = image_struct
            .column_by_name("bytes")
            .expect("No 'bytes' in struct");
        let binary_array = bytes_col.as_any().downcast_ref::<BinaryArray>().unwrap();

        for i in 0..batch.num_rows() {
            all_labels.push(label_array.value(i));
            let encoded_img_bytes = binary_array.value(i);
            let dynamic_img = image::load_from_memory(encoded_img_bytes)?;
            all_images.extend(dataset.vectorizer(dynamic_img));
        }
    }

    let (h, w, c) = dataset.hwc();
    let images = Tensor::from_slice(&all_images)
        .view([total_samples as i64, h, w, c])
        .to_kind(Kind::Float);

    let _labels = Tensor::from_slice(&all_labels).to_kind(Kind::Int64);

    Ok(images)
}
