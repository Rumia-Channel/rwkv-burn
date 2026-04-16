use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use burn::{
    config::Config,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};
use rwkv_tokenizer::WorldTokenizer;

use crate::{
    data::{DatasetFormat, TrainingCorpus},
    model::{RWKVv7, RWKVv7Config},
};

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub weights: Option<String>,
    pub checkpoint_dir: Option<String>,
    pub data_path: String,
    pub dataset_format: DatasetFormat,
    pub n_layer: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub vocab_size: usize,
    pub ctx_len: usize,
    pub batch_size: usize,
    pub steps: usize,
    pub learning_rate: f64,
    pub checkpoint_every: usize,
    pub seed: u64,
    pub magic_prime: Option<u64>,
}

impl TrainingConfig {
    pub fn model_config(&self) -> RWKVv7Config {
        RWKVv7Config::new(
            self.d_model,
            self.n_heads,
            self.d_model / self.n_heads,
            self.n_layer,
            self.vocab_size,
        )
    }
}

pub fn train<B: AutodiffBackend>(
    config: &TrainingConfig,
    tokenizer: &WorldTokenizer,
    device: B::Device,
) -> Result<()> {
    B::seed(&device, config.seed);

    let checkpoint_dir = checkpoint_dir(config);
    fs::create_dir_all(&checkpoint_dir)
        .with_context(|| format!("failed to create {}", checkpoint_dir.display()))?;

    let corpus = TrainingCorpus::load(&config.data_path, config.dataset_format, tokenizer)?;
    let mut model = load_or_init_model::<B>(config, &checkpoint_dir, &device)?;
    model
        .config()
        .save(checkpoint_dir.join("model-config.json"))
        .context("failed to save model config")?;

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let mut optimizer = AdamConfig::new().init();

    for step in 0..config.steps {
        let (inputs, targets) =
            corpus.sample_batch(config.batch_size, config.ctx_len, step, config.magic_prime)?;
        let input_tensor = Tensor::<B, 1, Int>::from_data(inputs.as_slice(), &device)
            .reshape([config.batch_size, config.ctx_len]);
        let target_tensor = Tensor::<B, 1, Int>::from_data(targets.as_slice(), &device)
            .reshape([config.batch_size, config.ctx_len]);

        let (logits, _) = model.forward_parallel_logits(input_tensor);
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let loss = loss_fn.forward(
            logits.reshape([batch_size * seq_len, vocab_size]),
            target_tensor.reshape([batch_size * seq_len]),
        );

        let loss_value = loss.clone().into_scalar();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(config.learning_rate, model, grads);

        if step % config.checkpoint_every.max(1) == 0 || step + 1 == config.steps {
            save_checkpoint(&model, &checkpoint_dir)
                .with_context(|| format!("failed to save checkpoint at step {step}"))?;
        }

        println!(
            "train step {}/{} loss {:.6}",
            step + 1,
            config.steps,
            loss_value
        );
    }

    Ok(())
}

fn checkpoint_dir(config: &TrainingConfig) -> PathBuf {
    config
        .checkpoint_dir
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("artifacts").join("rwkv-train"))
}

fn load_or_init_model<B: AutodiffBackend>(
    config: &TrainingConfig,
    checkpoint_dir: &PathBuf,
    device: &B::Device,
) -> Result<RWKVv7<B>> {
    let config_path = checkpoint_dir.join("model-config.json");
    let record_path = checkpoint_dir.join("model");

    if config_path.exists() && record_path.with_extension("mpk").exists() {
        let model_config =
            RWKVv7Config::load(config_path).context("failed to load saved model config")?;
        let model = model_config
            .init::<B>(device)
            .load_file(record_path, &CompactRecorder::new(), device)
            .context("failed to load burn checkpoint")?;
        return Ok(model);
    }

    if let Some(weight_path) = config.weights.as_deref() {
        return Ok(RWKVv7::<B>::new_from_safetensors(weight_path, device));
    }

    Ok(config.model_config().init::<B>(device))
}

fn save_checkpoint<B: Backend>(model: &RWKVv7<B>, checkpoint_dir: &PathBuf) -> Result<()> {
    model
        .clone()
        .save_file(checkpoint_dir.join("model"), &CompactRecorder::new())
        .context("failed to persist checkpoint")?;
    Ok(())
}
