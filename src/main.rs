mod data;
mod generator;
mod model;
mod parity;
mod training;

use std::{
    io::{self, Write},
    path::PathBuf,
};

use anyhow::{Context, Result, bail};
use burn::{
    backend::{Autodiff, LibTorch, Wgpu},
    config::Config as BurnConfig,
    prelude::*,
    record::CompactRecorder,
};
use clap::{Parser, ValueEnum};
use generator::InferenceMode;
use rwkv_tokenizer::WorldTokenizer;

use crate::{
    data::DatasetFormat,
    generator::Generator,
    model::{FrontPathStrategy, RWKVv7, RWKVv7Config},
    training::TrainingConfig,
};

#[derive(ValueEnum, Clone, Debug)]
enum DatasetFormatArg {
    Text,
    Binidx,
}

#[derive(ValueEnum, Clone, Debug)]
enum InferenceBackendArg {
    Libtorch,
    Wgpu,
}

impl From<DatasetFormatArg> for DatasetFormat {
    fn from(value: DatasetFormatArg) -> Self {
        match value {
            DatasetFormatArg::Text => DatasetFormat::Text,
            DatasetFormatArg::Binidx => DatasetFormat::BinIdx,
        }
    }
}

/// Command-line configuration for loading, training and running an RWKVv7 language model.
#[derive(Parser, Debug)]
#[command(
    version = "0.1.0",
    author = "dymat",
    about = "Run inference or train RWKVv7 with Burn + CubeCL backends."
)]
struct Config {
    #[arg(long, help = "Enable language-model training mode")]
    train: bool,

    #[arg(
        short = 'l',
        long = "n_layer",
        default_value_t = 12,
        help = "Number of RWKV layers"
    )]
    n_layer: usize,

    #[arg(
        short = 'd',
        long = "d_model",
        default_value_t = 768,
        help = "Embedding size"
    )]
    d_model: usize,

    #[arg(
        short = 'H',
        long = "n_heads",
        default_value_t = 12,
        help = "Number of attention heads"
    )]
    n_heads: usize,

    #[arg(
        short = 'v',
        long = "vocab_size",
        default_value_t = 65536,
        help = "Vocabulary size"
    )]
    vocab_size: usize,

    #[arg(
        long = "tokenizer_vocab_file",
        default_value_t = String::from("rwkv_vocab_v20230424.txt"),
        help = "Path to tokenizer vocab file"
    )]
    vocab_path: String,

    #[arg(
        short = 'w',
        long = "weights",
        help = "Path to safetensors weight file"
    )]
    weights: Option<String>,

    #[arg(
        long = "checkpoint_dir",
        help = "Path to a Burn checkpoint directory for loading or saving trained models"
    )]
    checkpoint_dir: Option<String>,

    #[arg(
        short = 't',
        long = "temperature",
        default_value_t = 0.6,
        help = "Temperature for token sampling"
    )]
    temperature: f32,

    #[arg(
        short = 'p',
        long = "top_p",
        default_value_t = 0.6,
        help = "Top-p sampling parameter"
    )]
    top_p: f32,

    #[arg(
        short = 'k',
        long = "top_k",
        default_value_t = 50,
        help = "Top-k sampling parameter"
    )]
    top_k: usize,

    #[arg(
        long = "max_new_tokens",
        default_value_t = 64,
        help = "Maximum number of tokens to generate per reply"
    )]
    max_new_tokens: usize,

    #[arg(
        long = "inference_mode",
        default_value_t = String::from("Mixed"),
        help = "Inference mode: Parallel, Sequential or Mixed"
    )]
    inference_mode: String,

    #[arg(
        long = "inference_backend",
        value_enum,
        default_value_t = InferenceBackendArg::Libtorch,
        help = "Inference backend: Libtorch or Wgpu"
    )]
    inference_backend: InferenceBackendArg,

    #[arg(long = "data_file", help = "Training corpus path")]
    data_file: Option<String>,

    #[arg(
        long = "dataset_format",
        value_enum,
        default_value_t = DatasetFormatArg::Text,
        help = "Training corpus format"
    )]
    dataset_format: DatasetFormatArg,

    #[arg(
        long = "ctx_len",
        default_value_t = 256,
        help = "Training context length"
    )]
    ctx_len: usize,

    #[arg(long = "batch_size", default_value_t = 4, help = "Training batch size")]
    batch_size: usize,

    #[arg(
        long = "train_steps",
        default_value_t = 1000,
        help = "Number of training steps"
    )]
    train_steps: usize,

    #[arg(
        long = "learning_rate",
        default_value_t = 1e-4,
        help = "Training learning rate"
    )]
    learning_rate: f64,

    #[arg(
        long = "checkpoint_every",
        default_value_t = 100,
        help = "Save a checkpoint every N training steps"
    )]
    checkpoint_every: usize,

    #[arg(long = "seed", default_value_t = 42, help = "Training RNG seed")]
    seed: u64,

    #[arg(
        long = "magic_prime",
        help = "Sample-compatible prime for deterministic RWKV dataset scheduling"
    )]
    magic_prime: Option<u64>,

    #[arg(
        long = "parity_check",
        help = "Compare LibTorch and WGPU intermediate tensors instead of interactive generation"
    )]
    parity_check: bool,

    #[arg(
        long = "parity_prompt",
        default_value_t = String::from("Hello"),
        help = "Prompt used for backend parity comparison"
    )]
    parity_prompt: String,

    #[arg(
        long = "parity_tolerance",
        default_value_t = 1e-4_f32,
        help = "Absolute tolerance used for backend parity reporting"
    )]
    parity_tolerance: f32,

    #[arg(
        long = "parity_max_tokens",
        default_value_t = 32,
        help = "Maximum number of prompt tokens to compare during parity checks (0 = all)"
    )]
    parity_max_tokens: usize,

    #[arg(
        long = "parity_verbose",
        help = "Print per-tensor backend parity details"
    )]
    parity_verbose: bool,

    #[arg(
        long = "stable_frontpath",
        help = "Use the numerically safer frontpath for sensitive sequential WGPU projections; auto-enabled for WGPU inference"
    )]
    stable_frontpath: bool,
}

impl Config {
    fn model_config(&self) -> RWKVv7Config {
        RWKVv7Config::new(
            self.d_model,
            self.n_heads,
            self.d_model / self.n_heads,
            self.n_layer,
            self.vocab_size,
        )
    }

    fn training_config(&self) -> Result<TrainingConfig> {
        let data_path = self
            .data_file
            .clone()
            .context("--data_file is required in training mode")?;

        if self.d_model % self.n_heads != 0 {
            bail!("d_model must be divisible by n_heads");
        }

        Ok(TrainingConfig {
            weights: self.weights.clone(),
            checkpoint_dir: self.checkpoint_dir.clone(),
            data_path,
            dataset_format: self.dataset_format.clone().into(),
            n_layer: self.n_layer,
            d_model: self.d_model,
            n_heads: self.n_heads,
            vocab_size: self.vocab_size,
            ctx_len: self.ctx_len,
            batch_size: self.batch_size,
            steps: self.train_steps,
            learning_rate: self.learning_rate,
            checkpoint_every: self.checkpoint_every,
            seed: self.seed,
            magic_prime: self.magic_prime,
        })
    }
}

fn main() -> Result<()> {
    let config = Config::parse();

    if config.train {
        run_training(&config)
    } else if config.parity_check {
        parity::run(&config)
    } else {
        run_generate(&config)
    }
}

fn run_generate(config: &Config) -> Result<()> {
    match config.inference_backend {
        InferenceBackendArg::Libtorch => run_generate_with_backend::<LibTorch>(config),
        InferenceBackendArg::Wgpu => run_generate_with_backend::<Wgpu>(config),
    }
}

fn run_generate_with_backend<B: Backend>(config: &Config) -> Result<()> {
    let device = Default::default();
    let mut model = load_or_init_model::<B>(config, &device)?;
    let wgpu_inference = matches!(config.inference_backend, InferenceBackendArg::Wgpu);
    let stable_frontpath = config.stable_frontpath || wgpu_inference;
    if stable_frontpath {
        model.set_front_path_strategy(FrontPathStrategy::GpuChunkedLinear);
    }
    let tokenizer =
        WorldTokenizer::new(Some(&config.vocab_path)).context("failed to load tokenizer")?;

    let mut generator = Generator::new(
        model.clone(),
        &tokenizer,
        config.temperature,
        config.top_p,
        config.top_k,
    );

    let requested_mode = match config.inference_mode.to_lowercase().as_str() {
        "mixed" => InferenceMode::Mixed,
        "parallel" => InferenceMode::Parallel,
        "sequential" => InferenceMode::Sequential,
        _ => {
            println!(
                "Inference mode '{}' not implemented. Falling back to 'Mixed'.",
                config.inference_mode
            );
            InferenceMode::Mixed
        }
    };

    let effective_mode = if wgpu_inference {
        if !matches!(requested_mode, InferenceMode::Sequential) {
            println!(
                "WGPU inference currently forces 'Sequential' mode to avoid unstable parallel prefill."
            );
        }
        if !config.stable_frontpath {
            println!("WGPU inference auto-enables the stable frontpath mitigation.");
        }
        InferenceMode::Sequential
    } else {
        requested_mode
    };
    generator.set_inference_mode(effective_mode);

    let mut transcript = String::new();
    loop {
        print!("User: ");
        let _ = io::stdout().flush();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                let trimmed_input = input.trim();

                if trimmed_input.eq_ignore_ascii_case("\\exit") {
                    break;
                }

                if trimmed_input.eq_ignore_ascii_case("\\reset") {
                    transcript.clear();
                    generator.reset_sequence_state();
                    println!("Resetting conversation transcript.");
                    continue;
                }

                let prompt_suffix = if transcript.is_empty() {
                    format!("User: {}\n\nAssistant:", trimmed_input)
                } else {
                    format!("\n\nUser: {}\n\nAssistant:", trimmed_input)
                };

                print!("Assistant: ");
                let _ = io::stdout().flush();

                let completion = if matches!(effective_mode, InferenceMode::Sequential) {
                    generator.generate_from_suffix(&prompt_suffix, config.max_new_tokens)
                } else {
                    let prompt = if transcript.is_empty() {
                        prompt_suffix.clone()
                    } else {
                        format!("{}{}", transcript, prompt_suffix)
                    };
                    generator.generate_from_prompt(&prompt, config.max_new_tokens)
                };
                print!("{}", completion);
                let _ = io::stdout().flush();

                transcript = if transcript.is_empty() {
                    format!("User: {}\n\nAssistant:{}", trimmed_input, completion)
                } else {
                    format!(
                        "{}\n\nUser: {}\n\nAssistant:{}",
                        transcript, trimmed_input, completion
                    )
                };
            }
            Err(err) => println!("Could not read input: {err}"),
        }

        println!();
        println!();
    }

    Ok(())
}

fn run_training(config: &Config) -> Result<()> {
    type BackendImpl = Autodiff<Wgpu>;

    let device = Default::default();
    let tokenizer =
        WorldTokenizer::new(Some(&config.vocab_path)).context("failed to load tokenizer")?;
    let training_config = config.training_config()?;

    training::train::<BackendImpl>(&training_config, &tokenizer, device)
}

fn load_or_init_model<B: Backend>(config: &Config, device: &B::Device) -> Result<RWKVv7<B>> {
    if let Some(checkpoint_dir) = config.checkpoint_dir.as_deref() {
        let checkpoint_dir = PathBuf::from(checkpoint_dir);
        let config_path = checkpoint_dir.join("model-config.json");
        let record_path = checkpoint_dir.join("model");

        if config_path.exists() && record_path.with_extension("mpk").exists() {
            let model_config = RWKVv7Config::load(&config_path)
                .with_context(|| format!("failed to load {}", config_path.display()))?;
            let model = model_config
                .init::<B>(device)
                .load_file(record_path, &CompactRecorder::new(), device)
                .context("failed to load Burn checkpoint")?;
            return Ok(model);
        }
    }

    if let Some(weight_path) = config.weights.as_deref() {
        return Ok(RWKVv7::<B>::new_from_safetensors(weight_path, device));
    }

    Ok(config.model_config().init::<B>(device))
}
