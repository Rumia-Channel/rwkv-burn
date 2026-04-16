# RWKVv7 Inference + Training (Burn Framework)

> A port of the [RWKV v7](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7) language model (100% attention free), implemented in [Rust](https://www.rust-lang.org/) with the [Burn](https://burn.dev) deep learning framework.

![rwkv](https://img.shields.io/badge/RWKV-v7-blue)
![burn](https://img.shields.io/badge/Burn-ML%20Framework-orange)
![rust](https://img.shields.io/badge/Rust-stable-informational)

---

## ✨ Features

- 🔥 **Pure Burn**: built with [Burn](https://burn.dev)
- ⚡ **CubeCL-ready backend path**: upgraded to Burn `0.20.1` with WGPU / CubeCL-oriented execution
- 🔤 **Tokenizer** support via [`rwkv-tokenizer`](https://crates.io/crates/rwkv-tokenizer)
- 🧠 Supports **stateful sequential** or **parallel generation** or **mixed generation**
- 🧪 Sampling with **top-k**, **temperature**, and **top-p**
- ⚙️ Load RWKV v7 models from [`safetensors`](https://github.com/huggingface/safetensors)
- 🏋️ **In-repo training** on text corpora or sample-compatible `binidx` datasets
- 💾 **Burn checkpoints** for continuing training and reusing trained models in inference

Inference now defaults to **LibTorch** for numerical stability, while training stays on the Burn autodiff + WGPU path.

For backend debugging, you can compare **LibTorch** and **WGPU** intermediate tensors on the same prompt:

```bash
cargo run -- --weights RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth.safetensors --parity_check --parity_prompt "Hello" --parity_max_tokens 24
```

Add `--parity_verbose` to print per-layer tensor diffs for `TimeMix` / WKV internals.

---

## Help wanted

Currently best results are achieved with [RWKV v7 World 0.1B Model](https://huggingface.co/BlinkDL/rwkv-7-world). 

I appriciate any help investigating the observed performance drop with larger models (0.4B, 1.5B, ...).


## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/dymat/rwkv-burn.git
cd rwkv-burn
cargo build --release
```

### Download and convert model weigths

Download the model weights from Huggingface: https://huggingface.co/BlinkDL/rwkv-7-world

Convert them into SafeTensors format:

```bash
uv run weights_to_safetensors.py path/to/model_weights.pth
```


## 🧠 Usage

### Download weights and convert to safetensors

```bash
wget https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth
uv run weights_to_safetensors.py RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth
```

### Run interactive inference from safetensors

```bash
cargo run --release -- \
  --weights /path/to/model_weights.pth.safetensors \
  --top_p 0.6 \
  --temperature 0.8
```

### Continue inference from a trained Burn checkpoint

```bash
cargo run --release -- \
  --checkpoint_dir artifacts/rwkv-train
```

### Train on a plain text corpus

```bash
cargo run --release -- \
  --train \
  --data_file sample.txt \
  --dataset_format text \
  --ctx_len 256 \
  --batch_size 4 \
  --train_steps 1000 \
  --learning_rate 1e-4 \
  --checkpoint_dir artifacts/rwkv-train
```

### Train on a sample-compatible binidx dataset

```bash
cargo run --release -- \
  --train \
  --data_file data/my_corpus \
  --dataset_format binidx \
  --ctx_len 256 \
  --batch_size 4 \
  --magic_prime 104729 \
  --checkpoint_dir artifacts/rwkv-train
```

## ⚙️ CLI Options

| Flag                        | Description                           | Default     |
|-----------------------------|---------------------------------------|-------------|
| `-l`, `--n_layer`           | Number of RWKV layers                 | `12`        |
| `-d`, `--d_model`           | Embedding (hidden) size               | `768`       |
| `-H`, `--n_heads`           | Number of attention heads             | `12`        |
| `-v`, `--vocab_size`        | Vocabulary size                       | `65536`     |
| `-w`, `--weights`           | Path to `.safetensors` weight file    | _optional_  |
| `-t`, `--temperature`       | Temperature for token sampling        | `0.6`       |
| `-p`, `--top_p`             | Top-p sampling parameter              | `0.6`       |
| `-k`, `--top_k`             | Top-k sampling parameter              | `50`        |
| `--checkpoint_dir`          | Burn checkpoint directory             | _optional_  |
| `--max_new_tokens`          | Max generated tokens per reply        | `64`        |
| `--inference_mode`          | Parallel, RNN, Mixed                  | `Mixed`     |
| `--tokenizer_vocab_file`    | Path to tokenizer vocab file          | `rwkv_vocab_v20230424.txt` |
| `--train`                   | Enable training mode                  | `false`     |
| `--data_file`               | Training corpus path                  | _optional_  |
| `--dataset_format`          | `text` or `binidx`                    | `text`      |
| `--ctx_len`                 | Training context length               | `256`       |
| `--batch_size`              | Training batch size                   | `4`         |
| `--train_steps`             | Training steps                        | `1000`      |
| `--learning_rate`           | Training learning rate                | `1e-4`      |
| `--checkpoint_every`        | Checkpoint interval                   | `100`       |
| `--seed`                    | Training RNG seed                     | `42`        |
| `--magic_prime`             | Deterministic RWKV dataset schedule   | _optional_  |


## 🛠️ Developer Notes

This project implements an RWKV-v7 inference and training pipeline using the [Burn](https://burn.dev) deep learning framework and [rwkv-tokenizer](github.com/cahya-wirawan/rwkv-tokenizer) for tokenization.

### Structure

- `main.rs`: Entry point for inference and training modes.
- `data.rs`: Text / binidx training corpus loading.
- `model`: Implements the RWKVv7 model architecture using `burn` modules.
- `training.rs`: Burn autodiff training loop and checkpoint management.
- `generator.rs`: Handles text generation (sequential and parallel modes).
- `rwkv_vocab_v20230424.txt`: Vocabulary file used by the tokenizer.
- `weights_to_safetensors.py`: Optional script to convert pre-trained weights into `.safetensors` format.

### Supported Features

- 🔄 RNN-style token-by-token generation
- 🔥 Top-k and top-p sampling
- 🧠 Inference state caching across generations
- 🎛 CLI configuration for model size, heads, tokenizer, weights
- 🏋️ Training with Burn autodiff and checkpoint reuse
- ⚙️ WKV kernel abstraction prepared for CubeCL-oriented optimization work


### 🛤️ Roadmap

This project is under active development. Below are the planned features and improvements:

#### Short-term Goals

- [x] Basic RWKVv7 model implementation with `burn` framework  
- [x] Sequential token-by-token text generation  
- [x] CLI interface with configurable parameters  
- [x] Integration with `rwkv-tokenizer`  
- [ ] Investigate poor performance on larger models (e.g. 0.4B, 1.5B)
- [ ] Implement parallel generation mode  
- [ ] Improve support for additional backends (e.g. WGPU); Inference with WGPU seems numerically unstable and produces bad results

#### Mid-term Goals

- [ ] Improve inference speed
- [ ] Save and load model states (checkpointing)  
- [ ] Model quantization for faster inference and smaller memory footprint  
- [ ] Implement batch generation support
- [ ] Improve sampling strategies (temperature annealing, beam search)  

#### Long-term Goals

- [ ] Full training support (fine-tuning on custom datasets, weight initialization)  
- [ ] Multi-GPU and distributed inference support  
- [ ] Model export to ONNX and interoperability with other frameworks  
- [ ] Web-based interface and API service for model inference  

### Ressources
- RWKV original repository: https://github.com/BlinkDL/RWKV-LM
- RWKV project website with many more ressources: https://www.rwkv.com/
- RWKV v7 paper: https://arxiv.org/abs/2503.14456
- Valuable re-implementation of RWKV models with focus on readability rather than performance: https://github.com/SmerkyG/RWKV_Explained

### Community and Contribution

Contributions and feedback are welcome!  
Feel free to open issues or submit pull requests for any feature requests or bug fixes.

---

_Last updated: 2025-06-01_
