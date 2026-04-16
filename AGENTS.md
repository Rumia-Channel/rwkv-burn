# AGENTS.md

## Project history

This repository was upgraded on branch `feat/rwkv-training-kernels` to turn it from inference-only RWKV Burn code into a project that can also **train RWKV inside the repo**, **save/load Burn checkpoints**, and **experiment with optimized kernel paths**.

### Major changes that were made

1. **Training pipeline added**
   - Added in-repo training with Burn autodiff.
   - Added dataset loading for plain text and sample-compatible `binidx`.
   - Added checkpoint save/load support and CLI entrypoints for training.

2. **Kernel abstraction introduced**
   - Extracted WKV / TimeMix recurrence behind `src/model/kernels.rs`.
   - This was done so future optimized CubeCL kernels can be introduced without rewriting the model surface again.

3. **Stable Burn/CubeCL dependency line chosen**
   - Moved to `burn 0.20.1`, `burn-cubecl 0.20.1`, `cubecl 0.9.0`.
   - Pre-release Burn versions were intentionally avoided in favor of the stable line.

4. **Dependency refresh**
   - Direct dependencies were audited conservatively.
   - `safetensors` was updated to `0.7.0`.
   - Other dependencies were kept on stable versions unless there was a clear stability or efficiency benefit.

5. **Inference stability restored**
   - Default inference backend was switched back to **LibTorch** because WGPU inference produced numerically broken output.
   - This fixed the obvious garbled-text regression.

6. **Backend parity tooling added**
   - Added `--parity_check` to compare **LibTorch** and **WGPU** intermediate tensors token-by-token.
   - Added `--parity_verbose` to inspect layer-by-layer tensor divergence inside TimeMix / WKV.

7. **WGPU front-path stabilization added**
   - Added `--stable_frontpath`.
   - This switches sensitive sequential projections to a host-side reference matvec path for:
     - `TimeMix` `receptance/key/value/output`
     - `ChannelMix` `key/value`
     - final `unembed`
   - Added `--inference_backend wgpu` so experimental WGPU inference can be run directly with this mitigation.

## Important findings

- The RWKV model design itself was **not** the primary failure.
- The main regression came from **backend-specific numerical drift on WGPU**.
- Early divergence appeared **before** WKV recurrence became dominant:
  - `tmix_r`
  - `tmix_k_raw`
  - `tmix_v_raw`
  - and later projection / normalization paths
- Therefore, **a CubeCL WKV kernel alone is not enough** unless the early projection and norm-sensitive paths are also stabilized.

## Current status

- **Default inference**: LibTorch
- **Training**: Burn autodiff + WGPU
- **Experimental WGPU inference**: available with `--inference_backend wgpu --stable_frontpath`

The `--stable_frontpath` mitigation improved parity substantially:

- 1-token worst diff: about `151` -> about `0.000103`
- 7-token prompt worst diff: about `0.000349`

This confirmed that the dominant issue was the WGPU front linear/projection path, not a fundamentally broken RWKV recurrence.

## Commands worth knowing

### Validation

```bash
cargo test --quiet && cargo build --quiet
```

### Parity check

```bash
cargo run -- --weights RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth.safetensors --parity_check --parity_prompt "Hello" --parity_max_tokens 24
```

### Parity check with front-path stabilization

```bash
cargo run -- --weights RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth.safetensors --parity_check --parity_prompt "Hello" --parity_max_tokens 24 --stable_frontpath
```

### Experimental WGPU inference

```bash
cargo run -- --weights RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth.safetensors --inference_backend wgpu --stable_frontpath --inference_mode sequential
```

## Important commits on this branch

- `fca1393` Add RWKV training and kernel refactor
- `fee345b` Refresh stable dependency versions
- `e069db4` Stabilize inference backend
- `484d76a` Add backend parity harness
- `ba166d6` Stabilize WGPU front path

## Likely next steps

- Fuse TimeMix preprocessing (`fused_addcmul` / `fused_k` equivalent) to reduce rounding points further.
- Revisit a dedicated CubeCL WKV kernel after front-path parity is already good.
- Replace the host-side reference matvec path with a faster GPU-side implementation only if parity remains within the same tolerance range.
