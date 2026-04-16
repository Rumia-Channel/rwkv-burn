# RWKV Burn upgrade history

Branch: `feat/rwkv-training-kernels`

## What was implemented
- Added in-repo RWKV training with Burn autodiff, dataset loading, checkpoint save/load, and CLI support.
- Refactored WKV/TimeMix into a kernel abstraction (`src/model/kernels.rs`) so optimized CubeCL-style kernels can be introduced later.
- Upgraded the project to the stable Burn/CubeCL stack used in this work: `burn 0.20.1`, `burn-cubecl 0.20.1`, `cubecl 0.9.0`.
- Refreshed direct dependencies conservatively, notably upgrading `safetensors` to `0.7.0` while keeping the rest on stable lines.
- Restored stable inference by switching default inference backend back to LibTorch.
- Added a backend parity harness to compare LibTorch and WGPU intermediate tensors token-by-token.
- Implemented a WGPU front-path stabilization option (`--stable_frontpath`) using host-side reference matvecs for sensitive sequential projections.

## Key findings
- RWKV design itself was not the main issue; the largest regression came from backend-specific numerical drift in WGPU inference.
- Early divergence appeared before WKV recurrence dominated: `tmix_r`, `tmix_k_raw`, `tmix_v_raw`, and later projection paths drifted first.
- A CubeCL WKV kernel alone is not enough unless early projection/norm paths are also stabilized.
- With `--stable_frontpath`, parity improved from roughly `151` worst diff on the first token to roughly `0.000103`, and a 7-token prompt check stayed around `0.000349` worst-case.

## Current recommended paths
- Default inference: LibTorch.
- Training: Burn autodiff + WGPU.
- Experimental WGPU inference / debugging: use `--inference_backend wgpu --stable_frontpath` and `--parity_check` to validate changes.

## Important commits
- `fca1393` Add RWKV training and kernel refactor
- `fee345b` Refresh stable dependency versions
- `e069db4` Stabilize inference backend
- `484d76a` Add backend parity harness
- `ba166d6` Stabilize WGPU front path

## Remaining follow-up ideas
- Fuse TimeMix preprocessing (`fused_addcmul` / `fused_k` equivalent) to reduce rounding points.
- Revisit a dedicated CubeCL WKV kernel after front-path parity is already good.
- Replace host-side reference matvecs with faster GPU-side paths only if parity remains within the same tolerance range.
