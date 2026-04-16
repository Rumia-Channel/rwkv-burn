use anyhow::{Result, bail};
use burn::backend::{LibTorch, Wgpu};
use rwkv_tokenizer::WorldTokenizer;

use crate::{
    Config,
    load_or_init_model,
    model::{FrontPathStrategy, StepTrace, TensorSnapshot},
};

struct TensorDiff {
    name: String,
    max_abs: f32,
    mean_abs: f32,
    worst_index: usize,
    lhs_value: f32,
    rhs_value: f32,
}

pub fn run(config: &Config) -> Result<()> {
    type StableBackend = LibTorch;
    type CandidateBackend = Wgpu;

    let stable_device = Default::default();
    let candidate_device = Default::default();

    let mut stable_model = load_or_init_model::<StableBackend>(config, &stable_device)?;
    let mut candidate_model = load_or_init_model::<CandidateBackend>(config, &candidate_device)?;
    if config.stable_frontpath {
        candidate_model.set_front_path_strategy(FrontPathStrategy::HostLinear);
    }
    let tokenizer = WorldTokenizer::new(Some(&config.vocab_path))?;

    let prompt = format!("User: {}\n\nAssistant:", config.parity_prompt.trim());
    let mut tokens: Vec<i32> = tokenizer
        .encode(&prompt)
        .iter()
        .map(|&token| i32::from(token))
        .collect();

    if config.parity_max_tokens > 0 && tokens.len() > config.parity_max_tokens {
        tokens.truncate(config.parity_max_tokens);
    }

    if tokens.is_empty() {
        bail!("parity prompt produced no tokens");
    }

    let mut stable_state = stable_model.get_init_state();
    let mut candidate_state = candidate_model.get_init_state();

    let mut worst_diff = 0.0_f32;
    let mut worst_label = String::new();
    let mut mismatch_found = false;

    println!(
        "Comparing LibTorch vs WGPU on {} prompt token(s) with tolerance {}",
        tokens.len(),
        config.parity_tolerance
    );

    for (token_index, token) in tokens.into_iter().enumerate() {
        let (_stable_logits, next_stable_state, stable_trace) =
            stable_model.forward_rnn_traced(token, stable_state, token_index);
        let (_candidate_logits, next_candidate_state, candidate_trace) =
            candidate_model.forward_rnn_traced(token, candidate_state, token_index);

        stable_state = next_stable_state;
        candidate_state = next_candidate_state;

        let decoded = tokenizer
            .decode(vec![token as u16])
            .map(|value| value.replace('\n', "\\n"))
            .unwrap_or_else(|_| format!("<{}>", token));

        let step_worst = compare_step(
            &stable_trace,
            &candidate_trace,
            config.parity_tolerance,
            config.parity_verbose,
        )?;

        let stable_logits = stable_trace
            .tensors
            .iter()
            .find(|snapshot| snapshot.name == "model_logits")
            .ok_or_else(|| anyhow::anyhow!("missing model_logits trace for LibTorch"))?;
        let candidate_logits = candidate_trace
            .tensors
            .iter()
            .find(|snapshot| snapshot.name == "model_logits")
            .ok_or_else(|| anyhow::anyhow!("missing model_logits trace for WGPU"))?;
        let logits_diff = diff_snapshot(stable_logits, candidate_logits)?;

        let step_max = step_worst
            .as_ref()
            .map(|diff| diff.max_abs)
            .unwrap_or(0.0)
            .max(logits_diff.max_abs);
        let step_label = if logits_diff.max_abs >= step_worst.as_ref().map(|item| item.max_abs).unwrap_or(0.0) {
            format!("model_logits (mean_abs={:.6})", logits_diff.mean_abs)
        } else {
            let diff = step_worst.as_ref().unwrap();
            format!(
                "{} (mean_abs={:.6}, idx={}, lhs={:.6}, rhs={:.6})",
                diff.name, diff.mean_abs, diff.worst_index, diff.lhs_value, diff.rhs_value
            )
        };

        println!(
            "token {:>3} {:>6} worst_abs_diff={:.6} at {}",
            token_index, decoded, step_max, step_label
        );

        if step_max > config.parity_tolerance {
            mismatch_found = true;
        }

        if step_max > worst_diff {
            worst_diff = step_max;
            worst_label = format!("token {} {}", token_index, step_label);
        }
    }

    println!("Overall worst_abs_diff={:.6} at {}", worst_diff, worst_label);

    if mismatch_found {
        bail!(
            "backend parity mismatch exceeded tolerance {} (worst {:.6})",
            config.parity_tolerance,
            worst_diff
        );
    }

    Ok(())
}

fn compare_step(
    stable: &StepTrace,
    candidate: &StepTrace,
    _tolerance: f32,
    verbose: bool,
) -> Result<Option<TensorDiff>> {
    if stable.token_index != candidate.token_index || stable.token != candidate.token {
        bail!("step trace mismatch between backends");
    }

    let mut worst: Option<TensorDiff> = None;

    for (lhs, rhs) in stable.tensors.iter().zip(candidate.tensors.iter()) {
        let diff = diff_snapshot(lhs, rhs)?;
        if verbose {
            println!(
                "  model {:<24} max_abs={:.6} mean_abs={:.6}",
                lhs.name, diff.max_abs, diff.mean_abs
            );
        }
        update_worst(&mut worst, diff);
    }

    for (lhs_layer, rhs_layer) in stable.layers.iter().zip(candidate.layers.iter()) {
        if lhs_layer.layer_id != rhs_layer.layer_id {
            bail!("layer trace mismatch between backends");
        }

        for (lhs, rhs) in lhs_layer.tensors.iter().zip(rhs_layer.tensors.iter()) {
            let mut diff = diff_snapshot(lhs, rhs)?;
            let scoped_name = format!("layer{}.{}", lhs_layer.layer_id, lhs.name);
            if verbose {
                println!(
                    "  {:<30} max_abs={:.6} mean_abs={:.6}",
                    scoped_name, diff.max_abs, diff.mean_abs
                );
            }
            diff.name = scoped_name;
            update_worst(&mut worst, diff);
        }
    }

    Ok(worst)
}

fn update_worst(current: &mut Option<TensorDiff>, next: TensorDiff) {
    if current
        .as_ref()
        .map(|value| next.max_abs > value.max_abs)
        .unwrap_or(true)
    {
        *current = Some(next);
    }
}

fn diff_snapshot(lhs: &TensorSnapshot, rhs: &TensorSnapshot) -> Result<TensorDiff> {
    if lhs.name != rhs.name {
        bail!("tensor name mismatch: {} vs {}", lhs.name, rhs.name);
    }

    if lhs.shape != rhs.shape {
        bail!(
            "tensor shape mismatch for {}: {:?} vs {:?}",
            lhs.name,
            lhs.shape,
            rhs.shape
        );
    }

    if lhs.values.len() != rhs.values.len() {
        bail!(
            "tensor value length mismatch for {}: {} vs {}",
            lhs.name,
            lhs.values.len(),
            rhs.values.len()
        );
    }

    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f32;
    let mut worst_index = 0_usize;
    let mut lhs_value = 0.0_f32;
    let mut rhs_value = 0.0_f32;

    for (index, (lhs_item, rhs_item)) in lhs.values.iter().zip(rhs.values.iter()).enumerate() {
        if !lhs_item.is_finite() || !rhs_item.is_finite() {
            bail!("non-finite value detected while comparing {}", lhs.name);
        }

        let abs = (lhs_item - rhs_item).abs();
        sum_abs += abs;

        if abs > max_abs {
            max_abs = abs;
            worst_index = index;
            lhs_value = *lhs_item;
            rhs_value = *rhs_item;
        }
    }

    let mean_abs = if lhs.values.is_empty() {
        0.0
    } else {
        sum_abs / lhs.values.len() as f32
    };

    Ok(TensorDiff {
        name: lhs.name.to_string(),
        max_abs,
        mean_abs,
        worst_index,
        lhs_value,
        rhs_value,
    })
}
