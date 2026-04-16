use burn::{
    module::{Ignored, Param},
    nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{DType, Tensor, activation},
};

use super::{
    kernels::WkvKernel,
    trace::TensorSnapshot,
};

#[derive(Clone, Copy, Debug, Default)]
pub enum FrontPathStrategy {
    #[default]
    Direct,
    HostLinear,
}

/// A module implementing the TimeMix layer for sequential processing.
///
/// This layer combines recurrent-style temporal mixing with attention-like computation,
/// using linear layers and tensor-based modulation. It is optimized for efficient inference
/// in both RNN and parallel modes, often used in RWKV-style architectures.
#[derive(Module, Debug)]
pub struct TimeMix<B: Backend> {
    pub layer_id: usize,
    pub n_layer: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_size: usize,
    pub d_decay_lora: usize,
    pub d_aaa_lora: usize,
    pub d_mv_lora: usize,
    pub d_gate_lora: usize,

    pub x_r: Param<Tensor<B, 3>>,
    pub x_w: Param<Tensor<B, 3>>,
    pub x_k: Param<Tensor<B, 3>>,
    pub x_v: Param<Tensor<B, 3>>,
    pub x_a: Param<Tensor<B, 3>>,
    pub x_g: Param<Tensor<B, 3>>,

    pub w0: Param<Tensor<B, 3>>,
    pub w1: Linear<B>,
    pub w2: Linear<B>,

    pub a0: Param<Tensor<B, 3>>,
    pub a1: Linear<B>,
    pub a2: Linear<B>,

    pub v0: Param<Tensor<B, 3>>,
    pub v1: Linear<B>,
    pub v2: Linear<B>,

    pub g1: Linear<B>,
    pub g2: Linear<B>,

    pub k_k: Param<Tensor<B, 3>>,
    pub k_a: Param<Tensor<B, 3>>,
    pub r_k: Param<Tensor<B, 2>>,

    wkv_op: Ignored<WkvKernel>,
    front_path: Ignored<FrontPathStrategy>,

    pub receptance: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub output: Linear<B>,
    pub group_norm: GroupNorm<B>,
}

impl<B: Backend> TimeMix<B> {
    /// Constructs a new `TimeMix` instance with specified model dimensions and LoRA capacities.
    ///
    /// # Arguments
    /// * `device` - The device on which tensors will be allocated.
    /// * `layer_id` - The index of this layer in the overall model.
    /// * `n_layer` - Total number of layers in the model.
    /// * `d_model` - The dimensionality of the input and output representations.
    /// * `n_heads` - Number of attention heads.
    /// * `head_size` - Dimensionality per attention head.
    /// * `d_decay_lora` - LoRA size for the decay weights.
    /// * `d_aaa_lora` - LoRA size for the attention "a" weights.
    /// * `d_mv_lora` - LoRA size for the value mixing weights.
    /// * `d_gate_lora` - LoRA size for the gating weights.
    ///
    /// # Returns
    /// A fully initialized `TimeMix` module.
    pub fn new(
        device: &B::Device,
        layer_id: usize,
        n_layer: usize,
        d_model: usize,
        n_heads: usize,
        head_size: usize,
        d_decay_lora: usize,
        d_aaa_lora: usize,
        d_mv_lora: usize,
        d_gate_lora: usize,
    ) -> TimeMix<B> {
        let ratio_0_to_1 = layer_id as f32 / n_layer.saturating_sub(1).max(1) as f32;
        let ratio_1_to_almost0 = 1.0 - (layer_id as f32 / n_layer.max(1) as f32);

        let x_r =
            Param::from_tensor(init_mix_vector::<B>(device, d_model, 0.2 * ratio_1_to_almost0));
        let x_w =
            Param::from_tensor(init_mix_vector::<B>(device, d_model, 0.9 * ratio_1_to_almost0));
        let x_k =
            Param::from_tensor(init_mix_vector::<B>(device, d_model, 0.7 * ratio_1_to_almost0));
        let x_v =
            Param::from_tensor(init_mix_vector::<B>(device, d_model, 0.7 * ratio_1_to_almost0));
        let x_a =
            Param::from_tensor(init_mix_vector::<B>(device, d_model, 0.9 * ratio_1_to_almost0));
        let x_g =
            Param::from_tensor(init_mix_vector::<B>(device, d_model, 0.2 * ratio_1_to_almost0));

        let w0 = Param::from_tensor(init_w0::<B>(
            device,
            d_model,
            head_size,
            ratio_0_to_1,
        ));
        let w1 = LinearConfig::new(d_model, d_decay_lora)
            .with_bias(false)
            .init::<B>(device);
        let w2 = LinearConfig::new(d_decay_lora, d_model)
            .with_bias(false)
            .init::<B>(device);

        let a0 = Param::from_tensor(init_a0::<B>(device, d_model, head_size));
        let a1 = LinearConfig::new(d_model, d_aaa_lora)
            .with_bias(false)
            .init::<B>(device);
        let a2 = LinearConfig::new(d_aaa_lora, d_model)
            .with_bias(false)
            .init::<B>(device);

        let v0 = Param::from_tensor(init_v0::<B>(device, d_model));
        let v1 = LinearConfig::new(d_model, d_mv_lora)
            .with_bias(false)
            .init::<B>(device);
        let v2 = LinearConfig::new(d_mv_lora, d_model)
            .with_bias(false)
            .init::<B>(device);

        let g1 = LinearConfig::new(d_model, d_gate_lora)
            .with_bias(false)
            .init::<B>(device);
        let g2 = LinearConfig::new(d_gate_lora, d_model)
            .with_bias(false)
            .init::<B>(device);

        let k_k = Param::from_tensor(init_kk::<B>(device, d_model));
        let k_a = Param::from_tensor(init_constant_1x1::<B>(device, d_model, 1.02));
        let r_k = Param::from_tensor(init_constant_2d::<B>(device, n_heads, head_size, -0.04));

        //time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        let receptance = LinearConfig::new(d_model, d_model)
            .with_bias(false)
            .init(device);

        let key = LinearConfig::new(d_model, d_model)
            .with_bias(false)
            .init(device);

        let value = LinearConfig::new(d_model, d_model)
            .with_bias(false)
            .init(device);

        let output = LinearConfig::new(d_model, d_model)
            .with_bias(false)
            .init(device);

        let group_norm = GroupNormConfig::new(n_heads, d_model)
            .with_epsilon(64e-5)
            .init(device);

        let wkv_op = Ignored(WkvKernel::new(n_heads, head_size));
        let front_path = Ignored(FrontPathStrategy::Direct);

        TimeMix {
            layer_id,
            n_layer,
            d_model,
            n_heads,
            head_size,
            d_decay_lora,
            d_aaa_lora,
            d_mv_lora,
            d_gate_lora,
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            w0,
            w1,
            w2,
            a0,
            a1,
            a2,
            v0,
            v1,
            v2,
            g1,
            g2,
            k_k,
            k_a,
            r_k,
            wkv_op,
            front_path,
            receptance,
            key,
            value,
            output,
            group_norm,
        }
    }

    /// Performs a forward pass over a batch of sequences in parallel mode.
    ///
    /// Applies temporal mixing and attention computation across time steps.
    /// This method is optimized for training or evaluation on batched sequence data.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape `[batch_size, time_steps, d_model]`.
    /// * `v_first` - Optional initial value vector for value mixing across steps.
    ///
    /// # Returns
    /// A tuple containing:
    /// * The output tensor of shape `[batch_size, time_steps, d_model]`.
    /// * The updated or original `v_first` tensor.
    pub fn forward_parallel(
        &self,
        x: Tensor<B, 3>,
        mut v_first: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 3>>, Tensor<B, 4>) {
        let [batch_size, time_steps, _] = x.clone().dims();

        let xx = x
            .clone()
            .slice([0..batch_size, 0..time_steps - 1])
            .pad((0, 0, 1, 0), 0);

        let xx = xx - x.clone();

        let xr = x.clone() + xx.clone().mul(self.x_r.val());
        let xw = x.clone() + xx.clone().mul(self.x_w.val());
        let xk = x.clone() + xx.clone().mul(self.x_k.val());
        let xv = x.clone() + xx.clone().mul(self.x_v.val());
        let xa = x.clone() + xx.clone().mul(self.x_a.val());
        let xg = x.clone() + xx.clone().mul(self.x_g.val());

        let r = self.receptance.forward(xr);
        let w = -activation::softplus(
            -(self.w0.val() + self.w2.forward(activation::tanh(self.w1.forward(xw)))),
            1.0,
        ) - 0.5;

        let mut v = self.value.forward(xv.clone());

        if let Some(_v_first) = v_first.clone() {
            v = v.clone()
                + (_v_first.clone() - v.clone()).mul(activation::sigmoid(
                    self.v0.val() + self.v2.forward(self.v1.forward(xv.clone())),
                ));
        } else {
            v_first = Some(v.clone());
        }

        let a = activation::sigmoid(self.a0.val() + self.a2.forward(self.a1.forward(xa)));
        let g = self.g2.forward(activation::sigmoid(self.g1.forward(xg)));

        let k = self.key.forward(xk);
        let kk = k.clone().mul(self.k_k.val());
        let kk = kk.reshape([batch_size, time_steps, self.n_heads, self.head_size]);
        let kk_norm = kk
            .clone()
            .mul(kk.clone())
            .sum_dim(3)
            .add_scalar(1e-12)
            .sqrt()
            .reshape([batch_size, time_steps, self.n_heads, 1]);
        let kk = kk.clone().div(kk_norm).reshape([batch_size, time_steps, self.d_model]);
        let k = k
            .clone()
            .mul((a.clone() - 1).mul(self.k_a.val()).add_scalar(1.0));

        let (x, state): (Tensor<B, 3>, Tensor<B, 4>) =
            self.wkv_op
                .clone()
                .forward_parallel(r.clone(), w, k.clone(), v.clone(), -kk.clone(), kk.mul(a));

        let x = self
            .group_norm
            .forward(x.reshape([batch_size * time_steps, self.d_model]))
            .reshape([batch_size, time_steps, self.d_model]);

        let x = x
            + ((r
                .reshape([batch_size, time_steps, self.n_heads, self.head_size])
                .mul(
                    k.reshape([batch_size, time_steps, self.n_heads, self.head_size])
                        .mul(self.r_k.val().unsqueeze_dims(&[0, 1])),
                ))
            .sum_dim(3)
            .mul(v.reshape([
                batch_size,
                time_steps,
                self.n_heads,
                self.head_size,
            ])))
            .reshape([batch_size, time_steps, self.d_model]);
        let x = self.output.forward(x.mul(g));

        (x, v_first, state)
    }

    /// Performs a forward pass over a single time step in recurrent (RNN) mode.
    ///
    /// This is used during inference when sequence elements are processed one at a time.
    /// Maintains internal state across time steps for temporal consistency.
    ///
    /// # Arguments
    /// * `x` - Current input vector at time `t`, shape `[d_model]`.
    /// * `x_prev` - Previous input vector at time `t-1`, shape `[d_model]`.
    /// * `v_first` - Optional reference value for value mixing.
    /// * `vk_state` - Accumulated attention state, shape `[n_heads, head_size, head_size]`.
    ///
    /// # Returns
    /// A tuple containing:
    /// * The new output vector, shape `[d_model]`.
    /// * The current input.
    /// * The updated attention state.
    /// * The updated or original `v_first` value.
    pub fn forward_rnn(
        &self,
        x: Tensor<B, 1>,
        x_prev: Tensor<B, 1>,
        mut v_first: Option<Tensor<B, 1>>,
        vk_state: Tensor<B, 3>,
    ) -> (
        Tensor<B, 1>,         // x_new
        Tensor<B, 1>,         // x
        Tensor<B, 3>,         // vk_state
        Option<Tensor<B, 1>>, // v_first
    ) {
        let x = x.reshape([self.d_model]);
        let x_prev = x_prev.reshape([self.d_model]);

        let xx = x_prev - x.clone();

        let xr = x.clone() + xx.clone().mul(self.x_r.val().reshape([self.d_model]));
        let xw = x.clone() + xx.clone().mul(self.x_w.val().reshape([self.d_model]));
        let xk = x.clone() + xx.clone().mul(self.x_k.val().reshape([self.d_model]));
        let xv = x.clone() + xx.clone().mul(self.x_v.val().reshape([self.d_model]));
        let xa = x.clone() + xx.clone().mul(self.x_a.val().reshape([self.d_model]));
        let xg = x.clone() + xx.clone().mul(self.x_g.val().reshape([self.d_model]));

        let r = self.front_linear(&self.receptance, xr);
        let w = self.w2.forward(activation::tanh(self.w1.forward(xw)));
        let k = self.front_linear(&self.key, xk);
        let mut v = self.front_linear(&self.value, xv.clone());
        let a = activation::sigmoid(
            self.a0.val().reshape([self.d_model]) + self.a2.forward(self.a1.forward(xa)),
        );
        let g = self.g2.forward(activation::sigmoid(self.g1.forward(xg)));

        let kk = k.clone().mul(self.k_k.val().reshape([self.d_model]));
        let kk = kk.reshape([self.n_heads, self.head_size]);
        let kk = kk
            .clone()
            .div(
                kk.clone()
                    .mul(kk.clone())
                    .sum_dim(1)
                    .add_scalar(1e-12)
                    .sqrt()
                    .reshape([self.n_heads, 1]),
            )
            .reshape([self.d_model]);
        let k = k.clone().mul(
            (a.clone() - 1)
                .mul(self.k_a.val().reshape([self.d_model]))
                .add_scalar(1.0),
        );

        if let Some(_v_first) = v_first.clone() {
            v = v.clone()
                + (_v_first - v.clone()).mul(activation::sigmoid(
                    self.v0.val().reshape([self.d_model])
                        + self.v2.forward(self.v1.forward(xv.clone())),
                ));
        } else {
            v_first = Some(v.clone());
        }

        let w = w + self.w0.val().reshape([self.d_model]);
        let (out, vk_state) = self.wkv_op.forward_rnn(
            r.clone(),
            w,
            k.clone(),
            v.clone(),
            -kk.clone(),
            kk.clone().mul(a.clone()),
            vk_state,
        );

        let out = self
            .group_norm
            .forward(out.reshape([1, self.d_model]))
            .reshape([self.d_model]);

        let out = out
            + r.mul(k)
                .mul(self.r_k.val().reshape([self.d_model]))
                .reshape([self.n_heads, self.head_size])
                .sum_dim(1)
                .mul(v.reshape([self.n_heads, self.head_size]))
                .reshape([self.d_model]);

        let out = self.front_linear(&self.output, out.mul(g));

        (out, x, vk_state, v_first)
    }

    pub fn forward_rnn_traced(
        &self,
        x: Tensor<B, 1>,
        x_prev: Tensor<B, 1>,
        mut v_first: Option<Tensor<B, 1>>,
        vk_state: Tensor<B, 3>,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 3>,
        Option<Tensor<B, 1>>,
        Vec<TensorSnapshot>,
    ) {
        let x = x.reshape([self.d_model]);
        let x_prev = x_prev.reshape([self.d_model]);
        let mut trace = Vec::new();

        trace.push(TensorSnapshot::from_tensor("tmix_input", &x));
        trace.push(TensorSnapshot::from_tensor("tmix_x_prev", &x_prev));
        trace.push(TensorSnapshot::from_tensor("tmix_vk_state_in", &vk_state));

        let xx = x_prev.clone() - x.clone();
        trace.push(TensorSnapshot::from_tensor("tmix_xx", &xx));

        let xr = x.clone() + xx.clone().mul(self.x_r.val().reshape([self.d_model]));
        let xw = x.clone() + xx.clone().mul(self.x_w.val().reshape([self.d_model]));
        let xk = x.clone() + xx.clone().mul(self.x_k.val().reshape([self.d_model]));
        let xv = x.clone() + xx.clone().mul(self.x_v.val().reshape([self.d_model]));
        let xa = x.clone() + xx.clone().mul(self.x_a.val().reshape([self.d_model]));
        let xg = x.clone() + xx.clone().mul(self.x_g.val().reshape([self.d_model]));

        trace.push(TensorSnapshot::from_tensor("tmix_xr", &xr));
        trace.push(TensorSnapshot::from_tensor("tmix_xw", &xw));
        trace.push(TensorSnapshot::from_tensor("tmix_xk", &xk));
        trace.push(TensorSnapshot::from_tensor("tmix_xv", &xv));
        trace.push(TensorSnapshot::from_tensor("tmix_xa", &xa));
        trace.push(TensorSnapshot::from_tensor("tmix_xg", &xg));

        let r = self.front_linear(&self.receptance, xr);
        let w_delta = self.w2.forward(activation::tanh(self.w1.forward(xw)));
        let k_raw = self.front_linear(&self.key, xk);
        let mut v = self.front_linear(&self.value, xv.clone());
        let a = activation::sigmoid(
            self.a0.val().reshape([self.d_model]) + self.a2.forward(self.a1.forward(xa)),
        );
        let g = self.g2.forward(activation::sigmoid(self.g1.forward(xg)));

        trace.push(TensorSnapshot::from_tensor("tmix_r", &r));
        trace.push(TensorSnapshot::from_tensor("tmix_w_delta", &w_delta));
        trace.push(TensorSnapshot::from_tensor("tmix_k_raw", &k_raw));
        trace.push(TensorSnapshot::from_tensor("tmix_v_raw", &v));
        trace.push(TensorSnapshot::from_tensor("tmix_a", &a));
        trace.push(TensorSnapshot::from_tensor("tmix_g", &g));

        let kk = k_raw.clone().mul(self.k_k.val().reshape([self.d_model]));
        let kk = kk.reshape([self.n_heads, self.head_size]);
        let kk = kk
            .clone()
            .div(
                kk.clone()
                    .mul(kk.clone())
                    .sum_dim(1)
                    .add_scalar(1e-12)
                    .sqrt()
                    .reshape([self.n_heads, 1]),
            )
            .reshape([self.d_model]);
        let k = k_raw.clone().mul(
            (a.clone() - 1)
                .mul(self.k_a.val().reshape([self.d_model]))
                .add_scalar(1.0),
        );

        trace.push(TensorSnapshot::from_tensor("tmix_kk", &kk));
        trace.push(TensorSnapshot::from_tensor("tmix_k", &k));

        if let Some(_v_first) = v_first.clone() {
            v = v.clone()
                + (_v_first - v.clone()).mul(activation::sigmoid(
                    self.v0.val().reshape([self.d_model])
                        + self.v2.forward(self.v1.forward(xv.clone())),
                ));
        } else {
            v_first = Some(v.clone());
        }

        trace.push(TensorSnapshot::from_tensor("tmix_v", &v));
        if let Some(v_first) = v_first.as_ref() {
            trace.push(TensorSnapshot::from_tensor("tmix_v_first", v_first));
        }

        let w = w_delta + self.w0.val().reshape([self.d_model]);
        let w_decay = activation::sigmoid(w.clone()).mul_scalar(-0.606531).exp();
        trace.push(TensorSnapshot::from_tensor("tmix_w_input", &w));
        trace.push(TensorSnapshot::from_tensor("tmix_w_decay", &w_decay));

        let (out, vk_state) = self.wkv_op.forward_rnn(
            r.clone(),
            w,
            k.clone(),
            v.clone(),
            -kk.clone(),
            kk.clone().mul(a.clone()),
            vk_state,
        );
        trace.push(TensorSnapshot::from_tensor("tmix_wkv_out", &out));
        trace.push(TensorSnapshot::from_tensor("tmix_vk_state_out", &vk_state));

        let out = self
            .group_norm
            .forward(out.reshape([1, self.d_model]))
            .reshape([self.d_model]);
        trace.push(TensorSnapshot::from_tensor("tmix_post_group_norm", &out));

        let out = out
            + r.mul(k)
                .mul(self.r_k.val().reshape([self.d_model]))
                .reshape([self.n_heads, self.head_size])
                .sum_dim(1)
                .mul(v.reshape([self.n_heads, self.head_size]))
                .reshape([self.d_model]);

        let out = self.front_linear(&self.output, out.mul(g));
        trace.push(TensorSnapshot::from_tensor("tmix_output", &out));

        (out, x, vk_state, v_first, trace)
    }

    pub fn set_front_path_strategy(&mut self, strategy: FrontPathStrategy) {
        self.front_path = Ignored(strategy);
    }

    fn front_linear(&self, linear: &Linear<B>, input: Tensor<B, 1>) -> Tensor<B, 1> {
        match self.front_path.0 {
            FrontPathStrategy::Direct => linear.forward(input),
            FrontPathStrategy::HostLinear => self.linear_forward_host(linear, input),
        }
    }

    fn linear_forward_host(&self, linear: &Linear<B>, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let device = input.device();
        let input = input.to_data().convert_dtype(DType::F32).to_vec::<f32>().unwrap();
        let weight = linear
            .weight
            .val()
            .to_data()
            .convert_dtype(DType::F32)
            .to_vec::<f32>()
            .unwrap();
        let [d_input, d_output] = linear.weight.val().dims();

        let mut output = if let Some(bias) = linear.bias.as_ref() {
            bias.val()
                .to_data()
                .convert_dtype(DType::F32)
                .to_vec::<f32>()
                .unwrap()
        } else {
            vec![0.0; d_output]
        };

        for i in 0..d_input {
            let input_value = input[i];
            let row_offset = i * d_output;
            for o in 0..d_output {
                output[o] += input_value * weight[row_offset + o];
            }
        }

        Tensor::<B, 1>::from_data(output.as_slice(), &device)
    }
}

fn init_mix_vector<B: Backend>(device: &B::Device, d_model: usize, exponent: f32) -> Tensor<B, 3> {
    let values = (0..d_model)
        .map(|index| {
            let position = index as f32 / d_model.max(1) as f32;
            1.0 - position.powf(exponent.max(1e-6))
        })
        .collect::<Vec<_>>();

    Tensor::<B, 1>::from_data(values.as_slice(), device).reshape([1, 1, d_model])
}

fn init_w0<B: Backend>(
    device: &B::Device,
    d_model: usize,
    head_size: usize,
    ratio_0_to_1: f32,
) -> Tensor<B, 3> {
    let denom = d_model.saturating_sub(1).max(1) as f32;
    let head_denom = head_size.saturating_sub(1).max(1) as f32;
    let values = (0..d_model)
        .map(|index| {
            let linear = index as f32 / denom;
            let zigzag = if head_size > 1 {
                let centered = ((index % head_size) as f32 - head_denom / 2.0) / (head_denom / 2.0);
                centered * centered.abs()
            } else {
                0.0
            };
            let curve = -6.0 + 6.0 * linear.powf(1.0 + ratio_0_to_1.powf(0.3));
            curve + 0.5 + zigzag * 2.5
        })
        .collect::<Vec<_>>();

    Tensor::<B, 1>::from_data(values.as_slice(), device).reshape([1, 1, d_model])
}

fn init_a0<B: Backend>(device: &B::Device, d_model: usize, head_size: usize) -> Tensor<B, 3> {
    let denom = d_model.saturating_sub(1).max(1) as f32;
    let head_denom = head_size.saturating_sub(1).max(1) as f32;
    let values = (0..d_model)
        .map(|index| {
            let linear = index as f32 / denom - 0.5;
            let zigzag = if head_size > 1 {
                let centered = ((index % head_size) as f32 - head_denom / 2.0) / (head_denom / 2.0);
                centered * centered.abs()
            } else {
                0.0
            };
            -0.19 + zigzag * 0.3 + linear * 0.4
        })
        .collect::<Vec<_>>();

    Tensor::<B, 1>::from_data(values.as_slice(), device).reshape([1, 1, d_model])
}

fn init_v0<B: Backend>(device: &B::Device, d_model: usize) -> Tensor<B, 3> {
    let denom = d_model.saturating_sub(1).max(1) as f32;
    let values = (0..d_model)
        .map(|index| 0.73 - (index as f32 / denom - 0.5) * 0.4)
        .collect::<Vec<_>>();

    Tensor::<B, 1>::from_data(values.as_slice(), device).reshape([1, 1, d_model])
}

fn init_kk<B: Backend>(device: &B::Device, d_model: usize) -> Tensor<B, 3> {
    let denom = d_model.saturating_sub(1).max(1) as f32;
    let values = (0..d_model)
        .map(|index| 0.71 - (index as f32 / denom - 0.5) * 0.1)
        .collect::<Vec<_>>();

    Tensor::<B, 1>::from_data(values.as_slice(), device).reshape([1, 1, d_model])
}

fn init_constant_1x1<B: Backend>(device: &B::Device, d_model: usize, value: f32) -> Tensor<B, 3> {
    let values = vec![value; d_model];
    Tensor::<B, 1>::from_data(values.as_slice(), device).reshape([1, 1, d_model])
}

fn init_constant_2d<B: Backend>(
    device: &B::Device,
    rows: usize,
    cols: usize,
    value: f32,
) -> Tensor<B, 2> {
    let values = vec![value; rows * cols];
    Tensor::<B, 1>::from_data(values.as_slice(), device).reshape([rows, cols])
}
