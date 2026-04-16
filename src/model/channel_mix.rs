use burn::{
    module::{Ignored, Param},
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::{DType, Tensor, activation},
};

use super::time_mix::FrontPathStrategy;

/// A feedforward (channel mixing) layer used in RWKV models.
///
/// This struct performs a per-token nonlinear transformation using a two-layer
/// MLP with ReLU squared activation, optionally conditioned on previous input
/// for RNN-style recurrence.
///
/// # Type Parameters
/// * `B` - The backend to use (e.g., NdArray or LibTorch).
#[derive(Module, Debug)]
pub struct ChannelMix<B: Backend> {
    /// Model dimensionality (hidden size).
    pub d_model: usize,
    /// Hidden size of the feedforward layer (usually 4 * d_model).
    pub dim_ffn: usize,
    /// Learnable tensor used to scale the delta between current and previous input.
    pub x_k: Param<Tensor<B, 3>>,
    /// Linear layer for projecting to the intermediate (hidden) dimension.
    pub key: Linear<B>,
    /// Linear layer for projecting back to the original model dimension.
    pub value: Linear<B>,
    front_path: Ignored<FrontPathStrategy>,
}

impl<B: Backend> ChannelMix<B> {
    /// Constructs a new `ChannelMix` module with initialized parameters.
    ///
    /// # Arguments
    /// * `device` - The target device to allocate tensors on.
    /// * `d_model` - The model's hidden dimension size.
    ///
    /// # Returns
    /// A new instance of `ChannelMix`.
    pub fn new(device: &B::Device, d_model: usize) -> ChannelMix<B> {
        let dim_ffn = 4 * d_model;
        let x_k = Param::from_tensor(init_channel_mix_xk::<B>(device, d_model));
        let key = LinearConfig::new(d_model, dim_ffn)
            .with_bias(false)
            .init::<B>(device);
        let value = LinearConfig::new(dim_ffn, d_model)
            .with_bias(false)
            .init::<B>(device);
        let front_path = Ignored(FrontPathStrategy::Direct);

        ChannelMix {
            d_model,
            dim_ffn,
            x_k,
            key,
            value,
            front_path,
        }
    }

    /// Forward pass of the channel mixing layer in parallel (non-recurrent) mode.
    ///
    /// Used during training or inference with full sequences. The input is mixed
    /// with its temporally shifted version using a learnable gating mechanism.
    ///
    /// # Arguments
    /// * `x` - A 3D tensor of shape `[batch_size, time_steps, d_model]`.
    ///
    /// # Returns
    /// A 3D tensor of shape `[batch_size, time_steps, d_model]` representing the transformed output.
    ///
    pub fn forward_parallel(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, time_steps, _] = x.clone().dims();

        let xx = x
            .clone()
            .slice([0..batch_size, 0..time_steps - 1])
            .pad((0, 0, 1, 0), 0);
        let xx = xx - x.clone();

        let k = activation::relu(self.key.forward(x.clone() + xx.mul(self.x_k.val())));
        let k = k.clone().mul(k);

        self.value.forward(k)
    }

    /// Forward pass in recurrent mode, used for step-by-step inference.
    ///
    /// Processes one token at a time by comparing the current and previous hidden states.
    ///
    /// # Arguments
    /// * `x` - The current input tensor of shape `[d_model]`.
    /// * `x_prev` - The previous input tensor of shape `[d_model]`.
    ///
    /// # Returns
    /// A tuple `(output, updated_state)` where:
    /// - `output` is the transformed tensor for this time step.
    /// - `updated_state` is the new value to be used as `x_prev` in the next step.
    pub fn forward_rnn(
        &self,
        x: Tensor<B, 1>,
        x_prev: Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let xx = x_prev - x.clone();
        let k = x.clone() + xx.mul(self.x_k.val().reshape([self.d_model]));
        let k = activation::relu(self.front_linear(&self.key, k)).powf_scalar(2.0);

        (self.front_linear(&self.value, k), x)
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

fn init_channel_mix_xk<B: Backend>(device: &B::Device, d_model: usize) -> Tensor<B, 3> {
    let values = (0..d_model)
        .map(|index| {
            let position = index as f32 / d_model.max(1) as f32;
            1.0 - position
        })
        .collect::<Vec<_>>();

    Tensor::<B, 1>::from_data(values.as_slice(), device).reshape([1, 1, d_model])
}
