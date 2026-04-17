use burn::{
    nn::Linear,
    prelude::*,
    tensor::{DType, Tensor},
};

#[derive(Clone, Copy, Debug, Default)]
pub enum FrontPathStrategy {
    #[default]
    Direct,
    GpuChunkedLinear,
    #[allow(dead_code)]
    HostLinear,
}

pub fn front_linear<B: Backend>(
    strategy: FrontPathStrategy,
    linear: &Linear<B>,
    input: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let [d_input, d_output] = linear.weight.val().dims();
    let output = front_linear_2d(strategy, linear, input.reshape([1, d_input]));
    output.reshape([d_output])
}

pub fn front_linear_3d<B: Backend>(
    strategy: FrontPathStrategy,
    linear: &Linear<B>,
    input: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [batch_size, time_steps, d_input] = input.dims();
    let [_, d_output] = linear.weight.val().dims();
    let output = front_linear_2d(
        strategy,
        linear,
        input.reshape([batch_size * time_steps, d_input]),
    );
    output.reshape([batch_size, time_steps, d_output])
}

fn front_linear_2d<B: Backend>(
    strategy: FrontPathStrategy,
    linear: &Linear<B>,
    input: Tensor<B, 2>,
) -> Tensor<B, 2> {
    match strategy {
        FrontPathStrategy::Direct => linear.forward(input),
        FrontPathStrategy::GpuChunkedLinear => linear_forward_gpu_chunked(linear, input),
        FrontPathStrategy::HostLinear => linear_forward_host(linear, input),
    }
}

fn linear_forward_gpu_chunked<B: Backend>(linear: &Linear<B>, input: Tensor<B, 2>) -> Tensor<B, 2> {
    let [batch_size, _] = input.dims();
    let [d_input, d_output] = linear.weight.val().dims();
    let out_dtype = input.dtype();
    let weight = linear.weight.val().cast(DType::F32);
    let device = weight.device();
    let chunk_size = preferred_chunk_size(d_input, d_output);

    let mut output = linear
        .bias
        .as_ref()
        .map(|bias| {
            bias.val()
                .cast(DType::F32)
                .reshape([1, d_output])
                .repeat_dim(0, batch_size)
        })
        .unwrap_or_else(|| Tensor::<B, 2>::zeros([batch_size, d_output], &device).cast(DType::F32));

    for start in (0..d_input).step_by(chunk_size) {
        let end = (start + chunk_size).min(d_input);
        let input_chunk = input
            .clone()
            .slice([0..batch_size, start..end])
            .cast(DType::F32);
        let weight_chunk = weight.clone().slice([start..end, 0..d_output]);
        let partial = input_chunk.matmul(weight_chunk);
        output = output + partial;
    }

    output.cast(out_dtype)
}

fn preferred_chunk_size(d_input: usize, d_output: usize) -> usize {
    let base = if d_output >= 32768 {
        16
    } else if d_output >= 4096 {
        32
    } else {
        64
    };

    base.min(d_input.max(1))
}

fn linear_forward_host<B: Backend>(linear: &Linear<B>, input: Tensor<B, 2>) -> Tensor<B, 2> {
    let [batch_size, _] = input.dims();
    let device = input.device();
    let input = input
        .to_data()
        .convert_dtype(DType::F32)
        .to_vec::<f32>()
        .unwrap();
    let weight = linear
        .weight
        .val()
        .to_data()
        .convert_dtype(DType::F32)
        .to_vec::<f32>()
        .unwrap();
    let [d_input, d_output] = linear.weight.val().dims();

    let bias = if let Some(bias) = linear.bias.as_ref() {
        bias.val()
            .to_data()
            .convert_dtype(DType::F32)
            .to_vec::<f32>()
            .unwrap()
    } else {
        vec![0.0; d_output]
    };

    let mut output = Vec::with_capacity(batch_size * d_output);

    for batch_index in 0..batch_size {
        let mut row = bias.clone();
        let input_offset = batch_index * d_input;
        for i in 0..d_input {
            let input_value = input[input_offset + i];
            let row_offset = i * d_output;
            for o in 0..d_output {
                row[o] += input_value * weight[row_offset + o];
            }
        }
        output.extend(row);
    }

    Tensor::<B, 1>::from_data(output.as_slice(), &device).reshape([batch_size, d_output])
}
