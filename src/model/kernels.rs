use burn::{
    prelude::*,
    tensor::{Tensor, activation, s},
};

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum KernelStrategy {
    Reference,
    CubeClPrepared,
}

#[derive(Clone, Debug)]
pub struct WkvKernel {
    n_heads: usize,
    head_size: usize,
    strategy: KernelStrategy,
}

impl WkvKernel {
    pub fn new(n_heads: usize, head_size: usize) -> Self {
        Self {
            n_heads,
            head_size,
            strategy: KernelStrategy::CubeClPrepared,
        }
    }

    pub fn forward_parallel<B: Backend>(
        &self,
        r: Tensor<B, 3>,
        w: Tensor<B, 3>,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
        a: Tensor<B, 3>,
        b: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        match self.strategy {
            KernelStrategy::Reference | KernelStrategy::CubeClPrepared => {
                self.forward_parallel_reference(r, w, k, v, a, b)
            }
        }
    }

    pub fn forward_rnn<B: Backend>(
        &self,
        r: Tensor<B, 1>,
        w: Tensor<B, 1>,
        k: Tensor<B, 1>,
        v: Tensor<B, 1>,
        a: Tensor<B, 1>,
        b: Tensor<B, 1>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 1>, Tensor<B, 3>) {
        match self.strategy {
            KernelStrategy::Reference | KernelStrategy::CubeClPrepared => {
                self.forward_rnn_reference(r, w, k, v, a, b, state)
            }
        }
    }

    fn forward_parallel_reference<B: Backend>(
        &self,
        r: Tensor<B, 3>,
        w: Tensor<B, 3>,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
        a: Tensor<B, 3>,
        b: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let [batch_size, time_steps, d_model] = r.clone().dims();

        let r: Tensor<B, 4> = r.reshape([batch_size, time_steps, self.n_heads, self.head_size]);
        let k: Tensor<B, 4> = k.reshape([batch_size, time_steps, self.n_heads, self.head_size]);
        let v: Tensor<B, 4> = v.reshape([batch_size, time_steps, self.n_heads, self.head_size]);
        let a: Tensor<B, 4> =
            a.reshape([batch_size, time_steps, self.n_heads, self.head_size]);
        let b: Tensor<B, 4> =
            b.reshape([batch_size, time_steps, self.n_heads, self.head_size]);
        let w: Tensor<B, 4> = Tensor::exp(-Tensor::exp(w.reshape([
            batch_size,
            time_steps,
            self.n_heads,
            self.head_size,
        ])));

        let mut out: Tensor<B, 4> = Tensor::zeros(
            [batch_size, time_steps, self.n_heads, self.head_size],
            &r.device(),
        );
        let mut state: Tensor<B, 4> = Tensor::zeros(
            [batch_size, self.n_heads, self.head_size, self.head_size],
            &r.device(),
        );

        for t in 0..time_steps {
            let kk = k
                .clone()
                .slice(s![.., t])
                .reshape([batch_size, self.n_heads, 1, self.head_size]);
            let rr = r
                .clone()
                .slice(s![.., t])
                .reshape([batch_size, self.n_heads, self.head_size, 1]);
            let vv = v
                .clone()
                .slice(s![.., t])
                .reshape([batch_size, self.n_heads, self.head_size, 1]);
            let aa = a
                .clone()
                .slice(s![.., t])
                .reshape([batch_size, self.n_heads, self.head_size, 1]);
            let bb = b
                .clone()
                .slice(s![.., t])
                .reshape([batch_size, self.n_heads, 1, self.head_size]);
            let ww = w
                .clone()
                .slice(s![.., t])
                .reshape([batch_size, self.n_heads, 1, self.head_size]);

            state = state.clone().mul(ww) + state.clone().matmul(aa.matmul(bb)) + vv.matmul(kk);

            out = out.slice_assign(
                [0..batch_size, t..t + 1, 0..self.n_heads, 0..self.head_size],
                state
                    .clone()
                    .matmul(rr)
                    .reshape([batch_size, 1, self.n_heads, self.head_size]),
            );
        }

        (out.reshape([batch_size, time_steps, d_model]), state)
    }

    fn forward_rnn_reference<B: Backend>(
        &self,
        r: Tensor<B, 1>,
        w: Tensor<B, 1>,
        k: Tensor<B, 1>,
        v: Tensor<B, 1>,
        a: Tensor<B, 1>,
        b: Tensor<B, 1>,
        state: Tensor<B, 3>,
    ) -> (Tensor<B, 1>, Tensor<B, 3>) {
        let w = activation::sigmoid(w).mul_scalar(-0.606531).exp();

        let vk = v
            .reshape([self.n_heads, self.head_size, 1])
            .matmul(k.reshape([self.n_heads, 1, self.head_size]));
        let ab = a
            .reshape([self.n_heads, self.head_size, 1])
            .matmul(b.reshape([self.n_heads, 1, self.head_size]));

        let state =
            state.clone().mul(w.reshape([self.n_heads, 1, self.head_size])) + state.matmul(ab) + vk;
        let out = state
            .clone()
            .matmul(r.reshape([self.n_heads, self.head_size, 1]))
            .reshape([self.n_heads * self.head_size]);

        (out, state)
    }
}
