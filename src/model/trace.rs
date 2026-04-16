use burn::{
    prelude::*,
    tensor::{DType, Tensor},
};

#[derive(Debug, Clone)]
pub struct TensorSnapshot {
    pub name: &'static str,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl TensorSnapshot {
    pub fn from_tensor<B: Backend, const D: usize>(name: &'static str, tensor: &Tensor<B, D>) -> Self {
        let data = tensor.clone().to_data().convert_dtype(DType::F32);
        let values = data.to_vec::<f32>().unwrap_or_else(|_| Vec::new());

        Self {
            name,
            shape: tensor.dims().into_iter().collect(),
            values,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerTrace {
    pub layer_id: usize,
    pub tensors: Vec<TensorSnapshot>,
}

impl LayerTrace {
    pub fn new(layer_id: usize) -> Self {
        Self {
            layer_id,
            tensors: Vec::new(),
        }
    }

    pub fn push<B: Backend, const D: usize>(&mut self, name: &'static str, tensor: &Tensor<B, D>) {
        self.tensors.push(TensorSnapshot::from_tensor(name, tensor));
    }

    pub fn extend(&mut self, tensors: Vec<TensorSnapshot>) {
        self.tensors.extend(tensors);
    }
}

#[derive(Debug, Clone)]
pub struct StepTrace {
    pub token_index: usize,
    pub token: i32,
    pub tensors: Vec<TensorSnapshot>,
    pub layers: Vec<LayerTrace>,
}

impl StepTrace {
    pub fn new(token_index: usize, token: i32) -> Self {
        Self {
            token_index,
            token,
            tensors: Vec::new(),
            layers: Vec::new(),
        }
    }

    pub fn push<B: Backend, const D: usize>(&mut self, name: &'static str, tensor: &Tensor<B, D>) {
        self.tensors.push(TensorSnapshot::from_tensor(name, tensor));
    }
}
