// scirust-core/src/tensor/tensor3d.rs
// Tensor3D wrapper (B, T, D) — layout row-major 2D (B*T, D)

use crate::autodiff::reverse::{Tape, Tensor, Var};

#[derive(Debug, Clone)]
pub struct Tensor3D {
    pub inner: Tensor, // (B*T, D)
    pub batch: usize,
    pub seq_len: usize,
    pub d_model: usize,
}

impl Tensor3D {
    pub fn new(inner: Tensor, batch: usize, seq_len: usize, d_model: usize) -> Self {
        Self {
            inner,
            batch,
            seq_len,
            d_model,
        }
    }

    pub fn zeros(batch: usize, seq_len: usize, d_model: usize) -> Self {
        Self {
            inner: Tensor::zeros(batch * seq_len, d_model),
            batch,
            seq_len,
            d_model,
        }
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.batch, self.seq_len, self.d_model)
    }
}

/// Variable 3D sur la tape.
#[derive(Debug, Clone, Copy)]
pub struct Var3D<'t> {
    pub var: Var<'t>,
    pub batch: usize,
    pub seq_len: usize,
    pub d_model: usize,
}

impl<'t> Var3D<'t> {
    pub fn new(var: Var<'t>, batch: usize, seq_len: usize, d_model: usize) -> Self {
        Self {
            var,
            batch,
            seq_len,
            d_model,
        }
    }

    pub fn input_3d(tape: &'t Tape, t3d: Tensor3D) -> Self {
        let var = tape.input(t3d.inner.clone());
        Self::new(var, t3d.batch, t3d.seq_len, t3d.d_model)
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.batch, self.seq_len, self.d_model)
    }
}

impl<'t> Var3D<'t> {
    pub fn as_var(&self) -> Var<'t> {
        self.var
    }

    pub fn from_var(var: Var<'t>, batch: usize, seq_len: usize, d_model: usize) -> Self {
        Self {
            var,
            batch,
            seq_len,
            d_model,
        }
    }
}
