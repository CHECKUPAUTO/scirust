// scirust-core/src/autodiff/reverse.rs
//
// Reverse-mode automatic differentiation — implémentation complète et fonctionnelle.
//
// CONTRAT DE CE FICHIER :
//   - Forward construit une "tape" de nodes, chacun avec son Op
//   - backward(idx) propage les gradients depuis le node idx vers tous ses ancêtres
//   - L'Op enregistrée DOIT correspondre à l'opération calculée (pas de mensonge)
//   - Chaque variante Op a son match arm dans propagate()
//
// VALIDATION : le test optimizer_minimizes_x_squared en bas du fichier
// est l'oracle. Si ce test passe, l'autograd marche. Si non, quelque chose
// est cassé fondamentalement.

use std::cell::RefCell;

// ==================================================================== //
//  Tensor — bloc de données 2D row-major                                //
// ==================================================================== //

#[derive(Debug, Clone)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_vec(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols,
            "Tensor::from_vec: len {} != rows·cols {}", data.len(), rows * cols);
        Self { rows, cols, data }
    }

    pub fn shape(&self) -> (usize, usize) { (self.rows, self.cols) }
    pub fn dims(&self)  -> (usize, usize) { (self.rows, self.cols) }
    pub fn nrows(&self) -> usize { self.rows }
    pub fn ncols(&self) -> usize { self.cols }

    /// Hadamard product (element-wise multiplication).
    pub fn hadamard(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape(), other.shape(),
            "hadamard: shapes mismatch {:?} vs {:?}", self.shape(), other.shape());
        let mut out = self.clone();
        for i in 0..out.data.len() {
            out.data[i] *= other.data[i];
        }
        out
    }

    pub fn sum(&self) -> f32 { self.data.iter().sum() }

    pub fn scale(&self, s: f32) -> Tensor {
        let mut out = self.clone();
        for x in &mut out.data { *x *= s; }
        out
    }

    pub fn reshape(&self, rows: usize, cols: usize) -> Tensor {
        assert_eq!(self.data.len(), rows * cols, "reshape: size mismatch");
        Tensor { rows, cols, data: self.data.clone() }
    }
}

impl Default for Tensor {
    fn default() -> Self { Self::zeros(1, 1) }
}

// ==================================================================== //
//  DeviceTensor — wrapper pour permettre future extension GPU           //
// ==================================================================== //

#[derive(Debug, Clone)]
pub struct DeviceTensor {
    pub inner: Tensor,
}

impl DeviceTensor {
    pub fn cpu(t: Tensor) -> Self { Self { inner: t } }
    pub fn as_cpu(&self) -> &Tensor { &self.inner }
    pub fn shape(&self) -> (usize, usize) { self.inner.shape() }
    pub fn scalar_value(&self) -> f32 { self.inner.data.iter().sum() }
}

// ==================================================================== //
//  SavedData — données auxiliaires nécessaires pour certains backwards //
// ==================================================================== //

#[derive(Debug, Clone)]
pub enum SavedData {
    None,
    Indices(Vec<u32>),    // pour Embedding
}

// ==================================================================== //
//  Op — opérations supportées par la tape                              //
//                                                                       //
//  RÈGLE D'OR : la variante Op enregistrée doit correspondre exactement //
//  à l'opération calculée. Pas de mensonge. Si l'enum dit Op::Sigmoid,  //
//  alors le backward de Sigmoid sera appelé.                            //
// ==================================================================== //

#[derive(Debug, Clone)]
pub enum Op {
    Input,                          // valeur d'entrée, pas de parents

    // Arithmétique élémentaire (binaire, mêmes shapes)
    Add(usize, usize),              // a + b
    Sub(usize, usize),              // a - b
    Mul(usize, usize),              // a ⊙ b (Hadamard, élément par élément)

    // Arithmétique avec scalaire
    Scale(usize, f32),              // a * s

    // Unaires
    Neg(usize),                     // -a
    Exp(usize),                     // exp(a)
    Log(usize),                     // log(a)
    Sqrt(usize),                    // sqrt(a)
    Sigmoid(usize),                 // 1 / (1 + exp(-a))
    ReLU(usize),                    // max(a, 0)

    // Réductions
    Sum(usize),                     // somme tous les éléments → (1, 1)

    // Algèbre matricielle
    MatMul(usize, usize),           // (M,K) @ (K,N) → (M,N)
    Transpose(usize),               // (M,N) → (N,M)

    // Broadcasting
    AddBias(usize, usize),          // a (M,N) + b (1,N), broadcast row-wise
}

// ==================================================================== //
//  Node — un node de la tape                                            //
// ==================================================================== //

#[derive(Debug, Clone)]
pub struct Node {
    pub op:    Op,
    pub shape: (usize, usize),
    pub saved: SavedData,
}

// ==================================================================== //
//  Tape — le graphe de calcul                                          //
// ==================================================================== //

#[derive(Debug)]
pub struct Tape {
    pub(crate) nodes:  RefCell<Vec<Node>>,
    pub(crate) values: RefCell<Vec<DeviceTensor>>,
    pub(crate) grads:  RefCell<Vec<Tensor>>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            nodes:  RefCell::new(Vec::new()),
            values: RefCell::new(Vec::new()),
            grads:  RefCell::new(Vec::new()),
        }
    }

    /// Push un nouvel input sur la tape (paramètre ou donnée d'entrée).
    pub fn input(&self, t: Tensor) -> Var {
        let idx = self.push(Op::Input, DeviceTensor::cpu(t), SavedData::None);
        Var { tape: self, idx }
    }

    /// Push un node interne. Crée le grad initialisé à zéro.
    pub(crate) fn push(&self, op: Op, value: DeviceTensor, saved: SavedData) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let idx = nodes.len();
        let shape = value.shape();
        nodes.push(Node { op, shape, saved });
        self.values.borrow_mut().push(value);
        self.grads.borrow_mut().push(Tensor::zeros(shape.0, shape.1));
        idx
    }

    /// Lit la valeur courante du node idx.
    pub fn value(&self, idx: usize) -> Tensor {
        self.values.borrow()[idx].as_cpu().clone()
    }

    /// Écrit une nouvelle valeur sur le node idx (utilisé par l'optimiseur).
    pub fn set_value(&self, idx: usize, value: Tensor) {
        self.values.borrow_mut()[idx] = DeviceTensor::cpu(value);
    }

    /// Lit le gradient courant accumulé sur le node idx.
    pub fn grad(&self, idx: usize) -> Tensor {
        self.grads.borrow()[idx].clone()
    }

    /// Force le gradient d'un node (utilisé pour le seed du backward).
    pub fn set_grad(&self, idx: usize, g: Tensor) {
        self.grads.borrow_mut()[idx] = g;
    }

    /// Réinitialise tous les gradients à zéro.
    pub fn zero_grad(&self) {
        let mut grads = self.grads.borrow_mut();
        for g in grads.iter_mut() {
            for x in &mut g.data { *x = 0.0; }
        }
    }

    pub fn shape(&self, idx: usize) -> (usize, usize) {
        self.values.borrow()[idx].shape()
    }

    pub fn num_nodes(&self) -> usize { self.nodes.borrow().len() }

    /// **Cœur de l'autograd.** Lance la propagation des gradients depuis
    /// le node `output_idx` vers tous ses ancêtres.
    ///
    /// Conditions :
    ///   - L'output doit être scalaire (shape (1, 1))
    ///   - Sinon, utiliser set_grad() avant pour seeder un gradient custom
    pub fn backward(&self, output_idx: usize) {
        // Seed : grad de l'output = 1 partout
        let out_shape = self.shape(output_idx);
        let mut seed = Tensor::zeros(out_shape.0, out_shape.1);
        for x in &mut seed.data { *x = 1.0; }
        self.set_grad(output_idx, seed);

        // Propage en ordre inverse depuis output_idx vers 0.
        // Tous les ancêtres potentiels sont à des indices < output_idx
        // (parce que les nodes sont créés dans l'ordre d'exécution forward).
        for i in (0..=output_idx).rev() {
            self.propagate(i);
        }
    }

    /// Propage le gradient du node `i` vers ses parents selon son Op.
    ///
    /// CONVENTION : grad_out = grads[i]. Pour chaque parent p, on calcule
    /// grad_p_contribution = ∂i/∂p · grad_out, et on l'AJOUTE à grads[p]
    /// (accumulation, pas remplacement, parce qu'un node peut avoir plusieurs
    /// successeurs qui contribuent à son gradient).
    fn propagate(&self, i: usize) {
        let op = self.nodes.borrow()[i].op.clone();
        let grad_out = self.grads.borrow()[i].clone();

        match op {
            Op::Input => {
                // Pas de parents — feuille du graphe. Rien à propager.
            }

            Op::Add(a, b) => {
                // c = a + b → ∂c/∂a = 1, ∂c/∂b = 1
                self.accumulate(a, &grad_out);
                self.accumulate(b, &grad_out);
            }

            Op::Sub(a, b) => {
                // c = a - b → ∂c/∂a = 1, ∂c/∂b = -1
                self.accumulate(a, &grad_out);
                let neg = grad_out.scale(-1.0);
                self.accumulate(b, &neg);
            }

            Op::Mul(a, b) => {
                // c[i] = a[i] * b[i] → ∂c/∂a[i] = b[i], ∂c/∂b[i] = a[i]
                let a_val = self.value(a);
                let b_val = self.value(b);
                let grad_a = grad_out.hadamard(&b_val);
                let grad_b = grad_out.hadamard(&a_val);
                self.accumulate(a, &grad_a);
                self.accumulate(b, &grad_b);
            }

            Op::Scale(a, s) => {
                // c = a * s → ∂c/∂a = s
                let grad_a = grad_out.scale(s);
                self.accumulate(a, &grad_a);
            }

            Op::Neg(a) => {
                // c = -a → ∂c/∂a = -1
                let grad_a = grad_out.scale(-1.0);
                self.accumulate(a, &grad_a);
            }

            Op::Exp(a) => {
                // c = exp(a) → ∂c/∂a = exp(a) = c
                let c_val = self.value(i);
                let grad_a = grad_out.hadamard(&c_val);
                self.accumulate(a, &grad_a);
            }

            Op::Log(a) => {
                // c = log(a) → ∂c/∂a = 1/a
                let a_val = self.value(a);
                let mut inv = a_val.clone();
                for x in &mut inv.data { *x = 1.0 / *x; }
                let grad_a = grad_out.hadamard(&inv);
                self.accumulate(a, &grad_a);
            }

            Op::Sqrt(a) => {
                // c = sqrt(a) → ∂c/∂a = 1/(2·sqrt(a)) = 1/(2c)
                let c_val = self.value(i);
                let mut factor = c_val.clone();
                for x in &mut factor.data { *x = 1.0 / (2.0 * *x); }
                let grad_a = grad_out.hadamard(&factor);
                self.accumulate(a, &grad_a);
            }

            Op::Sigmoid(a) => {
                // c = sigmoid(a) → ∂c/∂a = c · (1 - c)
                let c_val = self.value(i);
                let mut factor = c_val.clone();
                for x in &mut factor.data { *x = *x * (1.0 - *x); }
                let grad_a = grad_out.hadamard(&factor);
                self.accumulate(a, &grad_a);
            }

            Op::ReLU(a) => {
                // c = max(a, 0) → ∂c/∂a = 1 si a>0, 0 sinon
                let a_val = self.value(a);
                let mut mask = a_val.clone();
                for x in &mut mask.data { *x = if *x > 0.0 { 1.0 } else { 0.0 }; }
                let grad_a = grad_out.hadamard(&mask);
                self.accumulate(a, &grad_a);
            }

            Op::Sum(a) => {
                // c = sum_i(a_i) (scalaire) → ∂c/∂a_i = 1 partout
                // grad_out est (1,1) avec une seule valeur, on la diffuse sur a
                let g_scalar = grad_out.data[0];
                let a_shape = self.shape(a);
                let mut grad_a = Tensor::zeros(a_shape.0, a_shape.1);
                for x in &mut grad_a.data { *x = g_scalar; }
                self.accumulate(a, &grad_a);
            }

            Op::MatMul(a, b) => {
                // C = A @ B avec A (M,K), B (K,N), C (M,N)
                // ∂C/∂A = grad_out @ B^T   shape (M, K)
                // ∂C/∂B = A^T @ grad_out   shape (K, N)
                let a_val = self.value(a);
                let b_val = self.value(b);
                let grad_a = matmul(&grad_out, &transpose(&b_val));
                let grad_b = matmul(&transpose(&a_val), &grad_out);
                self.accumulate(a, &grad_a);
                self.accumulate(b, &grad_b);
            }

            Op::Transpose(a) => {
                // c = a^T → grad_a = grad_out^T
                let grad_a = transpose(&grad_out);
                self.accumulate(a, &grad_a);
            }

            Op::AddBias(a, b) => {
                // C = A + broadcast(b) où A est (M,N) et b est (1,N)
                // grad_a = grad_out (même shape que A)
                // grad_b = sum_axis_0(grad_out) (somme sur les rows)
                self.accumulate(a, &grad_out);
                let b_shape = self.shape(b);
                let mut grad_b = Tensor::zeros(b_shape.0, b_shape.1);
                for r in 0..grad_out.rows {
                    for c in 0..grad_out.cols {
                        grad_b.data[c] += grad_out.data[r * grad_out.cols + c];
                    }
                }
                self.accumulate(b, &grad_b);
            }
        }
    }

    /// Accumule un gradient sur le node `idx`.
    fn accumulate(&self, idx: usize, contribution: &Tensor) {
        let mut grads = self.grads.borrow_mut();
        let g = &mut grads[idx];
        assert_eq!(g.shape(), contribution.shape(),
            "accumulate: shape mismatch {:?} vs {:?} (idx={})",
            g.shape(), contribution.shape(), idx);
        for i in 0..g.data.len() {
            g.data[i] += contribution.data[i];
        }
    }
}

impl Default for Tape {
    fn default() -> Self { Self::new() }
}

// ==================================================================== //
//  Helpers : matmul et transpose libres                                 //
// ==================================================================== //

/// Matmul standard 2D, (M,K) @ (K,N) → (M,N).
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, k) = a.shape();
    let (k2, n) = b.shape();
    assert_eq!(k, k2, "matmul: inner dims mismatch ({:?} @ {:?})", a.shape(), b.shape());
    let mut out = Tensor::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += a.data[i * k + kk] * b.data[kk * n + j];
            }
            out.data[i * n + j] = s;
        }
    }
    out
}

/// Transpose 2D, (M,N) → (N,M).
pub fn transpose(a: &Tensor) -> Tensor {
    let (m, n) = a.shape();
    let mut out = Tensor::zeros(n, m);
    for i in 0..m {
        for j in 0..n {
            out.data[j * m + i] = a.data[i * n + j];
        }
    }
    out
}

// ==================================================================== //
//  Var — handle vers un node de la tape                                 //
//                                                                       //
//  Toutes les méthodes Var poussent la BONNE Op, pas un substitut.     //
// ==================================================================== //

#[derive(Debug, Clone, Copy)]
pub struct Var<'t> {
    pub tape: &'t Tape,
    pub idx:  usize,
}

impl<'t> Var<'t> {
    pub fn new(tape: &'t Tape, idx: usize) -> Self { Self { tape, idx } }
    pub fn idx(&self) -> usize { self.idx }
    pub fn shape(&self) -> (usize, usize) {
        self.tape.values.borrow()[self.idx].shape()
    }

    // ---------- Arithmétique binaire ---------- //

    pub fn add(self, other: Var<'t>) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let b = self.tape.value(other.idx);
        assert_eq!(a.shape(), b.shape(), "add: shapes mismatch");
        let mut out = a.clone();
        for i in 0..out.data.len() { out.data[i] += b.data[i]; }
        let idx = self.tape.push(Op::Add(self.idx, other.idx),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }

    pub fn sub(self, other: Var<'t>) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let b = self.tape.value(other.idx);
        assert_eq!(a.shape(), b.shape(), "sub: shapes mismatch");
        let mut out = a.clone();
        for i in 0..out.data.len() { out.data[i] -= b.data[i]; }
        let idx = self.tape.push(Op::Sub(self.idx, other.idx),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }

    /// Hadamard : multiplication élément par élément (mêmes shapes).
    pub fn mul(self, other: Var<'t>) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let b = self.tape.value(other.idx);
        assert_eq!(a.shape(), b.shape(), "mul: shapes mismatch");
        let out = a.hadamard(&b);
        let idx = self.tape.push(Op::Mul(self.idx, other.idx),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }

    /// Alias pour clarté en code Transformer.
    pub fn hadamard(self, other: Var<'t>) -> Var<'t> { self.mul(other) }

    // ---------- Arithmétique avec scalaire ---------- //

    pub fn scale(self, s: f32) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let out = a.scale(s);
        let idx = self.tape.push(Op::Scale(self.idx, s),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }

    // ---------- Unaires ---------- //

    pub fn neg(self) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let mut out = a.clone();
        for x in &mut out.data { *x = -*x; }
        let idx = self.tape.push(Op::Neg(self.idx),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }

    pub fn exp(self) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let mut out = a.clone();
        for x in &mut out.data { *x = x.exp(); }
        let idx = self.tape.push(Op::Exp(self.idx),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }

    pub fn log(self) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let mut out = a.clone();
        for x in &mut out.data { *x = x.ln(); }
        let idx = self.tape.push(Op::Log(self.idx),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }

    pub fn sqrt(self) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let mut out = a.clone();
        for x in &mut out.data { *x = x.sqrt(); }
        let idx = self.tape.push(Op::Sqrt(self.idx),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }

    pub fn sigmoid(self) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let mut out = a.clone();
        for x in &mut out.data { *x = 1.0 / (1.0 + (-*x).exp()); }
        let idx = self.tape.push(Op::Sigmoid(self.idx),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }

    pub fn relu(self) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let mut out = a.clone();
        for x in &mut out.data { *x = x.max(0.0); }
        let idx = self.tape.push(Op::ReLU(self.idx),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }

    // ---------- Réductions ---------- //

    /// Somme tous les éléments → scalaire (1,1).
    pub fn sum(self) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let s = a.sum();
        let out = Tensor::from_vec(vec![s], 1, 1);
        let idx = self.tape.push(Op::Sum(self.idx),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }

    // ---------- Algèbre matricielle ---------- //

    pub fn matmul(self, other: Var<'t>) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let b = self.tape.value(other.idx);
        let out = matmul(&a, &b);
        let idx = self.tape.push(Op::MatMul(self.idx, other.idx),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }

    pub fn transpose(self) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let out = transpose(&a);
        let idx = self.tape.push(Op::Transpose(self.idx),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }

    /// Add bias broadcast row-wise. self est (M,N), bias est (1,N).
    pub fn add_bias(self, bias: Var<'t>) -> Var<'t> {
        let a = self.tape.value(self.idx);
        let b = self.tape.value(bias.idx);
        assert_eq!(b.rows, 1, "add_bias: bias must have 1 row, got {}", b.rows);
        assert_eq!(a.cols, b.cols, "add_bias: cols mismatch ({} vs {})", a.cols, b.cols);
        let mut out = a.clone();
        for r in 0..a.rows {
            for c in 0..a.cols {
                out.data[r * a.cols + c] += b.data[c];
            }
        }
        let idx = self.tape.push(Op::AddBias(self.idx, bias.idx),
                                 DeviceTensor::cpu(out), SavedData::None);
        Var { tape: self.tape, idx }
    }
}

// ==================================================================== //
//  Tests internes                                                        //
//                                                                       //
//  optimizer_minimizes_x_squared est l'oracle. Si ce test passe,        //
//  l'autograd marche réellement.                                        //
// ==================================================================== //

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ---------- Forward correctness ---------- //

    #[test]
    fn add_forward() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![1.0, 2.0], 1, 2));
        let b = tape.input(Tensor::from_vec(vec![3.0, 4.0], 1, 2));
        let c = a.add(b);
        let v = tape.value(c.idx());
        assert_eq!(v.data, vec![4.0, 6.0]);
    }

    #[test]
    fn mul_forward_hadamard() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![2.0, 3.0], 1, 2));
        let b = tape.input(Tensor::from_vec(vec![4.0, 5.0], 1, 2));
        let c = a.mul(b);
        let v = tape.value(c.idx());
        assert_eq!(v.data, vec![8.0, 15.0]);
    }

    #[test]
    fn matmul_forward() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let b = tape.input(Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2));
        let c = a.matmul(b);
        let v = tape.value(c.idx());
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(v.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn sigmoid_forward() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![0.0], 1, 1));
        let b = a.sigmoid();
        let v = tape.value(b.idx());
        assert!(approx_eq(v.data[0], 0.5, 1e-6));
    }

    // ---------- Backward correctness (vérifications mathématiques) ---------- //

    #[test]
    fn add_backward() {
        // f = a + b, ∂f/∂a = 1, ∂f/∂b = 1
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![1.0], 1, 1));
        let b = tape.input(Tensor::from_vec(vec![2.0], 1, 1));
        let c = a.add(b);
        tape.backward(c.idx());
        assert_eq!(tape.grad(a.idx()).data, vec![1.0]);
        assert_eq!(tape.grad(b.idx()).data, vec![1.0]);
    }

    #[test]
    fn sub_backward() {
        // f = a - b, ∂f/∂a = 1, ∂f/∂b = -1
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![5.0], 1, 1));
        let b = tape.input(Tensor::from_vec(vec![3.0], 1, 1));
        let c = a.sub(b);
        tape.backward(c.idx());
        assert_eq!(tape.grad(a.idx()).data, vec![1.0]);
        assert_eq!(tape.grad(b.idx()).data, vec![-1.0]);
    }

    #[test]
    fn mul_backward() {
        // f = a * b, ∂f/∂a = b, ∂f/∂b = a
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![3.0], 1, 1));
        let b = tape.input(Tensor::from_vec(vec![5.0], 1, 1));
        let c = a.mul(b);
        tape.backward(c.idx());
        assert_eq!(tape.grad(a.idx()).data, vec![5.0]);   // = b
        assert_eq!(tape.grad(b.idx()).data, vec![3.0]);   // = a
    }

    #[test]
    fn scale_backward() {
        // f = a * 7, ∂f/∂a = 7
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![2.0], 1, 1));
        let c = a.scale(7.0);
        tape.backward(c.idx());
        assert_eq!(tape.grad(a.idx()).data, vec![7.0]);
    }

    #[test]
    fn neg_backward() {
        // f = -a, ∂f/∂a = -1
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![3.0], 1, 1));
        let c = a.neg();
        tape.backward(c.idx());
        assert_eq!(tape.grad(a.idx()).data, vec![-1.0]);
    }

    #[test]
    fn exp_backward() {
        // f = exp(a), ∂f/∂a = exp(a)
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![1.0], 1, 1));
        let c = a.exp();
        tape.backward(c.idx());
        assert!(approx_eq(tape.grad(a.idx()).data[0], 1.0_f32.exp(), 1e-5));
    }

    #[test]
    fn log_backward() {
        // f = log(a), ∂f/∂a = 1/a
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![4.0], 1, 1));
        let c = a.log();
        tape.backward(c.idx());
        assert!(approx_eq(tape.grad(a.idx()).data[0], 0.25, 1e-5));
    }

    #[test]
    fn sqrt_backward() {
        // f = sqrt(a), ∂f/∂a = 1/(2·sqrt(a))
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![9.0], 1, 1));
        let c = a.sqrt();
        tape.backward(c.idx());
        // 1/(2·3) = 1/6
        assert!(approx_eq(tape.grad(a.idx()).data[0], 1.0 / 6.0, 1e-5));
    }

    #[test]
    fn sigmoid_backward() {
        // f = sigmoid(0) = 0.5, ∂f/∂a = 0.5 · (1 - 0.5) = 0.25
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![0.0], 1, 1));
        let c = a.sigmoid();
        tape.backward(c.idx());
        assert!(approx_eq(tape.grad(a.idx()).data[0], 0.25, 1e-5));
    }

    #[test]
    fn relu_backward_positive_passes() {
        // f = relu(2.0) = 2.0, ∂f/∂a = 1
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![2.0], 1, 1));
        let c = a.relu();
        tape.backward(c.idx());
        assert_eq!(tape.grad(a.idx()).data, vec![1.0]);
    }

    #[test]
    fn relu_backward_negative_blocks() {
        // f = relu(-2.0) = 0, ∂f/∂a = 0
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![-2.0], 1, 1));
        let c = a.relu();
        tape.backward(c.idx());
        assert_eq!(tape.grad(a.idx()).data, vec![0.0]);
    }

    #[test]
    fn sum_backward() {
        // f = sum(a, b, c), ∂f/∂x = 1 partout
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let s = a.sum();
        tape.backward(s.idx());
        assert_eq!(tape.grad(a.idx()).data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn matmul_backward_shapes() {
        // Vérifie que les grads ont les bonnes shapes après matmul
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let b = tape.input(Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], 2, 2));
        let c = a.matmul(b);
        let s = c.sum();
        tape.backward(s.idx());
        assert_eq!(tape.grad(a.idx()).shape(), (2, 2));
        assert_eq!(tape.grad(b.idx()).shape(), (2, 2));
    }

    // ---------- Chain rule (composition d'ops) ---------- //

    #[test]
    fn chain_rule_x_squared() {
        // f(x) = x², ∂f/∂x = 2x
        // À x=3, f=9 et ∂f/∂x=6
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![3.0], 1, 1));
        let y = x.mul(x);
        tape.backward(y.idx());
        assert_eq!(tape.value(y.idx()).data, vec![9.0]);
        assert_eq!(tape.grad(x.idx()).data, vec![6.0]);    // 2x = 6
    }

    #[test]
    fn chain_rule_x_squared_via_sum() {
        // Même test mais via sum() pour avoir un scalar output
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![3.0, 4.0], 1, 2));
        let xx = x.mul(x);
        let loss = xx.sum();
        tape.backward(loss.idx());
        // ∂(x²+y²)/∂x = 2x = 6, ∂/∂y = 2y = 8
        assert_eq!(tape.grad(x.idx()).data, vec![6.0, 8.0]);
    }

    #[test]
    fn chain_rule_polynomial() {
        // f = (x + 2) * 3, à x=1 : f = 9, ∂f/∂x = 3
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0], 1, 1));
        let two = tape.input(Tensor::from_vec(vec![2.0], 1, 1));
        let sum_xt = x.add(two);
        let scaled = sum_xt.scale(3.0);
        tape.backward(scaled.idx());
        assert_eq!(tape.value(scaled.idx()).data, vec![9.0]);
        assert_eq!(tape.grad(x.idx()).data, vec![3.0]);
    }

    // ---------- Le test ORACLE : l'optimisation marche ---------- //

    #[test]
    fn optimizer_minimizes_x_squared() {
        // Test fondamental : si on applique manuellement un SGD à f(x) = x²
        // partant de x = 10, après plusieurs steps x doit tendre vers 0.
        //
        // Si ce test passe, on a vraiment :
        //   - Un forward correct
        //   - Un backward correct
        //   - La capacité d'écrire la valeur mise à jour sur la tape
        //   - Et donc la base d'un vrai optimizer

        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![10.0], 1, 1));
        let lr = 0.1;
        let n_steps = 50;

        for step in 0..n_steps {
            // Forward : f = x²
            // On calcule x.mul(x) en re-pushant à chaque itération.
            // Note : en pratique l'optimiseur écrira sur le node x lui-même
            // via set_value, ce qui évite de recréer la tape.
            //
            // Pour ce test, on vérifie l'algo de plus bas niveau possible :
            //   1. forward pour calculer loss
            //   2. backward pour calculer grad
            //   3. mise à jour de la valeur via set_value
            //   4. zero_grad pour la prochaine itération

            tape.zero_grad();
            // Re-fait le forward avec la nouvelle valeur de x
            // Mais comme la tape garde l'historique, on a besoin d'une astuce :
            // on lit la valeur actuelle de x, on crée une mini-tape éphémère.

            let val_x = tape.value(x.idx()).data[0];

            // Mini-forward isolé pour vérifier la dérivée
            let inner_tape = Tape::new();
            let xi = inner_tape.input(Tensor::from_vec(vec![val_x], 1, 1));
            let yi = xi.mul(xi);
            inner_tape.backward(yi.idx());
            let grad_x = inner_tape.grad(xi.idx()).data[0];

            // SGD update : x ← x - lr · grad
            let new_x = val_x - lr * grad_x;
            tape.set_value(x.idx(), Tensor::from_vec(vec![new_x], 1, 1));

            if step == n_steps - 1 {
                println!("Step {}: x = {:.6}, loss = {:.6}", step, new_x, new_x * new_x);
            }
        }

        let final_x = tape.value(x.idx()).data[0];
        // x devrait être proche de 0 (convergence quadratique avec lr=0.1)
        assert!(final_x.abs() < 0.01,
            "x = {} après {} steps, expected ~0", final_x, n_steps);
    }

    // ---------- Tests structurels ---------- //

    #[test]
    fn zero_grad_resets() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![3.0], 1, 1));
        let b = a.scale(2.0);
        tape.backward(b.idx());
        assert_eq!(tape.grad(a.idx()).data, vec![2.0]);
        tape.zero_grad();
        assert_eq!(tape.grad(a.idx()).data, vec![0.0]);
    }

    #[test]
    fn set_value_updates() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![1.0], 1, 1));
        assert_eq!(tape.value(a.idx()).data, vec![1.0]);
        tape.set_value(a.idx(), Tensor::from_vec(vec![42.0], 1, 1));
        assert_eq!(tape.value(a.idx()).data, vec![42.0]);
    }

    #[test]
    fn add_bias_forward() {
        // a (2,3) + b (1,3) → broadcast row-wise
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(
            vec![1.0, 2.0, 3.0,
                 4.0, 5.0, 6.0], 2, 3));
        let b = tape.input(Tensor::from_vec(vec![10.0, 20.0, 30.0], 1, 3));
        let c = a.add_bias(b);
        let v = tape.value(c.idx());
        assert_eq!(v.data, vec![11.0, 22.0, 33.0,  14.0, 25.0, 36.0]);
    }

    #[test]
    fn add_bias_backward() {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![1.0; 6], 2, 3));
        let b = tape.input(Tensor::from_vec(vec![1.0, 1.0, 1.0], 1, 3));
        let c = a.add_bias(b);
        let s = c.sum();
        tape.backward(s.idx());
        // grad_a = ones(2,3)
        assert_eq!(tape.grad(a.idx()).data, vec![1.0; 6]);
        // grad_b = sum sur axis=0 de grad_c = [2, 2, 2] (parce que 2 rows contribuent chacune 1)
        assert_eq!(tape.grad(b.idx()).data, vec![2.0, 2.0, 2.0]);
    }
}
