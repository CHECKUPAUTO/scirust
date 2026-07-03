// scirust-core/src/lazy/mod.rs
//
// Exécution différée — DAG d'opérations + 3 portes d'évaluation.

use crate::autodiff::reverse::{Tape, Tensor, Var};
use std::cell::{Ref, RefCell};
use std::rc::Rc;

pub mod plan;
pub use plan::{CachePolicy, Compiler, Plan, PlanStats};

#[derive(Clone, Debug)]
pub enum LazyOp {
    Const(Tensor),
    Feed { name: String, shape: (usize, usize) },
    Add(LazyId, LazyId),
    Sub(LazyId, LazyId),
    Mul(LazyId, LazyId),
    Scale(LazyId, f32),
    Relu(LazyId),
    Exp(LazyId),
    Log(LazyId),
    MatMul(LazyId, LazyId),
}

pub type LazyId = usize;

pub struct LazyGraph {
    nodes: RefCell<Vec<LazyNode>>,
    cache: RefCell<Vec<Option<Tensor>>>,
}

#[derive(Clone)]
pub(crate) struct LazyNode {
    pub op: LazyOp,
    pub shape: (usize, usize),
}

impl LazyGraph {
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            nodes: RefCell::new(Vec::new()),
            cache: RefCell::new(Vec::new()),
        })
    }

    fn push(&self, op: LazyOp, shape: (usize, usize)) -> LazyId {
        let mut nodes = self.nodes.borrow_mut();
        let mut cache = self.cache.borrow_mut();
        let id = nodes.len();
        nodes.push(LazyNode { op, shape });
        cache.push(None);
        id
    }

    pub(crate) fn nodes_borrow(&self) -> Ref<'_, Vec<LazyNode>> {
        self.nodes.borrow()
    }

    pub fn invalidate(&self) {
        for c in self.cache.borrow_mut().iter_mut()
        {
            *c = None;
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.borrow().len()
    }

    pub fn eval(&self, id: LazyId) -> Tensor {
        if let Some(t) = &self.cache.borrow()[id]
        {
            return t.clone();
        }
        let op = self.nodes.borrow()[id].op.clone();
        let result = match op
        {
            LazyOp::Const(t) => t,
            LazyOp::Feed { name, .. } =>
            {
                panic!("eval implicite ne supporte pas les feeds — utilise compile() pour '{name}'")
            },
            LazyOp::Add(a, b) => elementwise(self, a, b, |x, y| x + y),
            LazyOp::Sub(a, b) => elementwise(self, a, b, |x, y| x - y),
            LazyOp::Mul(a, b) => elementwise(self, a, b, |x, y| x * y),
            LazyOp::Scale(a, s) =>
            {
                let mut t = self.eval(a);
                for x in t.data.iter_mut()
                {
                    *x *= s;
                }
                t
            },
            LazyOp::Relu(a) =>
            {
                let mut t = self.eval(a);
                for x in t.data.iter_mut()
                {
                    *x = x.max(0.0);
                }
                t
            },
            LazyOp::Exp(a) =>
            {
                let mut t = self.eval(a);
                for x in t.data.iter_mut()
                {
                    *x = x.exp();
                }
                t
            },
            LazyOp::Log(a) =>
            {
                let mut t = self.eval(a);
                for x in t.data.iter_mut()
                {
                    *x = x.max(1e-12).ln();
                }
                t
            },
            LazyOp::MatMul(a, b) =>
            {
                let ta = self.eval(a);
                let tb = self.eval(b);
                naive_matmul(&ta, &tb)
            },
        };
        self.cache.borrow_mut()[id] = Some(result.clone());
        result
    }
}

fn elementwise<F: Fn(f32, f32) -> f32>(g: &LazyGraph, a: LazyId, b: LazyId, f: F) -> Tensor {
    let ta = g.eval(a);
    let tb = g.eval(b);
    assert_eq!(ta.shape(), tb.shape());
    let mut out = ta.clone();
    for i in 0..out.data.len()
    {
        out.data[i] = f(ta.data[i], tb.data[i]);
    }
    out
}

fn naive_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, k) = a.shape();
    let (k2, n) = b.shape();
    assert_eq!(k, k2);
    let mut c = Tensor::zeros(m, n);
    for i in 0..m
    {
        for j in 0..n
        {
            let mut acc = 0.0f32;
            for p in 0..k
            {
                acc += a.data[i * k + p] * b.data[p * n + j];
            }
            c.data[i * n + j] = acc;
        }
    }
    c
}

#[derive(Clone)]
pub struct LazyTensor {
    pub graph: Rc<LazyGraph>,
    pub id: LazyId,
}

impl LazyTensor {
    pub fn from_tensor(graph: Rc<LazyGraph>, t: Tensor) -> Self {
        let shape = t.shape();
        let id = graph.push(LazyOp::Const(t), shape);
        Self { graph, id }
    }

    pub fn feed(graph: Rc<LazyGraph>, name: String, shape: (usize, usize)) -> Self {
        let id = graph.push(LazyOp::Feed { name, shape }, shape);
        Self { graph, id }
    }

    pub fn from_var(graph: Rc<LazyGraph>, tape: &Tape, var: Var<'_>) -> Self {
        Self::from_tensor(graph, tape.value(var.idx()))
    }

    pub fn materialize_into<'t>(&self, tape: &'t Tape) -> Var<'t> {
        tape.input(self.value())
    }

    pub fn shape(&self) -> (usize, usize) {
        self.graph.nodes_borrow()[self.id].shape
    }

    pub fn value(&self) -> Tensor {
        self.graph.eval(self.id)
    }

    pub fn scalar(&self) -> f32 {
        let t = self.value();
        assert_eq!(t.shape(), (1, 1));
        t.data[0]
    }

    pub fn compile(&self) -> Plan {
        Compiler::new(&self.graph).compile(self.id)
    }
    pub fn compile_with(&self, policy: CachePolicy) -> Plan {
        Compiler::new(&self.graph)
            .with_cache_policy(policy)
            .compile(self.id)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        let id = self
            .graph
            .push(LazyOp::Add(self.id, other.id), self.shape());
        Self {
            graph: self.graph,
            id,
        }
    }
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        let id = self
            .graph
            .push(LazyOp::Sub(self.id, other.id), self.shape());
        Self {
            graph: self.graph,
            id,
        }
    }
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        let id = self
            .graph
            .push(LazyOp::Mul(self.id, other.id), self.shape());
        Self {
            graph: self.graph,
            id,
        }
    }
    pub fn scale(self, s: f32) -> Self {
        let id = self.graph.push(LazyOp::Scale(self.id, s), self.shape());
        Self {
            graph: self.graph,
            id,
        }
    }
    pub fn relu(self) -> Self {
        let id = self.graph.push(LazyOp::Relu(self.id), self.shape());
        Self {
            graph: self.graph,
            id,
        }
    }
    pub fn exp(self) -> Self {
        let id = self.graph.push(LazyOp::Exp(self.id), self.shape());
        Self {
            graph: self.graph,
            id,
        }
    }
    pub fn log(self) -> Self {
        let id = self.graph.push(LazyOp::Log(self.id), self.shape());
        Self {
            graph: self.graph,
            id,
        }
    }
    pub fn matmul(self, other: Self) -> Self {
        let (m, k) = self.shape();
        let (k2, n) = other.shape();
        // Validate inner dimensions at graph-build time, matching the eager
        // `naive_matmul` assert. Without this the COMPILED MatMul (which indexes
        // `b.data[p*n+j]` for `p in 0..k`) reads out of bounds or produces silent
        // garbage on a dimension mismatch, unlike the eager path which panics.
        assert_eq!(
            k, k2,
            "LazyTensor::matmul: inner dimensions mismatch — ({m}x{k}) · ({k2}x{n})"
        );
        let id = self.graph.push(LazyOp::MatMul(self.id, other.id), (m, n));
        Self {
            graph: self.graph,
            id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_eval_until_value() {
        let g = LazyGraph::new();
        let a = LazyTensor::from_tensor(g.clone(), Tensor::from_vec(vec![1.0; 4], 1, 4));
        let _ = a.relu().scale(2.0).exp();
        assert_eq!(g.node_count(), 4);
        // Personne n'a appelé .value() → cache vide
        assert!(g.cache.borrow().iter().all(|c| c.is_none()));
    }

    #[test]
    fn implicit_eval_works() {
        let g = LazyGraph::new();
        let a = LazyTensor::from_tensor(
            g.clone(),
            Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], 1, 4),
        );
        let y = a.relu().scale(2.0);
        assert_eq!(y.value().data, vec![0.0, 4.0, 0.0, 8.0]);
    }

    #[test]
    fn from_var_round_trip() {
        let tape = Tape::new();
        let v = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
        let g = LazyGraph::new();
        let l = LazyTensor::from_var(g.clone(), &tape, v);
        let v2 = l.scale(10.0).materialize_into(&tape);
        assert_eq!(tape.value(v2.idx()).data, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    #[should_panic(expected = "inner dimensions mismatch")]
    fn matmul_rejects_dimension_mismatch() {
        let g = LazyGraph::new();
        let a = LazyTensor::from_tensor(g.clone(), Tensor::from_vec(vec![1.0; 6], 2, 3)); // 2x3
        let b = LazyTensor::from_tensor(g.clone(), Tensor::from_vec(vec![1.0; 8], 4, 2)); // 4x2 (3 != 4)
        let _ = a.matmul(b);
    }

    #[test]
    fn matmul_valid_compiles_and_matches_eager() {
        let g = LazyGraph::new();
        let a = LazyTensor::from_tensor(
            g.clone(),
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
        );
        let b = LazyTensor::from_tensor(
            g.clone(),
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2),
        );
        let y = a.matmul(b);
        let eager = y.value().data;
        let compiled = y.compile().execute().data;
        assert_eq!(compiled, eager);
        assert_eq!(compiled, vec![22.0, 28.0, 49.0, 64.0]);
    }
}
