// scirust-core/src/tensor/tensor_nd.rs
//
// TensorND — tenseur N-dimensionnel dense row-major.
//
// Ce type est UNIQUEMENT un conteneur de données. Il n'est PAS intégré
// à l'autograd (qui reste 100% 2D via Tensor/Var/Tape).
// Les modules N-D (Conv2D, Transformer) utilisent TensorND pour stocker
// les poids/feature maps, et convertissent en 2D pour les ops autograd
// (pattern im2col déjà validé dans nn/.legacy/).

use crate::autodiff::reverse::Tensor;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorND {
    pub data: Arc<[f32]>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl TensorND {
    // ------------------------------------------------------------------
    //  Constructeurs
    // ------------------------------------------------------------------
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "TensorND::new: data.len() ({}) != shape product ({})",
            data.len(),
            numel
        );
        let strides = compute_strides(&shape);
        Self {
            data: Arc::from(data),
            shape,
            strides,
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            data: Arc::from(vec![0.0; numel]),
            shape: shape.to_vec(),
            strides: compute_strides(shape),
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            data: Arc::from(vec![1.0; numel]),
            shape: shape.to_vec(),
            strides: compute_strides(shape),
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::new(data, shape)
    }

    // ------------------------------------------------------------------
    //  Propriétés
    // ------------------------------------------------------------------
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    // ------------------------------------------------------------------
    //  Accès par indices
    // ------------------------------------------------------------------
    fn offset(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.ndim(),
            "TensorND::offset: expected {} indices, got {}",
            self.ndim(),
            indices.len()
        );
        let mut off = 0usize;
        for (i, &idx) in indices.iter().enumerate()
        {
            assert!(
                idx < self.shape[i],
                "index {} out of bounds for axis {} (dim {})",
                idx,
                i,
                self.shape[i]
            );
            off += idx * self.strides[i];
        }
        off
    }

    pub fn get(&self, indices: &[usize]) -> f32 {
        self.data[self.offset(indices)]
    }

    pub fn set(&mut self, indices: &[usize], value: f32) {
        let off = self.offset(indices);
        self.data_mut()[off] = value;
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        Arc::make_mut(&mut self.data)
    }

    // ------------------------------------------------------------------
    //  Reshape
    // ------------------------------------------------------------------
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, String> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel()
        {
            return Err(format!(
                "reshape: cannot reshape {:?} (numel={}) into {:?} (numel={})",
                self.shape,
                self.numel(),
                new_shape,
                new_numel
            ));
        }
        Ok(Self {
            data: Arc::clone(&self.data),
            shape: new_shape.to_vec(),
            strides: compute_strides(new_shape),
        })
    }

    pub fn flatten(&self) -> Self {
        self.reshape(&[self.numel()]).unwrap()
    }

    pub fn flatten_from(&self, start_axis: usize) -> Result<Self, String> {
        if start_axis > self.ndim()
        {
            return Err(format!(
                "flatten_from: start_axis {} > ndim {}",
                start_axis,
                self.ndim()
            ));
        }
        let prefix: usize = self.shape[..start_axis].iter().product();
        let suffix: usize = self.shape[start_axis..].iter().product();
        self.reshape(&[prefix, suffix])
    }

    // ------------------------------------------------------------------
    //  Transpose (permutation d'axes) — ZERO COPY via stride manipulation
    // ------------------------------------------------------------------
    pub fn transpose(&self, axes: &[usize]) -> Result<Self, String> {
        if axes.len() != self.ndim()
        {
            return Err(format!(
                "transpose: axes len {} != ndim {}",
                axes.len(),
                self.ndim()
            ));
        }
        let mut seen = vec![false; self.ndim()];
        for &a in axes
        {
            if a >= self.ndim()
            {
                return Err(format!(
                    "transpose: axis {} out of bounds (ndim {})",
                    a,
                    self.ndim()
                ));
            }
            if seen[a]
            {
                return Err(format!("transpose: axis {} appears twice", a));
            }
            seen[a] = true;
        }

        let new_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let new_strides: Vec<usize> = axes.iter().map(|&a| self.strides[a]).collect();

        // TRUE ZERO COPY: We share the same underlying data via Arc.
        Ok(Self {
            data: Arc::clone(&self.data),
            shape: new_shape,
            strides: new_strides,
        })
    }

    // ------------------------------------------------------------------
    //  Slice sur un axe
    // ------------------------------------------------------------------
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> Result<Self, String> {
        if axis >= self.ndim()
        {
            return Err(format!(
                "slice_axis: axis {} out of bounds (ndim {})",
                axis,
                self.ndim()
            ));
        }
        let dim = self.shape[axis];
        if start > end || end > dim
        {
            return Err(format!(
                "slice_axis: invalid range [{}, {}) for axis {} (dim {})",
                start, end, axis, dim
            ));
        }
        let slice_len = end - start;

        let mut new_shape = self.shape.clone();
        new_shape[axis] = slice_len;
        let new_numel: usize = new_shape.iter().product();
        let mut new_data = vec![0.0f32; new_numel];
        let new_strides = compute_strides(&new_shape);

        // Copie élément par élément
        let ndim = self.ndim();
        let mut new_indices = vec![0usize; ndim];
        let mut old_indices = vec![0usize; ndim];
        #[allow(clippy::needless_range_loop)]
        for flat_idx in 0..new_numel
        {
            let mut rem = flat_idx;
            for i in 0..ndim
            {
                new_indices[i] = rem / new_strides[i];
                rem %= new_strides[i];
            }

            old_indices.copy_from_slice(&new_indices);
            old_indices[axis] += start;
            let old_flat = self.offset(&old_indices);
            new_data[flat_idx] = self.data[old_flat];
        }

        Ok(Self {
            data: Arc::from(new_data),
            shape: new_shape,
            strides: new_strides,
        })
    }

    // ------------------------------------------------------------------
    //  Broadcast (vérification)
    // ------------------------------------------------------------------
    pub fn can_broadcast_to(&self, target_shape: &[usize]) -> bool {
        let ndim_self = self.ndim();
        let ndim_target = target_shape.len();
        if ndim_self > ndim_target
        {
            return false;
        }
        // Aligner à droite et vérifier les dimensions correspondantes
        for (&self_dim, &target_dim) in self.shape.iter().rev().zip(target_shape.iter().rev())
        {
            if self_dim != target_dim && self_dim != 1
            {
                return false;
            }
        }
        true
    }

    // ------------------------------------------------------------------
    //  Shape inference (numpy semantics) — building blocks for an N-D tape/IR
    // ------------------------------------------------------------------

    /// The shape two operands broadcast to (numpy semantics), or `None` if they
    /// are incompatible. Shapes are aligned from the right; missing leading axes
    /// count as `1`, and each output axis is the max of the two whenever one is
    /// `1` or they are equal.
    pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
        let nd = a.len().max(b.len());
        let pa = nd - a.len();
        let pb = nd - b.len();
        let mut out = vec![0usize; nd];
        for (i, slot) in out.iter_mut().enumerate()
        {
            let da = if i < pa { 1 } else { a[i - pa] };
            let db = if i < pb { 1 } else { b[i - pb] };
            *slot = if da == db
            {
                da
            }
            else if da == 1
            {
                db
            }
            else if db == 1
            {
                da
            }
            else
            {
                return None;
            };
        }
        Some(out)
    }

    /// Output shape of a (batched) matmul `a @ b` (numpy semantics): the last
    /// two axes are the matrix dims `(…, m, k) · (…, k, n) → (…, m, n)` and any
    /// leading batch axes must broadcast. `None` if the inner dims disagree,
    /// either operand has `ndim < 2`, or the batch axes are incompatible.
    pub fn matmul_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
        if a.len() < 2 || b.len() < 2
        {
            return None;
        }
        let (am, ak) = (a[a.len() - 2], a[a.len() - 1]);
        let (bk, bn) = (b[b.len() - 2], b[b.len() - 1]);
        if ak != bk
        {
            return None;
        }
        let mut out = Self::broadcast_shape(&a[..a.len() - 2], &b[..b.len() - 2])?;
        out.push(am);
        out.push(bn);
        Some(out)
    }

    /// Materialise a broadcast of `self` to `target_shape` (numpy semantics):
    /// size-1 axes and missing leading axes are replicated. Errors if `self`
    /// cannot broadcast to the target (see [`Self::can_broadcast_to`]).
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Self, String> {
        if !self.can_broadcast_to(target_shape)
        {
            return Err(format!(
                "cannot broadcast {:?} to {:?}",
                self.shape, target_shape
            ));
        }
        let nd = target_shape.len();
        let off = nd - self.ndim(); // right-alignment offset
        let out_strides = compute_strides(target_shape);
        let total: usize = target_shape.iter().product();
        let mut data = vec![0.0f32; total];
        for (flat, slot) in data.iter_mut().enumerate()
        {
            let mut rem = flat;
            let mut src_flat = 0usize;
            for (axis, &stride) in out_strides.iter().enumerate()
            {
                let idx = rem / stride;
                rem %= stride;
                if axis >= off
                {
                    let src_axis = axis - off;
                    // A size-1 source axis contributes index 0 (it is replicated).
                    let src_idx = if self.shape[src_axis] == 1 { 0 } else { idx };
                    src_flat += src_idx * self.strides[src_axis];
                }
            }
            *slot = self.data[src_flat];
        }
        Ok(Self {
            data: Arc::from(data),
            shape: target_shape.to_vec(),
            strides: out_strides,
        })
    }

    // ------------------------------------------------------------------
    //  Conversion 2D ↔ ND
    // ------------------------------------------------------------------
    pub fn from_tensor_2d(t: &Tensor) -> Self {
        let (rows, cols) = t.shape();
        Self {
            data: Arc::from(t.data.clone()),
            shape: vec![rows, cols],
            strides: compute_strides(&[rows, cols]),
        }
    }

    pub fn to_tensor_2d(&self) -> Result<Tensor, String> {
        if self.ndim() != 2
        {
            return Err(format!(
                "to_tensor_2d: expected ndim==2, got ndim=={} (shape {:?})",
                self.ndim(),
                self.shape
            ));
        }
        Ok(Tensor::from_vec(
            self.data.to_vec(),
            self.shape[0],
            self.shape[1],
        ))
    }

    /// Mode-`k` unfolding: rows = product of `shape[..k]`, cols = product of `shape[k..]`.
    /// Returns a clone of the data because row-major data preserves the unfolding layout.
    ///
    /// Example: shape `[2, 3, 4]`, unfold at `k=1` gives a `(2, 12)` matrix.
    pub fn unfold(&self, k: usize) -> (usize, usize, Vec<f32>) {
        assert!(k <= self.shape.len(), "unfold index {k} out of bounds");
        let rows: usize = self.shape[..k].iter().product::<usize>().max(1);
        let cols: usize = self.shape[k..].iter().product::<usize>().max(1);
        debug_assert_eq!(rows * cols, self.data.len());
        (rows, cols, self.data.to_vec())
    }

    /// Maximum absolute element value (used for tolerance checks).
    pub fn abs_max(&self) -> f32 {
        self.data.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()))
    }

    /// Frobenius norm: sqrt(sum of squares).
    pub fn frob_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Construct from a 2D matrix in row-major order: `data[i * cols + j]` = element `(i, j)`.
    pub fn from_matrix(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        Self::new(data, vec![rows, cols])
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == compute_strides(&self.shape)
    }

    pub fn to_contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }
        let numel = self.numel();
        let mut new_data = vec![0.0f32; numel];
        let new_strides = compute_strides(&self.shape);

        let ndim = self.ndim();
        let mut indices = vec![0usize; ndim];
        #[allow(clippy::needless_range_loop)]
        for i in 0..numel {
            let mut rem = i;
            for j in 0..ndim {
                indices[j] = rem / new_strides[j];
                rem %= new_strides[j];
            }
            new_data[i] = self.data[self.offset(&indices)];
        }

        Self {
            data: Arc::from(new_data),
            shape: self.shape.clone(),
            strides: new_strides,
        }
    }
}

// ------------------------------------------------------------------
//  Helper : calcul des strides row-major
// ------------------------------------------------------------------
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev()
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_zeros_ones() {
        let z = TensorND::zeros(&[2, 3, 4]);
        assert_eq!(z.shape(), &[2, 3, 4]);
        assert_eq!(z.ndim(), 3);
        assert_eq!(z.numel(), 24);
        assert!(z.data.iter().all(|&x| x == 0.0));

        let o = TensorND::ones(&[2, 3, 4]);
        assert!(o.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn reshape_valid() {
        let t = TensorND::zeros(&[2, 3, 4]);
        let r = t.reshape(&[6, 4]).unwrap();
        assert_eq!(r.shape(), &[6, 4]);
        assert_eq!(r.numel(), 24);

        let r2 = t.reshape(&[24]).unwrap();
        assert_eq!(r2.shape(), &[24]);
    }

    #[test]
    fn reshape_invalid_size() {
        let t = TensorND::zeros(&[2, 3, 4]);
        assert!(t.reshape(&[7, 4]).is_err());
    }

    #[test]
    fn transpose_4d() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = TensorND::new(data.clone(), vec![2, 3, 4]);
        let tr = t.transpose(&[2, 0, 1]).unwrap();
        assert_eq!(tr.shape(), &[4, 2, 3]);
        assert_eq!(tr.numel(), 24);

        // Vérification : tr[2, 0, 1] == t[0, 1, 2]
        assert!((tr.get(&[2, 0, 1]) - t.get(&[0, 1, 2])).abs() < 1e-6);
        // t[0, 1, 2] = 0*12 + 1*4 + 2 = 6
        assert_eq!(t.get(&[0, 1, 2]), 6.0);
        assert_eq!(tr.get(&[2, 0, 1]), 6.0);
    }

    #[test]
    fn slice_axis() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = TensorND::new(data, vec![2, 3, 4]);

        let s = t.slice_axis(1, 1, 3).unwrap();
        assert_eq!(s.shape(), &[2, 2, 4]);
        // s[0, 0, 0] == t[0, 1, 0] = 4
        assert_eq!(s.get(&[0, 0, 0]), 4.0);
        // s[0, 1, 0] == t[0, 2, 0] = 8
        assert_eq!(s.get(&[0, 1, 0]), 8.0);
    }

    #[test]
    fn get_set_indices() {
        let mut t = TensorND::zeros(&[2, 3, 4]);
        t.set(&[1, 2, 3], 42.0);
        assert_eq!(t.get(&[1, 2, 3]), 42.0);
        assert_eq!(t.get(&[0, 0, 0]), 0.0);
    }

    #[test]
    fn from_tensor_2d_round_trip() {
        let t2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let tnd = TensorND::from_tensor_2d(&t2d);
        assert_eq!(tnd.shape(), &[2, 3]);
        assert_eq!(tnd.data, Arc::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

        let back = tnd.to_tensor_2d().unwrap();
        assert_eq!(back.shape(), (2, 3));
        assert_eq!(back.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn flatten_from_axis() {
        let t = TensorND::zeros(&[2, 3, 4, 5]);
        let f = t.flatten_from(1).unwrap();
        assert_eq!(f.shape(), &[2, 60]);
        assert_eq!(f.numel(), 120);

        let f2 = t.flatten_from(2).unwrap();
        assert_eq!(f2.shape(), &[6, 20]);
    }

    #[test]
    fn can_broadcast_to_ok_and_fail() {
        let t = TensorND::zeros(&[3, 1, 4]);
        assert!(t.can_broadcast_to(&[2, 3, 5, 4]));
        assert!(t.can_broadcast_to(&[3, 1, 4]));
        assert!(!t.can_broadcast_to(&[3, 2, 5])); // 4 != 5
        assert!(!t.can_broadcast_to(&[1])); // ndim trop petit
    }

    #[test]
    fn broadcast_shape_rules() {
        let bs = TensorND::broadcast_shape;
        assert_eq!(bs(&[3, 1], &[1, 4]), Some(vec![3, 4]));
        assert_eq!(bs(&[2, 3, 4], &[4]), Some(vec![2, 3, 4])); // right-align + leading 1s
        assert_eq!(bs(&[1], &[]), Some(vec![1]));
        assert_eq!(bs(&[], &[]), Some(vec![]));
        assert_eq!(bs(&[3], &[4]), None); // 3 vs 4, neither is 1
        assert_eq!(bs(&[5, 2], &[5, 3]), None); // 2 vs 3
    }

    #[test]
    fn matmul_shape_rules() {
        let ms = TensorND::matmul_shape;
        assert_eq!(ms(&[2, 3], &[3, 4]), Some(vec![2, 4]));
        assert_eq!(ms(&[5, 2, 3], &[5, 3, 4]), Some(vec![5, 2, 4])); // batched
        assert_eq!(ms(&[5, 2, 3], &[3, 4]), Some(vec![5, 2, 4])); // batch broadcasts
        assert_eq!(ms(&[1, 2, 3], &[6, 3, 4]), Some(vec![6, 2, 4])); // batch 1 broadcasts
        assert_eq!(ms(&[2, 3], &[4, 5]), None); // inner dim 3 != 4
        assert_eq!(ms(&[3], &[3, 4]), None); // ndim < 2
        assert_eq!(ms(&[2, 2, 3], &[5, 3, 4]), None); // batch 2 vs 5
    }

    #[test]
    fn broadcast_to_materialises() {
        // Column vector [3,1] → [3,4]: each row value repeated across columns.
        let t = TensorND::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let b = t.broadcast_to(&[3, 4]).unwrap();
        assert_eq!(b.shape(), &[3, 4]);
        assert_eq!(
            b.data,
            Arc::from(vec![
                1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0
            ])
        );

        // Row vector [1,3] → [2,3]: the row replicated down.
        let r = TensorND::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        assert_eq!(
            r.broadcast_to(&[2, 3]).unwrap().data,
            Arc::from(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        );

        // Add leading axes: [4] → [2,3,4] replicates the vector 6 times.
        let v = TensorND::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
        let bv = v.broadcast_to(&[2, 3, 4]).unwrap();
        assert_eq!(bv.shape(), &[2, 3, 4]);
        assert_eq!(bv.numel(), 24);
        assert_eq!(&bv.data[0..4], &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(&bv.data[20..24], &[10.0, 20.0, 30.0, 40.0]);

        // Incompatible target is an error, not a panic: [3,1] aligned against
        // [2,3] gives 3-vs-2 on the leading axis → cannot broadcast.
        assert!(t.broadcast_to(&[2, 3]).is_err());
    }
}
