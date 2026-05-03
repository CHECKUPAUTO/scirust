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

#[derive(Debug, Clone, PartialEq)]
pub struct TensorND {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl TensorND {
    // ------------------------------------------------------------------
    //  Constructeurs
    // ------------------------------------------------------------------
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "TensorND::new: data.len() ({}) != shape product ({})",
            data.len(),
            numel
        );
        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let numel = shape.iter().product();
        Self {
            data: vec![0.0; numel],
            shape: shape.to_vec(),
            strides: compute_strides(shape),
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let numel = shape.iter().product();
        Self {
            data: vec![1.0; numel],
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
        for (i, &idx) in indices.iter().enumerate() {
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
        self.data[off] = value;
    }

    // ------------------------------------------------------------------
    //  Reshape
    // ------------------------------------------------------------------
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, String> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(format!(
                "reshape: cannot reshape {:?} (numel={}) into {:?} (numel={})",
                self.shape,
                self.numel(),
                new_shape,
                new_numel
            ));
        }
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            strides: compute_strides(new_shape),
        })
    }

    pub fn flatten(&self) -> Self {
        self.reshape(&[self.numel()]).unwrap()
    }

    pub fn flatten_from(&self, start_axis: usize) -> Result<Self, String> {
        if start_axis > self.ndim() {
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
    //  Transpose (permutation d'axes)
    // ------------------------------------------------------------------
    pub fn transpose(&self, axes: &[usize]) -> Result<Self, String> {
        if axes.len() != self.ndim() {
            return Err(format!(
                "transpose: axes len {} != ndim {}",
                axes.len(),
                self.ndim()
            ));
        }
        let mut seen = vec![false; self.ndim()];
        for &a in axes {
            if a >= self.ndim() {
                return Err(format!(
                    "transpose: axis {} out of bounds (ndim {})",
                    a,
                    self.ndim()
                ));
            }
            if seen[a] {
                return Err(format!("transpose: axis {} appears twice", a));
            }
            seen[a] = true;
        }

        let new_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let new_numel = self.numel();
        let mut new_data = vec![0.0f32; new_numel];
        let new_strides = compute_strides(&new_shape);

        // Parcours tous les éléments et les place à leur nouvelle position
        let ndim = self.ndim();
        let mut new_indices = vec![0usize; ndim];
        let mut old_indices = vec![0usize; ndim];
        #[allow(clippy::needless_range_loop)]
        for flat_idx in 0..new_numel {
            // Convertir flat_idx en indices dans le nouveau tenseur
            let mut rem = flat_idx;
            for i in 0..ndim {
                new_indices[i] = rem / new_strides[i];
                rem %= new_strides[i];
            }

            // Trouver l'indice original
            for i in 0..ndim {
                old_indices[axes[i]] = new_indices[i];
            }

            let old_flat = self.offset(&old_indices);
            new_data[flat_idx] = self.data[old_flat];
        }

        Ok(Self {
            data: new_data,
            shape: new_shape,
            strides: new_strides,
        })
    }

    // ------------------------------------------------------------------
    //  Slice sur un axe
    // ------------------------------------------------------------------
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> Result<Self, String> {
        if axis >= self.ndim() {
            return Err(format!(
                "slice_axis: axis {} out of bounds (ndim {})",
                axis,
                self.ndim()
            ));
        }
        let dim = self.shape[axis];
        if start > end || end > dim {
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
        for flat_idx in 0..new_numel {
            let mut rem = flat_idx;
            for i in 0..ndim {
                new_indices[i] = rem / new_strides[i];
                rem %= new_strides[i];
            }

            old_indices.copy_from_slice(&new_indices);
            old_indices[axis] += start;
            let old_flat = self.offset(&old_indices);
            new_data[flat_idx] = self.data[old_flat];
        }

        Ok(Self {
            data: new_data,
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
        if ndim_self > ndim_target {
            return false;
        }
        // Aligner à droite
        let offset = ndim_target - ndim_self;
        #[allow(clippy::needless_range_loop)]
        for i in 0..ndim_target {
            let self_dim = if i < offset {
                1
            } else {
                self.shape[i - offset]
            };
            let target_dim = target_shape[i];
            if self_dim != target_dim && self_dim != 1 {
                return false;
            }
        }
        true
    }

    // ------------------------------------------------------------------
    //  Conversion 2D ↔ ND
    // ------------------------------------------------------------------
    pub fn from_tensor_2d(t: &Tensor) -> Self {
        let (rows, cols) = t.shape();
        Self {
            data: t.data.clone(),
            shape: vec![rows, cols],
            strides: compute_strides(&[rows, cols]),
        }
    }

    pub fn to_tensor_2d(&self) -> Result<Tensor, String> {
        if self.ndim() != 2 {
            return Err(format!(
                "to_tensor_2d: expected ndim==2, got ndim=={} (shape {:?})",
                self.ndim(),
                self.shape
            ));
        }
        Ok(Tensor::from_vec(
            self.data.clone(),
            self.shape[0],
            self.shape[1],
        ))
    }
}

// ------------------------------------------------------------------
//  Helper : calcul des strides row-major
// ------------------------------------------------------------------
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
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
        assert_eq!(tnd.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

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
}
