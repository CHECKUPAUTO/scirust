//! Structure-of-arrays storage for differentiable dense tensors.
//!
//! Primal values are stored in one contiguous buffer and each tangent direction
//! in its own contiguous buffer. Kernels that only need primals never pull
//! tangent payloads into cache; kernels that propagate a subset of directions
//! can stream exactly those planes.

/// Dense tensor storage with one primal plane and `W` tangent planes.
#[derive(Debug, Clone, PartialEq)]
pub struct DifferentialTensor<T, const W: usize> {
    shape: Vec<usize>,
    primal: Vec<T>,
    tangents: [Vec<T>; W],
}

impl<T: Clone + Default, const W: usize> DifferentialTensor<T, W> {
    /// Allocate a zero/default-initialized tensor for `shape`.
    ///
    /// Allocation happens at construction only. Subsequent accessors and seed
    /// updates operate in-place on the existing planes.
    pub fn new_zeroed(shape: &[usize]) -> Self {
        let len = checked_len(shape);
        Self {
            shape: shape.to_vec(),
            primal: vec![T::default(); len],
            tangents: std::array::from_fn(|_| vec![T::default(); len]),
        }
    }

    /// Build from an existing primal buffer and zero/default tangent planes.
    pub fn from_primal(shape: &[usize], primal: Vec<T>) -> Self {
        let len = checked_len(shape);
        assert_eq!(
            primal.len(),
            len,
            "DifferentialTensor primal length does not match shape"
        );
        Self {
            shape: shape.to_vec(),
            primal,
            tangents: std::array::from_fn(|_| vec![T::default(); len]),
        }
    }

    /// Fill all tangent planes with the default value without reallocating.
    pub fn clear_tangents(&mut self) {
        for tangent in &mut self.tangents
        {
            tangent.fill(T::default());
        }
    }
}

impl<T, const W: usize> DifferentialTensor<T, W> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.primal.len()
    }

    pub fn is_empty(&self) -> bool {
        self.primal.is_empty()
    }

    pub const fn width() -> usize {
        W
    }

    /// Contiguous primal plane.
    pub fn primal(&self) -> &[T] {
        &self.primal
    }

    /// Contiguous mutable primal plane.
    pub fn primal_mut(&mut self) -> &mut [T] {
        &mut self.primal
    }

    /// Contiguous tangent plane for one direction.
    pub fn tangent(&self, lane: usize) -> &[T] {
        self.tangents.get(lane).unwrap_or_else(|| {
            panic!("DifferentialTensor tangent lane {lane} is outside width {W}")
        })
    }

    /// Contiguous mutable tangent plane for one direction.
    pub fn tangent_mut(&mut self, lane: usize) -> &mut [T] {
        self.tangents.get_mut(lane).unwrap_or_else(|| {
            panic!("DifferentialTensor tangent lane {lane} is outside width {W}")
        })
    }

    /// Expose all tangent planes without changing the SoA layout.
    pub fn tangents(&self) -> &[Vec<T>; W] {
        &self.tangents
    }

    /// Mutable access to all tangent planes.
    pub fn tangents_mut(&mut self) -> &mut [Vec<T>; W] {
        &mut self.tangents
    }
}

impl<T: Copy + Default, const W: usize> DifferentialTensor<T, W> {
    /// Set one sparse seed entry in-place.
    pub fn set_seed(&mut self, element: usize, lane: usize, value: T) {
        assert!(
            element < self.len(),
            "DifferentialTensor seed element {element} is outside length {}",
            self.len()
        );
        self.tangent_mut(lane)[element] = value;
    }

    /// Reset one tangent lane in-place.
    pub fn clear_lane(&mut self, lane: usize) {
        self.tangent_mut(lane).fill(T::default());
    }
}

fn checked_len(shape: &[usize]) -> usize {
    shape.iter().copied().fold(1usize, |acc, dim| {
        acc.checked_mul(dim)
            .expect("DifferentialTensor shape product overflows usize")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soa_planes_are_independent_and_contiguous() {
        let mut tensor = DifferentialTensor::<f32, 4>::new_zeroed(&[2, 3]);
        assert_eq!(tensor.len(), 6);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(DifferentialTensor::<f32, 4>::width(), 4);

        tensor.primal_mut()[2] = 7.0;
        tensor.set_seed(2, 3, 1.0);

        assert_eq!(tensor.primal()[2], 7.0);
        assert_eq!(tensor.tangent(3)[2], 1.0);
        assert_eq!(tensor.tangent(0)[2], 0.0);
        assert_ne!(tensor.primal().as_ptr(), tensor.tangent(3).as_ptr());
    }

    #[test]
    fn clear_tangents_reuses_existing_planes() {
        let mut tensor = DifferentialTensor::<f64, 3>::new_zeroed(&[8]);
        let ids = std::array::from_fn::<_, 3, _>(|lane| tensor.tangent(lane).as_ptr() as usize);
        for lane in 0..3
        {
            tensor.set_seed(lane, lane, 1.0);
        }
        tensor.clear_tangents();
        for (lane, id) in ids.iter().copied().enumerate()
        {
            assert!(tensor.tangent(lane).iter().all(|&x| x == 0.0));
            assert_eq!(tensor.tangent(lane).as_ptr() as usize, id);
        }
    }

    #[test]
    fn from_primal_preserves_caller_values() {
        let tensor = DifferentialTensor::<f32, 2>::from_primal(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor.primal(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(tensor.tangent(0).iter().all(|&x| x == 0.0));
        assert!(tensor.tangent(1).iter().all(|&x| x == 0.0));
    }

    #[test]
    #[should_panic(expected = "shape product overflows usize")]
    fn rejects_overflowing_shape() {
        let _ = DifferentialTensor::<f32, 1>::new_zeroed(&[usize::MAX, 2]);
    }
}
