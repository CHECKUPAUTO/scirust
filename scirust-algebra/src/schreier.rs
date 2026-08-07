//! Deterministic stabilizer-chain primitives for permutation groups.
//!
//! This module provides the core orbit/transversal and sifting machinery used by a
//! Schreier-Sims implementation. All storage is fixed at compile time.

use crate::core::Group;
use crate::discrete::Permutation;

/// Error returned when a fixed-capacity stabilizer-chain workspace is insufficient.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchreierError {
    /// A generator or transversal capacity was exhausted.
    CapacityExceeded,
}

/// One deterministic orbit/transversal level for a base point.
#[derive(Clone, Copy, Debug)]
pub struct OrbitLevel<const N: usize> {
    base: usize,
    orbit: [u16; N],
    orbit_len: usize,
    transversal: [Permutation<N>; N],
    present: [bool; N],
}

impl<const N: usize> OrbitLevel<N> {
    /// Build the orbit of `base` under the supplied generators.
    pub fn build<const G: usize>(
        base: usize,
        generators: &[Permutation<N>; G],
    ) -> Result<Self, SchreierError> {
        let identity = Permutation::<N>::identity_array();
        let mut level = Self {
            base,
            orbit: [0u16; N],
            orbit_len: 1,
            transversal: [identity; N],
            present: [false; N],
        };
        level.orbit[0] = base as u16;
        level.present[base] = true;
        level.transversal[base] = identity;

        let mut cursor = 0usize;
        while cursor < level.orbit_len {
            let point = level.orbit[cursor] as usize;
            let representative = level.transversal[point];
            let mut gi = 0usize;
            while gi < G {
                let next_point = generators[gi].apply(point);
                if !level.present[next_point] {
                    if level.orbit_len == N {
                        return Err(SchreierError::CapacityExceeded);
                    }
                    level.present[next_point] = true;
                    level.orbit[level.orbit_len] = next_point as u16;
                    level.orbit_len += 1;
                    level.transversal[next_point] = generators[gi].compose(&representative);
                }
                gi += 1;
            }
            cursor += 1;
        }
        Ok(level)
    }

    /// Base point for this level.
    pub const fn base(&self) -> usize {
        self.base
    }

    /// Orbit size.
    pub const fn orbit_len(&self) -> usize {
        self.orbit_len
    }

    /// Borrow the valid prefix of the orbit.
    pub fn orbit(&self) -> &[u16] {
        &self.orbit[..self.orbit_len]
    }

    /// Return a transversal taking the base point to `point` when it is in the orbit.
    pub fn transversal(&self, point: usize) -> Option<Permutation<N>> {
        if point < N && self.present[point] {
            Some(self.transversal[point])
        } else {
            None
        }
    }

    /// Sift one permutation through this orbit level.
    ///
    /// On success, the returned residue fixes the level base point. `None` means the
    /// image of the base is outside the known orbit and therefore the element is not
    /// represented by the current stabilizer chain.
    pub fn sift(&self, element: Permutation<N>) -> Option<Permutation<N>> {
        let image = element.apply(self.base);
        let transversal = self.transversal(image)?;
        Some(transversal.inverse().compose(&element))
    }
}

/// Deterministic base chain using the natural base `0, 1, ..., N-1`.
///
/// Each level is built from the subset of supplied strong generators that fixes all
/// earlier base points. This is a fixed-storage BSGS consumer: callers may provide a
/// strong generating set directly, and the chain then supports order and membership
/// by orbit-stabilizer sifting without enumerating the group.
#[derive(Clone, Copy, Debug)]
pub struct StabilizerChain<const N: usize> {
    levels: [OrbitLevel<N>; N],
}

impl<const N: usize> StabilizerChain<N> {
    /// Build a chain from a strong generating set supplied in caller-owned storage.
    ///
    /// The slice may contain redundant generators. Correctness of membership/order as
    /// a full BSGS requires it to be strong relative to the natural base.
    pub fn from_strong_generators(
        strong_generators: &[Permutation<N>],
    ) -> Result<Self, SchreierError> {
        let identity = Permutation::<N>::identity_array();
        let empty = OrbitLevel {
            base: 0,
            orbit: [0u16; N],
            orbit_len: 1,
            transversal: [identity; N],
            present: [false; N],
        };
        let mut levels = [empty; N];

        let mut base = 0usize;
        while base < N {
            let mut orbit = [0u16; N];
            let mut orbit_len = 1usize;
            let mut present = [false; N];
            let mut transversal = [identity; N];
            orbit[0] = base as u16;
            present[base] = true;
            transversal[base] = identity;

            let mut cursor = 0usize;
            while cursor < orbit_len {
                let point = orbit[cursor] as usize;
                let representative = transversal[point];
                let mut gi = 0usize;
                while gi < strong_generators.len() {
                    let generator = strong_generators[gi];
                    let mut fixes_prefix = true;
                    let mut prefix = 0usize;
                    while prefix < base {
                        if generator.apply(prefix) != prefix {
                            fixes_prefix = false;
                            break;
                        }
                        prefix += 1;
                    }
                    if fixes_prefix {
                        let next_point = generator.apply(point);
                        if !present[next_point] {
                            present[next_point] = true;
                            orbit[orbit_len] = next_point as u16;
                            orbit_len += 1;
                            transversal[next_point] = generator.compose(&representative);
                        }
                    }
                    gi += 1;
                }
                cursor += 1;
            }

            levels[base] = OrbitLevel {
                base,
                orbit,
                orbit_len,
                transversal,
                present,
            };
            base += 1;
        }

        Ok(Self { levels })
    }

    /// Exact group order implied by the supplied BSGS via orbit-stabilizer.
    pub fn order(&self) -> Option<usize> {
        let mut order = 1usize;
        let mut i = 0usize;
        while i < N {
            order = order.checked_mul(self.levels[i].orbit_len())?;
            i += 1;
        }
        Some(order)
    }

    /// Membership by deterministic stabilizer-chain sifting.
    pub fn contains(&self, element: &Permutation<N>) -> bool {
        let mut residue = *element;
        let mut i = 0usize;
        while i < N {
            let Some(next) = self.levels[i].sift(residue) else {
                return false;
            };
            residue = next;
            i += 1;
        }
        residue == Permutation::<N>::identity_array()
    }

    /// Borrow one chain level.
    pub fn level(&self, index: usize) -> Option<&OrbitLevel<N>> {
        self.levels.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_transversal_for_s3_is_complete() {
        let transposition = Permutation::new([1, 0, 2]).unwrap();
        let cycle = Permutation::new([1, 2, 0]).unwrap();
        let level = OrbitLevel::build(0, &[transposition, cycle]).unwrap();
        assert_eq!(level.orbit_len(), 3);
        assert!(level.transversal(0).is_some());
        assert!(level.transversal(1).is_some());
        assert!(level.transversal(2).is_some());
    }

    #[test]
    fn strong_generators_sift_s3_without_enumeration() {
        let transposition_01 = Permutation::new([1, 0, 2]).unwrap();
        let cycle_012 = Permutation::new([1, 2, 0]).unwrap();
        let transposition_12 = Permutation::new([0, 2, 1]).unwrap();
        let strong = [transposition_01, cycle_012, transposition_12];
        let chain = StabilizerChain::from_strong_generators(&strong).unwrap();
        assert_eq!(chain.order(), Some(6));
        assert!(chain.contains(&transposition_01));
        assert!(chain.contains(&cycle_012));
    }

    #[test]
    fn s3_chain_rejects_element_outside_embedded_subgroup() {
        let transposition = Permutation::new([1, 0, 2, 3]).unwrap();
        let chain = StabilizerChain::from_strong_generators(&[transposition]).unwrap();
        let outside = Permutation::new([0, 1, 3, 2]).unwrap();
        assert_eq!(chain.order(), Some(2));
        assert!(!chain.contains(&outside));
    }
}
