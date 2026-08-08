//! Deterministic Schreier-Sims primitives for permutation groups.
//!
//! This module provides fixed-storage orbit/transversal construction, stabilizer-chain
//! sifting, Schreier-generator production and deterministic strong-generating-set
//! completion. All algorithmic storage is selected at compile time.

use crate::core::Group;
use crate::discrete::Permutation;

/// Error returned when a fixed-capacity Schreier-Sims workspace is insufficient.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchreierError {
    /// Strong-generator storage or another fixed-capacity workspace was exhausted.
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
        if base >= N {
            return Err(SchreierError::CapacityExceeded);
        }
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
    /// image of the base is outside the known orbit and therefore exposes a missing
    /// strong generator at this level.
    pub fn sift(&self, element: Permutation<N>) -> Option<Permutation<N>> {
        let image = element.apply(self.base);
        let transversal = self.transversal(image)?;
        Some(transversal.inverse().compose(&element))
    }
}

/// Deterministic base chain using the natural base `0, 1, ..., N-1`.
///
/// Each level is built from the subset of supplied strong generators that fixes all
/// earlier base points. Once the set is strong, order and membership use only
/// orbit-stabilizer sifting and never enumerate the whole group.
#[derive(Clone, Copy, Debug)]
pub struct StabilizerChain<const N: usize> {
    levels: [OrbitLevel<N>; N],
}

impl<const N: usize> StabilizerChain<N> {
    /// Build a chain from a candidate strong generating set.
    ///
    /// The set need not already be strong when this constructor is used internally by
    /// [`Bsgs::build`]; missing stabilizer generators are discovered by Schreier sifting.
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
                    if fixes_prefix(&generator, base) {
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

    /// Exact group order implied by the BSGS via orbit-stabilizer.
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
        self.sift_residue_from(0, *element) == Permutation::<N>::identity_array()
    }

    /// Borrow one chain level.
    pub fn level(&self, index: usize) -> Option<&OrbitLevel<N>> {
        self.levels.get(index)
    }

    fn sift_residue_from(&self, start: usize, element: Permutation<N>) -> Permutation<N> {
        let mut residue = element;
        let mut level = start;
        while level < N {
            let Some(next) = self.levels[level].sift(residue) else {
                return residue;
            };
            residue = next;
            level += 1;
        }
        residue
    }
}

/// Completed deterministic base-and-strong-generating set.
///
/// `K` is the maximum number of stored strong generators. Construction starts from
/// ordinary generators, applies Schreier's lemma at every stabilizer level and inserts
/// non-trivial sift residues until no missing generator remains.
#[derive(Clone, Copy, Debug)]
pub struct Bsgs<const N: usize, const K: usize> {
    generators: [Permutation<N>; K],
    generator_len: usize,
    chain: StabilizerChain<N>,
}

impl<const N: usize, const K: usize> Bsgs<N, K> {
    /// Complete an ordinary generating set into a deterministic BSGS.
    ///
    /// Returns [`SchreierError::CapacityExceeded`] if more than `K` distinct strong
    /// generators are required. Identity and duplicate input generators are discarded.
    pub fn build(generators: &[Permutation<N>]) -> Result<Self, SchreierError> {
        let identity = Permutation::<N>::identity_array();
        let mut strong = [identity; K];
        let mut strong_len = 0usize;

        let mut input_index = 0usize;
        while input_index < generators.len() {
            let generator = generators[input_index];
            if generator != identity && !strong[..strong_len].contains(&generator) {
                if strong_len == K {
                    return Err(SchreierError::CapacityExceeded);
                }
                strong[strong_len] = generator;
                strong_len += 1;
            }
            input_index += 1;
        }

        loop {
            let chain = StabilizerChain::from_strong_generators(&strong[..strong_len])?;
            let mut inserted = false;
            let mut base = 0usize;

            'search: while base < N {
                let level = &chain.levels[base];
                let scan_len = strong_len;
                let mut orbit_index = 0usize;
                while orbit_index < level.orbit_len {
                    let point = level.orbit[orbit_index] as usize;
                    let transversal = level.transversal[point];
                    let mut generator_index = 0usize;
                    while generator_index < scan_len {
                        let generator = strong[generator_index];
                        if fixes_prefix(&generator, base) {
                            let image = generator.apply(point);
                            let image_transversal = level.transversal[image];
                            let schreier = image_transversal
                                .inverse()
                                .compose(&generator.compose(&transversal));
                            let residue = chain.sift_residue_from(base + 1, schreier);
                            if residue != identity && !strong[..strong_len].contains(&residue) {
                                if strong_len == K {
                                    return Err(SchreierError::CapacityExceeded);
                                }
                                strong[strong_len] = residue;
                                strong_len += 1;
                                inserted = true;
                                break 'search;
                            }
                        }
                        generator_index += 1;
                    }
                    orbit_index += 1;
                }
                base += 1;
            }

            if !inserted {
                return Ok(Self {
                    generators: strong,
                    generator_len: strong_len,
                    chain,
                });
            }
        }
    }

    /// Borrow the completed strong generating set.
    pub fn strong_generators(&self) -> &[Permutation<N>] {
        &self.generators[..self.generator_len]
    }

    /// Number of stored non-identity strong generators.
    pub const fn strong_generator_len(&self) -> usize {
        self.generator_len
    }

    /// Exact group order from the completed stabilizer chain.
    pub fn order(&self) -> Option<usize> {
        self.chain.order()
    }

    /// Exact membership test by BSGS sifting.
    pub fn contains(&self, element: &Permutation<N>) -> bool {
        self.chain.contains(element)
    }

    /// Borrow the completed stabilizer chain.
    pub const fn chain(&self) -> &StabilizerChain<N> {
        &self.chain
    }
}

fn fixes_prefix<const N: usize>(generator: &Permutation<N>, prefix_len: usize) -> bool {
    let mut point = 0usize;
    while point < prefix_len {
        if generator.apply(point) != point {
            return false;
        }
        point += 1;
    }
    true
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

    #[test]
    fn schreier_sims_completes_s3_from_ordinary_generators() {
        let transposition = Permutation::new([1, 0, 2]).unwrap();
        let cycle = Permutation::new([1, 2, 0]).unwrap();
        let bsgs = Bsgs::<3, 8>::build(&[transposition, cycle]).unwrap();
        let stabilizer_generator = Permutation::new([0, 2, 1]).unwrap();
        assert_eq!(bsgs.order(), Some(6));
        assert!(bsgs.contains(&transposition));
        assert!(bsgs.contains(&cycle));
        assert!(bsgs.contains(&stabilizer_generator));
        assert!(bsgs.strong_generator_len() >= 3);
    }

    #[test]
    fn schreier_sims_completes_dihedral_group_d4() {
        let rotation = Permutation::new([1, 2, 3, 0]).unwrap();
        let reflection = Permutation::new([0, 3, 2, 1]).unwrap();
        let bsgs = Bsgs::<4, 16>::build(&[rotation, reflection]).unwrap();
        let outside = Permutation::new([1, 0, 2, 3]).unwrap();
        assert_eq!(bsgs.order(), Some(8));
        assert!(bsgs.contains(&rotation));
        assert!(bsgs.contains(&reflection));
        assert!(!bsgs.contains(&outside));
    }

    #[test]
    fn schreier_sims_reports_generator_capacity_exhaustion() {
        let transposition = Permutation::new([1, 0, 2]).unwrap();
        let cycle = Permutation::new([1, 2, 0]).unwrap();
        assert_eq!(
            Bsgs::<3, 2>::build(&[transposition, cycle]),
            Err(SchreierError::CapacityExceeded)
        );
    }
}
