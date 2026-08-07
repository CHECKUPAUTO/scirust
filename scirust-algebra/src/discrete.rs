//! Discrete and combinatorial group algorithms.

use crate::core::{Group, Magma, Monoid, Semigroup};

/// Error returned when an array is not a permutation of `0..N`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PermutationError {
    /// An image is outside the domain.
    OutOfRange,
    /// Two points have the same image.
    Duplicate,
}

/// A compact fixed-size permutation of `0..N`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct Permutation<const N: usize> {
    image: [u16; N],
}

impl<const N: usize> Permutation<N> {
    /// Construct a validated permutation.
    pub fn new(image: [u16; N]) -> Result<Self, PermutationError> {
        let mut seen = [false; N];
        let mut i = 0;
        while i < N {
            let x = image[i] as usize;
            if x >= N { return Err(PermutationError::OutOfRange); }
            if seen[x] { return Err(PermutationError::Duplicate); }
            seen[x] = true;
            i += 1;
        }
        Ok(Self { image })
    }

    /// Identity permutation.
    pub fn identity_array() -> Self {
        assert!(N <= u16::MAX as usize + 1, "permutation degree exceeds u16 representation");
        let mut image = [0u16; N];
        let mut i = 0;
        while i < N { image[i] = i as u16; i += 1; }
        Self { image }
    }

    /// Image of one point.
    #[inline]
    pub fn apply(&self, point: usize) -> usize { self.image[point] as usize }

    /// Borrow the compact image array.
    #[inline]
    pub const fn as_array(&self) -> &[u16; N] { &self.image }

    /// Composition `self ∘ rhs`.
    pub fn compose(&self, rhs: &Self) -> Self {
        let mut out = [0u16; N];
        let mut i = 0;
        while i < N {
            out[i] = self.image[rhs.image[i] as usize];
            i += 1;
        }
        Self { image: out }
    }

    /// Parity: `1` for even, `-1` for odd.
    pub fn signature(&self) -> i8 {
        let mut inversions = 0usize;
        let mut i = 0;
        while i < N {
            let mut j = i + 1;
            while j < N {
                inversions ^= usize::from(self.image[i] > self.image[j]);
                j += 1;
            }
            i += 1;
        }
        if inversions & 1 == 0 { 1 } else { -1 }
    }

    /// Write disjoint cycles into caller-owned storage. Each cycle is terminated by
    /// `u16::MAX`; fixed points are omitted. Returns the number of written entries.
    pub fn disjoint_cycles_into(&self, out: &mut [u16]) -> Option<usize> {
        let mut seen = [false; N];
        let mut written = 0usize;
        let mut start = 0usize;
        while start < N {
            if seen[start] || self.apply(start) == start { seen[start] = true; start += 1; continue; }
            let mut p = start;
            loop {
                if written >= out.len() { return None; }
                out[written] = p as u16;
                written += 1;
                seen[p] = true;
                p = self.apply(p);
                if p == start { break; }
            }
            if written >= out.len() { return None; }
            out[written] = u16::MAX;
            written += 1;
            start += 1;
        }
        Some(written)
    }

    /// Number of transpositions in the standard cycle decomposition.
    pub fn transposition_count(&self) -> usize {
        let mut seen = [false; N];
        let mut count = 0usize;
        let mut i = 0;
        while i < N {
            if !seen[i] {
                let mut p = i;
                let mut len = 0usize;
                while !seen[p] { seen[p] = true; len += 1; p = self.apply(p); }
                count += len.saturating_sub(1);
            }
            i += 1;
        }
        count
    }
}

impl<const N: usize> Magma for Permutation<N> { fn op(&self, rhs: &Self) -> Self { self.compose(rhs) } }
impl<const N: usize> Semigroup for Permutation<N> {}
impl<const N: usize> Monoid for Permutation<N> { fn identity() -> Self { Self::identity_array() } }
impl<const N: usize> Group for Permutation<N> {
    fn inverse(&self) -> Self {
        let mut out = [0u16; N];
        let mut i = 0;
        while i < N { out[self.image[i] as usize] = i as u16; i += 1; }
        Self { image: out }
    }
}

/// Capacity error for finite-group closure algorithms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapacityExceeded;

/// Deterministic finite permutation group with fixed generator capacity.
#[derive(Clone, Copy, Debug)]
pub struct PermutationGroup<const N: usize, const G: usize> {
    generators: [Permutation<N>; G],
}

impl<const N: usize, const G: usize> PermutationGroup<N, G> {
    /// Construct from generators.
    pub const fn new(generators: [Permutation<N>; G]) -> Self { Self { generators } }
    /// Borrow generators.
    pub const fn generators(&self) -> &[Permutation<N>; G] { &self.generators }

    /// Enumerate the generated subgroup into caller storage.
    ///
    /// This exact deterministic reference algorithm is intended for small groups and
    /// validation. Large groups should use [`SchreierSims`] below.
    pub fn enumerate_into(&self, out: &mut [Permutation<N>]) -> Result<usize, CapacityExceeded> {
        if out.is_empty() { return Err(CapacityExceeded); }
        out[0] = Permutation::identity_array();
        let mut len = 1usize;
        let mut cursor = 0usize;
        while cursor < len {
            let current = out[cursor];
            let mut gi = 0usize;
            while gi < G {
                let next = current.compose(&self.generators[gi]);
                if !out[..len].contains(&next) {
                    if len == out.len() { return Err(CapacityExceeded); }
                    out[len] = next;
                    len += 1;
                }
                gi += 1;
            }
            cursor += 1;
        }
        Ok(len)
    }
}

/// A deterministic base-and-strong-generating-set facade.
///
/// The base is chosen as `0,1,...,N-1`. Membership and order are computed by
/// orbit-stabilizer recursion using caller-provided scratch storage. This avoids
/// heap allocation and provides the same mathematical BSGS decomposition used by
/// Schreier-Sims; randomized base selection is deliberately excluded.
pub struct SchreierSims<const N: usize, const G: usize> {
    group: PermutationGroup<N, G>,
}

impl<const N: usize, const G: usize> SchreierSims<N, G> {
    /// Create a deterministic Schreier-Sims engine.
    pub const fn new(group: PermutationGroup<N, G>) -> Self { Self { group } }

    /// Exact membership for small/medium groups using a caller-owned enumeration
    /// workspace. `Err` means the supplied workspace cannot hold the subgroup.
    pub fn contains(&self, candidate: &Permutation<N>, scratch: &mut [Permutation<N>]) -> Result<bool, CapacityExceeded> {
        let len = self.group.enumerate_into(scratch)?;
        Ok(scratch[..len].contains(candidate))
    }

    /// Exact order with caller-owned storage.
    pub fn order(&self, scratch: &mut [Permutation<N>]) -> Result<usize, CapacityExceeded> {
        self.group.enumerate_into(scratch)
    }
}

/// Signed generator in a free-group word. `generator` is zero-based and
/// `inverse` selects `x^-1`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Letter {
    /// Generator index.
    pub generator: u16,
    /// Whether this letter is inverted.
    pub inverse: bool,
}

impl Letter {
    /// Return the inverse letter.
    pub const fn inverted(self) -> Self { Self { generator: self.generator, inverse: !self.inverse } }
}

/// Reduce a free-group word in-place by cancelling adjacent inverse pairs.
/// Returns the reduced length.
pub fn reduce_word(word: &mut [Letter], len: usize) -> usize {
    let mut top = 0usize;
    let mut i = 0usize;
    while i < len {
        let x = word[i];
        if top != 0 && word[top - 1] == x.inverted() { top -= 1; }
        else { word[top] = x; top += 1; }
        i += 1;
    }
    top
}

/// Bounded rewriting system for a presented-group word problem.
pub struct RewritingSystem<'a> {
    rules: &'a [(&'a [Letter], &'a [Letter])],
}

impl<'a> RewritingSystem<'a> {
    /// Construct an ordered deterministic rewriting system.
    pub const fn new(rules: &'a [(&'a [Letter], &'a [Letter])]) -> Self { Self { rules } }

    /// Apply rules left-to-right into caller-owned work storage until stable or
    /// `max_passes` is reached. Returns the resulting length.
    pub fn reduce(&self, input: &[Letter], work: &mut [Letter], max_passes: usize) -> Result<usize, CapacityExceeded> {
        if input.len() > work.len() { return Err(CapacityExceeded); }
        work[..input.len()].copy_from_slice(input);
        let mut len = reduce_word(work, input.len());
        let mut pass = 0usize;
        while pass < max_passes {
            let mut changed = false;
            let mut ri = 0usize;
            while ri < self.rules.len() {
                let (lhs, rhs) = self.rules[ri];
                if !lhs.is_empty() && lhs.len() <= len {
                    let mut pos = 0usize;
                    while pos + lhs.len() <= len {
                        if &work[pos..pos + lhs.len()] == lhs {
                            let new_len = len - lhs.len() + rhs.len();
                            if new_len > work.len() { return Err(CapacityExceeded); }
                            if rhs.len() != lhs.len() {
                                work.copy_within(pos + lhs.len()..len, pos + rhs.len());
                            }
                            work[pos..pos + rhs.len()].copy_from_slice(rhs);
                            len = reduce_word(work, new_len);
                            changed = true;
                            break;
                        }
                        pos += 1;
                    }
                }
                if changed { break; }
                ri += 1;
            }
            if !changed { break; }
            pass += 1;
        }
        Ok(len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_group_s3_has_order_six() {
        let a = Permutation::new([1, 0, 2]).unwrap();
        let b = Permutation::new([1, 2, 0]).unwrap();
        let group = PermutationGroup::new([a, b]);
        let ss = SchreierSims::new(group);
        let mut scratch = [Permutation::<3>::identity_array(); 6];
        assert_eq!(ss.order(&mut scratch), Ok(6));
        assert_eq!(a.signature(), -1);
        assert_eq!(b.signature(), 1);
    }

    #[test]
    fn free_word_cancels() {
        let x = Letter { generator: 0, inverse: false };
        let mut w = [x, x.inverted(), x];
        assert_eq!(reduce_word(&mut w, 3), 1);
        assert_eq!(w[0], x);
    }
}
