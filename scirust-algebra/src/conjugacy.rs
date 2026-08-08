//! Deterministic conjugacy-class decomposition for finite permutation groups.
//!
//! The routines operate on a caller-provided enumeration of the group. They are
//! intentionally exact reference algorithms: no heap allocation, no hashing, and a
//! stable class order determined by the first unclassified group element.

use crate::core::Group;
use crate::discrete::Permutation;

/// Error returned when the supplied finite-group enumeration is inconsistent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConjugacyError {
    /// The output class-label storage is shorter than the element enumeration.
    OutputTooSmall,
    /// A conjugate produced from the enumeration could not be found in that enumeration.
    IncompleteEnumeration,
}

/// Summary of a deterministic conjugacy-class decomposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConjugacySummary {
    /// Number of conjugacy classes.
    pub class_count: usize,
    /// Number of group elements classified.
    pub group_order: usize,
}

/// Partition an enumerated finite permutation group into conjugacy classes.
///
/// `elements` must contain every element of the group exactly once. On success,
/// `class_of[i]` is the zero-based class index of `elements[i]`. Classes are numbered
/// in first-occurrence order, making the result reproducible for a fixed enumeration.
pub fn conjugacy_classes_into<const N: usize>(
    elements: &[Permutation<N>],
    class_of: &mut [usize],
) -> Result<ConjugacySummary, ConjugacyError> {
    if class_of.len() < elements.len() {
        return Err(ConjugacyError::OutputTooSmall);
    }
    let unclassified = usize::MAX;
    let mut i = 0usize;
    while i < elements.len() {
        class_of[i] = unclassified;
        i += 1;
    }

    let mut class_count = 0usize;
    let mut seed_index = 0usize;
    while seed_index < elements.len() {
        if class_of[seed_index] != unclassified {
            seed_index += 1;
            continue;
        }

        class_of[seed_index] = class_count;
        let representative = elements[seed_index];
        let mut gi = 0usize;
        while gi < elements.len() {
            let g = elements[gi];
            let conjugate = g.compose(&representative).compose(&g.inverse());
            let mut found = None;
            let mut k = 0usize;
            while k < elements.len() {
                if elements[k] == conjugate {
                    found = Some(k);
                    break;
                }
                k += 1;
            }
            let index = found.ok_or(ConjugacyError::IncompleteEnumeration)?;
            class_of[index] = class_count;
            gi += 1;
        }

        class_count += 1;
        seed_index += 1;
    }

    Ok(ConjugacySummary {
        class_count,
        group_order: elements.len(),
    })
}

/// Write conjugacy-class sizes into caller storage.
///
/// Returns the number of classes represented by `class_of`. The class labels are
/// expected to be contiguous from zero, as produced by [`conjugacy_classes_into`].
pub fn class_sizes_into(class_of: &[usize], sizes: &mut [usize]) -> Option<usize> {
    let mut i = 0usize;
    while i < sizes.len() {
        sizes[i] = 0;
        i += 1;
    }
    let mut class_count = 0usize;
    i = 0;
    while i < class_of.len() {
        let class = class_of[i];
        if class >= sizes.len() {
            return None;
        }
        sizes[class] += 1;
        class_count = class_count.max(class + 1);
        i += 1;
    }
    Some(class_count)
}

/// Return the centralizer size of an element from an exact group enumeration.
pub fn centralizer_size<const N: usize>(
    elements: &[Permutation<N>],
    element: &Permutation<N>,
) -> usize {
    let mut count = 0usize;
    let mut i = 0usize;
    while i < elements.len() {
        let g = elements[i];
        if g.compose(element) == element.compose(&g) {
            count += 1;
        }
        i += 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::PermutationGroup;

    #[test]
    fn s3_has_three_conjugacy_classes() {
        let transposition = Permutation::new([1, 0, 2]).unwrap();
        let cycle = Permutation::new([1, 2, 0]).unwrap();
        let group = PermutationGroup::new([transposition, cycle]);
        let mut elements = [Permutation::<3>::identity_array(); 6];
        let len = group.enumerate_into(&mut elements).unwrap();
        assert_eq!(len, 6);

        let mut class_of = [usize::MAX; 6];
        let summary = conjugacy_classes_into(&elements, &mut class_of).unwrap();
        assert_eq!(summary.class_count, 3);
        assert_eq!(summary.group_order, 6);

        let mut sizes = [0usize; 3];
        assert_eq!(class_sizes_into(&class_of, &mut sizes), Some(3));
        sizes.sort_unstable();
        assert_eq!(sizes, [1, 2, 3]);
    }

    #[test]
    fn s3_centralizers_have_expected_sizes() {
        let transposition = Permutation::new([1, 0, 2]).unwrap();
        let cycle = Permutation::new([1, 2, 0]).unwrap();
        let group = PermutationGroup::new([transposition, cycle]);
        let mut elements = [Permutation::<3>::identity_array(); 6];
        group.enumerate_into(&mut elements).unwrap();
        assert_eq!(centralizer_size(&elements, &transposition), 2);
        assert_eq!(centralizer_size(&elements, &cycle), 3);
    }
}
