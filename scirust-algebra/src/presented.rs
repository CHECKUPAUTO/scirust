//! Presented groups and deterministic fixed-capacity Todd-Coxeter enumeration.
//!
//! The implementation in this module deliberately uses caller-selected compile-time
//! capacities. It performs no heap allocation while enumerating cosets. Columns are
//! encoded as generator/inverse pairs: generator `g` uses column `2*g` and its inverse
//! uses column `2*g + 1`.

use crate::discrete::Letter;

const UNDEFINED: u16 = u16::MAX;

/// Error produced by fixed-capacity coset enumeration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CosetError {
    /// The compile-time alphabet width is inconsistent with a referenced generator.
    GeneratorOutOfRange,
    /// The fixed coset capacity was exhausted before enumeration completed.
    CapacityExceeded,
    /// The selected coset capacity cannot be represented by the compact `u16` table.
    CapacityTooLarge,
}

/// Compact Todd-Coxeter coset table with union-find coincidence handling.
///
/// `C` is the maximum number of provisional cosets and `A` is the number of columns.
/// For a presentation with `n` generators, `A` should normally be `2*n`.
#[derive(Clone, Debug)]
pub struct CosetTable<const C: usize, const A: usize> {
    table: [[u16; A]; C],
    parent: [u16; C],
    used: usize,
}

impl<const C: usize, const A: usize> CosetTable<C, A> {
    /// Construct a table containing the distinguished coset zero.
    pub fn new() -> Result<Self, CosetError> {
        if C == 0 || C > u16::MAX as usize {
            return Err(if C == 0 {
                CosetError::CapacityExceeded
            } else {
                CosetError::CapacityTooLarge
            });
        }
        let mut parent = [UNDEFINED; C];
        parent[0] = 0;
        Ok(Self {
            table: [[UNDEFINED; A]; C],
            parent,
            used: 1,
        })
    }

    /// Number of provisional rows allocated so far, including rows later merged.
    pub const fn provisional_len(&self) -> usize {
        self.used
    }

    /// Return the representative of a coset.
    pub fn representative(&mut self, coset: usize) -> usize {
        let root = self.find(coset);
        root as usize
    }

    /// Return a transition when it has already been defined.
    pub fn transition(&mut self, coset: usize, column: usize) -> Option<usize> {
        if coset >= self.used || column >= A {
            return None;
        }
        let root = self.find(coset) as usize;
        let value = self.table[root][column];
        if value == UNDEFINED {
            None
        } else {
            Some(self.find(value as usize) as usize)
        }
    }

    /// Count distinct live cosets after all coincidences currently known.
    pub fn live_coset_count(&mut self) -> usize {
        let mut count = 0usize;
        let mut i = 0usize;
        while i < self.used {
            if self.find(i) as usize == i {
                count += 1;
            }
            i += 1;
        }
        count
    }

    fn inverse_column(column: usize) -> usize {
        column ^ 1
    }

    fn column(letter: Letter) -> Result<usize, CosetError> {
        let base = usize::from(letter.generator)
            .checked_mul(2)
            .ok_or(CosetError::GeneratorOutOfRange)?;
        let column = base + usize::from(letter.inverse);
        if column >= A {
            Err(CosetError::GeneratorOutOfRange)
        } else {
            Ok(column)
        }
    }

    fn find(&mut self, coset: usize) -> u16 {
        let mut root = coset;
        while self.parent[root] as usize != root {
            root = self.parent[root] as usize;
        }
        let root_u16 = root as u16;
        let mut cursor = coset;
        while self.parent[cursor] != root_u16 {
            let next = self.parent[cursor] as usize;
            self.parent[cursor] = root_u16;
            cursor = next;
        }
        root_u16
    }

    fn allocate_coset(&mut self) -> Result<u16, CosetError> {
        if self.used == C {
            return Err(CosetError::CapacityExceeded);
        }
        let index = self.used;
        self.used += 1;
        self.parent[index] = index as u16;
        Ok(index as u16)
    }

    fn install_pair(&mut self, from: u16, column: usize, to: u16) {
        let from_root = self.find(from as usize) as usize;
        let to_root = self.find(to as usize) as usize;
        self.table[from_root][column] = to_root as u16;
        self.table[to_root][Self::inverse_column(column)] = from_root as u16;
    }

    fn follow_or_define(&mut self, from: u16, column: usize) -> Result<u16, CosetError> {
        let root = self.find(from as usize) as usize;
        let existing = self.table[root][column];
        if existing != UNDEFINED {
            return Ok(self.find(existing as usize));
        }
        let fresh = self.allocate_coset()?;
        self.install_pair(root as u16, column, fresh);
        Ok(fresh)
    }

    fn coincide(&mut self, left: u16, right: u16) {
        let mut pending_left = [UNDEFINED; C];
        let mut pending_right = [UNDEFINED; C];
        pending_left[0] = left;
        pending_right[0] = right;
        let mut pending_len = 1usize;

        while pending_len != 0 {
            pending_len -= 1;
            let a = self.find(pending_left[pending_len] as usize);
            let b = self.find(pending_right[pending_len] as usize);
            if a == b {
                continue;
            }

            let (keep, merge) = if a < b { (a, b) } else { (b, a) };
            self.parent[merge as usize] = keep;

            let mut column = 0usize;
            while column < A {
                let keep_value = self.table[keep as usize][column];
                let merge_value = self.table[merge as usize][column];
                match (keep_value, merge_value) {
                    (UNDEFINED, UNDEFINED) => {}
                    (UNDEFINED, value) => {
                        let value_root = self.find(value as usize);
                        self.table[keep as usize][column] = value_root;
                        self.table[value_root as usize][Self::inverse_column(column)] = keep;
                    }
                    (value, UNDEFINED) => {
                        let value_root = self.find(value as usize);
                        self.table[keep as usize][column] = value_root;
                        self.table[value_root as usize][Self::inverse_column(column)] = keep;
                    }
                    (x, y) => {
                        let xr = self.find(x as usize);
                        let yr = self.find(y as usize);
                        self.table[keep as usize][column] = xr;
                        self.table[xr as usize][Self::inverse_column(column)] = keep;
                        if xr != yr && pending_len < C {
                            pending_left[pending_len] = xr;
                            pending_right[pending_len] = yr;
                            pending_len += 1;
                        }
                    }
                }
                column += 1;
            }
        }
    }

    fn enforce_word(&mut self, start: u16, word: &[Letter], target: u16) -> Result<(), CosetError> {
        if word.is_empty() {
            self.coincide(start, target);
            return Ok(());
        }

        let mut current = self.find(start as usize);
        let mut index = 0usize;
        while index + 1 < word.len() {
            let column = Self::column(word[index])?;
            current = self.follow_or_define(current, column)?;
            index += 1;
        }

        let column = Self::column(word[word.len() - 1])?;
        let current_root = self.find(current as usize) as usize;
        let existing = self.table[current_root][column];
        let target_root = self.find(target as usize);
        if existing == UNDEFINED {
            self.install_pair(current_root as u16, column, target_root);
        } else {
            self.coincide(existing, target_root);
        }
        Ok(())
    }
}

/// Deterministic fixed-capacity Todd-Coxeter enumerator.
///
/// Relators are interpreted as words equal to the identity. Subgroup generators are
/// words constrained to fix coset zero. Enumeration repeatedly scans all currently
/// live cosets until a complete pass creates neither new rows nor new coincidences.
pub struct ToddCoxeter<'a, const C: usize, const A: usize> {
    relators: &'a [&'a [Letter]],
    subgroup_generators: &'a [&'a [Letter]],
}

impl<'a, const C: usize, const A: usize> ToddCoxeter<'a, C, A> {
    /// Construct an enumerator from presentation relators and subgroup generators.
    pub const fn new(
        relators: &'a [&'a [Letter]],
        subgroup_generators: &'a [&'a [Letter]],
    ) -> Self {
        Self {
            relators,
            subgroup_generators,
        }
    }

    /// Enumerate cosets into a newly initialized fixed-capacity table.
    pub fn enumerate(&self) -> Result<CosetTable<C, A>, CosetError> {
        let mut table = CosetTable::<C, A>::new()?;

        let mut subgroup_index = 0usize;
        while subgroup_index < self.subgroup_generators.len() {
            table.enforce_word(0, self.subgroup_generators[subgroup_index], 0)?;
            subgroup_index += 1;
        }

        loop {
            let before_used = table.provisional_len();
            let before_live = table.live_coset_count();
            let scan_limit = table.provisional_len();
            let mut coset = 0usize;
            while coset < scan_limit {
                if table.find(coset) as usize == coset {
                    let mut relator_index = 0usize;
                    while relator_index < self.relators.len() {
                        table.enforce_word(coset as u16, self.relators[relator_index], coset as u16)?;
                        relator_index += 1;
                    }
                }
                coset += 1;
            }

            let after_used = table.provisional_len();
            let after_live = table.live_coset_count();
            if before_used == after_used && before_live == after_live {
                break;
            }
        }

        Ok(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Letter = Letter {
        generator: 0,
        inverse: false,
    };
    const B: Letter = Letter {
        generator: 1,
        inverse: false,
    };

    #[test]
    fn cyclic_group_c2_has_two_cosets() {
        let r0 = [A, A];
        let relators: [&[Letter]; 1] = [&r0];
        let enumerator = ToddCoxeter::<8, 2>::new(&relators, &[]);
        let mut table = enumerator.enumerate().unwrap();
        assert_eq!(table.live_coset_count(), 2);
        assert_eq!(table.transition(0, 0), Some(1));
        assert_eq!(table.transition(1, 0), Some(0));
    }

    #[test]
    fn klein_four_group_has_four_cosets() {
        let r0 = [A, A];
        let r1 = [B, B];
        let r2 = [A, B, A, B];
        let relators: [&[Letter]; 3] = [&r0, &r1, &r2];
        let enumerator = ToddCoxeter::<32, 4>::new(&relators, &[]);
        let mut table = enumerator.enumerate().unwrap();
        assert_eq!(table.live_coset_count(), 4);
    }

    #[test]
    fn subgroup_generated_by_a_has_index_two_in_v4() {
        let r0 = [A, A];
        let r1 = [B, B];
        let r2 = [A, B, A, B];
        let subgroup = [A];
        let relators: [&[Letter]; 3] = [&r0, &r1, &r2];
        let subgroup_generators: [&[Letter]; 1] = [&subgroup];
        let enumerator = ToddCoxeter::<32, 4>::new(&relators, &subgroup_generators);
        let mut table = enumerator.enumerate().unwrap();
        assert_eq!(table.live_coset_count(), 2);
    }
}
