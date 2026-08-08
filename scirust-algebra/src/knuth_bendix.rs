//! Deterministic fixed-capacity Knuth-Bendix completion for presented groups.
//!
//! The implementation uses compile-time capacities for both the rule set and words.
//! Rules are oriented by shortlex order, words are freely reduced before rewriting,
//! and critical pairs are generated from all compatible overlaps of rule left sides.

use core::cmp::Ordering;

use crate::discrete::Letter;

const EMPTY: Letter = Letter {
    generator: 0,
    inverse: false,
};

/// Error returned by bounded Knuth-Bendix completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnuthBendixError {
    /// The compile-time rule capacity was exhausted.
    RuleCapacityExceeded,
    /// A word exceeded the selected compile-time word capacity.
    WordCapacityExceeded,
    /// Completion did not converge within the requested number of passes.
    CompletionLimitReached,
}

/// Fixed-capacity freely reduced word.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedWord<const W: usize> {
    letters: [Letter; W],
    len: usize,
}

impl<const W: usize> FixedWord<W> {
    /// Empty word.
    pub const fn empty() -> Self {
        Self {
            letters: [EMPTY; W],
            len: 0,
        }
    }

    /// Construct and freely reduce a word.
    pub fn from_slice(input: &[Letter]) -> Result<Self, KnuthBendixError> {
        let mut out = Self::empty();
        let mut i = 0usize;
        while i < input.len() {
            out.push_reduced(input[i])?;
            i += 1;
        }
        Ok(out)
    }

    /// Borrow the valid word prefix.
    pub fn as_slice(&self) -> &[Letter] {
        &self.letters[..self.len]
    }

    /// Word length.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the word is empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push_raw(&mut self, letter: Letter) -> Result<(), KnuthBendixError> {
        if self.len == W {
            return Err(KnuthBendixError::WordCapacityExceeded);
        }
        self.letters[self.len] = letter;
        self.len += 1;
        Ok(())
    }

    fn push_reduced(&mut self, letter: Letter) -> Result<(), KnuthBendixError> {
        if self.len != 0 && self.letters[self.len - 1] == letter.inverted() {
            self.len -= 1;
            return Ok(());
        }
        self.push_raw(letter)
    }

    fn append_reduced(&mut self, word: &[Letter]) -> Result<(), KnuthBendixError> {
        let mut i = 0usize;
        while i < word.len() {
            self.push_reduced(word[i])?;
            i += 1;
        }
        Ok(())
    }

    fn replace_range(
        &mut self,
        start: usize,
        remove_len: usize,
        replacement: &[Letter],
    ) -> Result<(), KnuthBendixError> {
        let mut rebuilt = Self::empty();
        rebuilt.append_reduced(&self.as_slice()[..start])?;
        rebuilt.append_reduced(replacement)?;
        rebuilt.append_reduced(&self.as_slice()[start + remove_len..])?;
        *self = rebuilt;
        Ok(())
    }
}

/// One oriented rewrite rule `lhs -> rhs`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RewriteRule<const W: usize> {
    lhs: FixedWord<W>,
    rhs: FixedWord<W>,
}

impl<const W: usize> RewriteRule<W> {
    /// Left side.
    pub const fn lhs(&self) -> &FixedWord<W> {
        &self.lhs
    }

    /// Right side.
    pub const fn rhs(&self) -> &FixedWord<W> {
        &self.rhs
    }
}

/// Fixed-capacity deterministic Knuth-Bendix engine.
#[derive(Clone, Copy, Debug)]
pub struct KnuthBendix<const R: usize, const W: usize> {
    rules: [RewriteRule<W>; R],
    rule_len: usize,
}

impl<const R: usize, const W: usize> KnuthBendix<R, W> {
    /// Construct an empty completion engine.
    pub const fn new() -> Self {
        const fn empty_rule<const W: usize>() -> RewriteRule<W> {
            RewriteRule {
                lhs: FixedWord::empty(),
                rhs: FixedWord::empty(),
            }
        }
        Self {
            rules: [empty_rule::<W>(); R],
            rule_len: 0,
        }
    }

    /// Number of currently stored oriented rules.
    pub const fn rule_len(&self) -> usize {
        self.rule_len
    }

    /// Borrow the active rule set.
    pub fn rules(&self) -> &[RewriteRule<W>] {
        &self.rules[..self.rule_len]
    }

    /// Add an equation and orient it by deterministic shortlex order.
    pub fn add_equation(
        &mut self,
        left: &[Letter],
        right: &[Letter],
    ) -> Result<bool, KnuthBendixError> {
        let mut lhs = FixedWord::<W>::from_slice(left)?;
        let mut rhs = FixedWord::<W>::from_slice(right)?;
        self.normalize(&mut lhs)?;
        self.normalize(&mut rhs)?;
        self.insert_oriented(lhs, rhs)
    }

    /// Reduce a word to the normal form induced by the current rule set.
    pub fn normal_form(&self, input: &[Letter]) -> Result<FixedWord<W>, KnuthBendixError> {
        let mut word = FixedWord::<W>::from_slice(input)?;
        self.normalize(&mut word)?;
        Ok(word)
    }

    /// Complete the current rule set by resolving critical overlaps.
    ///
    /// Returns the number of active rules once a full pass discovers no new equation.
    pub fn complete(&mut self, max_passes: usize) -> Result<usize, KnuthBendixError> {
        let mut pass = 0usize;
        while pass < max_passes {
            let snapshot_len = self.rule_len;
            let mut added = false;
            let mut i = 0usize;
            while i < snapshot_len {
                let mut j = 0usize;
                while j < snapshot_len {
                    if self.resolve_rule_pair(i, j)? {
                        added = true;
                    }
                    j += 1;
                }
                i += 1;
            }
            if !added {
                return Ok(self.rule_len);
            }
            pass += 1;
        }
        Err(KnuthBendixError::CompletionLimitReached)
    }

    fn insert_oriented(
        &mut self,
        mut left: FixedWord<W>,
        mut right: FixedWord<W>,
    ) -> Result<bool, KnuthBendixError> {
        if left == right {
            return Ok(false);
        }
        if shortlex_cmp(&left, &right) == Ordering::Less {
            core::mem::swap(&mut left, &mut right);
        }
        let candidate = RewriteRule {
            lhs: left,
            rhs: right,
        };
        if self.rules[..self.rule_len].contains(&candidate) {
            return Ok(false);
        }
        if self.rule_len == R {
            return Err(KnuthBendixError::RuleCapacityExceeded);
        }
        self.rules[self.rule_len] = candidate;
        self.rule_len += 1;
        Ok(true)
    }

    fn normalize(&self, word: &mut FixedWord<W>) -> Result<(), KnuthBendixError> {
        loop {
            let mut changed = false;
            let mut ri = 0usize;
            while ri < self.rule_len {
                let rule = self.rules[ri];
                if rule.lhs.is_empty() || rule.lhs.len() > word.len() {
                    ri += 1;
                    continue;
                }
                let mut pos = 0usize;
                while pos + rule.lhs.len() <= word.len() {
                    if &word.as_slice()[pos..pos + rule.lhs.len()] == rule.lhs.as_slice() {
                        word.replace_range(pos, rule.lhs.len(), rule.rhs.as_slice())?;
                        changed = true;
                        break;
                    }
                    pos += 1;
                }
                if changed {
                    break;
                }
                ri += 1;
            }
            if !changed {
                return Ok(());
            }
        }
    }

    fn resolve_rule_pair(
        &mut self,
        left_index: usize,
        right_index: usize,
    ) -> Result<bool, KnuthBendixError> {
        let first = self.rules[left_index];
        let second = self.rules[right_index];
        let a = first.lhs.as_slice();
        let b = second.lhs.as_slice();
        if a.is_empty() || b.is_empty() {
            return Ok(false);
        }

        let min_shift = -((b.len() as isize) - 1);
        let max_shift = a.len() as isize - 1;
        let mut shift = min_shift;
        let mut inserted = false;
        while shift <= max_shift {
            if overlap_matches(a, b, shift) {
                let start = core::cmp::min(0isize, shift);
                let end = core::cmp::max(a.len() as isize, shift + b.len() as isize);
                let total_len = (end - start) as usize;
                if total_len > W {
                    return Err(KnuthBendixError::WordCapacityExceeded);
                }

                let a_pos = (-start) as usize;
                let b_pos = (shift - start) as usize;
                let mut superword = FixedWord::<W>::empty();
                let mut k = 0usize;
                while k < total_len {
                    let world = start + k as isize;
                    let letter = if world >= 0 && world < a.len() as isize {
                        a[world as usize]
                    } else {
                        b[(world - shift) as usize]
                    };
                    superword.push_raw(letter)?;
                    k += 1;
                }

                let mut branch_a = superword;
                branch_a.replace_range(a_pos, a.len(), first.rhs.as_slice())?;
                self.normalize(&mut branch_a)?;

                let mut branch_b = superword;
                branch_b.replace_range(b_pos, b.len(), second.rhs.as_slice())?;
                self.normalize(&mut branch_b)?;

                if self.insert_oriented(branch_a, branch_b)? {
                    inserted = true;
                }
            }
            shift += 1;
        }
        Ok(inserted)
    }
}

impl<const R: usize, const W: usize> Default for KnuthBendix<R, W> {
    fn default() -> Self {
        Self::new()
    }
}

fn shortlex_cmp<const W: usize>(left: &FixedWord<W>, right: &FixedWord<W>) -> Ordering {
    match left.len().cmp(&right.len()) {
        Ordering::Equal => {
            let mut i = 0usize;
            while i < left.len() {
                let a = letter_key(left.as_slice()[i]);
                let b = letter_key(right.as_slice()[i]);
                match a.cmp(&b) {
                    Ordering::Equal => i += 1,
                    other => return other,
                }
            }
            Ordering::Equal
        }
        other => other,
    }
}

fn letter_key(letter: Letter) -> (u16, bool) {
    (letter.generator, letter.inverse)
}

fn overlap_matches(left: &[Letter], right: &[Letter], shift: isize) -> bool {
    let overlap_start = core::cmp::max(0isize, shift);
    let overlap_end = core::cmp::min(left.len() as isize, shift + right.len() as isize);
    if overlap_start >= overlap_end {
        return false;
    }
    let mut p = overlap_start;
    while p < overlap_end {
        if left[p as usize] != right[(p - shift) as usize] {
            return false;
        }
        p += 1;
    }
    true
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
    fn free_reduction_is_applied_before_rewriting() {
        let kb = KnuthBendix::<8, 16>::new();
        let word = [A, A.inverted(), B];
        let normal = kb.normal_form(&word).unwrap();
        assert_eq!(normal.as_slice(), &[B]);
    }

    #[test]
    fn shortlex_orients_equation_towards_shorter_side() {
        let mut kb = KnuthBendix::<8, 16>::new();
        assert!(kb.add_equation(&[A, A], &[]).unwrap());
        let normal = kb.normal_form(&[A, A, A]).unwrap();
        assert_eq!(normal.as_slice(), &[A]);
    }

    #[test]
    fn completion_resolves_basic_overlap() {
        let mut kb = KnuthBendix::<16, 16>::new();
        kb.add_equation(&[A, B], &[A]).unwrap();
        kb.add_equation(&[B, A], &[B]).unwrap();
        let _ = kb.complete(8).unwrap();
        let left = kb.normal_form(&[A, B, A]).unwrap();
        let right = kb.normal_form(&[A, B]).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn equivalent_words_share_normal_form_after_completion() {
        let mut kb = KnuthBendix::<32, 24>::new();
        kb.add_equation(&[A, A], &[]).unwrap();
        kb.add_equation(&[B, B], &[]).unwrap();
        kb.add_equation(&[B, A], &[A, B]).unwrap();
        let _ = kb.complete(8).unwrap();
        let x = kb.normal_form(&[B, A, B, A]).unwrap();
        let y = kb.normal_form(&[A, B, A, B]).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn capacity_exhaustion_is_explicit() {
        let mut kb = KnuthBendix::<1, 8>::new();
        kb.add_equation(&[A, A], &[]).unwrap();
        assert_eq!(
            kb.add_equation(&[B, B], &[]),
            Err(KnuthBendixError::RuleCapacityExceeded)
        );
    }
}
