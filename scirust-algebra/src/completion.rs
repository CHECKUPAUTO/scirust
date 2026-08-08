//! Certification and simplification helpers for bounded Knuth-Bendix systems.
//!
//! These routines deliberately rebuild temporary systems from caller-visible rules.
//! That makes exclusion of one rule explicit during inter-reduction while preserving
//! the fixed-capacity, allocation-free execution model of `KnuthBendix`.

use crate::discrete::Letter;
use crate::knuth_bendix::{FixedWord, KnuthBendix, KnuthBendixError};

const EMPTY: Letter = Letter {
    generator: 0,
    inverse: false,
};

/// Inter-reduce a completed rewriting system.
///
/// For each rule, the left and right sides are normalized against all *other* rules.
/// Trivial equations disappear, duplicate oriented equations are coalesced by
/// `KnuthBendix::add_equation`, and the result is completed again before return.
/// The process is deterministic because rules are visited in storage order.
pub fn interreduce<const R: usize, const W: usize>(
    source: &KnuthBendix<R, W>,
    max_passes: usize,
) -> Result<KnuthBendix<R, W>, KnuthBendixError> {
    let mut result = KnuthBendix::<R, W>::new();
    let rules = source.rules();
    let mut i = 0usize;
    while i < rules.len() {
        let mut others = KnuthBendix::<R, W>::new();
        let mut j = 0usize;
        while j < rules.len() {
            if i != j {
                let rule = rules[j];
                others.add_equation(rule.lhs().as_slice(), rule.rhs().as_slice())?;
            }
            j += 1;
        }

        let lhs = others.normal_form(rules[i].lhs().as_slice())?;
        let rhs = others.normal_form(rules[i].rhs().as_slice())?;
        result.add_equation(lhs.as_slice(), rhs.as_slice())?;
        i += 1;
    }
    result.complete(max_passes)?;
    Ok(result)
}

/// Check local confluence of the current bounded rewriting system.
///
/// Every non-empty overlap of two left-hand sides is expanded into its two one-step
/// descendants. The descendants are normalized with the complete current rule set and
/// must agree. By Newman's lemma, this is a confluence certificate when the oriented
/// system is terminating; shortlex-oriented rules provide the intended well-founded
/// reduction order for systems accepted by this module.
pub fn critical_pairs_confluent<const R: usize, const W: usize>(
    system: &KnuthBendix<R, W>,
) -> Result<bool, KnuthBendixError> {
    let rules = system.rules();
    let mut i = 0usize;
    while i < rules.len() {
        let mut j = 0usize;
        while j < rules.len() {
            if !rule_pair_confluent(system, i, j)? {
                return Ok(false);
            }
            j += 1;
        }
        i += 1;
    }
    Ok(true)
}

fn rule_pair_confluent<const R: usize, const W: usize>(
    system: &KnuthBendix<R, W>,
    left_index: usize,
    right_index: usize,
) -> Result<bool, KnuthBendixError> {
    let first = system.rules()[left_index];
    let second = system.rules()[right_index];
    let a = first.lhs().as_slice();
    let b = second.lhs().as_slice();
    if a.is_empty() || b.is_empty() {
        return Ok(true);
    }

    let min_shift = -((b.len() as isize) - 1);
    let max_shift = a.len() as isize - 1;
    let mut shift = min_shift;
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
            let mut superword = [EMPTY; W];
            let mut k = 0usize;
            while k < total_len {
                let world = start + k as isize;
                superword[k] = if world >= 0 && world < a.len() as isize {
                    a[world as usize]
                } else {
                    b[(world - shift) as usize]
                };
                k += 1;
            }

            let left_branch = replace::<W>(
                &superword[..total_len],
                a_pos,
                a.len(),
                first.rhs().as_slice(),
            )?;
            let right_branch = replace::<W>(
                &superword[..total_len],
                b_pos,
                b.len(),
                second.rhs().as_slice(),
            )?;
            let left_nf = system.normal_form(left_branch.as_slice())?;
            let right_nf = system.normal_form(right_branch.as_slice())?;
            if left_nf != right_nf {
                return Ok(false);
            }
        }
        shift += 1;
    }
    Ok(true)
}

fn replace<const W: usize>(
    input: &[Letter],
    start: usize,
    remove_len: usize,
    replacement: &[Letter],
) -> Result<FixedWord<W>, KnuthBendixError> {
    let new_len = input.len() - remove_len + replacement.len();
    if new_len > W {
        return Err(KnuthBendixError::WordCapacityExceeded);
    }
    let mut raw = [EMPTY; W];
    let mut out = 0usize;
    let mut i = 0usize;
    while i < start {
        raw[out] = input[i];
        out += 1;
        i += 1;
    }
    i = 0;
    while i < replacement.len() {
        raw[out] = replacement[i];
        out += 1;
        i += 1;
    }
    i = start + remove_len;
    while i < input.len() {
        raw[out] = input[i];
        out += 1;
        i += 1;
    }
    FixedWord::from_slice(&raw[..out])
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
    fn completed_klein_system_has_joinable_critical_pairs() {
        let mut kb = KnuthBendix::<32, 24>::new();
        kb.add_equation(&[A, A], &[]).unwrap();
        kb.add_equation(&[B, B], &[]).unwrap();
        kb.add_equation(&[B, A], &[A, B]).unwrap();
        kb.complete(8).unwrap();
        assert_eq!(critical_pairs_confluent(&kb), Ok(true));
    }

    #[test]
    fn incomplete_overlap_is_detected() {
        let mut kb = KnuthBendix::<16, 16>::new();
        kb.add_equation(&[A, B], &[A]).unwrap();
        kb.add_equation(&[B, A], &[B]).unwrap();
        assert_eq!(critical_pairs_confluent(&kb), Ok(false));
    }

    #[test]
    fn interreduction_preserves_normal_forms() {
        let mut kb = KnuthBendix::<32, 24>::new();
        kb.add_equation(&[A, A], &[]).unwrap();
        kb.add_equation(&[B, B], &[]).unwrap();
        kb.add_equation(&[B, A], &[A, B]).unwrap();
        kb.complete(8).unwrap();
        let reduced = interreduce(&kb, 8).unwrap();
        let word = [B, A, B, A, A, B];
        assert_eq!(kb.normal_form(&word), reduced.normal_form(&word));
        assert_eq!(critical_pairs_confluent(&reduced), Ok(true));
    }
}
