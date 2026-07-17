//! Experiment 7 — subspace and low-bit structure (spec §Experiment 7).
//!
//! We test whether structured input sets (even sublattice, scalar-only,
//! imaginary-only, a quaternionic Fano-line subalgebra, `2^t`-divisible) are
//! preserved by the round function, and count fixed points / short cycles of the
//! reduced permutation. Preserved structure is a diffusion weakness; a structure
//! surviving all rounds would be a kill.

use crate::algebra::Oct;
use crate::algebra::word::Word;
use crate::analysis::util::{Coverage, sample_octs};

/// Result of a subspace-preservation probe.
#[derive(Clone, Debug)]
pub struct SubspaceResult {
    /// Name of the structured set.
    pub name: String,
    /// `true` iff `f(member)` stayed in the set for every tested member.
    pub preserved: bool,
    /// Number of members tested.
    pub members_tested: usize,
    /// Coverage of the members set.
    pub coverage: Coverage,
}

/// Members of the "all coefficients divisible by `2^t`" sublattice, enumerated
/// for `W2` (small) or sampled-and-masked otherwise.
fn divisible_members<W: Word>(t: u32, seed: u64, sample: usize) -> (Vec<Oct<W>>, Coverage) {
    let bits = W::BITS;
    if t >= bits
    {
        return (vec![Oct::zero()], Coverage::Exhaustive);
    }
    // each coefficient is a multiple of 2^t: value = m * 2^t, m in [0, 2^(bits-t))
    let per = 1u64 << (bits - t);
    let total = per.checked_pow(8);
    if let Some(tot) = total
    {
        if tot <= 100_000
        {
            let mut out = Vec::with_capacity(tot as usize);
            for code in 0..tot
            {
                let mut c = [W::ZERO; 8];
                let mut rem = code;
                for slot in c.iter_mut()
                {
                    let m = rem % per;
                    rem /= per;
                    *slot = W::from_u64((m << t) & mask_for(bits));
                }
                out.push(Oct::from_coeffs(c));
            }
            return (out, Coverage::Exhaustive);
        }
    }
    // sample: mask each coefficient down to a multiple of 2^t
    let mut pts = sample_octs::<W>(seed, sample);
    for p in pts.iter_mut()
    {
        for slot in p.c.iter_mut()
        {
            let v = (slot.to_u64() >> t) << t;
            *slot = W::from_u64(v);
        }
    }
    (
        pts,
        Coverage::Sampled {
            count: sample,
            seed,
        },
    )
}

fn mask_for(bits: u32) -> u64 {
    if bits >= 64
    {
        u64::MAX
    }
    else
    {
        (1u64 << bits) - 1
    }
}

/// Generic membership-preservation test: does `f` map every tested `member`
/// (built by `gen`) to something satisfying `pred`?
fn preserved_by<W: Word>(
    name: &str,
    members: Vec<Oct<W>>,
    coverage: Coverage,
    pred: impl Fn(&Oct<W>) -> bool,
    f: impl Fn(Oct<W>) -> Oct<W>,
) -> SubspaceResult {
    let mut preserved = true;
    for m in &members
    {
        if !pred(&f(*m))
        {
            preserved = false;
            break;
        }
    }
    SubspaceResult {
        name: name.to_string(),
        preserved,
        members_tested: members.len(),
        coverage,
    }
}

/// Predicate helpers.
fn all_even<W: Word>(o: &Oct<W>) -> bool {
    o.c.iter().all(|w| w.to_u64() & 1 == 0)
}
fn scalar_only<W: Word>(o: &Oct<W>) -> bool {
    o.c[1..].iter().all(|w| w.to_u64() == 0)
}
fn imaginary_only<W: Word>(o: &Oct<W>) -> bool {
    o.c[0].to_u64() == 0
}
/// Fano-line subalgebra spanned by `{e0, e1, e2, e4}` (the line `(1,2,4)`).
fn fano_124<W: Word>(o: &Oct<W>) -> bool {
    [3usize, 5, 6, 7].iter().all(|&i| o.c[i].to_u64() == 0)
}

/// Run the full battery of subspace-preservation probes on `f`.
pub fn run_battery<W: Word>(
    f: impl Fn(Oct<W>) -> Oct<W> + Copy,
    seed: u64,
    sample: usize,
) -> Vec<SubspaceResult> {
    let mut out = Vec::new();

    // even sublattice 2·R^8
    let (mem, cov) = divisible_members::<W>(1, seed, sample);
    out.push(preserved_by(
        "even-sublattice(2·R^8)",
        mem,
        cov,
        all_even,
        f,
    ));

    // scalar-only inputs
    let bits = W::BITS;
    let scalar_total = 1u64 << bits;
    let (mem, cov) = if scalar_total <= 100_000
    {
        (
            (0..scalar_total)
                .map(|v| {
                    let mut c = [W::ZERO; 8];
                    c[0] = W::from_u64(v);
                    Oct::from_coeffs(c)
                })
                .collect(),
            Coverage::Exhaustive,
        )
    }
    else
    {
        let mut pts = sample_octs::<W>(seed ^ 1, sample);
        for p in pts.iter_mut()
        {
            for i in 1..8
            {
                p.c[i] = W::ZERO;
            }
        }
        (
            pts,
            Coverage::Sampled {
                count: sample,
                seed: seed ^ 1,
            },
        )
    };
    out.push(preserved_by("scalar-only", mem, cov, scalar_only, f));

    // imaginary-only inputs (sampled: set c0=0)
    let mut pts = sample_octs::<W>(
        seed ^ 2,
        sample.min(
            if scalar_total <= 100_000
            {
                usize::MAX
            }
            else
            {
                sample
            },
        ),
    );
    for p in pts.iter_mut()
    {
        p.c[0] = W::ZERO;
    }
    let cov = Coverage::Sampled {
        count: pts.len(),
        seed: seed ^ 2,
    };
    out.push(preserved_by("imaginary-only", pts, cov, imaginary_only, f));

    // Fano-line subalgebra {e0,e1,e2,e4}
    let per = 1u64 << bits;
    let (mem, cov) = if per.checked_pow(4).map(|t| t <= 100_000).unwrap_or(false)
    {
        let tot = per.pow(4);
        let mut v = Vec::with_capacity(tot as usize);
        for code in 0..tot
        {
            let mut c = [W::ZERO; 8];
            let mut rem = code;
            for &i in &[0usize, 1, 2, 4]
            {
                c[i] = W::from_u64(rem % per);
                rem /= per;
            }
            v.push(Oct::from_coeffs(c));
        }
        (v, Coverage::Exhaustive)
    }
    else
    {
        let mut pts = sample_octs::<W>(seed ^ 3, sample);
        for p in pts.iter_mut()
        {
            for &i in &[3usize, 5, 6, 7]
            {
                p.c[i] = W::ZERO;
            }
        }
        (
            pts,
            Coverage::Sampled {
                count: sample,
                seed: seed ^ 3,
            },
        )
    };
    out.push(preserved_by(
        "fano-line-subalgebra{e0,e1,e2,e4}",
        mem,
        cov,
        fano_124,
        f,
    ));

    out
}

/// Fixed-point / short-cycle census result for a reduced permutation.
#[derive(Clone, Debug)]
pub struct CycleResult {
    /// Number of sampled inputs that were fixed points `P(x)=x`.
    pub fixed_points: u64,
    /// Number of sampled inputs on a 2-cycle `P(P(x))=x, P(x)≠x`.
    pub two_cycles: u64,
    /// Number of samples.
    pub samples: usize,
}

/// Census fixed points and 2-cycles of a permutation `p` over sampled states.
pub fn cycle_census<W: Word>(
    p: impl Fn(Oct<W>) -> Oct<W>,
    seed: u64,
    samples: usize,
) -> CycleResult {
    let pts = sample_octs::<W>(seed, samples);
    let mut fixed = 0u64;
    let mut two = 0u64;
    for x in &pts
    {
        let px = p(*x);
        if px == *x
        {
            fixed += 1;
        }
        else if p(px) == *x
        {
            two += 1;
        }
    }
    CycleResult {
        fixed_points: fixed,
        two_cycles: two,
        samples: pts.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::word::{W2, W8};
    use crate::fixtures::{Fixture, FixtureId};
    use crate::permutation::round::f_round;

    #[test]
    fn full_round_breaks_scalar_subspace() {
        let m = Fixture::new(FixtureId::PseudoRandom(6)).round_material::<W2>(0);
        let results = run_battery::<W2>(move |x| f_round(x, &m), 1, 4096);
        // At least the scalar-only structure should be broken by the round.
        let scalar = results.iter().find(|r| r.name == "scalar-only").unwrap();
        assert!(!scalar.preserved, "F should not keep scalars scalar");
    }

    #[test]
    fn cycle_census_runs() {
        let m = Fixture::new(FixtureId::PseudoRandom(8)).round_material::<W8>(0);
        let c = cycle_census::<W8>(move |x| f_round(x, &m), 3, 2000);
        assert_eq!(c.samples, 2000);
    }
}
