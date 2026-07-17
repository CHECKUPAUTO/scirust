//! Shared helpers for the experiments: deterministic test-point generation and
//! coverage bookkeeping. No OS entropy is ever used (spec §18).

use crate::algebra::Oct;
use crate::algebra::word::Word;
use crate::fixtures::SplitMix64;

/// Whether a set of test points exhausts the domain or merely samples it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Coverage {
    /// Every element of the domain was tested.
    Exhaustive,
    /// A deterministic pseudo-random sample was tested.
    Sampled {
        /// Number of points.
        count: usize,
        /// Seed used.
        seed: u64,
    },
}

impl Coverage {
    /// Short label for reports.
    pub fn label(self) -> String {
        match self
        {
            Coverage::Exhaustive => "exhaustive".to_string(),
            Coverage::Sampled { count, seed } =>
            {
                format!("sampled(n={count}, seed=0x{seed:016x})")
            },
        }
    }
}

/// `true` iff the single-octonion domain `(Z/2^k)^8` can be exhausted cheaply
/// (only `k = 2`, i.e. `2^16` points).
pub fn oct_domain_exhaustible<W: Word>() -> bool {
    8 * W::BITS <= 16
}

/// Enumerate the entire single-octonion domain for `W2` (65536 points). For any
/// other width this returns an empty vector (use [`sample_octs`] instead).
pub fn enumerate_octs<W: Word>() -> Vec<Oct<W>> {
    if !oct_domain_exhaustible::<W>()
    {
        return Vec::new();
    }
    let bits = W::BITS;
    let total = 1u64 << (8 * bits);
    let mut out = Vec::with_capacity(total as usize);
    for code in 0..total
    {
        let mut c = [W::ZERO; 8];
        for (i, slot) in c.iter_mut().enumerate()
        {
            let shift = (i as u32) * bits;
            let mask = (1u64 << bits) - 1;
            *slot = W::from_u64((code >> shift) & mask);
        }
        out.push(Oct::from_coeffs(c));
    }
    out
}

/// A deterministic pseudo-random sample of octonions.
pub fn sample_octs<W: Word>(seed: u64, count: usize) -> Vec<Oct<W>> {
    let mut rng = SplitMix64::new(seed);
    (0..count)
        .map(|_| Oct::<W>::from_u64s(std::array::from_fn(|_| rng.next_u64())))
        .collect()
}

/// Return exhaustive points for `W2`, else a deterministic sample.
pub fn test_points<W: Word>(seed: u64, sample: usize) -> (Vec<Oct<W>>, Coverage) {
    if oct_domain_exhaustible::<W>()
    {
        (enumerate_octs::<W>(), Coverage::Exhaustive)
    }
    else
    {
        (
            sample_octs::<W>(seed, sample),
            Coverage::Sampled {
                count: sample,
                seed,
            },
        )
    }
}
