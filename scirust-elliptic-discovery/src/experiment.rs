//! Deterministic local corpus construction and experiment manifests.

use std::collections::BTreeSet;

use scirust_sim::rng::SplitMix64;

use crate::canonical::{CanonicalEncoder, sha256};
use crate::{CorpusKind, LocalResearchCase, ToyCurve, ToyPrime};

const EXHAUSTIVE_PRIMES: [u64; 4] = [5, 7, 11, 13];
const HOLDOUT_PRIMES: [u64; 19] = [
    17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
];
const SCALE_PRIMES: [u64; 6] = [127, 251, 509, 1021, 2039, 4093];

/// Fully specified deterministic experiment manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperimentManifest {
    case: LocalResearchCase,
}

impl ExperimentManifest {
    pub const SCHEMA_VERSION: u32 = 1;

    /// Creates a manifest from an already validated local-only case.
    pub const fn new(case: LocalResearchCase) -> Self {
        Self { case }
    }

    /// Local authorization and limits.
    pub const fn research_case(&self) -> LocalResearchCase {
        self.case
    }

    /// Canonical manifest bytes, including crate version and every limit.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::with_domain(b"SCIRUST-ELLIPTIC-DISCOVERY/MANIFEST/V1");
        encoder.u32(Self::SCHEMA_VERSION);
        encoder.bytes(env!("CARGO_PKG_VERSION").as_bytes());
        encoder.u64(self.case.seed());
        encoder.u8(self.case.corpus().tag());
        encoder.u32(self.case.curves_per_prime());
        encoder.u64(self.case.tuple_budget());
        encoder.finish()
    }

    /// Integrity fingerprint of the manifest.
    pub fn fingerprint(&self) -> [u8; 32] {
        sha256(&self.canonical_bytes())
    }
}

/// Canonical summary of one locally generated curve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CorpusCurve {
    curve: ToyCurve,
    group_order: u64,
}

impl CorpusCurve {
    /// Validated toy curve.
    pub const fn curve(self) -> ToyCurve {
        self.curve
    }

    /// Exact order obtained by enumeration.
    pub const fn group_order(self) -> u64 {
        self.group_order
    }
}

/// Immutable deterministic corpus in canonical `(p, a, b)` order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Corpus {
    manifest: ExperimentManifest,
    curves: Vec<CorpusCurve>,
}

impl Corpus {
    /// Generates only built-in local toy curves.
    pub fn generate(manifest: ExperimentManifest) -> Self {
        let case = manifest.research_case();
        let curves = match case.corpus()
        {
            CorpusKind::ExhaustiveSmall => exhaustive_curves(),
            CorpusKind::IndependentHoldout =>
            {
                sampled_curves(&HOLDOUT_PRIMES, case.seed(), case.curves_per_prime())
            },
            CorpusKind::ScaleLadder =>
            {
                sampled_curves(&SCALE_PRIMES, case.seed(), case.curves_per_prime())
            },
        };
        Self { manifest, curves }
    }

    /// Manifest which produced this corpus.
    pub const fn manifest(&self) -> &ExperimentManifest {
        &self.manifest
    }

    /// Curves in stable canonical order.
    pub fn curves(&self) -> &[CorpusCurve] {
        &self.curves
    }

    /// Total number of exactly enumerated points across curves.
    pub fn total_points(&self) -> u64 {
        self.curves.iter().map(|entry| entry.group_order).sum()
    }

    /// Canonical corpus identity, independent of allocation and platform.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut encoder = CanonicalEncoder::with_domain(b"SCIRUST-ELLIPTIC-DISCOVERY/CORPUS/V1");
        encoder.bytes(&self.manifest.canonical_bytes());
        encoder.u64(u64::try_from(self.curves.len()).expect("curve count fits in u64"));
        for entry in &self.curves
        {
            let curve = entry.curve;
            encoder.u64(curve.prime().value());
            encoder.u64(curve.a());
            encoder.u64(curve.b());
            encoder.u64(entry.group_order);
        }
        encoder.finish()
    }

    /// Integrity fingerprint of canonical corpus bytes.
    pub fn fingerprint(&self) -> [u8; 32] {
        sha256(&self.canonical_bytes())
    }
}

fn exhaustive_curves() -> Vec<CorpusCurve> {
    let mut curves = Vec::new();
    for modulus in EXHAUSTIVE_PRIMES
    {
        let prime = ToyPrime::new(modulus).expect("exhaustive modulus is prime");
        for a in 0..modulus
        {
            for b in 0..modulus
            {
                if let Ok(curve) = ToyCurve::new(prime, a, b)
                {
                    curves.push(CorpusCurve {
                        curve,
                        group_order: curve.group_order(),
                    });
                }
            }
        }
    }
    curves
}

fn sampled_curves(primes: &[u64], seed: u64, curves_per_prime: u32) -> Vec<CorpusCurve> {
    let mut rng = SplitMix64::new(seed);
    let mut curves = Vec::new();
    for &modulus in primes
    {
        let prime = ToyPrime::new(modulus).expect("sample modulus is a bounded prime");
        let total_pairs = modulus * modulus;
        let target = u64::from(curves_per_prime).min(total_pairs);
        let mut examined = BTreeSet::new();
        let mut selected = BTreeSet::new();
        while u64::try_from(examined.len()).expect("examined count fits in u64") < total_pairs
            && u64::try_from(selected.len()).expect("selected count fits in u64") < target
        {
            let pair = (rng.next_u64() % modulus, rng.next_u64() % modulus);
            if examined.insert(pair) && ToyCurve::new(prime, pair.0, pair.1).is_ok()
            {
                selected.insert(pair);
            }
        }
        for (a, b) in selected
        {
            let curve = ToyCurve::new(prime, a, b).expect("selected curve is nonsingular");
            curves.push(CorpusCurve {
                curve,
                group_order: curve.group_order(),
            });
        }
    }
    curves
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(seed: u64) -> ExperimentManifest {
        ExperimentManifest::new(
            LocalResearchCase::new(seed, CorpusKind::IndependentHoldout, 2, 100)
                .expect("valid local case"),
        )
    }

    #[test]
    fn every_built_in_modulus_is_prime() {
        for modulus in EXHAUSTIVE_PRIMES
            .into_iter()
            .chain(HOLDOUT_PRIMES)
            .chain(SCALE_PRIMES)
        {
            assert!(ToyPrime::new(modulus).is_ok(), "composite modulus: {modulus}");
        }
    }

    #[test]
    fn equal_manifests_produce_byte_identical_corpora() {
        let left = Corpus::generate(manifest(42));
        let right = Corpus::generate(manifest(42));
        assert_eq!(left.canonical_bytes(), right.canonical_bytes());
        assert_eq!(left.fingerprint(), right.fingerprint());
    }

    #[test]
    fn seed_changes_sampled_corpus() {
        assert_ne!(
            Corpus::generate(manifest(42)).fingerprint(),
            Corpus::generate(manifest(43)).fingerprint()
        );
    }

    #[test]
    fn sampled_curves_are_canonically_ordered() {
        let corpus = Corpus::generate(manifest(7));
        let keys: Vec<_> = corpus
            .curves()
            .iter()
            .map(|entry| {
                let curve = entry.curve();
                (curve.prime().value(), curve.a(), curve.b())
            })
            .collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted);
    }
}
