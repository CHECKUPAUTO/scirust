//! Deterministic, explicit **experimental** round-material fixtures for Phase 1
//! (spec §12 note: "Use deterministic explicit experimental key and
//! round-constant fixtures ... does not require the complete HKDF key schedule").
//!
//! These are NOT production keys and NOT derived by the spec's HKDF schedule.
//! They are reproducible, versioned test fixtures whose only purpose is to feed
//! the matrix/invariant/degree experiments with concrete key material. The
//! matrix and invariant experiments are independent of any specific key
//! schedule, so building one before the gating tests would be premature (spec
//! §10, §12).
//!
//! Determinism source: an inlined SplitMix64 (Steele et al., 2014). SciRust also
//! ships SplitMix64 in `scirust-stats`, but it is inlined here to keep this
//! adversarial research crate self-contained and dependency-light; it is a test
//! fixture generator, explicitly NOT a cryptographic KDF.

use crate::algebra::Oct;
use crate::algebra::word::Word;

/// SplitMix64 deterministic generator (fixture use only, not cryptographic).
#[derive(Copy, Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
    /// Next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Named Phase-1 fixture identifiers (spec §12: required fixture families).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum FixtureId {
    /// All key material zero.
    Zero,
    /// Incrementing coefficient words.
    Incrementing,
    /// Every coefficient has only its high bit set.
    HighBit,
    /// `K1`, `K2` forced to **odd** (unit) norm — invertible left/right maps.
    OddNorm,
    /// `K1`, `K2` forced to **even** norm — non-unit / zero-divisor-prone maps.
    EvenNormZeroDiv,
    /// Deterministic pseudo-random material from the given seed.
    PseudoRandom(u64),
    /// The **real** HKDF-SHA-256 key schedule (spec §10) from a 32-byte master
    /// key and 16-byte tweak. Used for the official test vectors.
    Hkdf([u8; 32], [u8; 16]),
}

impl FixtureId {
    /// Stable identifier string for reports/CLI.
    pub fn label(self) -> String {
        match self
        {
            FixtureId::Zero => "zero".to_string(),
            FixtureId::Incrementing => "incrementing".to_string(),
            FixtureId::HighBit => "highbit".to_string(),
            FixtureId::OddNorm => "odd-norm".to_string(),
            FixtureId::EvenNormZeroDiv => "even-norm-zerodiv".to_string(),
            FixtureId::PseudoRandom(s) => format!("pseudo-random-0x{s:016x}"),
            FixtureId::Hkdf(k, _) =>
            {
                format!("hkdf-{:02x}{:02x}{:02x}{:02x}", k[0], k[1], k[2], k[3])
            },
        }
    }
    /// Parse from a CLI token.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str()
        {
            "zero" => Some(FixtureId::Zero),
            "incrementing" | "inc" => Some(FixtureId::Incrementing),
            "highbit" | "high" => Some(FixtureId::HighBit),
            "odd-norm" | "oddnorm" | "odd" => Some(FixtureId::OddNorm),
            "even-norm-zerodiv" | "evennorm" | "even" | "zerodiv" =>
            {
                Some(FixtureId::EvenNormZeroDiv)
            },
            _ =>
            {
                if let Some(rest) = s.strip_prefix("pseudo-random-")
                {
                    let rest = rest.trim_start_matches("0x");
                    u64::from_str_radix(rest, 16)
                        .ok()
                        .map(FixtureId::PseudoRandom)
                }
                else if let Some(rest) = s.strip_prefix("prng:")
                {
                    rest.parse::<u64>().ok().map(FixtureId::PseudoRandom)
                }
                else
                {
                    None
                }
            },
        }
    }
}

/// Per-round key material used by the v0.1 round function (spec §12.2).
#[derive(Copy, Clone, Debug)]
pub struct RoundMaterial<W: Word> {
    /// `K_{r,0}` (added to the branch).
    pub k0: Oct<W>,
    /// `K_{r,1}` (left multiplier).
    pub k1: Oct<W>,
    /// `K_{r,2}` (right multiplier).
    pub k2: Oct<W>,
    /// `RC_r` (round-constant XOR).
    pub rc: Oct<W>,
}

/// Even–Mansour whitening octonions (spec §12.3).
#[derive(Copy, Clone, Debug)]
pub struct Whitening<W: Word> {
    /// Input-side whitening for `L`.
    pub in_l: Oct<W>,
    /// Input-side whitening for `R`.
    pub in_r: Oct<W>,
    /// Output-side whitening for `L`.
    pub out_l: Oct<W>,
    /// Output-side whitening for `R`.
    pub out_r: Oct<W>,
}

/// A fixture: a deterministic source of round material and whitening.
#[derive(Copy, Clone, Debug)]
pub struct Fixture {
    /// The identifier this fixture was built from.
    pub id: FixtureId,
}

impl Fixture {
    /// Build a fixture from an id.
    pub fn new(id: FixtureId) -> Self {
        Fixture { id }
    }

    /// Domain-separated deterministic seed for one `(round, slot)` cell.
    fn cell_seed(base: u64, round: u32, slot: u32) -> u64 {
        let mut m = SplitMix64::new(
            base ^ (u64::from(round).wrapping_mul(0x1000_0001))
                ^ (u64::from(slot).wrapping_mul(0x9E37_79B1)),
        );
        m.next_u64()
    }

    fn oct_from_seed<W: Word>(seed: u64) -> Oct<W> {
        let mut m = SplitMix64::new(seed);
        Oct::from_u64s(std::array::from_fn(|_| m.next_u64()))
    }

    /// Force `Σ c_i²` to the requested parity by nudging `c[0]`.
    ///
    /// `N(x) mod 2 = (# odd coefficients) mod 2`, so flipping the parity of one
    /// coefficient flips the norm parity. We flip `c[0]` by XOR-ing its low bit.
    fn force_norm_parity<W: Word>(mut o: Oct<W>, want_odd: bool) -> Oct<W> {
        let odd_count: u32 = o.c.iter().map(|w| (w.to_u64() & 1) as u32).sum();
        let is_odd = odd_count % 2 == 1;
        if is_odd != want_odd
        {
            o.c[0] = o.c[0].xor(W::ONE);
        }
        o
    }

    /// Deterministic round material for round `r` (spec §12.2 slots K0,K1,K2,RC).
    pub fn round_material<W: Word>(&self, r: u32) -> RoundMaterial<W> {
        match self.id
        {
            FixtureId::Zero => RoundMaterial {
                k0: Oct::zero(),
                k1: Oct::zero(),
                k2: Oct::zero(),
                rc: Oct::zero(),
            },
            FixtureId::Incrementing =>
            {
                let base = u64::from(r).wrapping_mul(32);
                let mk = |slot: u64| {
                    Oct::<W>::from_u64s(std::array::from_fn(|i| base + slot * 8 + i as u64 + 1))
                };
                RoundMaterial {
                    k0: mk(0),
                    k1: mk(1),
                    k2: mk(2),
                    rc: mk(3),
                }
            },
            FixtureId::HighBit =>
            {
                let hb = if W::BITS >= 64
                {
                    1u64 << 63
                }
                else
                {
                    1u64 << (W::BITS - 1)
                };
                let o = Oct::<W>::from_u64s([hb; 8]);
                RoundMaterial {
                    k0: o,
                    k1: o,
                    k2: o,
                    rc: o,
                }
            },
            FixtureId::OddNorm =>
            {
                let k0 =
                    Self::oct_from_seed::<W>(Self::cell_seed(0x00DD_0000, r, 0).wrapping_add(1));
                let k1 = Self::force_norm_parity(
                    Self::oct_from_seed::<W>(Self::cell_seed(0x0DD1, r, 1)),
                    true,
                );
                let k2 = Self::force_norm_parity(
                    Self::oct_from_seed::<W>(Self::cell_seed(0x0DD2, r, 2)),
                    true,
                );
                let rc = Self::oct_from_seed::<W>(Self::cell_seed(0x0DDC, r, 3));
                RoundMaterial { k0, k1, k2, rc }
            },
            FixtureId::EvenNormZeroDiv =>
            {
                let k0 = Self::oct_from_seed::<W>(Self::cell_seed(0xEEE0, r, 0));
                let k1 = Self::force_norm_parity(
                    Self::oct_from_seed::<W>(Self::cell_seed(0xEEE1, r, 1)),
                    false,
                );
                let k2 = Self::force_norm_parity(
                    Self::oct_from_seed::<W>(Self::cell_seed(0xEEE2, r, 2)),
                    false,
                );
                let rc = Self::oct_from_seed::<W>(Self::cell_seed(0xEEEC, r, 3));
                RoundMaterial { k0, k1, k2, rc }
            },
            FixtureId::PseudoRandom(seed) =>
            {
                let mk = |slot: u32| Self::oct_from_seed::<W>(Self::cell_seed(seed, r, slot));
                RoundMaterial {
                    k0: mk(0),
                    k1: mk(1),
                    k2: mk(2),
                    rc: mk(3),
                }
            },
            FixtureId::Hkdf(key, tweak) =>
            {
                crate::derivation::KeySchedule::new(&key, &tweak).round_material::<W>(r)
            },
        }
    }

    /// Deterministic whitening octonions (spec §12.3). Uses reserved slots.
    pub fn whitening<W: Word>(&self) -> Whitening<W> {
        match self.id
        {
            FixtureId::Zero => Whitening {
                in_l: Oct::zero(),
                in_r: Oct::zero(),
                out_l: Oct::zero(),
                out_r: Oct::zero(),
            },
            FixtureId::Hkdf(key, tweak) =>
            {
                crate::derivation::KeySchedule::new(&key, &tweak).whitening::<W>()
            },
            _ =>
            {
                let base = match self.id
                {
                    FixtureId::PseudoRandom(s) => s,
                    FixtureId::Incrementing => 0x1111,
                    FixtureId::HighBit => 0x2222,
                    FixtureId::OddNorm => 0x3333,
                    FixtureId::EvenNormZeroDiv => 0x4444,
                    FixtureId::Zero => 0,
                    // Hkdf is handled by its own arm above and never reaches here.
                    FixtureId::Hkdf(..) => 0,
                };
                Whitening {
                    in_l: Self::oct_from_seed::<W>(Self::cell_seed(base, 0xFFFF_FFFF, 0)),
                    in_r: Self::oct_from_seed::<W>(Self::cell_seed(base, 0xFFFF_FFFF, 1)),
                    out_l: Self::oct_from_seed::<W>(Self::cell_seed(base, 0xFFFF_FFFE, 0)),
                    out_r: Self::oct_from_seed::<W>(Self::cell_seed(base, 0xFFFF_FFFE, 1)),
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::word::W8;

    #[test]
    fn splitmix_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100
        {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        assert_ne!(SplitMix64::new(1).next_u64(), SplitMix64::new(2).next_u64());
    }

    #[test]
    fn norm_parity_forced() {
        let f = Fixture::new(FixtureId::OddNorm);
        for r in 0..8
        {
            let m = f.round_material::<W8>(r);
            assert_eq!(m.k1.norm().to_u64() & 1, 1, "k1 norm should be odd");
            assert_eq!(m.k2.norm().to_u64() & 1, 1, "k2 norm should be odd");
        }
        let g = Fixture::new(FixtureId::EvenNormZeroDiv);
        for r in 0..8
        {
            let m = g.round_material::<W8>(r);
            assert_eq!(m.k1.norm().to_u64() & 1, 0, "k1 norm should be even");
            assert_eq!(m.k2.norm().to_u64() & 1, 0, "k2 norm should be even");
        }
    }

    #[test]
    fn fixture_reproducible() {
        let f = Fixture::new(FixtureId::PseudoRandom(0xdead_beef));
        let m1 = f.round_material::<W8>(3);
        let m2 = f.round_material::<W8>(3);
        assert_eq!(m1.k1.to_u64s(), m2.k1.to_u64s());
    }
}
