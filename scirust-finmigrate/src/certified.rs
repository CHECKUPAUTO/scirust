//! Certified round-trip bounds for fixed-point decimal fields — the
//! `scirust-core::certified_numerics` idea (a *proven* round-trip error bound
//! plus an explicit domain of validity) transferred into this crate's
//! **exact-decimal, float-free** world.
//!
//! The core crate bounds the round-trip error of an `f64` representation change
//! with `κ_rt(x)·B_ENC + B_DEC` ulps. That machinery is binary-floating-point
//! and would introduce the very thing this crate forbids in the money path
//! (see the module header of [`crate`] and `tests/no_float_guard.rs`). So this
//! module does **not** depend on it; it states the analogous certificate for a
//! `PIC S9(int)V(frac)` fixed-point field, where the arithmetic is
//! [`rust_decimal::Decimal`] and every bound below is *exact*, not an estimate:
//!
//! * **Exactness domain.** A value already on the field's grid (in magnitude
//!   range, no fractional digit finer than `frac`) round-trips through the
//!   store operation with **zero** error — the decimal analogue of the core's
//!   "κ_rt = 1, admissible" case.
//! * **Rounded store bound.** Storing any in-range value with COBOL `ROUNDED`
//!   ([`RoundingStrategy::MidpointAwayFromZero`]) moves it by at most **half a
//!   ULP** of the field (`½·10⁻ᶠʳᵃᶜ`). This is [`FixedPointField::round_trip_bound_rounded`].
//! * **Truncated store bound.** The COBOL default (truncate toward zero,
//!   [`RoundingStrategy::ToZero`]) moves it by strictly less than **one ULP**
//!   (`10⁻ᶠʳᵃᶜ`) — [`FixedPointField::round_trip_bound_truncated`].
//!
//! The certificate is meaningful precisely because it is stated against the
//! **same** store operations the money path actually uses: the tests below pin
//! [`money_field`]'s rounding to `crate::store_money_rounded` /
//! `store_money_trunc` bit-for-bit, and check (over a deterministic sweep, in
//! the spirit of ANEE Phase D's D3 "bounds sound on all observed data") that
//! the observed round-trip error never exceeds the certified bound.

use rust_decimal::{Decimal, RoundingStrategy};

/// A fixed-point decimal field `PIC S9(int_digits)V(frac_digits)`: signed,
/// with `int_digits` digits left of the implied point and `frac_digits`
/// right of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedPointField {
    /// Digits left of the decimal point (the `9(n)` count).
    pub int_digits: u32,
    /// Digits right of the decimal point (the `V9(m)` count) — the scale.
    pub frac_digits: u32,
}

impl FixedPointField {
    /// A field with the given integer/fractional digit counts.
    ///
    /// `int_digits + frac_digits` must be at most 28 (the exact-precision
    /// ceiling of [`rust_decimal::Decimal`]); a wider field cannot be
    /// represented exactly and would defeat the point of a certificate.
    pub const fn new(int_digits: u32, frac_digits: u32) -> Self {
        assert!(
            int_digits + frac_digits <= 28,
            "field exceeds Decimal's 28-digit exact-precision ceiling"
        );
        Self {
            int_digits,
            frac_digits,
        }
    }

    /// One unit in the last place of the field: `10⁻ᶠʳᵃᶜ`.
    pub fn ulp(&self) -> Decimal {
        Decimal::new(1, self.frac_digits)
    }

    /// The largest representable magnitude: `(10^(int+frac) − 1) · 10⁻ᶠʳᵃᶜ`
    /// (e.g. `999_999_999.99` for `PIC S9(9)V99`).
    pub fn max_magnitude(&self) -> Decimal {
        let digits = self.int_digits + self.frac_digits;
        let numerator = 10i128.pow(digits) - 1;
        Decimal::from_i128_with_scale(numerator, self.frac_digits)
    }

    /// Whether `x` is within the field's magnitude range (independent of its
    /// fractional precision).
    pub fn in_domain(&self, x: Decimal) -> bool {
        x.abs() <= self.max_magnitude()
    }

    /// Store `x` with COBOL `ROUNDED` semantics (nearest, ties away from zero).
    pub fn store_rounded(&self, x: Decimal) -> Decimal {
        x.round_dp_with_strategy(self.frac_digits, RoundingStrategy::MidpointAwayFromZero)
    }

    /// Store `x` with the COBOL default (truncate toward zero).
    pub fn store_truncated(&self, x: Decimal) -> Decimal {
        x.round_dp_with_strategy(self.frac_digits, RoundingStrategy::ToZero)
    }

    /// Whether `x` sits exactly on the field's grid: in range, and already
    /// equal to its own rounded store (no fractional digit finer than `frac`).
    /// Such values round-trip with **zero** error under either store.
    pub fn is_exactly_representable(&self, x: Decimal) -> bool {
        self.in_domain(x) && self.store_rounded(x) == x
    }

    /// Certified round-trip bound for the `ROUNDED` store: `½·10⁻ᶠʳᵃᶜ`. For
    /// every in-range `x`, `|store_rounded(x) − x| ≤` this value.
    pub fn round_trip_bound_rounded(&self) -> Decimal {
        // ½·10⁻ᶠʳᵃᶜ = 5·10⁻⁽ᶠʳᵃᶜ⁺¹⁾, exact.
        Decimal::new(5, self.frac_digits + 1)
    }

    /// Certified round-trip bound for the truncating store: `10⁻ᶠʳᵃᶜ` (one
    /// ULP). For every in-range `x`, `|store_truncated(x) − x| <` this value.
    pub fn round_trip_bound_truncated(&self) -> Decimal {
        self.ulp()
    }
}

/// The `PIC S9(9)V99` money field this crate stores balances and interest in
/// ([`crate::MONEY_SCALE`]). Its rounding certificate covers
/// `crate::store_money_rounded` / `store_money_trunc` exactly (pinned by test).
pub fn money_field() -> FixedPointField {
    FixedPointField::new(9, crate::MONEY_SCALE)
}

/// The `PIC SV9(5)` annual-rate field ([`crate::RATE_SCALE`]).
pub fn rate_field() -> FixedPointField {
    FixedPointField::new(0, crate::RATE_SCALE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    #[test]
    fn field_constants_match_the_crate_money_field() {
        let f = money_field();
        assert_eq!(f.frac_digits, crate::MONEY_SCALE);
        assert_eq!(f.ulp(), Decimal::from_str("0.01").unwrap());
        assert_eq!(
            f.round_trip_bound_rounded(),
            Decimal::from_str("0.005").unwrap()
        );
        // max_magnitude equals the crate's own private money_max (999_999_999.99).
        assert_eq!(
            f.max_magnitude(),
            Decimal::from_str("999999999.99").unwrap()
        );
    }

    /// The certificate is only meaningful if it describes the *actual* store
    /// operations the money path uses. Pin both to the crate's functions.
    #[test]
    fn certified_field_stores_match_the_crate_money_stores() {
        let f = money_field();
        for raw in [
            "0",
            "1.005",
            "-1.005",
            "2.994",
            "2.995",
            "2.996",
            "-2.995",
            "123.456789",
            "0.004999",
            "0.005",
            "999999999.994",
            "-999999999.994",
        ]
        {
            let x = Decimal::from_str(raw).unwrap();
            assert_eq!(
                f.store_rounded(x),
                crate::store_money_rounded(x),
                "rounded store diverged from the crate for {raw}"
            );
            assert_eq!(
                f.store_truncated(x),
                crate::store_money_trunc(x),
                "truncated store diverged from the crate for {raw}"
            );
        }
    }

    /// Exactness domain: grid values round-trip with zero error under both
    /// stores (the decimal analogue of the core's κ_rt = 1 admissible case).
    #[test]
    fn on_grid_values_round_trip_exactly() {
        let f = money_field();
        for raw in [
            "0",
            "0.01",
            "-0.01",
            "1.50",
            "1234.56",
            "999999999.99",
            "-42.00",
        ]
        {
            let x = Decimal::from_str(raw).unwrap();
            assert!(f.is_exactly_representable(x), "{raw} should be on-grid");
            assert_eq!(f.store_rounded(x), x);
            assert_eq!(f.store_truncated(x), x);
        }
        // A value finer than the grid is NOT exactly representable.
        assert!(!f.is_exactly_representable(Decimal::from_str("1.234").unwrap()));
    }

    /// The certificate holds on every sampled value: observed round-trip error
    /// never exceeds the certified bound, for both stores, over a deterministic
    /// sweep of over-precise decimals (ANEE Phase D D3 discipline: bounds sound
    /// on all observed data). Also confirms the rounded bound is *tight* — the
    /// half-cent tie realizes it exactly.
    #[test]
    fn observed_round_trip_error_never_exceeds_the_certified_bound() {
        let f = money_field();
        let rounded_bound = f.round_trip_bound_rounded();
        let trunc_bound = f.round_trip_bound_truncated();
        let mut tight_witness = false;

        // Sweep k·10⁻⁶ over a wide integer range: dense sub-cent structure.
        for k in -250_000i64..=250_000
        {
            let x = Decimal::from_i128_with_scale(i128::from(k), 6);

            let r_err = (f.store_rounded(x) - x).abs();
            assert!(
                r_err <= rounded_bound,
                "rounded round-trip error {r_err} exceeds bound {rounded_bound} at {x}"
            );
            if r_err == rounded_bound
            {
                tight_witness = true;
            }

            let t_err = (f.store_truncated(x) - x).abs();
            assert!(
                t_err < trunc_bound || t_err.is_zero(),
                "truncated round-trip error {t_err} not below one ULP {trunc_bound} at {x}"
            );
        }
        assert!(
            tight_witness,
            "the rounded bound should be realized exactly by a half-ULP tie"
        );
    }

    #[test]
    fn rate_field_certificate_is_well_formed() {
        let f = rate_field();
        assert_eq!(f.ulp(), Decimal::from_str("0.00001").unwrap());
        assert_eq!(
            f.round_trip_bound_rounded(),
            Decimal::from_str("0.000005").unwrap()
        );
        // PIC SV9(5): no integer digits, so |x| < 1.
        assert_eq!(f.max_magnitude(), Decimal::from_str("0.99999").unwrap());
        assert!(f.is_exactly_representable(Decimal::from_str("0.12345").unwrap()));
        assert!(!f.is_exactly_representable(Decimal::from_str("0.123456").unwrap()));
    }
}
