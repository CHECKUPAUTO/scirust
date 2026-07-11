//! # scirust-finmigrate — INTACCR migration unit
//!
//! Audit-gated COBOL→Rust port of the monthly interest-accrual core
//! (`cobol/INTACCR.cbl`). The legacy semantic contract is pinned in
//! `cobol/SEMANTICS.md`; the pre-migration audit is `audit_report.md`; the
//! decision log is `audit_trail.log`.
//!
//! ## Guarantees
//! * **No floating point in the money path.** Every monetary value is
//!   [`rust_decimal::Decimal`]. There is deliberately no `f32`/`f64` anywhere in
//!   this module (audit_report Gap-1).
//! * **COBOL-exact rounding.** `ROUNDED` ⇒ [`RoundingStrategy::MidpointAwayFromZero`]
//!   (NEAREST-AWAY-FROM-ZERO, the IBM default); the un-`ROUNDED` companion ⇒
//!   [`RoundingStrategy::ToZero`] (truncation). One rounding event, at the store.
//! * **Reversibility.** [`MigrationUnit`] can dispatch to the Rust port or fall
//!   back to a `Legacy` shim, so a unit can be swapped back in production if an
//!   audit fails (Phase 3).
//! * **Error reconstruction.** Every call captures its input context in an
//!   [`AccrualTrace`]; a production divergence can be replayed bit-for-bit in the
//!   sandbox via [`replay`].

use rust_decimal::{Decimal, RoundingStrategy};

/// Fractional digits of a `PIC ...V99` money field.
pub const MONEY_SCALE: u32 = 2;
/// Fractional digits of the `PIC SV9(5)` rate field.
pub const RATE_SCALE: u32 = 5;

/// `|value|` must not exceed `PIC S9(9)V99` = 999_999_999.99.
fn money_max() -> Decimal {
    Decimal::new(99_999_999_999, MONEY_SCALE)
}

fn twelve() -> Decimal {
    Decimal::from(12)
}

/// Result of one accrual, mirroring the WORKING-STORAGE outputs of INTACCR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accrual {
    /// `WS-MONTHLY-INT` — interest posted, ROUNDED (away from zero) to 2 dp.
    pub monthly_int: Decimal,
    /// `WS-MONTHLY-TRUNC` — interest under the COBOL default (truncated) to 2 dp.
    pub monthly_trunc: Decimal,
    /// `WS-NEW-BALANCE` — principal plus the ROUNDED interest.
    pub new_balance: Decimal,
}

/// A stored money field would exceed `PIC S9(9)V99`.
///
/// The legacy program codes no `ON SIZE ERROR`, so it would *silently* truncate
/// high-order digits. The port refuses to corrupt silently and stops loudly
/// instead (audit_report Gap-5). This is a documented, intentional divergence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccrualError {
    /// Which field overflowed, and the offending value (pre-truncation).
    SizeError { field: &'static str, value: Decimal },
}

/// Coerce a value into a fixed-scale money field using COBOL ROUNDED semantics.
fn store_money_rounded(value: Decimal) -> Decimal {
    value.round_dp_with_strategy(MONEY_SCALE, RoundingStrategy::MidpointAwayFromZero)
}

/// Coerce into a money field using the COBOL default (truncate toward zero).
fn store_money_trunc(value: Decimal) -> Decimal {
    value.round_dp_with_strategy(MONEY_SCALE, RoundingStrategy::ToZero)
}

fn check_size(field: &'static str, value: Decimal) -> Result<Decimal, AccrualError> {
    if value.abs() > money_max()
    {
        Err(AccrualError::SizeError { field, value })
    }
    else
    {
        Ok(value)
    }
}

/// Port of INTACCR's PROCEDURE DIVISION. Decimal-exact, COBOL-faithful.
///
/// ```
/// use rust_decimal::Decimal;
/// use std::str::FromStr;
/// use scirust_finmigrate::accrue;
/// // +0.005 half-cent tie rounds AWAY from zero (not banker's) -> 0.01.
/// let a = accrue(Decimal::from_str("100.00").unwrap(),
///                Decimal::from_str("0.00060").unwrap()).unwrap();
/// assert_eq!(a.monthly_int, Decimal::from_str("0.01").unwrap());
/// assert_eq!(a.monthly_trunc, Decimal::from_str("0.00").unwrap());
/// ```
pub fn accrue(principal: Decimal, annual_rate: Decimal) -> Result<Accrual, AccrualError> {
    // Fields carry their declared fixed scale (enforced on store, Gap-2).
    let principal = store_money_rounded(principal);
    let annual_rate =
        annual_rate.round_dp_with_strategy(RATE_SCALE, RoundingStrategy::MidpointAwayFromZero);
    check_size("principal", principal)?;

    // principal * rate is exact; / 12 carried at full Decimal precision. Rounding
    // happens ONCE, at each store below — never per sub-operation (Gap-4).
    let intermediate = (principal * annual_rate) / twelve();

    let monthly_int = check_size("monthly_int", store_money_rounded(intermediate))?;
    let monthly_trunc = check_size("monthly_trunc", store_money_trunc(intermediate))?;
    // Both operands are already 2 dp; the sum is exact at 2 dp (Gap: none).
    let new_balance = check_size("new_balance", principal + monthly_int)?;

    Ok(Accrual {
        monthly_int,
        monthly_trunc,
        new_balance,
    })
}

// ============================================================================
// Phase 3 — Reversibility
// ============================================================================

/// Backend behind a migration unit. Lets production swap the Rust port back to
/// the legacy path if an audit fails, without touching call sites.
pub trait AccrualBackend {
    fn accrue(&self, principal: Decimal, annual_rate: Decimal) -> Result<Accrual, AccrualError>;
    fn name(&self) -> &'static str;
}

/// The migrated Rust implementation.
pub struct RustAccrual;
impl AccrualBackend for RustAccrual {
    fn accrue(&self, p: Decimal, r: Decimal) -> Result<Accrual, AccrualError> {
        accrue(p, r)
    }
    fn name(&self) -> &'static str {
        "rust"
    }
}

/// A slot for the legacy path (e.g. an FFI/COBOL bridge or a service call). Left
/// unimplemented on purpose: reversibility is an *architecture*, and wiring the
/// real legacy call is a deployment decision, not something to fake here.
pub struct LegacyAccrual;
impl AccrualBackend for LegacyAccrual {
    fn accrue(&self, _p: Decimal, _r: Decimal) -> Result<Accrual, AccrualError> {
        unimplemented!("bind the legacy INTACCR path (FFI/service) at deploy time")
    }
    fn name(&self) -> &'static str {
        "legacy"
    }
}

/// Which backend a [`MigrationUnit`] dispatches to. Flip to [`Route::Legacy`] to
/// reverse the migration in production.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Rust,
    Legacy,
}

/// A reversible, traced migration unit. Dispatches to the selected backend and
/// records the input context of every call for error reconstruction.
pub struct MigrationUnit {
    route: Route,
    rust: RustAccrual,
    legacy: LegacyAccrual,
}

impl Default for MigrationUnit {
    fn default() -> Self {
        Self {
            route: Route::Rust,
            rust: RustAccrual,
            legacy: LegacyAccrual,
        }
    }
}

impl MigrationUnit {
    pub fn new(route: Route) -> Self {
        Self {
            route,
            ..Self::default()
        }
    }

    /// Reverse (or re-apply) the migration at runtime.
    pub fn set_route(&mut self, route: Route) {
        self.route = route;
    }

    pub fn route(&self) -> Route {
        self.route
    }

    /// Run one accrual, returning the result together with a replayable trace of
    /// the exact input state (Phase 3 — error reconstruction).
    pub fn run(
        &self,
        principal: Decimal,
        annual_rate: Decimal,
    ) -> (Result<Accrual, AccrualError>, AccrualTrace) {
        let backend: &dyn AccrualBackend = match self.route
        {
            Route::Rust => &self.rust,
            Route::Legacy => &self.legacy,
        };
        let result = backend.accrue(principal, annual_rate);
        let trace = AccrualTrace {
            backend: backend.name(),
            principal,
            annual_rate,
            ok: result.is_ok(),
        };
        (result, trace)
    }
}

/// The exact input context of one call — everything needed to reproduce it in
/// the sandbox. Serialize this from production logs and feed it to [`replay`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccrualTrace {
    pub backend: &'static str,
    pub principal: Decimal,
    pub annual_rate: Decimal,
    pub ok: bool,
}

impl std::fmt::Display for AccrualTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "INTACCR backend={} principal={} annual_rate={} ok={}",
            self.backend, self.principal, self.annual_rate, self.ok
        )
    }
}

/// Deterministically reproduce a call from its captured input context. Given a
/// production trace of a divergent/erroring call, this re-runs the Rust port on
/// the identical inputs so the failure can be debugged in the sandbox.
pub fn replay(trace: &AccrualTrace) -> Result<Accrual, AccrualError> {
    accrue(trace.principal, trace.annual_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn d(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }

    #[test]
    fn rounded_is_away_from_zero_not_bankers() {
        // 0.005 -> 0.01, 0.025 -> 0.03 (banker's would give 0.00 and 0.02).
        assert_eq!(
            accrue(d("100.00"), d("0.00060")).unwrap().monthly_int,
            d("0.01")
        );
        assert_eq!(
            accrue(d("100.00"), d("0.00300")).unwrap().monthly_int,
            d("0.03")
        );
    }

    #[test]
    fn negative_tie_rounds_away_from_zero() {
        let a = accrue(d("-100.00"), d("0.00060")).unwrap();
        assert_eq!(a.monthly_int, d("-0.01"));
        assert_eq!(a.monthly_trunc, d("0.00")); // -0.005 truncated toward zero
    }

    #[test]
    fn truncation_differs_from_rounding() {
        let a = accrue(d("1200.40"), d("0.02999")).unwrap();
        assert_eq!(a.monthly_int, d("3.00"));
        assert_eq!(a.monthly_trunc, d("2.99"));
    }

    #[test]
    fn posting_is_exact() {
        let a = accrue(d("2500.75"), d("0.03500")).unwrap();
        assert_eq!(a.monthly_int, d("7.29"));
        assert_eq!(a.new_balance, d("2508.04"));
    }

    #[test]
    fn size_error_on_overflow_instead_of_silent_truncation() {
        // 999_999_999.99 at 50% annual -> monthly ~41.6M, new_balance overflows.
        let e = accrue(d("999999999.99"), d("0.50000")).unwrap_err();
        assert_eq!(
            e,
            AccrualError::SizeError {
                field: "new_balance",
                value: d("1041666666.66"),
            }
        );
    }

    #[test]
    fn max_principal_zero_rate_is_valid_boundary() {
        let a = accrue(d("999999999.99"), d("0.00000")).unwrap();
        assert_eq!(a.new_balance, d("999999999.99"));
    }

    #[test]
    fn reversibility_switches_backend() {
        let mut unit = MigrationUnit::default();
        assert_eq!(unit.route(), Route::Rust);
        let (res, trace) = unit.run(d("2500.75"), d("0.03500"));
        assert_eq!(res.unwrap().monthly_int, d("7.29"));
        assert_eq!(trace.backend, "rust");
        unit.set_route(Route::Legacy);
        assert_eq!(unit.route(), Route::Legacy);
    }

    #[test]
    fn replay_reproduces_from_trace() {
        let unit = MigrationUnit::default();
        let (orig, trace) = unit.run(d("100.00"), d("0.00060"));
        assert_eq!(replay(&trace), orig);
    }
}
