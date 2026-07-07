//! Options — Black-Scholes-Merton pricing, the Greeks, implied volatility, and
//! book-level risk.
//!
//! Everything so far trades the underlying. Options are a different instrument
//! class: a leveraged, convex, *volatility-sensitive* claim. An agent that can
//! price them and read their Greeks can do things spot trading can't — hedge a
//! book's directional risk with a fraction of the capital, express a view on
//! volatility itself, or cap downside with a defined-risk structure. This module
//! gives the standard desk toolkit:
//!
//! * **Black-Scholes-Merton** pricing of European calls/puts, with a continuous
//!   carry/dividend yield `q` (funding, staking yield, or 0).
//! * **The Greeks** — delta, gamma, vega, theta, rho — in market conventions
//!   (vega per 1 vol-point, theta per calendar day, rho per 1 rate-point).
//! * **Implied volatility** — invert the price to the vol the market is pricing,
//!   by robust bracketed bisection (no-arbitrage bounds checked first).
//! * **Analysis** — moneyness, intrinsic/time value, break-even, and the
//!   risk-neutral probability of finishing in the money.
//! * **Book risk** — aggregate a portfolio of option legs into net Greeks and
//!   the spot trade that delta-neutralizes it (delta hedging).
//!
//! Deterministic: closed-form pricing and a fixed-iteration solver — same inputs
//! ⇒ same numbers. Internally computed in `f64` so the Greeks and the IV solve
//! keep their precision, then narrowed to `f32`.

use serde::{Deserialize, Serialize};

/// Standard normal PDF.
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Error function (Abramowitz & Stegun 7.1.26, max error ~1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * xa);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-xa * xa).exp();
    sign * y
}

/// Standard normal CDF.
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// A call or a put.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    pub fn parse(s: &str) -> Option<OptionType> {
        match s.trim().to_lowercase().as_str()
        {
            "call" | "c" => Some(OptionType::Call),
            "put" | "p" => Some(OptionType::Put),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self
        {
            OptionType::Call => "call",
            OptionType::Put => "put",
        }
    }

    fn is_call(self) -> bool {
        matches!(self, OptionType::Call)
    }
}

/// Inputs to the Black-Scholes-Merton model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BsInputs {
    /// Spot price of the underlying.
    pub spot: f32,
    /// Strike price.
    pub strike: f32,
    /// Time to expiry, in **years**.
    pub time_to_expiry: f32,
    /// Continuously-compounded risk-free rate (e.g. 0.05 = 5%).
    pub rate: f32,
    /// Volatility (annualized, e.g. 0.6 = 60%).
    pub vol: f32,
    /// Continuous carry/dividend yield (funding/staking, or 0).
    pub dividend: f32,
}

impl BsInputs {
    /// `(d1, d2, sigma·√T)` in `f64`. `None` if `T ≤ 0` or `σ ≤ 0` (degenerate:
    /// price is intrinsic, no diffusion).
    fn d1_d2(&self) -> Option<(f64, f64, f64)> {
        let s = self.spot as f64;
        let k = self.strike as f64;
        let t = self.time_to_expiry as f64;
        let sigma = self.vol as f64;
        if t <= 0.0 || sigma <= 0.0 || s <= 0.0 || k <= 0.0
        {
            return None;
        }
        let r = self.rate as f64;
        let q = self.dividend as f64;
        let vsqrt = sigma * t.sqrt();
        let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / vsqrt;
        let d2 = d1 - vsqrt;
        Some((d1, d2, vsqrt))
    }
}

/// Intrinsic value at expiry: `max(S−K, 0)` for a call, `max(K−S, 0)` for a put.
pub fn intrinsic_value(spot: f32, strike: f32, ot: OptionType) -> f32 {
    if ot.is_call()
    {
        (spot - strike).max(0.0)
    }
    else
    {
        (strike - spot).max(0.0)
    }
}

/// Black-Scholes-Merton price of a European option.
pub fn price(inp: &BsInputs, ot: OptionType) -> f32 {
    let Some((d1, d2, _)) = inp.d1_d2()
    else
    {
        // No time or no vol ⇒ worth its (discounted) intrinsic; with the tiny
        // horizons this guards, undiscounted intrinsic is the right limit.
        return intrinsic_value(inp.spot, inp.strike, ot);
    };
    let s = inp.spot as f64;
    let k = inp.strike as f64;
    let t = inp.time_to_expiry as f64;
    let r = inp.rate as f64;
    let q = inp.dividend as f64;
    let disc_r = (-r * t).exp();
    let disc_q = (-q * t).exp();
    let p = if ot.is_call()
    {
        s * disc_q * norm_cdf(d1) - k * disc_r * norm_cdf(d2)
    }
    else
    {
        k * disc_r * norm_cdf(-d2) - s * disc_q * norm_cdf(-d1)
    };
    p.max(0.0) as f32
}

/// The Greeks, in market-quoting conventions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Greeks {
    /// ∂price/∂spot — call in `(0,1)`, put in `(−1,0)`.
    pub delta: f32,
    /// ∂delta/∂spot — always ≥ 0 for a long option.
    pub gamma: f32,
    /// Price change per **1 percentage-point** change in volatility.
    pub vega_per_pct: f32,
    /// Price change per **calendar day** of time decay (usually negative long).
    pub theta_per_day: f32,
    /// Price change per **1 percentage-point** change in the rate.
    pub rho_per_pct: f32,
}

impl Greeks {
    fn zero() -> Self {
        Self {
            delta: 0.0,
            gamma: 0.0,
            vega_per_pct: 0.0,
            theta_per_day: 0.0,
            rho_per_pct: 0.0,
        }
    }
}

/// The Greeks of a European option.
pub fn greeks(inp: &BsInputs, ot: OptionType) -> Greeks {
    let Some((d1, d2, vsqrt)) = inp.d1_d2()
    else
    {
        // Degenerate: delta is the moneyness step, everything else ~0.
        let itm = if ot.is_call()
        {
            inp.spot > inp.strike
        }
        else
        {
            inp.spot < inp.strike
        };
        let delta = if !itm
        {
            0.0
        }
        else if ot.is_call()
        {
            1.0
        }
        else
        {
            -1.0
        };
        return Greeks {
            delta,
            ..Greeks::zero()
        };
    };
    let s = inp.spot as f64;
    let k = inp.strike as f64;
    let t = inp.time_to_expiry as f64;
    let r = inp.rate as f64;
    let q = inp.dividend as f64;
    let sigma = inp.vol as f64;
    let disc_r = (-r * t).exp();
    let disc_q = (-q * t).exp();
    let pdf_d1 = norm_pdf(d1);

    let delta = if ot.is_call()
    {
        disc_q * norm_cdf(d1)
    }
    else
    {
        -disc_q * norm_cdf(-d1)
    };
    let gamma = disc_q * pdf_d1 / (s * vsqrt);
    let vega = s * disc_q * pdf_d1 * t.sqrt(); // per 1.0 vol
    // Theta per year.
    let theta_common = -(s * disc_q * pdf_d1 * sigma) / (2.0 * t.sqrt());
    let theta_year = if ot.is_call()
    {
        theta_common - r * k * disc_r * norm_cdf(d2) + q * s * disc_q * norm_cdf(d1)
    }
    else
    {
        theta_common + r * k * disc_r * norm_cdf(-d2) - q * s * disc_q * norm_cdf(-d1)
    };
    let rho = if ot.is_call()
    {
        k * t * disc_r * norm_cdf(d2)
    }
    else
    {
        -k * t * disc_r * norm_cdf(-d2)
    };

    Greeks {
        delta: delta as f32,
        gamma: gamma as f32,
        vega_per_pct: (vega / 100.0) as f32,
        theta_per_day: (theta_year / 365.0) as f32,
        rho_per_pct: (rho / 100.0) as f32,
    }
}

/// Implied volatility: the `vol` that makes the model price equal `market_price`.
/// Robust bracketed bisection over `[1e-4, 5.0]`; returns `None` if the price
/// violates the no-arbitrage bounds (no vol can produce it) or the inputs are
/// degenerate.
pub fn implied_vol(
    market_price: f32,
    spot: f32,
    strike: f32,
    time_to_expiry: f32,
    rate: f32,
    dividend: f32,
    ot: OptionType,
) -> Option<f32> {
    if time_to_expiry <= 0.0 || spot <= 0.0 || strike <= 0.0 || market_price < 0.0
    {
        return None;
    }
    let s = spot as f64;
    let k = strike as f64;
    let t = time_to_expiry as f64;
    let r = rate as f64;
    let q = dividend as f64;
    let disc_r = (-r * t).exp();
    let disc_q = (-q * t).exp();
    // No-arbitrage price bounds for a European option.
    let (lo_bound, hi_bound) = if ot.is_call()
    {
        ((s * disc_q - k * disc_r).max(0.0), s * disc_q)
    }
    else
    {
        ((k * disc_r - s * disc_q).max(0.0), k * disc_r)
    };
    let mp = market_price as f64;
    // Small epsilon tolerance at the boundaries.
    if mp < lo_bound - 1e-6 || mp > hi_bound + 1e-6
    {
        return None;
    }

    let mut lo = 1e-4f64;
    let mut hi = 5.0f64;
    let f = |sigma: f64| -> f64 {
        let inp = BsInputs {
            spot,
            strike,
            time_to_expiry,
            rate,
            vol: sigma as f32,
            dividend,
        };
        price(&inp, ot) as f64 - mp
    };
    let (mut flo, fhi) = (f(lo), f(hi));
    // Price is monotincreasing in vol: expect flo ≤ 0 ≤ fhi to bracket a root.
    if flo > 0.0 || fhi < 0.0
    {
        return None;
    }
    for _ in 0..100
    {
        let mid = 0.5 * (lo + hi);
        let fm = f(mid);
        if fm.abs() < 1e-7 || (hi - lo) < 1e-7
        {
            return Some(mid as f32);
        }
        if (fm < 0.0) == (flo < 0.0)
        {
            lo = mid;
            flo = fm;
        }
        else
        {
            hi = mid;
        }
    }
    Some((0.5 * (lo + hi)) as f32)
}

/// Full single-option analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionAnalysis {
    pub option_type: OptionType,
    pub price: f32,
    pub greeks: Greeks,
    /// Spot ÷ strike.
    pub moneyness: f32,
    pub intrinsic_value: f32,
    pub time_value: f32,
    /// Underlying price at expiry where the buyer breaks even (ignoring carry).
    pub break_even: f32,
    /// Risk-neutral probability of finishing in the money (`N(d2)` call).
    pub prob_itm: f32,
}

/// Price an option and report its full analysis.
pub fn analyze(inp: &BsInputs, ot: OptionType) -> OptionAnalysis {
    let px = price(inp, ot);
    let intrinsic = intrinsic_value(inp.spot, inp.strike, ot);
    let moneyness = if inp.strike.abs() > 1e-9
    {
        inp.spot / inp.strike
    }
    else
    {
        0.0
    };
    let break_even = if ot.is_call()
    {
        inp.strike + px
    }
    else
    {
        inp.strike - px
    };
    let prob_itm = match inp.d1_d2()
    {
        Some((_, d2, _)) =>
        {
            let p = if ot.is_call()
            {
                norm_cdf(d2)
            }
            else
            {
                norm_cdf(-d2)
            };
            p as f32
        },
        None =>
        {
            if intrinsic > 0.0
            {
                1.0
            }
            else
            {
                0.0
            }
        },
    };
    OptionAnalysis {
        option_type: ot,
        price: px,
        greeks: greeks(inp, ot),
        moneyness,
        intrinsic_value: intrinsic,
        time_value: (px - intrinsic).max(0.0),
        break_even,
        prob_itm,
    }
}

/// One position in an options book: a signed quantity of a specific contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionLeg {
    /// Signed contracts: positive = long, negative = short.
    pub quantity: f32,
    pub inputs: BsInputs,
    pub option_type: OptionType,
}

/// Net risk of an options book.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookGreeks {
    pub net_value: f32,
    pub net_delta: f32,
    pub net_gamma: f32,
    pub net_vega_per_pct: f32,
    pub net_theta_per_day: f32,
    pub net_rho_per_pct: f32,
    /// Spot units to trade to delta-neutralize: `−net_delta` (negative ⇒ sell
    /// spot, positive ⇒ buy spot). Spot has delta 1.
    pub delta_hedge_spot: f32,
}

/// Aggregate a book of option legs into net Greeks and the delta hedge.
pub fn book_greeks(legs: &[OptionLeg]) -> BookGreeks {
    let mut b = BookGreeks {
        net_value: 0.0,
        net_delta: 0.0,
        net_gamma: 0.0,
        net_vega_per_pct: 0.0,
        net_theta_per_day: 0.0,
        net_rho_per_pct: 0.0,
        delta_hedge_spot: 0.0,
    };
    for leg in legs
    {
        let q = leg.quantity;
        let px = price(&leg.inputs, leg.option_type);
        let g = greeks(&leg.inputs, leg.option_type);
        b.net_value += q * px;
        b.net_delta += q * g.delta;
        b.net_gamma += q * g.gamma;
        b.net_vega_per_pct += q * g.vega_per_pct;
        b.net_theta_per_day += q * g.theta_per_day;
        b.net_rho_per_pct += q * g.rho_per_pct;
    }
    b.delta_hedge_spot = -b.net_delta;
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atm() -> BsInputs {
        BsInputs {
            spot: 100.0,
            strike: 100.0,
            time_to_expiry: 1.0,
            rate: 0.05,
            vol: 0.2,
            dividend: 0.0,
        }
    }

    #[test]
    fn norm_cdf_known_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((norm_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!((norm_cdf(-1.96) - 0.025).abs() < 1e-3);
    }

    #[test]
    fn black_scholes_textbook_values() {
        // S=K=100, T=1, r=5%, sigma=20% ⇒ call≈10.4506, put≈5.5735.
        let inp = atm();
        assert!((price(&inp, OptionType::Call) - 10.4506).abs() < 1e-2);
        assert!((price(&inp, OptionType::Put) - 5.5735).abs() < 1e-2);
    }

    #[test]
    fn put_call_parity() {
        // C − P = S·e^{−qT} − K·e^{−rT}.
        let inp = BsInputs {
            dividend: 0.03,
            ..atm()
        };
        let c = price(&inp, OptionType::Call) as f64;
        let p = price(&inp, OptionType::Put) as f64;
        let lhs = c - p;
        let rhs = inp.spot as f64 * (-inp.dividend as f64 * inp.time_to_expiry as f64).exp()
            - inp.strike as f64 * (-inp.rate as f64 * inp.time_to_expiry as f64).exp();
        assert!((lhs - rhs).abs() < 1e-2, "parity {lhs} vs {rhs}");
    }

    #[test]
    fn greek_signs_and_ranges() {
        let g_call = greeks(&atm(), OptionType::Call);
        let g_put = greeks(&atm(), OptionType::Put);
        assert!(g_call.delta > 0.0 && g_call.delta < 1.0);
        assert!(g_put.delta < 0.0 && g_put.delta > -1.0);
        assert!(g_call.gamma > 0.0);
        assert!(g_call.vega_per_pct > 0.0);
        assert!(
            g_call.theta_per_day < 0.0,
            "long call theta {}",
            g_call.theta_per_day
        );
        // Call/put gamma and vega are equal.
        assert!((g_call.gamma - g_put.gamma).abs() < 1e-6);
        assert!((g_call.vega_per_pct - g_put.vega_per_pct).abs() < 1e-4);
    }

    #[test]
    fn call_delta_deep_itm_near_one() {
        let inp = BsInputs {
            spot: 200.0,
            ..atm()
        };
        let g = greeks(&inp, OptionType::Call);
        assert!(g.delta > 0.9, "deep-itm delta {}", g.delta);
    }

    #[test]
    fn implied_vol_round_trips() {
        let mut inp = atm();
        inp.vol = 0.35;
        let mkt = price(&inp, OptionType::Call);
        let iv = implied_vol(
            mkt,
            inp.spot,
            inp.strike,
            inp.time_to_expiry,
            inp.rate,
            0.0,
            OptionType::Call,
        )
        .unwrap();
        assert!((iv - 0.35).abs() < 1e-3, "iv {iv}");
    }

    #[test]
    fn implied_vol_rejects_arbitrage_price() {
        // A call can't cost more than the (discounted) spot.
        let iv = implied_vol(150.0, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call);
        assert!(iv.is_none());
    }

    #[test]
    fn expiry_price_is_intrinsic() {
        let inp = BsInputs {
            spot: 110.0,
            time_to_expiry: 0.0,
            ..atm()
        };
        assert!((price(&inp, OptionType::Call) - 10.0).abs() < 1e-4);
        assert!(price(&inp, OptionType::Put).abs() < 1e-4);
    }

    #[test]
    fn analysis_fields_consistent() {
        let a = analyze(
            &BsInputs {
                spot: 110.0,
                ..atm()
            },
            OptionType::Call,
        );
        assert!((a.moneyness - 1.1).abs() < 1e-4);
        assert!((a.intrinsic_value - 10.0).abs() < 1e-4);
        assert!(a.time_value > 0.0);
        assert!(a.price >= a.intrinsic_value);
        assert!(a.prob_itm > 0.5 && a.prob_itm < 1.0);
        assert!((a.break_even - (100.0 + a.price)).abs() < 1e-4);
    }

    #[test]
    fn book_delta_hedge_neutralizes() {
        // Long 2 ATM calls (delta ~0.6 each) ⇒ net delta ~1.2 ⇒ sell ~1.2 spot.
        let legs = vec![OptionLeg {
            quantity: 2.0,
            inputs: atm(),
            option_type: OptionType::Call,
        }];
        let b = book_greeks(&legs);
        assert!(b.net_delta > 0.0);
        assert!((b.delta_hedge_spot + b.net_delta).abs() < 1e-6);
        assert!(b.net_value > 0.0);
        assert!(b.net_gamma > 0.0);
    }

    #[test]
    fn book_straddle_is_delta_light_vega_heavy() {
        // Long ATM call + long ATM put at zero carry ⇒ near delta-neutral (an
        // ATM-spot straddle is only delta-neutral at zero rate; the forward
        // drift otherwise tilts it long), and strongly long vega.
        let zr = BsInputs { rate: 0.0, ..atm() };
        let legs = vec![
            OptionLeg {
                quantity: 1.0,
                inputs: zr.clone(),
                option_type: OptionType::Call,
            },
            OptionLeg {
                quantity: 1.0,
                inputs: zr,
                option_type: OptionType::Put,
            },
        ];
        let b = book_greeks(&legs);
        assert!(b.net_delta.abs() < 0.15, "straddle delta {}", b.net_delta);
        assert!(b.net_vega_per_pct > 0.0);
    }

    #[test]
    fn deterministic() {
        let a = analyze(&atm(), OptionType::Call);
        let b = analyze(&atm(), OptionType::Call);
        assert_eq!(a.price, b.price);
        assert_eq!(a.greeks.delta, b.greeks.delta);
    }
}
