//! Honest lottery mathematics: exact odds, expected value, draw-fairness
//! testing.
//!
//! # What this module will never do
//!
//! Fair lottery draws are **independent, uniformly random events**: the
//! machine has no memory, so every combination has exactly the same
//! probability at every draw, regardless of the entire history. No algorithm
//! — frequency analysis, "hot/cold numbers", machine learning, or anything
//! else — can predict outcomes better than chance; products claiming
//! otherwise monetize the gambler's fallacy (Clotfelter & Cook, *The
//! Gambler's Fallacy in Lottery Play*, Management Science 39(12), 1993).
//! Accordingly there is no `predict` function here and there never will be.
//!
//! # What honest mathematics can do
//!
//! - compute the **exact probability of every outcome** of a
//!   `k`-of-`n` (+ bonus) game — the hypergeometric law
//!   (see [`crate::discrete::Hypergeometric`]);
//! - quantify the **expected value of a ticket** against a prize table
//!   (negative for every real game — that margin is how lotteries are
//!   funded);
//! - **test a draw history for uniformity** (χ² on number frequencies), the
//!   standard audit that regulators run — which detects a *broken or rigged*
//!   generator, not a predictable one.
//!
//! Odds computed here reproduce the officially published tables exactly
//! (Powerball 1 in 292,201,338; EuroMillions 1 in 139,838,160; Loto français
//! 1 in 19,068,840 — see the tests).

use crate::comb::binomial;
use crate::discrete::{DiscreteDistribution, Hypergeometric};
use crate::htest::{TestResult, chi_square_gof};

/// A fixed-matrix lottery: `main_picks` numbers drawn from `main_pool`, plus
/// an optional independent bonus machine (`bonus_picks` from `bonus_pool`).
///
/// The player is assumed to pick as many numbers as are drawn (the standard
/// single-ticket rules of 6/49, EuroMillions, Powerball, Loto…). Matching
/// `m` main numbers and `b` bonus numbers is then the product of two
/// independent hypergeometric masses.
#[derive(Debug, Clone, Copy)]
pub struct LotteryGame {
    main_pool: u64,
    main_picks: u64,
    bonus_pool: u64,
    bonus_picks: u64,
}

impl LotteryGame {
    /// Single-pool game: draw `main_picks` from `main_pool` (e.g. 6-of-49).
    pub fn new(main_pool: u64, main_picks: u64) -> Self {
        Self::with_bonus(main_pool, main_picks, 0, 0)
    }

    /// Game with an independent bonus machine (e.g. EuroMillions: 5 of 50
    /// plus 2 "stars" of 12). Pass `bonus_pool = bonus_picks = 0` for none.
    pub fn with_bonus(main_pool: u64, main_picks: u64, bonus_pool: u64, bonus_picks: u64) -> Self {
        assert!(
            main_picks >= 1 && main_picks <= main_pool,
            "LotteryGame: require 1 ≤ main_picks ≤ main_pool"
        );
        assert!(
            bonus_picks <= bonus_pool,
            "LotteryGame: require bonus_picks ≤ bonus_pool"
        );
        assert!(
            (bonus_pool == 0) == (bonus_picks == 0),
            "LotteryGame: a bonus machine needs both a pool and a pick count"
        );
        Self {
            main_pool,
            main_picks,
            bonus_pool,
            bonus_picks,
        }
    }

    /// The classic single-pool 6-of-49 matrix.
    pub fn lotto_6of49() -> Self {
        Self::new(49, 6)
    }
    /// Loto français : 5 of 49 + 1 « numéro chance » of 10 (matrix since
    /// 2008). Jackpot odds 1 in 19,068,840.
    pub fn loto_france() -> Self {
        Self::with_bonus(49, 5, 10, 1)
    }
    /// EuroMillions: 5 of 50 + 2 stars of 12 (matrix since Sept 2016).
    /// Jackpot odds 1 in 139,838,160.
    pub fn euromillions() -> Self {
        Self::with_bonus(50, 5, 12, 2)
    }
    /// US Powerball: 5 of 69 + 1 Powerball of 26 (matrix since Oct 2015).
    /// Jackpot odds 1 in 292,201,338.
    pub fn powerball() -> Self {
        Self::with_bonus(69, 5, 26, 1)
    }

    /// Exact number of equally likely outcomes,
    /// `C(main_pool, main_picks) · C(bonus_pool, bonus_picks)`;
    /// `None` only on u128 overflow (no real-world matrix comes close).
    pub fn total_combinations(&self) -> Option<u128> {
        let main = binomial(self.main_pool, self.main_picks)?;
        if self.bonus_pool == 0
        {
            return Some(main);
        }
        main.checked_mul(binomial(self.bonus_pool, self.bonus_picks)?)
    }

    /// Probability that one ticket matches exactly `main_matched` main
    /// numbers and exactly `bonus_matched` bonus numbers.
    ///
    /// Zero outside the attainable range. For a game without a bonus
    /// machine, pass `bonus_matched = 0`.
    pub fn p_match(&self, main_matched: u64, bonus_matched: u64) -> f64 {
        let p_main =
            Hypergeometric::new(self.main_pool, self.main_picks, self.main_picks).pmf(main_matched);
        let p_bonus = if self.bonus_pool == 0
        {
            f64::from(u8::from(bonus_matched == 0))
        }
        else
        {
            Hypergeometric::new(self.bonus_pool, self.bonus_picks, self.bonus_picks)
                .pmf(bonus_matched)
        };
        p_main * p_bonus
    }

    /// The published-style "1 in X" figure for an exact match tier:
    /// `1 / p_match` (`∞` for unattainable tiers).
    pub fn odds_against(&self, main_matched: u64, bonus_matched: u64) -> f64 {
        1.0 / self.p_match(main_matched, bonus_matched)
    }

    /// Probability of the jackpot (all main and all bonus numbers matched).
    pub fn p_jackpot(&self) -> f64 {
        self.p_match(self.main_picks, self.bonus_picks)
    }

    /// Expected gross winnings of one ticket against a prize table.
    ///
    /// Tiers must be disjoint `(main_matched, bonus_matched)` outcomes; any
    /// outcome not listed pays nothing.
    pub fn expected_gain(&self, tiers: &[PrizeTier]) -> f64 {
        tiers
            .iter()
            .map(|t| self.p_match(t.main_matched, t.bonus_matched) * t.prize)
            .sum()
    }

    /// Expected **net** result of buying one ticket:
    /// `expected_gain − ticket_price`.
    ///
    /// Negative for every real lottery — the house margin is structural, and
    /// no choice of numbers changes it.
    pub fn expected_net(&self, tiers: &[PrizeTier], ticket_price: f64) -> f64 {
        self.expected_gain(tiers) - ticket_price
    }
}

/// One prize tier: exact match counts and the prize paid for them.
#[derive(Debug, Clone, Copy)]
pub struct PrizeTier {
    /// Exact number of main numbers matched.
    pub main_matched: u64,
    /// Exact number of bonus numbers matched (0 for games without a bonus).
    pub bonus_matched: u64,
    /// Prize paid for this tier, in whatever currency unit the caller uses.
    pub prize: f64,
}

/// χ² uniformity test on a draw history: `counts_per_number[i]` is how many
/// times ball `i` appeared over the whole history.
///
/// Under the null hypothesis of a fair machine every ball is equally likely,
/// so the expected count is uniform. A small p-value flags a *defective or
/// rigged* generator — it does **not** make future draws predictable, and a
/// fair machine will still produce ~5% of p-values below 0.05.
///
/// Caveat: within one draw the balls are drawn without replacement, so counts
/// are weakly negatively correlated across numbers; the χ² reference
/// distribution is the standard, slightly conservative auditor's
/// approximation. Requires every expected count ≥ 5 to be meaningful (the
/// usual χ² rule of thumb); returns `None` if there are fewer than two
/// numbers or no observations, like [`chi_square_gof`].
pub fn draw_frequency_chi_square(counts_per_number: &[u64]) -> Option<TestResult> {
    if counts_per_number.len() < 2
    {
        return None;
    }
    let observed: Vec<f64> = counts_per_number.iter().map(|&c| c as f64).collect();
    let total: f64 = observed.iter().sum();
    if total <= 0.0
    {
        return None;
    }
    let expected = vec![total / observed.len() as f64; observed.len()];
    chi_square_gof(&observed, &expected, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    // Oracle values: official published odds tables (powerball.com prize
    // chart, Wikipedia EuroMillions, FDJ), each re-verified by exact
    // big-integer arithmetic — see docs/ references in the module header.

    #[test]
    fn published_jackpot_combination_counts_are_exact() {
        assert_eq!(
            LotteryGame::lotto_6of49().total_combinations(),
            Some(13_983_816)
        );
        assert_eq!(
            LotteryGame::loto_france().total_combinations(),
            Some(19_068_840)
        );
        assert_eq!(
            LotteryGame::euromillions().total_combinations(),
            Some(139_838_160)
        );
        assert_eq!(
            LotteryGame::powerball().total_combinations(),
            Some(292_201_338)
        );
    }

    #[test]
    fn jackpot_probability_is_one_over_total() {
        for game in [
            LotteryGame::lotto_6of49(),
            LotteryGame::loto_france(),
            LotteryGame::euromillions(),
            LotteryGame::powerball(),
        ]
        {
            let total = game.total_combinations().unwrap() as f64;
            assert!(close(game.p_jackpot(), 1.0 / total, 1e-10));
        }
    }

    #[test]
    fn powerball_tiers_match_official_chart() {
        // Exact winning-combination counts per tier (re-derived from the
        // matrix and cross-checked against powerball.com's prize chart).
        let g = LotteryGame::powerball();
        let total = 292_201_338.0;
        for (m, b, combos) in [
            (5u64, 1u64, 1.0),
            (5, 0, 25.0),
            (4, 1, 320.0),
            (4, 0, 8_000.0),
            (3, 1, 20_160.0),
            (3, 0, 504_000.0),
            (2, 1, 416_640.0),
            (1, 1, 3_176_880.0),
            (0, 1, 7_624_512.0),
        ]
        {
            assert!(
                close(g.p_match(m, b), combos / total, 1e-10),
                "tier {m}+{b}"
            );
        }
        // The official chart prints odds to 2 decimals; ours must round to it.
        for (m, b, published) in [
            (5u64, 0u64, 11_688_053.52),
            (4, 1, 913_129.18),
            (3, 0, 579.76),
            (1, 1, 91.98),
            (0, 1, 38.32),
        ]
        {
            let rounded = (g.odds_against(m, b) * 100.0).round() / 100.0;
            assert!(
                (rounded - published).abs() < 1e-6,
                "tier {m}+{b}: {rounded} vs published {published}"
            );
        }
    }

    #[test]
    fn euromillions_and_loto_match_published_odds() {
        let e = LotteryGame::euromillions();
        // 5 mains + 1 star: exactly 1 in 6,991,908 (20 winning combos).
        assert!(close(e.odds_against(5, 1), 6_991_908.0, 1e-9));
        assert!(close(e.odds_against(5, 0), 3_107_514.666_666_667, 1e-9));
        assert!(close(e.odds_against(4, 2), 621_502.933_333_333, 1e-9));
        let l = LotteryGame::loto_france();
        // 5 bons numéros sans le numéro chance : 9 combos ⇒ 1 sur 2 118 760.
        assert!(close(l.odds_against(5, 0), 2_118_760.0, 1e-9));
        assert!(close(l.odds_against(4, 1), 86_676.545_454_545_45, 1e-8));
    }

    #[test]
    fn six_of_49_match_three_is_the_wikipedia_fraction() {
        // 8815/499422 ≈ 0.0176504, i.e. 1 in ~56.66.
        let g = LotteryGame::lotto_6of49();
        assert!(close(g.p_match(3, 0), 8_815.0 / 499_422.0, 1e-12));
        assert!(close(g.odds_against(3, 0), 56.655_927_396_483_264, 1e-10));
        // Unattainable tier: more matches than picks.
        assert_eq!(g.p_match(7, 0), 0.0);
        assert_eq!(g.odds_against(7, 0), f64::INFINITY);
    }

    #[test]
    fn outcome_probabilities_sum_to_one() {
        // Every (main, bonus) outcome, jackpot included, partitions the
        // sample space — the strongest single invariant of the model.
        let g = LotteryGame::euromillions();
        let mut total = 0.0;
        for m in 0..=5
        {
            for b in 0..=2
            {
                total += g.p_match(m, b);
            }
        }
        assert!(close(total, 1.0, 1e-12));
    }

    #[test]
    fn expected_net_is_negative_for_a_realistic_game() {
        // Stylized Loto FR fixed prizes (jackpot at a typical 5 M€) — the
        // point is the sign and magnitude, not the exact payout schedule.
        let l = LotteryGame::loto_france();
        let tiers = [
            PrizeTier {
                main_matched: 5,
                bonus_matched: 1,
                prize: 5_000_000.0,
            },
            PrizeTier {
                main_matched: 5,
                bonus_matched: 0,
                prize: 100_000.0,
            },
            PrizeTier {
                main_matched: 4,
                bonus_matched: 1,
                prize: 1_000.0,
            },
            PrizeTier {
                main_matched: 4,
                bonus_matched: 0,
                prize: 500.0,
            },
            PrizeTier {
                main_matched: 3,
                bonus_matched: 1,
                prize: 50.0,
            },
            PrizeTier {
                main_matched: 3,
                bonus_matched: 0,
                prize: 20.0,
            },
            PrizeTier {
                main_matched: 2,
                bonus_matched: 1,
                prize: 10.0,
            },
            PrizeTier {
                main_matched: 2,
                bonus_matched: 0,
                prize: 5.0,
            },
            PrizeTier {
                main_matched: 0,
                bonus_matched: 1,
                prize: 2.2,
            },
            PrizeTier {
                main_matched: 1,
                bonus_matched: 1,
                prize: 2.2,
            },
        ];
        let net = l.expected_net(&tiers, 2.2);
        assert!(net < 0.0, "expected net {net} should be negative");
        // The player keeps well under 80% of the stake in expectation.
        assert!(l.expected_gain(&tiers) < 0.8 * 2.2);
    }

    #[test]
    fn fairness_test_accepts_uniform_and_rejects_rigged() {
        // 49 numbers, ~200 appearances each with small jitter: fair.
        let fair: Vec<u64> = (0..49).map(|i| 200 + (i % 5)).collect();
        let r = draw_frequency_chi_square(&fair).unwrap();
        assert!(r.p_value > 0.05, "fair history rejected: p = {}", r.p_value);
        // One ball appearing 3x too often: flagrantly broken.
        let mut rigged = vec![200u64; 49];
        rigged[7] = 600;
        let r = draw_frequency_chi_square(&rigged).unwrap();
        assert!(
            r.p_value < 1e-6,
            "rigged history accepted: p = {}",
            r.p_value
        );
        // Degenerate inputs.
        assert!(draw_frequency_chi_square(&[5]).is_none());
        assert!(draw_frequency_chi_square(&[0, 0, 0]).is_none());
    }
}
