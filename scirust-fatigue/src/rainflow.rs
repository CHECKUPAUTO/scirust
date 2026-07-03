//! Comptage de cycles rainflow (ASTM E1049-85 §5.4.4, "Standard Practices
//! for Cycle Counting in Fatigue Analysis") — la méthode standard pour
//! réduire un historique de charge irrégulier à un ensemble de cycles
//! (plage, moyenne, compte) exploitable par la règle de Miner
//! ([`super::miner`]).
//!
//! Port fidèle de l'algorithme à pile (deque) décrit à la §5.4.4 de la
//! norme, vérifié contre la bibliothèque de référence `rainflow` (PyPI,
//! MIT, implémentation ASTM E1049-85 dédiée, github.com/iamlikeme/rainflow)
//! sur deux séquences indépendantes avant portage — voir les tests.
//!
//! ## Algorithme
//! 1. [`reversals`] réduit un signal brut aux points de retournement
//!    (inversions de signe de la dérivée ; premier et dernier point
//!    toujours inclus, points en plateau ignorés).
//! 2. [`rainflow_count`] applique l'algorithme à trois points sur cette
//!    séquence de retournements : à chaque nouveau point, forme les
//!    plages `X` (la plus récente) et `Y` (la précédente) à partir des
//!    trois derniers points non écartés ; si `X >= Y`, compte `Y` comme
//!    un cycle complet (ou une moitié de cycle si `Y` touche le début de
//!    l'historique) et écarte les points de `Y` ; sinon, lit le point
//!    suivant. À la fin, les plages restantes sont comptées comme des
//!    demi-cycles.

use std::collections::VecDeque;

/// Un point de retournement : `(index dans la série d'origine, valeur)`.
pub type Reversal = (usize, f64);

/// Un cycle rainflow : plage, moyenne, et compte (`1.0` = cycle complet,
/// `0.5` = demi-cycle).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cycle {
    pub range: f64,
    pub mean: f64,
    pub count: f64,
    pub start_index: usize,
    pub end_index: usize,
}

/// Réduit un signal brut à ses points de retournement (inversions de la
/// dérivée). Le premier et le dernier point du signal sont toujours des
/// retournements ; les points en plateau (valeur identique au point
/// précédent) sont ignorés.
pub fn reversals(series: &[f64]) -> Vec<Reversal> {
    let n = series.len();
    if n == 0
    {
        return Vec::new();
    }
    if n == 1
    {
        return vec![(0, series[0])];
    }

    let mut out = Vec::new();
    out.push((0, series[0]));

    let mut x = series[1];
    let mut d_last = x - series[0];

    if n == 2
    {
        out.push((1, x));
        return out;
    }

    for (i, &x_next) in series.iter().enumerate().skip(2)
    {
        if x_next != x
        {
            let d_next = x_next - x;
            if d_last * d_next < 0.0
            {
                out.push((i - 1, x));
            }
            x = x_next;
            d_last = d_next;
        }
    }
    out.push((n - 1, series[n - 1]));
    out
}

fn make_cycle(a: Reversal, b: Reversal, count: f64) -> Cycle {
    Cycle {
        range: (a.1 - b.1).abs(),
        mean: 0.5 * (a.1 + b.1),
        count,
        start_index: a.0,
        end_index: b.0,
    }
}

/// Comptage rainflow (ASTM E1049-85 §5.4.4) sur une séquence déjà réduite
/// aux points de retournement (voir [`reversals`]).
pub fn rainflow_count(points: &[Reversal]) -> Vec<Cycle> {
    let mut deque: VecDeque<Reversal> = VecDeque::new();
    let mut cycles = Vec::new();

    for &point in points
    {
        deque.push_back(point);
        loop
        {
            let n = deque.len();
            if n < 3
            {
                break;
            }
            let (x1, x2, x3) = (deque[n - 3].1, deque[n - 2].1, deque[n - 1].1);
            let big_x = (x3 - x2).abs();
            let big_y = (x2 - x1).abs();
            if big_x < big_y
            {
                break;
            }
            else if n == 3
            {
                cycles.push(make_cycle(deque[0], deque[1], 0.5));
                deque.pop_front();
            }
            else
            {
                cycles.push(make_cycle(deque[n - 3], deque[n - 2], 1.0));
                let last = deque.pop_back().expect("len checked >= 3 above");
                deque.pop_back();
                deque.pop_back();
                deque.push_back(last);
            }
        }
    }

    while deque.len() > 1
    {
        cycles.push(make_cycle(deque[0], deque[1], 0.5));
        deque.pop_front();
    }

    cycles
}

/// Combine [`reversals`] et [`rainflow_count`] pour compter directement
/// les cycles d'un signal brut.
pub fn count_cycles(series: &[f64]) -> Vec<Cycle> {
    rainflow_count(&reversals(series))
}

/// Agrège les cycles par plage exacte, sommant leurs comptes — la forme
/// `(plage, compte total)` attendue par [`super::miner::miner_damage`].
pub fn aggregate_by_range(cycles: &[Cycle]) -> Vec<(f64, f64)> {
    let mut ranges: Vec<f64> = cycles.iter().map(|c| c.range).collect();
    ranges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ranges.dedup();
    ranges
        .iter()
        .map(|&r| {
            let total: f64 = cycles
                .iter()
                .filter(|c| c.range == r)
                .map(|c| c.count)
                .sum();
            (r, total)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn assert_cycles_approx(got: &[Cycle], want: &[(f64, f64, f64, usize, usize)]) {
        assert_eq!(got.len(), want.len(), "got {got:?}");
        for (c, &(range, mean, count, start, end)) in got.iter().zip(want)
        {
            assert_relative_eq!(c.range, range, epsilon = 1e-9);
            assert_relative_eq!(c.mean, mean, epsilon = 1e-9);
            assert_relative_eq!(c.count, count, epsilon = 1e-9);
            assert_eq!(c.start_index, start);
            assert_eq!(c.end_index, end);
        }
    }

    /// Vérifié contre `rainflow.extract_cycles` (PyPI `rainflow` 3.2.0)
    /// avant portage.
    #[test]
    fn matches_reference_implementation_on_worked_example_one() {
        let series = [2.0, -1.0, 5.0, -2.0, 3.0, -3.0, 4.0, -1.0];
        let cycles = count_cycles(&series);
        assert_cycles_approx(
            &cycles,
            &[
                (3.0, 0.5, 0.5, 0, 1),
                (6.0, 2.0, 0.5, 1, 2),
                (5.0, 0.5, 1.0, 3, 4),
                (8.0, 1.0, 0.5, 2, 5),
                (7.0, 0.5, 0.5, 5, 6),
                (5.0, 1.5, 0.5, 6, 7),
            ],
        );
        let agg = aggregate_by_range(&cycles);
        let expected = [(3.0, 0.5), (5.0, 1.5), (6.0, 0.5), (7.0, 0.5), (8.0, 0.5)];
        assert_eq!(agg.len(), expected.len());
        for (&(r, c), &(er, ec)) in agg.iter().zip(&expected)
        {
            assert_relative_eq!(r, er, epsilon = 1e-9);
            assert_relative_eq!(c, ec, epsilon = 1e-9);
        }
    }

    /// Vérifié contre `rainflow.extract_cycles` sur une deuxième séquence
    /// indépendante avant portage.
    #[test]
    fn matches_reference_implementation_on_worked_example_two() {
        let series = [0.0, -2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0, 0.0];
        let cycles = count_cycles(&series);
        assert_cycles_approx(
            &cycles,
            &[
                (2.0, -1.0, 0.5, 0, 1),
                (3.0, -0.5, 0.5, 1, 2),
                (4.0, -1.0, 0.5, 2, 3),
                (4.0, 1.0, 1.0, 5, 6),
                (8.0, 1.0, 0.5, 3, 4),
                (9.0, 0.5, 0.5, 4, 7),
                (8.0, 0.0, 0.5, 7, 8),
                (6.0, 1.0, 0.5, 8, 9),
                (2.0, -1.0, 0.5, 9, 10),
            ],
        );
    }

    #[test]
    fn reversals_include_first_and_last_and_skip_plateaus() {
        let series = [1.0, 1.0, 3.0, 3.0, 3.0, -2.0, -2.0, 4.0];
        let r = reversals(&series);
        // First (1.0) and last (4.0) always included; the plateau runs
        // collapse to a single reversal each at the point where they end.
        assert_eq!(r.first().copied(), Some((0, 1.0)));
        assert_eq!(r.last().copied(), Some((7, 4.0)));
    }

    #[test]
    fn empty_and_singleton_series_are_handled() {
        assert!(reversals(&[]).is_empty());
        assert_eq!(reversals(&[5.0]), vec![(0, 5.0)]);
        assert!(count_cycles(&[]).is_empty());
    }
}
