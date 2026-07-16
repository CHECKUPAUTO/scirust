// scirust-simd/src/transformed/branch.rs
//
// # Branches d'inversion pour les transformations non injectives
//
// `1/Γ(x+1)` et `ln Γ(x+1)` ne sont **pas** globalement inversibles sur leur
// domaine `x > −1` : `Γ(x+1)` possède un extremum unique en
// `x* = ` [`special::GAMMA_ARGMIN`]` − 1 ≈ 0.4616`, de part et d'autre duquel
// la fonction est monotone. Un même encodé possède donc (en général) **deux**
// antécédents. On rend l'ambiguïté explicite via [`GammaBranch`], et le
// décodage se fait par bissection **déterministe** sur la branche choisie.

use super::special::GAMMA_ARGMIN;

/// Branche d'inversion d'une transformation à extremum unique (les Gamma).
///
/// L'extremum est en `x* = GAMMA_ARGMIN − 1 ≈ 0.4616`. `Lower` couvre
/// `x ∈ (−1, x*]`, `Upper` couvre `x ∈ [x*, ∞)`. `Upper` est la branche
/// **principale** (régime factoriel `x ≥ x*`, le plus courant).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GammaBranch {
    /// Branche gauche `x ∈ (−1, x*]` (petits arguments, proche de la singularité).
    Lower,
    /// Branche droite `x ∈ [x*, ∞)` — **principale** (régime factoriel).
    #[default]
    Upper,
}

/// Abscisse `x*` de l'extremum, exprimée dans la variable latente `x` (et non
/// `z = x + 1`) : `x* = GAMMA_ARGMIN − 1`.
pub const GAMMA_TURN_X: f64 = GAMMA_ARGMIN - 1.0;

/// Bissection déterministe : `g` change de signe entre `lo` et `hi`. Renvoie
/// `x` tel que `g(x) ≈ 0`. 100 itérations ⇒ ~2⁻¹⁰⁰ de l'intervalle initial
/// (bien au-delà de la précision `f64`), donc **reproductible au bit près**.
pub(crate) fn bisect<G: Fn(f64) -> f64>(g: G, mut lo: f64, mut hi: f64) -> f64 {
    let mut glo = g(lo);
    for _ in 0..100
    {
        let mid = 0.5 * (lo + hi);
        let gm = g(mid);
        if (gm > 0.0) == (glo > 0.0)
        {
            lo = mid;
            glo = gm;
        }
        else
        {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Inverse une fonction **unimodale** `phi` (extremum en `x_star`, monotone sur
/// chaque branche, domaine ouvert en `domain_inf`) pour la valeur `target`, sur
/// la branche demandée.
///
/// Renvoie `None` si `target` n'est pas atteignable sur cette branche (pas
/// d'encadrement) — c'est-à-dire hors de l'image de la branche.
pub(crate) fn invert_unimodal<P: Fn(f64) -> f64>(
    phi: P,
    x_star: f64,
    domain_inf: f64,
    target: f64,
    branch: GammaBranch,
) -> Option<f64> {
    let g = |x: f64| phi(x) - target;
    let g_star = g(x_star);
    match branch
    {
        GammaBranch::Lower =>
        {
            // Encadrement [domain_inf⁺, x*] : près de la singularité φ diverge du
            // côté opposé à φ(x*), donc si target est dans l'image de la branche,
            // g(domain_inf⁺) et g(x*) sont de signes opposés.
            let lo = domain_inf + 1e-9;
            let g_lo = g(lo);
            if (g_lo > 0.0) == (g_star > 0.0)
            {
                return None;
            }
            Some(bisect(g, lo, x_star))
        },
        GammaBranch::Upper =>
        {
            // Encadrement [x*, hi] : on fait croître hi géométriquement jusqu'au
            // changement de signe (borne haute déterministe 1e15).
            let mut hi = x_star + 1.0;
            let mut bracketed = false;
            for _ in 0..200
            {
                if (g(hi) > 0.0) != (g_star > 0.0)
                {
                    bracketed = true;
                    break;
                }
                hi = x_star + (hi - x_star) * 2.0;
                if !hi.is_finite() || hi > 1e15
                {
                    break;
                }
            }
            if !bracketed
            {
                return None;
            }
            Some(bisect(g, x_star, hi))
        },
    }
}
