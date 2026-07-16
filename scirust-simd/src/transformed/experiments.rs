// scirust-simd/src/transformed/experiments.rs
//
// # Expériences déterministes — défaut de transformation
//
// Compare **Modèle A** (`φ(A⋆B)`) et **Modèle B** (`φ(A)⋆φ(B)`) sur des
// quaternions, octonions et sédénions tirés de façon **déterministe** (LCG à
// graine fixe), sous plusieurs transformations. Agrège le défaut (absolu,
// relatif, `L∞`, distorsion de norme) et sait produire un **CSV**.
//
// Tout est reproductible au bit près : même graine ⇒ mêmes tirages ⇒ mêmes
// statistiques, indépendamment de l'architecture et du parallélisme.
//
// Les latents sont tirés dans `[−amp, amp]` avec `amp = 0.6/√N` : cela borne les
// composantes du produit à ≈ 0.36, garantissant qu'elles restent dans le
// domaine `x > −1` des transformations Gamma (encodage toujours défini).

use super::hypercomplex::{Hypercomplex, model_a_product, model_b_product};
use super::identity::Identity;
use super::log_gamma::LogGamma;
use super::metrics::defect_report;
use super::reciprocal_gamma::ReciprocalGamma;
use super::scalar::TransformedScalar;
use super::transform::ScalarTransform;

/// Générateur congruentiel linéaire déterministe.
struct Lcg(u64);
impl Lcg {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// `f64` déterministe dans `[−1, 1)`.
    #[inline]
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// Tire un élément latent de dimension `N`, composantes dans `[−amp, amp]`.
fn random_latent<F, const N: usize>(
    rng: &mut Lcg,
    amp: f64,
) -> Hypercomplex<TransformedScalar<f64, F>, N> {
    let mut c = [TransformedScalar::<f64, F>::from_latent(0.0); N];
    for slot in &mut c
    {
        *slot = TransformedScalar::from_latent(rng.unit() * amp);
    }
    Hypercomplex(c)
}

/// Statistiques agrégées d'une expérience (une transformation × une dimension).
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentStats {
    /// Nom de la transformation.
    pub transform: &'static str,
    /// Dimension de l'algèbre (4, 8, 16).
    pub dim: usize,
    /// Nombre d'échantillons **effectivement** mesurés (domaine valide).
    pub samples: usize,
    /// Défaut absolu moyen `‖Δ‖`.
    pub mean_abs: f64,
    /// Défaut absolu maximal.
    pub max_abs: f64,
    /// Défaut relatif moyen.
    pub mean_rel: f64,
    /// Défaut relatif maximal.
    pub max_rel: f64,
    /// Distorsion de norme moyenne.
    pub mean_norm_distortion: f64,
}

/// Exécute une expérience `samples` tirages pour la transformation `F` en
/// dimension `N`. Les tirages dont un encodage sort du domaine sont ignorés
/// (comptés hors `samples`).
#[must_use]
pub fn run_experiment<F, const N: usize>(samples: usize, seed: u64) -> ExperimentStats
where
    F: ScalarTransform<f64>,
{
    let amp = 0.6 / (N as f64).sqrt();
    let mut rng = Lcg(seed);
    let (mut sum_abs, mut max_abs) = (0.0f64, 0.0f64);
    let (mut sum_rel, mut max_rel) = (0.0f64, 0.0f64);
    let mut sum_nd = 0.0f64;
    let mut count = 0usize;

    for _ in 0..samples
    {
        let a = random_latent::<F, N>(&mut rng, amp);
        let b = random_latent::<F, N>(&mut rng, amp);
        if let (Ok(ma), Ok(mb)) = (model_a_product(a, b), model_b_product(a, b))
        {
            let r = defect_report(&ma, &mb);
            sum_abs += r.abs_l2;
            max_abs = max_abs.max(r.abs_l2);
            sum_rel += r.rel_l2;
            max_rel = max_rel.max(r.rel_l2);
            sum_nd += r.norm_distortion;
            count += 1;
        }
    }

    let n = count.max(1) as f64;
    ExperimentStats {
        transform: F::NAME,
        dim: N,
        samples: count,
        mean_abs: sum_abs / n,
        max_abs,
        mean_rel: sum_rel / n,
        max_rel,
        mean_norm_distortion: sum_nd / n,
    }
}

/// Suite complète : Identity, ReciprocalGamma, LogGamma × {quaternion, octonion,
/// sédénion}. Graines fixes ⇒ résultat déterministe.
#[must_use]
pub fn run_suite(samples: usize) -> Vec<ExperimentStats> {
    vec![
        run_experiment::<Identity, 4>(samples, 0x5151_0004),
        run_experiment::<Identity, 8>(samples, 0x5151_0008),
        run_experiment::<Identity, 16>(samples, 0x5151_0010),
        run_experiment::<ReciprocalGamma, 4>(samples, 0x5247_0004),
        run_experiment::<ReciprocalGamma, 8>(samples, 0x5247_0008),
        run_experiment::<ReciprocalGamma, 16>(samples, 0x5247_0010),
        run_experiment::<LogGamma, 4>(samples, 0x4c47_0004),
        run_experiment::<LogGamma, 8>(samples, 0x4c47_0008),
        run_experiment::<LogGamma, 16>(samples, 0x4c47_0010),
    ]
}

/// Sérialise une suite en CSV (en-tête inclus).
#[must_use]
pub fn suite_csv(stats: &[ExperimentStats]) -> String {
    let mut out = String::from(
        "transform,dim,samples,mean_abs,max_abs,mean_rel,max_rel,mean_norm_distortion\n",
    );
    for s in stats
    {
        out.push_str(&format!(
            "{},{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n",
            s.transform,
            s.dim,
            s.samples,
            s.mean_abs,
            s.max_abs,
            s.mean_rel,
            s.max_rel,
            s.mean_norm_distortion,
        ));
    }
    out
}
