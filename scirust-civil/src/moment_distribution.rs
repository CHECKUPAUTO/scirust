//! **Méthode de Cross** — **répartition des moments** (Hardy Cross) pour poutres
//! et portiques continus à nœuds **fixes** (sans translation). Ce module fournit
//! les **briques élémentaires d'un cycle** de la méthode : **rigidité** d'une
//! barre (encastrée-encastrée ou à extrémité articulée), **facteur de
//! répartition** à un nœud, **moment d'équilibrage** et **moment reporté** à
//! l'extrémité opposée.
//!
//! ```text
//! rigidité (encastr.-encastr.)  K   = 4·E·I / L
//! rigidité (extrémité articulée) K'  = 3·E·I / L
//! facteur de répartition         DF  = Ki / ΣK
//! moment d'équilibrage           ΔM  = −Mu · DF
//! moment reporté (carry-over)    Mco = Md / 2
//! ```
//!
//! `E` module d'élasticité (Young) du matériau (Pa), `I` moment quadratique
//! (inertie) de la section (m⁴), `L` portée de la barre (m), `K` rigidité de
//! barre (N·m par radian de rotation, soit N·m), `Ki` rigidité d'une barre
//! aboutissant au nœud, `ΣK` = `total_joint_stiffness` somme des rigidités des
//! barres au nœud, `DF` facteur de répartition (sans dimension, Σ DF = 1 au
//! nœud), `Mu` = `unbalanced_moment` moment déséquilibré au nœud (N·m), `ΔM`
//! moment d'équilibrage réparti sur une barre (N·m), `Md` = `distributed_moment`
//! moment réparti sur une barre (N·m), `Mco` moment reporté à l'extrémité
//! opposée (N·m).
//!
//! **Convention** : **SI cohérent** — `E` en **Pa**, `I` en **m⁴**, `L` en **m**,
//! d'où une rigidité `K = 4EI/L` en **N·m**. Les moments (`Mu`, `Md`, `ΔM`,
//! `Mco`) sont exprimés en **N·m** ; le module étant **linéaire**, tout jeu
//! d'unités **cohérent** (par ex. `E` en MPa, `I` en mm⁴, `L` en mm → `K` en
//! N·mm) convient pourvu que l'appelant reste cohérent. Types `f64`.
//!
//! **Limite honnête** : la méthode de Hardy Cross est **itérative** ; ce module
//! **n'exécute pas** l'itération complète (répartir / reporter / re-répartir
//! jusqu'à convergence) — il fournit les **fonctions d'un seul cycle**, que
//! l'appelant orchestre. Les **modules d'élasticité** `E`, les **inerties** `I`
//! et les **moments d'encastrement parfait** (« fixed-end moments ») sont
//! **fournis par l'appelant** (tables de charges usuelles, EN 1992/1993 selon le
//! matériau) — aucune valeur n'est inventée ici. Hypothèses : nœuds **sans
//! déplacement** (pas de translation de portique, pas de terme de moment dû au
//! déplacement `6EIΔ/L²`), barres **prismatiques** (`E`, `I` constants sur la
//! portée), comportement **élastique linéaire**. Les portiques à nœuds
//! déplaçables exigent la méthode de Cross avec correction de translation, non
//! traitée ici.

/// Rigidité d'une barre **encastrée-encastrée** (EN 1992/1993, analyse
/// élastique) : `K = 4·E·I / L`.
///
/// `elastic_modulus` = `E` module d'élasticité (Pa), `inertia` = `I` moment
/// quadratique de la section (m⁴), `length` = `L` portée de la barre (m) ;
/// renvoie la rigidité `K` en N·m (par radian).
///
/// Panique si un argument est non fini, si `E ≤ 0`, si `I ≤ 0`, ou si `L ≤ 0`
/// (grandeurs physiquement strictement positives).
pub fn cross_stiffness_factor(elastic_modulus: f64, inertia: f64, length: f64) -> f64 {
    assert!(
        elastic_modulus.is_finite() && elastic_modulus > 0.0,
        "le module d'élasticité E doit être fini et > 0"
    );
    assert!(
        inertia.is_finite() && inertia > 0.0,
        "l'inertie I doit être finie et > 0"
    );
    assert!(
        length.is_finite() && length > 0.0,
        "la portée L doit être finie et > 0"
    );
    4.0 * elastic_modulus * inertia / length
}

/// Rigidité d'une barre à **extrémité articulée** (rotule à l'about opposé) :
/// `K' = 3·E·I / L`. La rigidité réduite (facteur 3 au lieu de 4) traduit
/// l'absence de report de moment vers un appui articulé.
///
/// `elastic_modulus` = `E` module d'élasticité (Pa), `inertia` = `I` moment
/// quadratique (m⁴), `length` = `L` portée (m) ; renvoie la rigidité `K'` en N·m.
///
/// Panique si un argument est non fini, si `E ≤ 0`, si `I ≤ 0`, ou si `L ≤ 0`
/// (grandeurs physiquement strictement positives).
pub fn cross_stiffness_factor_pinned(elastic_modulus: f64, inertia: f64, length: f64) -> f64 {
    assert!(
        elastic_modulus.is_finite() && elastic_modulus > 0.0,
        "le module d'élasticité E doit être fini et > 0"
    );
    assert!(
        inertia.is_finite() && inertia > 0.0,
        "l'inertie I doit être finie et > 0"
    );
    assert!(
        length.is_finite() && length > 0.0,
        "la portée L doit être finie et > 0"
    );
    3.0 * elastic_modulus * inertia / length
}

/// Facteur de répartition d'une barre à un nœud : `DF = Ki / ΣK`. À un nœud
/// équilibré, la somme des facteurs de répartition des barres aboutissantes vaut
/// 1.
///
/// `member_stiffness` = `Ki` rigidité de la barre considérée (N·m),
/// `total_joint_stiffness` = `ΣK` somme des rigidités des barres au nœud (N·m) ;
/// renvoie le facteur de répartition `DF` (sans dimension).
///
/// Panique si un argument est non fini, si `member_stiffness < 0`, si
/// `total_joint_stiffness ≤ 0`, ou si `member_stiffness > total_joint_stiffness`
/// (une rigidité de barre ne peut excéder la somme au nœud).
pub fn cross_distribution_factor(member_stiffness: f64, total_joint_stiffness: f64) -> f64 {
    assert!(
        member_stiffness.is_finite() && member_stiffness >= 0.0,
        "la rigidité de barre Ki doit être finie et ≥ 0"
    );
    assert!(
        total_joint_stiffness.is_finite() && total_joint_stiffness > 0.0,
        "la rigidité totale au nœud ΣK doit être finie et > 0"
    );
    assert!(
        member_stiffness <= total_joint_stiffness,
        "la rigidité de barre Ki ne peut excéder la rigidité totale au nœud ΣK"
    );
    member_stiffness / total_joint_stiffness
}

/// Moment **reporté** (« carry-over ») à l'extrémité opposée d'une barre
/// **prismatique** encastrée : `Mco = Md / 2` (facteur de report 1/2).
///
/// `distributed_moment` = `Md` moment réparti à l'about proche (N·m) ; renvoie
/// le moment reporté `Mco` à l'about opposé (N·m), de même signe.
///
/// Panique si `distributed_moment` est non fini.
pub fn cross_carry_over(distributed_moment: f64) -> f64 {
    assert!(
        distributed_moment.is_finite(),
        "le moment réparti Md doit être fini"
    );
    distributed_moment / 2.0
}

/// Moment d'**équilibrage** attribué à une barre lors d'un cycle :
/// `ΔM = −Mu · DF`. Le moment déséquilibré `Mu` au nœud est réparti sur les
/// barres au prorata de leur facteur de répartition, avec un **signe opposé**
/// pour rétablir l'équilibre du nœud.
///
/// `unbalanced_moment` = `Mu` moment déséquilibré au nœud (N·m),
/// `distribution_factor` = `DF` facteur de répartition de la barre (sans
/// dimension) ; renvoie le moment d'équilibrage `ΔM` de la barre (N·m).
///
/// Panique si un argument est non fini, ou si `distribution_factor` sort de
/// l'intervalle `[0, 1]` (un facteur de répartition est une proportion).
pub fn cross_balancing_moment(unbalanced_moment: f64, distribution_factor: f64) -> f64 {
    assert!(
        unbalanced_moment.is_finite(),
        "le moment déséquilibré Mu doit être fini"
    );
    assert!(
        distribution_factor.is_finite() && (0.0..=1.0).contains(&distribution_factor),
        "le facteur de répartition DF doit être fini et dans [0, 1]"
    );
    -unbalanced_moment * distribution_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pinned_stiffness_is_three_quarters_of_fixed() {
        // Pour un même E, I, L, la rigidité d'une extrémité articulée (3EI/L)
        // vaut exactement 3/4 de celle d'une barre encastrée (4EI/L).
        let e = 210.0e9;
        let i = 8.0e-5;
        let l = 6.0;
        let k_fixed = cross_stiffness_factor(e, i, l);
        let k_pinned = cross_stiffness_factor_pinned(e, i, l);
        assert_relative_eq!(k_pinned / k_fixed, 0.75, epsilon = 1e-9);
    }

    #[test]
    fn distribution_factors_sum_to_one() {
        // À un nœud, la somme des facteurs de répartition des barres vaut 1.
        // Trois barres de rigidités 2, 3 et 5 → ΣK = 10.
        let sum = 10.0;
        let df1 = cross_distribution_factor(2.0, sum);
        let df2 = cross_distribution_factor(3.0, sum);
        let df3 = cross_distribution_factor(5.0, sum);
        assert_relative_eq!(df1 + df2 + df3, 1.0, epsilon = 1e-9);
        // Chaque facteur est bien la proportion attendue.
        assert_relative_eq!(df1, 0.2, epsilon = 1e-9);
    }

    #[test]
    fn balancing_moments_offset_unbalance() {
        // Les moments d'équilibrage répartis sur toutes les barres somment à
        // −Mu (puisque Σ DF = 1), annulant le moment déséquilibré au nœud.
        let mu = 12_000.0;
        let df1 = 0.4;
        let df2 = 0.6;
        let dm1 = cross_balancing_moment(mu, df1);
        let dm2 = cross_balancing_moment(mu, df2);
        assert_relative_eq!(dm1 + dm2, -mu, epsilon = 1e-6);
    }

    #[test]
    fn carry_over_is_half_and_linear() {
        // Le moment reporté vaut la moitié du moment réparti et conserve le signe.
        assert_relative_eq!(cross_carry_over(8_000.0), 4_000.0, epsilon = 1e-9);
        assert_relative_eq!(cross_carry_over(-8_000.0), -4_000.0, epsilon = 1e-9);
        // Linéarité : doubler Md double Mco.
        assert_relative_eq!(
            cross_carry_over(16_000.0),
            2.0 * cross_carry_over(8_000.0),
            epsilon = 1e-9
        );
    }

    #[test]
    fn worked_two_span_joint_cycle() {
        // Nœud B d'une poutre continue de deux travées, sections identiques.
        //   Travée BA : encastrée-encastrée, L = 4 m → K = 4EI/4 = EI.
        //   Travée BC : extrémité C articulée,  L = 6 m → K' = 3EI/6 = 0,5·EI.
        // On pose EI = 20 000 N·m (E·I cohérent) pour un calcul chiffré.
        let ei = 20_000.0_f64;
        let e = 20_000.0_f64; // en posant I = 1, E = EI numériquement.
        let i = 1.0_f64;

        let k_ba = cross_stiffness_factor(e, i, 4.0);
        let k_bc = cross_stiffness_factor_pinned(e, i, 6.0);
        // K_BA = 4·20000·1/4 = 20 000 ; K_BC = 3·20000·1/6 = 10 000.
        assert_relative_eq!(k_ba, 20_000.0, epsilon = 1e-6);
        assert_relative_eq!(k_bc, 10_000.0, epsilon = 1e-6);
        let _ = ei;

        // ΣK = 20 000 + 10 000 = 30 000.
        let sum = k_ba + k_bc;
        assert_relative_eq!(sum, 30_000.0, epsilon = 1e-6);

        // DF_BA = 20 000 / 30 000 = 2/3 ; DF_BC = 10 000 / 30 000 = 1/3.
        let df_ba = cross_distribution_factor(k_ba, sum);
        let df_bc = cross_distribution_factor(k_bc, sum);
        assert_relative_eq!(df_ba, 2.0 / 3.0, epsilon = 1e-9);
        assert_relative_eq!(df_bc, 1.0 / 3.0, epsilon = 1e-9);

        // Moment déséquilibré au nœud B : Mu = +9 000 N·m (somme des moments
        // d'encastrement parfait, fournis par l'appelant).
        let mu = 9_000.0_f64;

        // Équilibrage :
        //   ΔM_BA = −9 000 · 2/3 = −6 000 ; ΔM_BC = −9 000 · 1/3 = −3 000.
        let dm_ba = cross_balancing_moment(mu, df_ba);
        let dm_bc = cross_balancing_moment(mu, df_bc);
        assert_relative_eq!(dm_ba, -6_000.0, epsilon = 1e-3);
        assert_relative_eq!(dm_bc, -3_000.0, epsilon = 1e-3);
        // Les équilibrages annulent bien le déséquilibre : −6 000 − 3 000 = −9 000.
        assert_relative_eq!(dm_ba + dm_bc, -9_000.0, epsilon = 1e-3);

        // Report de ΔM_BA vers l'about encastré A : Mco_A = −6 000 / 2 = −3 000.
        let mco_a = cross_carry_over(dm_ba);
        assert_relative_eq!(mco_a, -3_000.0, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "la portée L doit être finie et > 0")]
    fn zero_length_panics() {
        cross_stiffness_factor(210.0e9, 8.0e-5, 0.0);
    }
}
