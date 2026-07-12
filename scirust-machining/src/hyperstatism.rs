//! Isostatisme et hyperstatisme des mécanismes — analyse d'un système de solides
//! reliés par des [`liaisons`](crate::liaisons) : nombre de boucles
//! indépendantes, mobilité et **degré d'hyperstaticité**.
//!
//! Pour un mécanisme de `p` solides (bâti compris) reliés par `Nl` liaisons, le
//! nombre de **boucles indépendantes** (nombre cyclomatique du graphe des
//! liaisons) vaut :
//!
//! ```text
//! μ = Nl − p + 1
//! ```
//!
//! En notant `Ic = Σ mᵢ` la somme des mobilités des liaisons (inconnues
//! cinématiques) et `m` la **mobilité** utile du mécanisme (nombre de mouvements
//! d'entrée indépendants), le **degré d'hyperstaticité** `h` est :
//!
//! ```text
//! h = m + 6·μ − Ic
//! ```
//!
//! Forme statique équivalente, avec `Ns = Σ (6 − mᵢ)` les inconnues d'action :
//!
//! ```text
//! h = Ns − 6·(p − 1) + m
//! ```
//!
//! `h = 0` : mécanisme **isostatique** (montage sans contrainte interne).
//! `h > 0` : **hyperstatique** de degré `h` (montage imposant des contraintes ;
//! exige une précision géométrique accrue). Le calcul spatial (6 équations par
//! boucle) est utilisé ; un mécanisme plan y apparaît souvent hyperstatique.
//!
//! **Limite honnête** : ces relations donnent le degré d'hyperstaticité par le
//! **comptage** des mobilités et des boucles, en supposant les liaisons
//! indépendantes (rang maximal). Un mécanisme à liaisons redondantes/singulières
//! (configurations particulières) peut avoir un rang inférieur ; l'analyse fine
//! du rang du système d'équations, elle, sort du périmètre de ce comptage.

use crate::liaisons::Liaison;

/// Nombre de boucles indépendantes `μ = Nl − p + 1` d'un mécanisme de
/// `num_solids` solides (bâti compris) et `num_joints` liaisons.
///
/// Renvoie `0` pour une chaîne ouverte (arbre cinématique, `Nl = p − 1`).
/// Panique si `num_solids == 0` ou si `num_joints + 1 < num_solids` (graphe non
/// connexe : moins de liaisons qu'un arbre couvrant).
pub fn independent_loops(num_solids: u32, num_joints: u32) -> u32 {
    assert!(
        num_solids > 0,
        "un mécanisme a au moins un solide (le bâti)"
    );
    assert!(
        num_joints + 1 >= num_solids,
        "graphe non connexe : il faut au moins (p − 1) liaisons"
    );
    num_joints + 1 - num_solids
}

/// Inconnues cinématiques `Ic = Σ mᵢ` : somme des mobilités des liaisons.
pub fn kinematic_unknowns(joints: &[Liaison]) -> u32 {
    joints.iter().map(|l| l.mobility() as u32).sum()
}

/// Inconnues statiques `Ns = Σ (6 − mᵢ)` : somme des inconnues d'action des
/// liaisons.
pub fn static_unknowns(joints: &[Liaison]) -> u32 {
    joints.iter().map(|l| l.static_unknowns() as u32).sum()
}

/// Degré d'hyperstaticité `h = m + 6·μ − Ic` d'un mécanisme de `num_solids`
/// solides (bâti compris), relié par `joints`, de mobilité utile `mobility`.
///
/// Renvoie un entier signé : `0` isostatique, `> 0` hyperstatique de degré `h`.
/// Une valeur négative signale une incohérence des données (mobilité annoncée
/// supérieure à la mobilité géométrique — à revoir). Panique sur graphe non
/// connexe (voir [`independent_loops`]).
pub fn degree_of_hyperstaticity(joints: &[Liaison], num_solids: u32, mobility: u32) -> i32 {
    let mu = independent_loops(num_solids, joints.len() as u32) as i32;
    let ic = kinematic_unknowns(joints) as i32;
    mobility as i32 + 6 * mu - ic
}

/// `true` si le mécanisme est **isostatique** (`h == 0`).
pub fn is_isostatic(joints: &[Liaison], num_solids: u32, mobility: u32) -> bool {
    degree_of_hyperstaticity(joints, num_solids, mobility) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liaisons::Liaison::*;

    #[test]
    fn loops_of_a_single_closed_chain() {
        // 4 solides, 4 liaisons → une boucle.
        assert_eq!(independent_loops(4, 4), 1);
        // chaîne ouverte : Nl = p − 1 → 0 boucle.
        assert_eq!(independent_loops(4, 3), 0);
    }

    #[test]
    fn planar_four_bar_is_hyperstatic_of_degree_three() {
        // 4 pivots en boucle, mobilité 1 : h = 1 + 6·1 − 4 = 3 (spatial).
        let joints = [Pivot, Pivot, Pivot, Pivot];
        assert_eq!(degree_of_hyperstaticity(&joints, 4, 1), 3);
    }

    #[test]
    fn appui_plan_plus_rotule_is_isostatic() {
        // 2 solides, une boucle (appui plan + rotule), mobilité 0 :
        // Ic = 3 + 3 = 6 ; h = 0 + 6·1 − 6 = 0.
        let joints = [AppuiPlan, Rotule];
        assert_eq!(degree_of_hyperstaticity(&joints, 2, 0), 0);
        assert!(is_isostatic(&joints, 2, 0));
    }

    #[test]
    fn static_and_kinematic_forms_agree() {
        // h = Ns − 6·(p−1) + m doit égaler la forme cinématique.
        let joints = [Pivot, Pivot, Pivot, Pivot];
        let (p, m) = (4u32, 1u32);
        let h_kin = degree_of_hyperstaticity(&joints, p, m);
        let h_stat = static_unknowns(&joints) as i32 - 6 * (p as i32 - 1) + m as i32;
        assert_eq!(h_kin, h_stat);
    }

    #[test]
    fn kinematic_and_static_unknowns_are_complementary() {
        // Pour chaque liaison, mobilité + inconnues statiques = 6.
        let joints = [Pivot, Rotule, AppuiPlan, Glissiere];
        assert_eq!(
            kinematic_unknowns(&joints) + static_unknowns(&joints),
            6 * joints.len() as u32
        );
    }

    #[test]
    #[should_panic(expected = "non connexe")]
    fn disconnected_graph_panics() {
        // 4 solides mais seulement 2 liaisons (< p − 1 = 3).
        independent_loops(4, 2);
    }
}
