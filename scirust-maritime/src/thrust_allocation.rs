//! Allocation de poussée par pseudo-inverse pondérée — le problème
//! standard de positionnement dynamique (DP) : répartir une force
//! généralisée désirée (surge/sway/lacet) entre plusieurs propulseurs en
//! minimisant un coût quadratique pondéré (Fossen, *Handbook of Marine
//! Craft Hydrodynamics and Motion Control*, Wiley, 2011, chap. 12;
//! Sørdalen, "Optimal Thrust Allocation for Marine Vessels", Control
//! Engineering Practice 5(9), 1997).
//!
//! ## Formulation
//! `n` propulseurs, chacun à position `(lx_i, ly_i)` (repère navire,
//! origine au centre de gravité) et angle d'azimut fixe `angle_i` ;
//! chaque propulseur produit une force scalaire `f_i` le long de son
//! azimut. La matrice de configuration `B` (3×n) relie le vecteur de
//! poussées `u = [f_1..f_n]` à la force généralisée :
//! `Fx = Σ f_i·cos(angle_i)`, `Fy = Σ f_i·sin(angle_i)`,
//! `Mz = Σ f_i·(lx_i·sin(angle_i) − ly_i·cos(angle_i))`.
//!
//! Pour `n > 3` (configuration sur-actionnée, le cas normal en DP), le
//! système `B·u = τ_d` a une infinité de solutions ; on choisit celle qui
//! minimise `uᵀWu` (coût de poussée pondéré, `W` diagonale positive) :
//! `u* = W⁻¹Bᵀ(BW⁻¹Bᵀ)⁻¹τ_d`. Pour `W = I` c'est exactement la
//! pseudo-inverse de Moore-Penrose à norme minimale — vérifié
//! numériquement (numpy `pinv`) avant portage.
//!
//! **Limite honnête** : allocation statique instantanée seulement — ni
//! saturation de poussée par propulseur, ni zones d'ombre (masquage
//! hydrodynamique entre propulseurs), ni la boucle de commande DP
//! complète (observateur, modèle de référence, PID/MPC 3-DDL en amont)
//! qui produirait `τ_d` : ce module prend `τ_d` comme donnée d'entrée.

use scirust_solvers::SolverError;
use scirust_solvers::linalg::Matrix;

/// Un propulseur : position dans le repère navire et angle d'azimut fixe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thruster {
    pub lx: f64,
    pub ly: f64,
    pub angle_rad: f64,
    /// Poids de coût `w_i > 0` — plus grand décourage l'usage de ce propulseur.
    pub weight: f64,
}

fn configuration_matrix(thrusters: &[Thruster]) -> Matrix {
    Matrix::from_fn(3, thrusters.len(), |row, col| {
        let t = &thrusters[col];
        let (c, s) = (t.angle_rad.cos(), t.angle_rad.sin());
        match row
        {
            0 => c,
            1 => s,
            _ => t.lx * s - t.ly * c,
        }
    })
}

/// Alloue la force généralisée désirée `tau_d = [Fx, Fy, Mz]` entre les
/// propulseurs, minimisant le coût pondéré `Σ w_i·f_i²`. Erreur si la
/// configuration est dégénérée (moins de 3 propulseurs indépendants —
/// `BW⁻¹Bᵀ` singulière).
pub fn allocate_thrust(thrusters: &[Thruster], tau_d: [f64; 3]) -> Result<Vec<f64>, SolverError> {
    let n = thrusters.len();
    let b = configuration_matrix(thrusters);
    let w_inv = Matrix::from_fn(n, n, |i, j| {
        if i == j
        {
            1.0 / thrusters[i].weight
        }
        else
        {
            0.0
        }
    });
    let bt = b.transpose();
    let bwbt = b.matmul(&w_inv)?.matmul(&bt)?;
    let bwbt_inv = bwbt.inverse()?;
    let rhs = bwbt_inv.matvec(&tau_d)?;
    w_inv.matmul(&bt)?.matvec(&rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_PI_2;

    /// Deux propulseurs de tunnel (proue/poupe, azimut fixe travers) plus
    /// deux propulseurs principaux azimutaux (poupe, axe longitudinal) —
    /// une configuration DP à 4 propulseurs sur-actionnée classique.
    /// Valeurs vérifiées indépendamment (numpy) avant portage.
    fn overactuated_rig() -> Vec<Thruster> {
        vec![
            Thruster {
                lx: 6.0,
                ly: 0.0,
                angle_rad: FRAC_PI_2,
                weight: 1.0,
            },
            Thruster {
                lx: -6.0,
                ly: 0.0,
                angle_rad: FRAC_PI_2,
                weight: 1.0,
            },
            Thruster {
                lx: -5.0,
                ly: -2.0,
                angle_rad: 0.0,
                weight: 1.0,
            },
            Thruster {
                lx: -5.0,
                ly: 2.0,
                angle_rad: 0.0,
                weight: 1.0,
            },
        ]
    }

    #[test]
    fn equal_weights_match_the_minimum_norm_solution() {
        let thrusters = overactuated_rig();
        let tau_d = [3.0, 10.0, 1.0];
        let u = allocate_thrust(&thrusters, tau_d).unwrap();
        let expected = [5.075, 4.925, 1.525, 1.475];
        for (got, want) in u.iter().zip(&expected)
        {
            assert_relative_eq!(got, want, epsilon = 1e-6);
        }
    }

    #[test]
    fn allocation_always_satisfies_the_demanded_generalized_force() {
        let thrusters = overactuated_rig();
        let tau_d = [3.0, 10.0, 1.0];
        let u = allocate_thrust(&thrusters, tau_d).unwrap();
        let b = configuration_matrix(&thrusters);
        let achieved = b.matvec(&u).unwrap();
        for (got, want) in achieved.iter().zip(&tau_d)
        {
            assert_relative_eq!(got, want, epsilon = 1e-9);
        }
    }

    #[test]
    fn heavily_penalizing_one_thruster_shifts_load_to_its_partner() {
        let mut thrusters = overactuated_rig();
        thrusters[2].weight = 100.0; // penalize thruster 3 (shares Fx with thruster 4)
        let tau_d = [3.0, 10.0, 1.0];
        let u = allocate_thrust(&thrusters, tau_d).unwrap();
        let expected = [5.572_173_44, 4.427_826_56, 0.033_479_69, 2.966_520_31];
        for (got, want) in u.iter().zip(&expected)
        {
            assert_relative_eq!(got, want, epsilon = 1e-6);
        }
        // Fx = f3 + f4 = 3.0 is a hard constraint regardless of weights;
        // the heavy penalty shifts almost all of it onto thruster 4.
        assert!(u[2] < u[3] / 10.0);
    }

    #[test]
    fn rejects_a_degenerate_configuration() {
        // All three thrusters point the same way: B has rank 1, BW^-1B^T is singular.
        let thrusters = vec![
            Thruster {
                lx: 0.0,
                ly: 0.0,
                angle_rad: 0.0,
                weight: 1.0,
            },
            Thruster {
                lx: 0.0,
                ly: 0.0,
                angle_rad: 0.0,
                weight: 1.0,
            },
            Thruster {
                lx: 0.0,
                ly: 0.0,
                angle_rad: 0.0,
                weight: 1.0,
            },
        ];
        assert!(allocate_thrust(&thrusters, [1.0, 0.0, 0.0]).is_err());
    }
}
