//! Analyse du seuil de rentabilité — modèle linéaire coût-volume-profit :
//! quantité et chiffre d'affaires d'équilibre, marge sur coût variable et
//! marge de sécurité.
//!
//! ```text
//! marge sur coût variable   m   = p - v
//! quantité d'équilibre      BEQ = FC / (p - v)
//! chiffre d'affaires seuil  BER = BEQ · p
//! marge de sécurité         MoS = (Qa - BEQ) / Qa
//! ```
//!
//! `FC` coût fixe total (€), `p` prix de vente unitaire (€/unité), `v` coût
//! variable unitaire (€/unité), `m` marge sur coût variable unitaire (€/unité),
//! `BEQ` quantité au point mort (unités), `BER` chiffre d'affaires au point
//! mort (€), `Qa` quantité réellement écoulée (unités), `MoS` marge de sécurité
//! (fraction sans dimension, `1` = tout le volume est au-dessus du seuil).
//!
//! **Convention** : unités monétaires cohérentes (mêmes € partout) et prix
//! supérieur au coût variable unitaire (`p > v`). **Limite honnête** : modèle
//! linéaire coût-volume-profit à prix et coût variable unitaires constants (pas
//! d'effet d'échelle, ni de remise, ni de saut de charge). Les coûts fixes
//! `FC`, le prix `p` et le coût variable `v` sont FOURNIS par l'appelant ;
//! aucune valeur « par défaut » n'est inventée.

/// Marge sur coût variable unitaire `m = p - v`.
///
/// Contribution de chaque unité vendue à la couverture des coûts fixes.
///
/// Panique si `price_per_unit < 0`, `variable_cost_per_unit < 0` ou si
/// `variable_cost_per_unit >= price_per_unit` (marge nulle ou négative).
pub fn margin_contribution(price_per_unit: f64, variable_cost_per_unit: f64) -> f64 {
    assert!(
        price_per_unit >= 0.0,
        "le prix de vente unitaire doit être positif ou nul"
    );
    assert!(
        variable_cost_per_unit >= 0.0,
        "le coût variable unitaire doit être positif ou nul"
    );
    assert!(
        variable_cost_per_unit < price_per_unit,
        "le coût variable unitaire doit être strictement inférieur au prix de vente"
    );
    price_per_unit - variable_cost_per_unit
}

/// Quantité d'équilibre `BEQ = FC / (p - v)`.
///
/// Nombre d'unités à vendre pour que la marge sur coût variable couvre
/// exactement les coûts fixes (profit nul).
///
/// Panique si `fixed_cost < 0`, `price_per_unit < 0`,
/// `variable_cost_per_unit < 0` ou `variable_cost_per_unit >= price_per_unit`.
pub fn break_even_quantity(
    fixed_cost: f64,
    price_per_unit: f64,
    variable_cost_per_unit: f64,
) -> f64 {
    assert!(fixed_cost >= 0.0, "le coût fixe doit être positif ou nul");
    let margin = margin_contribution(price_per_unit, variable_cost_per_unit);
    fixed_cost / margin
}

/// Chiffre d'affaires au point mort `BER = BEQ · p`.
///
/// Chiffre d'affaires correspondant à la quantité d'équilibre.
///
/// Panique si `fixed_cost < 0`, `price_per_unit < 0`,
/// `variable_cost_per_unit < 0` ou `variable_cost_per_unit >= price_per_unit`.
pub fn break_even_revenue(
    fixed_cost: f64,
    price_per_unit: f64,
    variable_cost_per_unit: f64,
) -> f64 {
    break_even_quantity(fixed_cost, price_per_unit, variable_cost_per_unit) * price_per_unit
}

/// Marge de sécurité `MoS = (Qa - BEQ) / Qa`.
///
/// Fraction du volume écoulé qui excède le seuil de rentabilité ; une valeur
/// négative signale une exploitation à perte (`Qa < BEQ`).
///
/// Panique si `actual_quantity <= 0` ou `break_even_qty < 0`.
pub fn margin_of_safety(actual_quantity: f64, break_even_qty: f64) -> f64 {
    assert!(
        actual_quantity > 0.0,
        "la quantité écoulée doit être strictement positive"
    );
    assert!(
        break_even_qty >= 0.0,
        "la quantité d'équilibre doit être positive ou nulle"
    );
    (actual_quantity - break_even_qty) / actual_quantity
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn realistic_break_even_case() {
        // FC = 12000 €, p = 50 €/unité, v = 30 €/unité.
        // m = 20 €/unité ; BEQ = 12000/20 = 600 unités ; BER = 600·50 = 30000 €.
        let fc = 12_000.0;
        let (p, v) = (50.0, 30.0);
        assert_relative_eq!(margin_contribution(p, v), 20.0, epsilon = 1e-9);
        assert_relative_eq!(break_even_quantity(fc, p, v), 600.0, epsilon = 1e-9);
        assert_relative_eq!(break_even_revenue(fc, p, v), 30_000.0, epsilon = 1e-9);
    }

    #[test]
    fn revenue_is_quantity_times_price() {
        // Identité : BER = BEQ · p par construction.
        let (fc, p, v) = (9_000.0, 45.0, 27.0);
        let beq = break_even_quantity(fc, p, v);
        assert_relative_eq!(break_even_revenue(fc, p, v), beq * p, epsilon = 1e-9);
    }

    #[test]
    fn margin_covers_fixed_cost_at_break_even() {
        // Au seuil, la marge totale BEQ · m couvre exactement le coût fixe FC.
        let (fc, p, v) = (15_000.0, 60.0, 40.0);
        let beq = break_even_quantity(fc, p, v);
        let total_margin = beq * margin_contribution(p, v);
        assert_relative_eq!(total_margin, fc, epsilon = 1e-9);
    }

    #[test]
    fn break_even_quantity_inversely_proportional_to_margin() {
        // BEQ ∝ 1/m : doubler la marge (via v) divise la quantité seuil par deux.
        let fc = 8_000.0;
        let beq1 = break_even_quantity(fc, 50.0, 40.0); // m = 10
        let beq2 = break_even_quantity(fc, 50.0, 30.0); // m = 20
        assert_relative_eq!(beq2, beq1 / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn margin_of_safety_zero_at_break_even() {
        // Vendre exactement BEQ donne MoS = 0 ; le double donne MoS = 0.5.
        let (fc, p, v) = (12_000.0, 50.0, 30.0);
        let beq = break_even_quantity(fc, p, v);
        assert_relative_eq!(margin_of_safety(beq, beq), 0.0, epsilon = 1e-9);
        assert_relative_eq!(margin_of_safety(2.0 * beq, beq), 0.5, epsilon = 1e-9);
    }

    #[test]
    fn margin_of_safety_negative_below_break_even() {
        // Exploitation à perte : Qa < BEQ ⇒ MoS < 0.
        let mos = margin_of_safety(400.0, 600.0);
        assert!(mos < 0.0);
        assert_relative_eq!(mos, -0.5, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "coût variable unitaire doit être strictement inférieur au prix")]
    fn non_positive_margin_panics() {
        break_even_quantity(10_000.0, 30.0, 30.0);
    }
}
