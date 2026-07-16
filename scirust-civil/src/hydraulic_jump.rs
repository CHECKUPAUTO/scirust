//! Ressaut hydraulique en **canal rectangulaire horizontal** — passage brutal
//! d'un écoulement torrentiel (rapide, `Fr > 1`) à un écoulement fluvial (lent,
//! `Fr < 1`). Nombre de Froude amont, hauteurs conjuguées (équation de
//! **Bélanger**), perte de charge dissipée, longueur empirique et rendement.
//!
//! ```text
//! Froude amont          Fr1 = V / √(g·y1)
//! hauteurs conjuguées   y2/y1 = ½·(√(1 + 8·Fr1²) − 1)      (Bélanger)
//! perte de charge       ΔE = (y2 − y1)³ / (4·y1·y2)
//! longueur (empirique)  Lj ≈ 6·(y2 − y1)
//! rendement             η = 1 − ΔE / E1
//! ```
//!
//! `V` vitesse moyenne amont (m/s), `y1` profondeur amont (torrentielle, m),
//! `y2` profondeur aval (fluviale, hauteur conjuguée, m), `g` accélération de
//! la pesanteur (m/s²), `Fr1` nombre de Froude amont (sans dimension), `ΔE`
//! perte de charge spécifique dissipée dans le ressaut (m), `Lj` longueur du
//! ressaut (m), `E1` charge spécifique amont (m), `η` rendement (sans
//! dimension, dans `[0 ; 1[`).
//!
//! **Convention** : SI strict et cohérent — mètres (m) et secondes (s).
//! Les charges et pertes de charge sont exprimées en hauteur d'eau (m). Types
//! `f64`.
//!
//! **Limite honnête** : canal **rectangulaire horizontal** (radier sans pente),
//! écoulement amont **torrentiel** (`Fr1 > 1` requis pour qu'un ressaut existe).
//! L'accélération de la pesanteur `g` est **fournie par l'appelant** (jamais une
//! valeur « par défaut » inventée), de même que les profondeurs et la vitesse
//! **fournies** par la mesure ou le calcul hydraulique amont. L'équation de
//! Bélanger découle de la **conservation de la quantité de mouvement** entre les
//! sections conjuguées ; la longueur `Lj ≈ 6·(y2 − y1)` est une corrélation
//! **empirique** (ordre de grandeur usuel). Ce module **néglige le frottement de
//! fond** sur la longueur du ressaut et ne traite pas les sections non
//! rectangulaires ni les radiers en pente.

/// Nombre de Froude amont `Fr1 = V / √(g·y1)` (sans dimension).
///
/// `Fr1 > 1` caractérise un écoulement torrentiel (condition d'existence d'un
/// ressaut), `Fr1 < 1` un écoulement fluvial, `Fr1 = 1` le régime critique.
///
/// Panique si `velocity < 0`, `depth <= 0` ou `gravity <= 0`.
pub fn jump_froude_number(velocity: f64, depth: f64, gravity: f64) -> f64 {
    assert!(velocity >= 0.0, "la vitesse V doit être positive ou nulle");
    assert!(
        depth > 0.0,
        "la profondeur y doit être strictement positive"
    );
    assert!(
        gravity > 0.0,
        "l'accélération de la pesanteur g doit être strictement positive"
    );
    velocity / (gravity * depth).sqrt()
}

/// Rapport des hauteurs conjuguées `y2/y1 = ½·(√(1 + 8·Fr1²) − 1)`
/// (équation de Bélanger, canal rectangulaire — sans dimension).
///
/// Le résultat est `> 1` dès que `Fr1 > 1` : la profondeur aval `y2` est
/// supérieure à la profondeur amont `y1`.
///
/// Panique si `upstream_froude <= 1` (écoulement amont non torrentiel : aucun
/// ressaut ne se forme).
pub fn jump_sequent_depth_ratio(upstream_froude: f64) -> f64 {
    assert!(
        upstream_froude > 1.0,
        "le nombre de Froude amont Fr1 doit être strictement supérieur à 1 (régime torrentiel requis)"
    );
    0.5 * ((1.0 + 8.0 * upstream_froude * upstream_froude).sqrt() - 1.0)
}

/// Perte de charge dissipée dans le ressaut
/// `ΔE = (y2 − y1)³ / (4·y1·y2)` (m).
///
/// Formule exacte pour un canal rectangulaire horizontal : l'énergie mécanique
/// perdue par turbulence et brassage est toujours positive lorsque `y2 > y1`.
///
/// Panique si `upstream_depth <= 0`, `downstream_depth <= 0` ou si
/// `downstream_depth < upstream_depth` (le ressaut fait toujours croître la
/// profondeur).
pub fn jump_energy_loss(upstream_depth: f64, downstream_depth: f64) -> f64 {
    assert!(
        upstream_depth > 0.0,
        "la profondeur amont y1 doit être strictement positive"
    );
    assert!(
        downstream_depth > 0.0,
        "la profondeur aval y2 doit être strictement positive"
    );
    assert!(
        downstream_depth >= upstream_depth,
        "la profondeur aval y2 doit être supérieure ou égale à la profondeur amont y1"
    );
    (downstream_depth - upstream_depth).powi(3) / (4.0 * upstream_depth * downstream_depth)
}

/// Longueur approchée du ressaut `Lj ≈ 6·(y2 − y1)` (m, corrélation empirique).
///
/// Panique si `upstream_depth < 0`, `downstream_depth < 0` ou si
/// `downstream_depth < upstream_depth`.
pub fn jump_length_approx(downstream_depth: f64, upstream_depth: f64) -> f64 {
    assert!(
        upstream_depth >= 0.0,
        "la profondeur amont y1 doit être positive ou nulle"
    );
    assert!(
        downstream_depth >= 0.0,
        "la profondeur aval y2 doit être positive ou nulle"
    );
    assert!(
        downstream_depth >= upstream_depth,
        "la profondeur aval y2 doit être supérieure ou égale à la profondeur amont y1"
    );
    6.0 * (downstream_depth - upstream_depth)
}

/// Rendement du ressaut `η = 1 − ΔE / E1` (sans dimension), fraction de la
/// charge spécifique amont conservée à l'aval.
///
/// `E1` est la charge spécifique amont (m), `ΔE` la perte de charge (m).
/// Le résultat vaut `1` pour une dissipation nulle et décroît vers `0` quand la
/// dissipation approche la charge amont.
///
/// Panique si `upstream_energy <= 0` ou si `energy_loss < 0`.
pub fn jump_efficiency(energy_loss: f64, upstream_energy: f64) -> f64 {
    assert!(
        upstream_energy > 0.0,
        "la charge spécifique amont E1 doit être strictement positive"
    );
    assert!(
        energy_loss >= 0.0,
        "la perte de charge ΔE doit être positive ou nulle"
    );
    1.0 - energy_loss / upstream_energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn froude_unity_at_critical_flow() {
        // Régime critique : V = √(g·y) ⇒ Fr = 1.
        let g = 9.81_f64;
        let y = 2.0_f64;
        let v = (g * y).sqrt();
        assert_relative_eq!(jump_froude_number(v, y, g), 1.0, max_relative = 1e-12);
    }

    #[test]
    fn belanger_reciprocity_between_conjugate_depths() {
        // Réciprocité de Bélanger : si l'on repart de l'aval avec Fr2 et son
        // rapport conjugué, on retrouve la profondeur amont. On vérifie ici la
        // relation de quantité de mouvement 2·(y2/y1)·(1 + y2/y1) = 2·Fr1²·y1/y2
        // sous sa forme directe : y2/y1 vérifie r² + r − 2·Fr1² = 0.
        let fr1 = 4.0_f64;
        let r = jump_sequent_depth_ratio(fr1);
        // r doit annuler le polynôme de Bélanger r² + r − 2·Fr1².
        assert_relative_eq!(r * r + r - 2.0 * fr1 * fr1, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn no_loss_and_full_efficiency_at_critical_limit() {
        // À la limite Fr1 → 1⁺, y2 → y1 : la perte de charge tend vers zéro et
        // le rendement vers l'unité.
        let loss = jump_energy_loss(1.0, 1.0);
        assert_relative_eq!(loss, 0.0, epsilon = 1e-15);
        assert_relative_eq!(jump_efficiency(loss, 5.0), 1.0, max_relative = 1e-15);
    }

    #[test]
    fn length_proportional_to_height_rise() {
        // Lj ∝ (y2 − y1) : doubler le saut de hauteur double la longueur.
        let l1 = jump_length_approx(2.0, 1.0); // saut de 1 m
        let l2 = jump_length_approx(3.0, 1.0); // saut de 2 m
        assert_relative_eq!(l2, 2.0 * l1, max_relative = 1e-12);
        assert_relative_eq!(l1, 6.0, max_relative = 1e-12);
    }

    #[test]
    fn worked_case_fr1_equals_three() {
        // Canal rectangulaire horizontal, g = 9,81, y1 = 1,0 m.
        // Pour Fr1 = 3 : V = 3·√(9,81·1) = 3·3,132092 = 9,396276 m/s.
        let g = 9.81_f64;
        let y1 = 1.0_f64;
        let v1 = 9.396276_f64;
        let fr1 = jump_froude_number(v1, y1, g);
        assert_relative_eq!(fr1, 3.0, max_relative = 1e-3);

        // y2/y1 = ½·(√(1 + 8·9) − 1) = ½·(√73 − 1)
        //       = ½·(8,544004 − 1) = 3,772002 ⇒ y2 = 3,772002 m.
        let ratio = jump_sequent_depth_ratio(fr1);
        assert_relative_eq!(ratio, 3.772002, max_relative = 1e-3);
        let y2 = ratio * y1;

        // ΔE = (y2 − y1)³/(4·y1·y2) = (2,772002)³/(4·1·3,772002)
        //    = 21,300047/15,088007 = 1,411720 m.
        let loss = jump_energy_loss(y1, y2);
        assert_relative_eq!(loss, 1.411720, max_relative = 1e-3);

        // Lj ≈ 6·(y2 − y1) = 6·2,772002 = 16,632011 m.
        let length = jump_length_approx(y2, y1);
        assert_relative_eq!(length, 16.632011, max_relative = 1e-3);

        // E1 = y1 + V²/(2g) = 1 + 88,289994/19,62 = 1 + 4,5 = 5,5 m.
        // η = 1 − ΔE/E1 = 1 − 1,411720/5,5 = 1 − 0,256676 = 0,743324.
        let e1 = y1 + v1 * v1 / (2.0 * g);
        assert_relative_eq!(e1, 5.5, max_relative = 1e-3);
        let eta = jump_efficiency(loss, e1);
        assert_relative_eq!(eta, 0.743324, max_relative = 1e-3);
    }

    #[test]
    #[should_panic(
        expected = "le nombre de Froude amont Fr1 doit être strictement supérieur à 1 (régime torrentiel requis)"
    )]
    fn subcritical_froude_panics() {
        // Fr1 = 0,8 < 1 : écoulement fluvial, aucun ressaut possible.
        let _ = jump_sequent_depth_ratio(0.8);
    }
}
