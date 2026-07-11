//! Temps d'usinage — durée de coupe (temps machine) des opérations élémentaires
//! de tournage, fraisage et perçage, à partir de la longueur usinée et de la
//! cinématique d'avance.
//!
//! Le temps de coupe d'une passe est la longueur parcourue divisée par la
//! vitesse d'avance :
//!
//! ```text
//! t = L / Vf = L / (f · N)        (min, avec L en mm et Vf en mm/min)
//! ```
//!
//! La longueur `L` inclut, à la charge de l'appelant, les approches et
//! dépassements (engagement/dégagement de l'outil).
//!
//! **Limite honnête** : ce module ne calcule que le **temps de coupe** (temps
//! copeau). Le temps de cycle réel ajoute les temps morts — approche rapide,
//! changements d'outil, indexation, chargement/déchargement — qui relèvent de
//! la gamme et ne sont pas déductibles des seuls paramètres de coupe.

/// Temps de coupe d'une passe (min) : `t = L / (f·N)`, longueur `length` (mm),
/// avance par tour `feed_per_rev` (mm) et rotation `n` (tr/min).
///
/// Panique si `feed_per_rev <= 0` ou `n <= 0`.
pub fn pass_time_min(length_mm: f64, feed_per_rev_mm: f64, n_rpm: f64) -> f64 {
    assert!(
        feed_per_rev_mm > 0.0,
        "l'avance doit être strictement positive"
    );
    assert!(n_rpm > 0.0, "la rotation doit être strictement positive");
    length_mm / (feed_per_rev_mm * n_rpm)
}

/// Nombre de passes (arrondi supérieur) pour retirer une profondeur totale
/// `total_depth` (mm) par passes de `depth_per_pass` (mm).
///
/// Panique si `depth_per_pass <= 0`.
pub fn number_of_passes(total_depth_mm: f64, depth_per_pass_mm: f64) -> u32 {
    assert!(
        depth_per_pass_mm > 0.0,
        "la profondeur de passe doit être strictement positive"
    );
    (total_depth_mm / depth_per_pass_mm).ceil().max(0.0) as u32
}

/// Temps de coupe d'un tournage cylindrique (min) sur plusieurs passes :
/// `t = passes · L / (f·N)`, la profondeur totale `total_depth` étant retirée
/// par passes de `depth_per_pass`.
pub fn turning_time_min(
    length_mm: f64,
    feed_per_rev_mm: f64,
    n_rpm: f64,
    total_depth_mm: f64,
    depth_per_pass_mm: f64,
) -> f64 {
    let passes = number_of_passes(total_depth_mm, depth_per_pass_mm) as f64;
    passes * pass_time_min(length_mm, feed_per_rev_mm, n_rpm)
}

/// Temps de coupe d'un perçage (min) : `t = L / (f·N)`, `depth` étant la course
/// axiale (l'appelant y ajoute l'approche et, en trou débouchant, la pointe du
/// foret ≈ 0,3·D).
pub fn drilling_time_min(depth_mm: f64, feed_per_rev_mm: f64, n_rpm: f64) -> f64 {
    pass_time_min(depth_mm, feed_per_rev_mm, n_rpm)
}

/// Temps de coupe d'un fraisage (min) : `t = L / Vf`, longueur `length` (mm) et
/// vitesse d'avance `feed_velocity` (mm/min) — voir
/// [`crate::kinematics::feed_velocity_mm_min`].
///
/// Panique si `feed_velocity <= 0`.
pub fn milling_time_min(length_mm: f64, feed_velocity_mm_min: f64) -> f64 {
    assert!(
        feed_velocity_mm_min > 0.0,
        "la vitesse d'avance doit être strictement positive"
    );
    length_mm / feed_velocity_mm_min
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pass_time_is_length_over_feed_velocity() {
        // L=100 mm, f=0,2 mm/tr, N=500 tr/min → 100/100 = 1 min.
        assert_relative_eq!(pass_time_min(100.0, 0.2, 500.0), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn passes_round_up() {
        // 5 mm par passes de 2 mm → 3 passes (2+2+1).
        assert_eq!(number_of_passes(5.0, 2.0), 3);
        // pile 6/2 = 3 passes.
        assert_eq!(number_of_passes(6.0, 2.0), 3);
        // rien à retirer → 0 passe.
        assert_eq!(number_of_passes(0.0, 2.0), 0);
    }

    #[test]
    fn turning_multiplies_pass_time_by_pass_count() {
        // 3 passes × 1 min = 3 min.
        assert_relative_eq!(
            turning_time_min(100.0, 0.2, 500.0, 5.0, 2.0),
            3.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn drilling_time_is_a_single_pass() {
        // 30 mm à 0,1 mm/tr, 1000 tr/min → 0,3 min.
        assert_relative_eq!(drilling_time_min(30.0, 0.1, 1000.0), 0.3, epsilon = 1e-9);
    }

    #[test]
    fn milling_time_is_length_over_feed() {
        // 200 mm à 400 mm/min → 0,5 min.
        assert_relative_eq!(milling_time_min(200.0, 400.0), 0.5, epsilon = 1e-9);
    }
}
