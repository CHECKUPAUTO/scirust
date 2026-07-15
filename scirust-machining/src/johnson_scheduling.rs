//! Ordonnancement en flow-shop à deux machines (contexte règle de Johnson) :
//! makespan d'une séquence de tâches donnée et temps d'inactivité de la
//! seconde machine.
//!
//! ```text
//! temps de fin machine 1   C1(k) = C1(k-1) + p1(σ_k)
//! temps de fin machine 2   C2(k) = max(C2(k-1), C1(k)) + p2(σ_k)
//! makespan (durée totale)  Cmax  = C2(n)
//! identité équivalente     Cmax  = max_{1≤k≤n} [ Σ_{j≤k} p1(σ_j) + Σ_{j≥k} p2(σ_j) ]
//! inactivité machine 2     I2    = Cmax - Σ_k p2(k)
//! ```
//!
//! `p1(i)`, `p2(i)` temps opératoires de la tâche `i` sur la machine 1 puis la
//! machine 2 (unité de temps cohérente, p. ex. min ou s), `σ` la séquence
//! (permutation des indices de tâches), `C1(k)`/`C2(k)` dates de fin de la
//! `k`-ième tâche de la séquence sur chaque machine (même unité), `Cmax`
//! makespan de la séquence (même unité), `I2` temps total d'inactivité de la
//! machine 2 avant/pendant le programme (même unité).
//!
//! **Convention** : flow-shop à deux machines, ordre machine 1 → machine 2 pour
//! toutes les tâches, aucune préemption, machines disponibles à l'instant `0`.
//! **Limite honnête** : seul le makespan d'une séquence FOURNIE est calculé ;
//! les temps opératoires `p1`, `p2` et la séquence `σ` sont fournis par
//! l'appelant. La détermination de la séquence optimale (tri de Johnson) reste
//! à la charge de l'appelant ; aucun temps ni aucune séquence « par défaut »
//! n'est inventé ici.

/// Vérifie que `sequence` est une permutation des indices `0..n` sans doublon.
///
/// Panique si un indice est hors bornes ou apparaît plusieurs fois.
fn assert_valid_permutation(sequence: &[usize], n: usize) {
    let mut seen = vec![false; n];
    for &idx in sequence
    {
        assert!(
            idx < n,
            "chaque indice de la séquence doit être inférieur au nombre de tâches"
        );
        assert!(
            !seen[idx],
            "la séquence doit être une permutation sans doublon des indices de tâches"
        );
        seen[idx] = true;
    }
}

/// Makespan `Cmax = C2(n)` d'un flow-shop à deux machines pour une séquence
/// donnée, par la récurrence `C2(k) = max(C2(k-1), C1(k)) + p2(σ_k)`.
///
/// `machine1_times[i]` et `machine2_times[i]` sont les temps opératoires de la
/// tâche `i` sur la machine 1 puis la machine 2 ; `sequence` est l'ordre de
/// passage (permutation des indices de tâches).
///
/// Panique si `machine1_times` est vide, si les trois tranches n'ont pas la
/// même longueur, si un temps est négatif, ou si `sequence` n'est pas une
/// permutation valide des indices de tâches.
pub fn flowshop_makespan_two_machines(
    machine1_times: &[f64],
    machine2_times: &[f64],
    sequence: &[usize],
) -> f64 {
    let n = machine1_times.len();
    assert!(n >= 1, "au moins une tâche est requise");
    assert_eq!(
        machine2_times.len(),
        n,
        "les temps machine 1 et machine 2 doivent avoir la même longueur"
    );
    assert_eq!(
        sequence.len(),
        n,
        "la séquence doit contenir exactement une position par tâche"
    );
    assert_valid_permutation(sequence, n);
    for &t in machine1_times
    {
        assert!(
            t >= 0.0,
            "les temps opératoires machine 1 doivent être positifs ou nuls"
        );
    }
    for &t in machine2_times
    {
        assert!(
            t >= 0.0,
            "les temps opératoires machine 2 doivent être positifs ou nuls"
        );
    }

    let mut completion_m1 = 0.0_f64;
    let mut completion_m2 = 0.0_f64;
    for &task in sequence
    {
        completion_m1 += machine1_times[task];
        completion_m2 = completion_m1.max(completion_m2) + machine2_times[task];
    }
    completion_m2
}

/// Temps total d'inactivité de la machine 2 `I2 = Cmax - Σ_k p2(k)`.
///
/// Différence entre le makespan et la charge totale de la machine 2 : temps
/// pendant lequel la seconde machine attend (au démarrage puis entre tâches).
///
/// Panique dans les mêmes conditions que [`flowshop_makespan_two_machines`]
/// (tranches vides, longueurs incohérentes, temps négatif, séquence invalide).
pub fn flowshop_idle_time_machine2(
    machine1_times: &[f64],
    machine2_times: &[f64],
    sequence: &[usize],
) -> f64 {
    let makespan = flowshop_makespan_two_machines(machine1_times, machine2_times, sequence);
    let total_processing_m2: f64 = machine2_times.iter().sum();
    makespan - total_processing_m2
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn realistic_makespan_case() {
        // p1 = [4, 2, 6], p2 = [3, 5, 1], séquence 0→1→2.
        // C1: 4, 6, 12 ; C2: max(0,4)+3=7, max(7,6)+5=12, max(12,12)+1=13.
        let p1 = [4.0, 2.0, 6.0];
        let p2 = [3.0, 5.0, 1.0];
        let seq = [0, 1, 2];
        assert_relative_eq!(
            flowshop_makespan_two_machines(&p1, &p2, &seq),
            13.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn single_task_makespan_is_sum() {
        // Une seule tâche : Cmax = p1 + p2 (aucun chevauchement possible).
        let p1 = [5.0];
        let p2 = [3.0];
        assert_relative_eq!(
            flowshop_makespan_two_machines(&p1, &p2, &[0]),
            8.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn idle_time_matches_makespan_minus_load() {
        // Identité : I2 = Cmax - Σ p2.
        let p1 = [4.0, 2.0, 6.0];
        let p2 = [3.0, 5.0, 1.0];
        let seq = [0, 1, 2];
        let makespan = flowshop_makespan_two_machines(&p1, &p2, &seq);
        let total_p2: f64 = p2.iter().sum();
        assert_relative_eq!(
            flowshop_idle_time_machine2(&p1, &p2, &seq),
            makespan - total_p2,
            epsilon = 1e-9
        );
    }

    #[test]
    fn recurrence_matches_max_formula() {
        // Identité de Johnson : Cmax = max_k [ Σ_{j≤k} p1 + Σ_{j≥k} p2 ].
        let p1 = [3.0, 5.0, 1.0, 6.0, 7.0];
        let p2 = [6.0, 2.0, 2.0, 6.0, 5.0];
        let seq = [2, 0, 3, 4, 1];
        let n = seq.len();

        let mut prefix_p1 = vec![0.0_f64; n + 1];
        let mut suffix_p2 = vec![0.0_f64; n + 1];
        for k in 0..n
        {
            prefix_p1[k + 1] = prefix_p1[k] + p1[seq[k]];
        }
        for k in (0..n).rev()
        {
            suffix_p2[k] = suffix_p2[k + 1] + p2[seq[k]];
        }
        let mut cmax_formula = f64::MIN;
        for k in 1..=n
        {
            cmax_formula = cmax_formula.max(prefix_p1[k] + suffix_p2[k - 1]);
        }

        assert_relative_eq!(
            flowshop_makespan_two_machines(&p1, &p2, &seq),
            cmax_formula,
            epsilon = 1e-9
        );
    }

    #[test]
    fn makespan_scales_linearly_with_times() {
        // Proportionnalité : multiplier tous les temps par λ multiplie Cmax par λ.
        let p1 = [4.0, 2.0, 6.0];
        let p2 = [3.0, 5.0, 1.0];
        let seq = [0, 1, 2];
        let lambda = 2.5_f64;
        let scaled_p1: Vec<f64> = p1.iter().map(|&t| lambda * t).collect();
        let scaled_p2: Vec<f64> = p2.iter().map(|&t| lambda * t).collect();
        let base = flowshop_makespan_two_machines(&p1, &p2, &seq);
        assert_relative_eq!(
            flowshop_makespan_two_machines(&scaled_p1, &scaled_p2, &seq),
            lambda * base,
            epsilon = 1e-9
        );
    }

    #[test]
    fn idle_time_is_never_negative() {
        // Cmax ≥ Σ p2 toujours (la machine 2 démarre au plus tôt après p1 de la
        // 1re tâche), donc I2 ≥ 0.
        let p1 = [7.0, 1.0, 3.0];
        let p2 = [2.0, 8.0, 4.0];
        let seq = [1, 2, 0];
        assert!(flowshop_idle_time_machine2(&p1, &p2, &seq) >= 0.0);
    }

    #[test]
    #[should_panic(expected = "permutation sans doublon")]
    fn duplicate_index_in_sequence_panics() {
        let p1 = [4.0, 2.0, 6.0];
        let p2 = [3.0, 5.0, 1.0];
        flowshop_makespan_two_machines(&p1, &p2, &[0, 1, 1]);
    }
}
