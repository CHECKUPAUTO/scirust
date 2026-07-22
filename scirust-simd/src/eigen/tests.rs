// scirust-simd/src/eigen/tests.rs
//
// Validation de `lanczos_eigen_symmetric`. Stratégie :
// * matrice diagonale connue (valeurs propres exactes, vecteurs canoniques) ;
// * `steps == n` (base complète) : trace = Σ valeurs propres (invariant
//   algébrique exact) + résidu `A·v − λ·v` faible pour chaque couple ;
// * `steps < n` (cas d'usage réel) : résidu faible sur les couples
//   **extrêmes** seulement (les intérieurs peuvent ne pas avoir convergé) ;
// * `f32` vs `f64` sur la même matrice (petits entiers, exacts dans les deux) ;
// * cas `steps == 1` (quotient de Rayleigh du vecteur de départ) ;
// * préconditions (paniques).

use super::*;

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn small(&mut self) -> f64 {
        ((self.next_u64() >> 40) as f64 / (1u64 << 24) as f64) - 0.5
    }
}

fn random_symmetric_f64(rng: &mut Lcg, n: usize) -> Vec<f64> {
    let b: Vec<f64> = (0..n * n).map(|_| rng.small()).collect();
    let mut a = vec![0.0f64; n * n];
    for i in 0..n
    {
        for j in 0..n
        {
            a[i * n + j] = b[i * n + j] + b[j * n + i];
        }
    }
    a
}

fn matvec_ref(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n
    {
        let mut acc = 0.0;
        for j in 0..n
        {
            acc += a[i * n + j] * x[j];
        }
        y[i] = acc;
    }
    y
}

fn residual_norm(a: &[f64], lambda: f64, v: &[f64], n: usize) -> f64 {
    let av = matvec_ref(a, v, n);
    let mut acc = 0.0f64;
    for k in 0..n
    {
        let d = av[k] - lambda * v[k];
        acc += d * d;
    }
    acc.sqrt()
}

#[test]
fn known_diagonal_matrix() {
    // diag(3,7,-2) : valeurs propres exactes {7,3,-2} (triées décroissant),
    // A·v = λ·v vérifiable directement.
    let n = 3;
    let a = vec![3.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, -2.0];
    let (vals, vecs) = lanczos_eigen_symmetric::<f64>(&a, n, n, 1e-10, 100);
    assert_eq!(vals.len(), 3);
    let want = [7.0, 3.0, -2.0];
    for i in 0..3
    {
        assert!(
            (vals[i] - want[i]).abs() < 1e-8,
            "λ{i}={} vs {}",
            vals[i],
            want[i]
        );
    }
    for i in 0..3
    {
        let r = residual_norm(&a, vals[i], &vecs[i], n);
        assert!(r < 1e-6, "i={i}: résidu {r}");
    }
}

#[test]
fn full_steps_sum_matches_trace_and_low_residual() {
    let mut rng = Lcg(0xE16E_0002);
    for &n in &[2usize, 3, 5, 8]
    {
        let a = random_symmetric_f64(&mut rng, n);
        let (vals, vecs) = lanczos_eigen_symmetric::<f64>(&a, n, n, 1e-12, 200);
        assert_eq!(
            vals.len(),
            n,
            "n={n}: rupture chanceuse inattendue sur données aléatoires"
        );

        let trace: f64 = (0..n).map(|i| a[i * n + i]).sum();
        let sum_eig: f64 = vals.iter().sum();
        assert!(
            (trace - sum_eig).abs() < 1e-6 * (n as f64).max(1.0),
            "n={n}: trace {trace} vs Σλ {sum_eig}"
        );

        for i in 0..n
        {
            let r = residual_norm(&a, vals[i], &vecs[i], n);
            assert!(r < 1e-4, "n={n} i={i}: résidu {r}");
        }
    }
}

#[test]
fn partial_steps_extreme_eigenpairs_have_small_residual() {
    // Propriété classique de Lanczos : avec `steps ≪ n`, les couples
    // propres **extrêmes** (les premiers/derniers de la sortie, triée
    // décroissante) convergent bien avant les intérieurs.
    //
    // `steps = 15` (sur `n = 20`) est choisi empiriquement : le résidu des
    // couples extrêmes décroît **régulièrement** avec `steps` pour cette
    // graine (mesuré : ~3e-1 à 5 pas, ~7e-2 à 10, ~1e-3 à 15, ~9e-13 à 20 —
    // convergence normale, pas un bug), et 15 pas donne une marge confortable
    // sous la tolérance tout en restant nettement `< n`.
    let mut rng = Lcg(0xE16E_0003);
    let n = 20;
    let a = random_symmetric_f64(&mut rng, n);
    let steps = 15;
    let (vals, vecs) = lanczos_eigen_symmetric::<f64>(&a, n, steps, 1e-12, 200);
    assert!(vals.len() <= steps);
    let m = vals.len();
    for &i in &[0usize, 1, m - 2, m - 1]
    {
        let r = residual_norm(&a, vals[i], &vecs[i], n);
        assert!(r < 1e-2, "i={i}: résidu {r} (λ={})", vals[i]);
    }
}

#[test]
fn f32_matches_f64_within_tolerance() {
    // Coefficients petits entiers, exactement représentables en f32 ET f64 :
    // les deux instanciations doivent converger vers les mêmes valeurs
    // propres, à la précision f32 près.
    let n = 4;
    #[rustfmt::skip]
    let a32: Vec<f32> = vec![
        4.0, 1.0, 0.0, 0.0,
        1.0, 3.0, 1.0, 0.0,
        0.0, 1.0, 2.0, 1.0,
        0.0, 0.0, 1.0, 1.0,
    ];
    let a64: Vec<f64> = a32.iter().map(|&x| x as f64).collect();
    let (vals32, _) = lanczos_eigen_symmetric::<f32>(&a32, n, n, 1e-5, 100);
    let (vals64, _) = lanczos_eigen_symmetric::<f64>(&a64, n, n, 1e-12, 100);
    assert_eq!(vals32.len(), n);
    assert_eq!(vals64.len(), n);
    for i in 0..n
    {
        assert!(
            (f64::from(vals32[i]) - vals64[i]).abs() < 1e-3,
            "i={i}: f32={} f64={}",
            vals32[i],
            vals64[i]
        );
    }
}

#[test]
fn single_step_gives_rayleigh_quotient_of_start_vector() {
    // steps=1 : un seul vecteur de Lanczos (le vecteur de départ
    // déterministe (1,…,1)/√n) — la seule "valeur propre" renvoyée est son
    // quotient de Rayleigh vᵀAv (v déjà unitaire).
    let n = 3;
    let a = vec![2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 9.0];
    let (vals, vecs) = lanczos_eigen_symmetric::<f64>(&a, n, 1, 1e-10, 50);
    assert_eq!(vals.len(), 1);
    assert_eq!(vecs.len(), 1);
    let v0 = 1.0 / (n as f64).sqrt();
    let start = vec![v0; n];
    let av = matvec_ref(&a, &start, n);
    let rayleigh: f64 = (0..n).map(|i| start[i] * av[i]).sum();
    assert!(
        (vals[0] - rayleigh).abs() < 1e-9,
        "{} vs {rayleigh}",
        vals[0]
    );
}

#[test]
#[should_panic(expected = "longueur")]
fn shape_mismatch_panics() {
    let _ = lanczos_eigen_symmetric::<f64>(&[1.0, 2.0, 3.0], 2, 1, 1e-8, 10);
}

#[test]
#[should_panic(expected = "steps")]
fn steps_zero_panics() {
    let a = vec![1.0f64, 0.0, 0.0, 1.0];
    let _ = lanczos_eigen_symmetric::<f64>(&a, 2, 0, 1e-8, 10);
}

#[test]
#[should_panic(expected = "steps")]
fn steps_too_large_panics() {
    let a = vec![1.0f64, 0.0, 0.0, 1.0];
    let _ = lanczos_eigen_symmetric::<f64>(&a, 2, 3, 1e-8, 10);
}
