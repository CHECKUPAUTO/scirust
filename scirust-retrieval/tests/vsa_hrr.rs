//! HRR binding + cleanup, measured — the stable port of the winning
//! structure-encoding pipeline from the `scirust-hypermemory` program.
//!
//! Scenario: a trace superposes several role⊛filler bindings; unbinding a role
//! yields a noisy filler estimate that the cleanup memory must snap back to
//! the exact stored filler. The harness measures recovery accuracy across
//! dimensions (HRR capacity grows with dimension — the very reason the
//! research program's 16-dimensional ceiling mattered) and under broadband
//! noise on the trace. All profiles are deterministic and printed for the CI
//! log.

use scirust_retrieval::vsa::{
    CleanupMemory, circular_convolution, circular_correlation, superpose,
};

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }

    fn unit_vector(&mut self, dim: usize) -> Vec<f32> {
        loop
        {
            let v: Vec<f32> = (0..dim).map(|_| self.next_f32()).collect();
            let n = scirust_retrieval::vector::norm(&v);
            if n > 0.0
            {
                return v.iter().map(|x| x / n).collect();
            }
        }
    }
}

/// Fraction of correctly recovered fillers over `trials` traces of
/// `pairs` role⊛filler bindings each, with broadband noise of the given
/// `amplitude` added to the trace before unbinding.
fn recovery_accuracy(dim: usize, pairs: usize, amplitude: f32, trials: usize, seed: u64) -> f64 {
    let mut rng = Lcg::new(seed);

    // Codebook of candidate fillers, stored in the cleanup memory.
    let n_fillers = 32usize;
    let fillers: Vec<Vec<f32>> = (0..n_fillers).map(|_| rng.unit_vector(dim)).collect();
    let mut cleanup = CleanupMemory::new(dim);
    for (i, f) in fillers.iter().enumerate()
    {
        cleanup.insert(i as u64, f).unwrap();
    }

    let mut correct = 0usize;
    let mut total = 0usize;
    for _ in 0..trials
    {
        let roles: Vec<Vec<f32>> = (0..pairs).map(|_| rng.unit_vector(dim)).collect();
        let chosen: Vec<usize> = (0..pairs)
            .map(|_| (rng.next_u64() % n_fillers as u64) as usize)
            .collect();
        let bound: Vec<Vec<f32>> = roles
            .iter()
            .zip(&chosen)
            .map(|(r, &c)| circular_convolution(r, &fillers[c]).unwrap())
            .collect();
        let mut trace = superpose(&bound).unwrap();
        for x in &mut trace
        {
            *x += amplitude * rng.next_f32();
        }

        for (r, &c) in roles.iter().zip(&chosen)
        {
            let noisy = circular_correlation(r, &trace).unwrap();
            // Threshold 0.0: pure nearest-neighbour (accuracy is what is measured).
            if let Some(hit) = cleanup.clean(&noisy, 0.0).unwrap()
            {
                if hit.id == c as u64
                {
                    correct += 1;
                }
            }
            total += 1;
        }
    }
    correct as f64 / total as f64
}

#[test]
fn recovery_improves_with_dimension() {
    // 3 bound pairs, noiseless trace: the only noise is superposition
    // cross-talk, which HRR dilutes as the dimension grows — the capacity
    // story that bounded the 16-dimensional sedenion program.
    let mut profile = String::new();
    let mut accs = Vec::new();
    for dim in [16usize, 64, 256]
    {
        let acc = recovery_accuracy(dim, 3, 0.0, 60, 81);
        profile.push_str(&format!("dim {dim}: {acc:.3}  "));
        accs.push(acc);
    }
    println!("HRR recovery, 3 pairs, no noise: {profile}");
    assert!(
        accs[2] > accs[0],
        "capacity must grow with dimension: dim16 {:.3}, dim256 {:.3}",
        accs[0],
        accs[2]
    );
    assert!(
        accs[2] > 0.95,
        "at dim 256 recovery should be near-perfect, got {:.3}",
        accs[2]
    );
}

#[test]
fn cleanup_carries_recovery_under_trace_noise() {
    // dim 64, 3 pairs, growing broadband noise on the trace.
    let mut profile = String::new();
    let mut accs = Vec::new();
    for &amp in &[0.0f32, 0.25, 0.5]
    {
        let acc = recovery_accuracy(64, 3, amp, 60, 82);
        profile.push_str(&format!("amp {amp}: {acc:.3}  "));
        accs.push(acc);
    }
    println!("HRR recovery, dim 64, 3 pairs: {profile}");
    assert!(
        accs[0] >= accs[2],
        "accuracy must not improve with noise: {:.3} -> {:.3}",
        accs[0],
        accs[2]
    );
    // Measured ≈ 0.689 at amp 0.5: heavy trace noise degrades the unbound
    // estimate past what cleanup can always rescue — stated, not hidden.
    assert!(
        accs[2] > 0.65,
        "cleanup should keep recovery substantial at amp 0.5, got {:.3}",
        accs[2]
    );
}

#[test]
fn the_whole_experiment_is_bit_reproducible() {
    let a = recovery_accuracy(64, 3, 0.25, 30, 83);
    let b = recovery_accuracy(64, 3, 0.25, 30, 83);
    assert!((a - b).abs() < f64::EPSILON, "must be bit-reproducible");
}
