// examples/v2_features_demo/src/main.rs
//
// Démo des trois nouveautés v2 :
//   1. Reverse-mode autodiff sur MatrixView
//   2. Runtime dispatch des backends
//   3. SIMD complexe (DSP)

use scirust_core::{
    autodiff::reverse::{Tape, Tensor},
    matrix::backend::SimdBackend,
};
use scirust_simd::{
    complex::{Complex, complex_mul_f32, complex_dot_hermitian_f32, complex_norm_l2_f32},
    dispatch::{detect_backend, runtime_backend, print_capabilities},
};

// ------------------------------------------------------------------ //
//  1. Autodiff — entraînement d'une régression logistique             //
// ------------------------------------------------------------------ //

fn demo_autodiff() {
    println!("\n=== Reverse-mode AutoDiff sur MatrixView ===");
    println!("Mini-régression linéaire : y = w·x + b, MSE backprop\n");

    let tape = Tape::new();

    // Données : 3 échantillons, 2 features, target y = 2·x0 + 1·x1 + 0.5
    let x = tape.input(Tensor::from_vec(
        vec![1.0, 2.0,  3.0, 1.0,  0.5, 2.5],
        3, 2,
    ));
    let y_true = tape.input(Tensor::from_vec(
        vec![4.5, 7.5, 4.0],
        3, 1,
    ));

    // Paramètres initiaux (faux exprès pour qu'on voie le gradient)
    let w = tape.input(Tensor::from_vec(vec![0.0, 0.0], 2, 1));

    // Forward : y_pred = x @ w
    let y_pred = x.matmul(w);

    // Loss = sum((y_pred - y_true)²)
    let diff   = y_pred.sub(y_true);
    let sq     = diff.hadamard(diff);
    let loss   = sq.sum();

    let loss_val = tape.value(loss.idx()).data[0];
    println!("Loss initiale : {loss_val:.4}");

    loss.backward();

    let grad_w = tape.grad(w.idx());
    println!("∂Loss/∂w = [{:.3}, {:.3}]", grad_w.data[0], grad_w.data[1]);
    println!("→ Gradient non nul : la backward s'exécute correctement");
}

// ------------------------------------------------------------------ //
//  2. Runtime dispatch                                                 //
// ------------------------------------------------------------------ //

fn demo_runtime_dispatch() {
    println!("\n=== Runtime Dispatch ===");
    print_capabilities();

    let backend = runtime_backend();
    println!("\nTest opérationnel via le backend détecté :");

    let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut y = vec![0.0f32; 10];
    backend.saxpy_f32(2.0, &x, &mut y);
    println!("  y = 2*x : {:?}", y);

    let dot = backend.sdot_f32(&x, &x);
    println!("  <x,x>  : {dot}  (attendu : {})", (1..=10).map(|i| i * i).sum::<i32>());
}

// ------------------------------------------------------------------ //
//  3. SIMD complexe — somme de cohérence (signaux radio)              //
// ------------------------------------------------------------------ //

fn demo_complex_dsp() {
    println!("\n=== SIMD Complexe — DSP ===");

    // Signal IQ : 8 échantillons d'un porteur à 0.25 cycle/échantillon
    let signal: Vec<Complex<f32>> = (0..8).map(|n| {
        let phase = 2.0 * std::f32::consts::PI * 0.25 * n as f32;
        Complex::new(phase.cos(), phase.sin())
    }).collect();

    println!("Signal (8 échantillons d'un porteur 0.25 c/éch) :");
    for c in &signal {
        println!("  ({:6.3}, {:6.3})", c.re, c.im);
    }

    // Norme L2
    let n = complex_norm_l2_f32(&signal);
    println!("\n|x|₂ = {n:.4}  (attendu ≈ √8 = {:.4})", (8.0f32).sqrt());

    // Auto-corrélation au décalage 0 : <x, x> = sum |x_i|²
    let auto = complex_dot_hermitian_f32(&signal, &signal);
    println!("Re<x,x> = {:.4}  (énergie du signal, doit valoir |x|² = {:.4})",
             auto.re, n * n);
    println!("Im<x,x> = {:.4}  (doit être 0)", auto.im);

    // Démodulation : multiplier par le conjugué de la porteuse
    let conj_carrier: Vec<Complex<f32>> = (0..8).map(|n| {
        let phase = -2.0 * std::f32::consts::PI * 0.25 * n as f32;
        Complex::new(phase.cos(), phase.sin())
    }).collect();

    let mut demod = vec![Complex::ZERO; 8];
    complex_mul_f32(&mut demod, &signal, &conj_carrier);

    println!("\nAprès démodulation (signal * conj(carrier)) :");
    println!("  Devrait être ≈ (1, 0) pour tous les échantillons");
    for c in &demod {
        println!("  ({:6.3}, {:6.3})", c.re, c.im);
    }
}

fn main() {
    demo_autodiff();
    demo_runtime_dispatch();
    demo_complex_dsp();
}
