//! # Couches neuronales accélérées SIMD (intégration `scirust-simd`)
//!
//! Câble concrètement les noyaux x86_64 de [`scirust_simd`] dans le crate
//! applicatif : une **couche dense entraînable** dont le *forward* passe par le
//! GEMM fusionné (`Y = act(X·W + b)` en un seul passage,
//! [`scirust_simd::gemm::sgemm_bias_act`]) et le *backward* par les gradients
//! du même crate ([`scirust_simd::grad`]). Le tout hérite du dispatch runtime
//! (AVX-512 → AVX2 → SSE2 → NEON → scalaire).
//!
//! C'est la démonstration que la brique bas-niveau (kernels SIMD) et la couche
//! haut-niveau (apprentissage) s'assemblent : `forward` pour l'inférence,
//! `backward` pour l'entraînement, tous deux vérifiés dans les tests.

use scirust_simd::gemm::{Activation, sgemm_bias_act};
use scirust_simd::grad::{linear_backward, relu_backward, silu_backward};
use scirust_simd::matrix::view::{MatrixView, MatrixViewMut};

/// Activation d'une [`DenseLayer`]. Restreinte aux fonctions dont le crate SIMD
/// fournit **et** le forward **et** le backward (pour une couche entraînable).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    Identity,
    Relu,
    Silu,
}

impl Act {
    fn to_simd(self) -> Activation {
        match self
        {
            Act::Identity => Activation::Identity,
            Act::Relu => Activation::Relu,
            Act::Silu => Activation::Silu,
        }
    }
}

/// Couche dense `Y = act(X·W + b)` à poids possédés (row-major).
///
/// * `w` : `in_dim × out_dim`, `b` : `out_dim`.
/// * `X` : `batch × in_dim`, `Y` : `batch × out_dim`.
#[derive(Clone, Debug)]
pub struct DenseLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub act: Act,
}

/// Gradients renvoyés par [`DenseLayer::backward`].
pub struct DenseGrads {
    /// Gradient de l'entrée `X` (`batch × in_dim`).
    pub dx: Vec<f32>,
    /// Gradient des poids `W` (`in_dim × out_dim`).
    pub dw: Vec<f32>,
    /// Gradient du biais `b` (`out_dim`).
    pub db: Vec<f32>,
}

impl DenseLayer {
    /// Nouvelle couche avec poids explicites.
    pub fn new(in_dim: usize, out_dim: usize, w: Vec<f32>, b: Vec<f32>, act: Act) -> Self {
        assert_eq!(w.len(), in_dim * out_dim, "DenseLayer: W shape");
        assert_eq!(b.len(), out_dim, "DenseLayer: b shape");
        Self {
            in_dim,
            out_dim,
            w,
            b,
            act,
        }
    }

    /// **Forward** : `Y = act(X·W + b)` (`batch × out_dim`), via le GEMM fusionné.
    pub fn forward(&self, x: &[f32], batch: usize) -> Vec<f32> {
        assert_eq!(x.len(), batch * self.in_dim, "forward: X shape");
        let mut y = vec![0.0f32; batch * self.out_dim];
        sgemm_bias_act(
            1.0,
            MatrixView::new(x, batch, self.in_dim),
            MatrixView::new(&self.w, self.in_dim, self.out_dim),
            &self.b,
            self.act.to_simd(),
            MatrixViewMut::new(&mut y, batch, self.out_dim),
        );
        y
    }

    /// Pré-activation `Z = X·W + b` (`batch × out_dim`), nécessaire au backward.
    fn preact(&self, x: &[f32], batch: usize) -> Vec<f32> {
        let mut z = vec![0.0f32; batch * self.out_dim];
        sgemm_bias_act(
            1.0,
            MatrixView::new(x, batch, self.in_dim),
            MatrixView::new(&self.w, self.in_dim, self.out_dim),
            &self.b,
            Activation::Identity,
            MatrixViewMut::new(&mut z, batch, self.out_dim),
        );
        z
    }

    /// **Backward** : à partir de `dy` (gradient de `Y`, `batch × out_dim`),
    /// renvoie les gradients de l'entrée et des paramètres. Enchaîne l'activation
    /// backward (`dz = dy ⊙ act'(z)`) puis le backward linéaire, tous deux issus
    /// de `scirust_simd::grad`.
    pub fn backward(&self, x: &[f32], batch: usize, dy: &[f32]) -> DenseGrads {
        assert_eq!(x.len(), batch * self.in_dim, "backward: X shape");
        assert_eq!(dy.len(), batch * self.out_dim, "backward: dY shape");

        // dz = dy ⊙ act'(z).
        let dz = match self.act
        {
            Act::Identity => dy.to_vec(),
            Act::Relu =>
            {
                let z = self.preact(x, batch);
                let mut dz = vec![0.0f32; z.len()];
                relu_backward(&z, dy, &mut dz);
                dz
            },
            Act::Silu =>
            {
                let z = self.preact(x, batch);
                let mut dz = vec![0.0f32; z.len()];
                silu_backward(&z, dy, &mut dz);
                dz
            },
        };

        let mut dx = vec![0.0f32; batch * self.in_dim];
        let mut dw = vec![0.0f32; self.in_dim * self.out_dim];
        let mut db = vec![0.0f32; self.out_dim];
        linear_backward(
            x,
            batch,
            self.in_dim,
            &self.w,
            self.out_dim,
            &dz,
            &mut dx,
            &mut dw,
            &mut db,
        );
        DenseGrads { dx, dw, db }
    }

    /// Un pas de descente de gradient (SGD) : `W -= lr·dW`, `b -= lr·db`.
    pub fn sgd_step(&mut self, grads: &DenseGrads, lr: f32) {
        for (w, dw) in self.w.iter_mut().zip(&grads.dw)
        {
            *w -= lr * dw;
        }
        for (b, db) in self.b.iter_mut().zip(&grads.db)
        {
            *b -= lr * db;
        }
    }
}

/// Perceptron multi-couche : pile de [`DenseLayer`] entraînée **end-to-end**
/// (forward → perte MSE → backward chaîné couche par couche → SGD). Les
/// dimensions doivent s'enchaîner (`layers[i].out_dim == layers[i+1].in_dim`).
///
/// La rétropropagation applique la règle de la chaîne : le gradient `dX` d'une
/// couche devient le `dY` de la couche précédente — chaque backward de couche
/// étant lui-même validé par gradcheck ([`DenseLayer::backward`]).
#[derive(Clone, Debug)]
pub struct Mlp {
    pub layers: Vec<DenseLayer>,
}

impl Mlp {
    /// Construit un MLP à partir de couches dont les dimensions s'enchaînent.
    pub fn new(layers: Vec<DenseLayer>) -> Self {
        assert!(!layers.is_empty(), "Mlp: au moins une couche requise");
        for w in layers.windows(2)
        {
            assert_eq!(
                w[0].out_dim, w[1].in_dim,
                "Mlp: dimensions non enchaînées ({} -> {})",
                w[0].out_dim, w[1].in_dim
            );
        }
        Self { layers }
    }

    /// **Forward** (inférence) : `x` (`batch × in_dim`) → sortie
    /// (`batch × out_dim` de la dernière couche).
    pub fn forward(&self, x: &[f32], batch: usize) -> Vec<f32> {
        let mut cur = x.to_vec();
        for l in &self.layers
        {
            cur = l.forward(&cur, batch);
        }
        cur
    }

    /// Forward en mémorisant l'entrée de chaque couche (nécessaire au backward).
    /// Renvoie `layers.len()+1` tenseurs : `[x, a₁, …, a_L]` (le dernier = sortie).
    fn forward_cache(&self, x: &[f32], batch: usize) -> Vec<Vec<f32>> {
        let mut acts = Vec::with_capacity(self.layers.len() + 1);
        acts.push(x.to_vec());
        for l in &self.layers
        {
            let next = l.forward(acts.last().unwrap(), batch);
            acts.push(next);
        }
        acts
    }

    /// Perte MSE et **gradients de toutes les couches** (dans l'ordre `0..L`),
    /// pour une cible `target` (`batch × out_dim`), **sans** mettre à jour les
    /// poids. `dL/dY = (2/N)·(pred − target)`.
    pub fn mse_loss_and_grads(
        &self,
        x: &[f32],
        batch: usize,
        target: &[f32],
    ) -> (f32, Vec<DenseGrads>) {
        let acts = self.forward_cache(x, batch);
        let pred = acts.last().unwrap();
        assert_eq!(pred.len(), target.len(), "mse: target shape");
        let n = pred.len() as f32;
        let loss: f32 = pred
            .iter()
            .zip(target)
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f32>()
            / n;

        // Gradient de la perte par rapport à la sortie.
        let mut dy: Vec<f32> = pred
            .iter()
            .zip(target)
            .map(|(p, t)| 2.0 / n * (p - t))
            .collect();

        // Backward chaîné, de la dernière couche à la première.
        let mut grads_rev = Vec::with_capacity(self.layers.len());
        for l in (0..self.layers.len()).rev()
        {
            let g = self.layers[l].backward(&acts[l], batch, &dy);
            dy = g.dx.clone(); // dX de la couche l = dY de la couche l-1
            grads_rev.push(g);
        }
        grads_rev.reverse(); // remet dans l'ordre 0..L
        (loss, grads_rev)
    }

    /// Applique un pas SGD à chaque couche à partir des gradients (ordre `0..L`).
    pub fn sgd_step(&mut self, grads: &[DenseGrads], lr: f32) {
        assert_eq!(
            grads.len(),
            self.layers.len(),
            "sgd_step: un gradient par couche"
        );
        for (l, g) in self.layers.iter_mut().zip(grads)
        {
            l.sgd_step(g, lr);
        }
    }

    /// Un pas d'entraînement complet (forward + backward + update) sur la MSE.
    /// Renvoie la perte **avant** la mise à jour.
    pub fn train_step_mse(&mut self, x: &[f32], batch: usize, target: &[f32], lr: f32) -> f32 {
        let (loss, grads) = self.mse_loss_and_grads(x, batch, target);
        self.sgd_step(&grads, lr);
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.021).sin() * 0.5)
            .collect()
    }

    fn layer(in_dim: usize, out_dim: usize, act: Act) -> DenseLayer {
        DenseLayer::new(
            in_dim,
            out_dim,
            mk(in_dim * out_dim, 1.0),
            mk(out_dim, 2.0),
            act,
        )
    }

    fn naive_forward(l: &DenseLayer, x: &[f32], batch: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; batch * l.out_dim];
        for i in 0..batch
        {
            for j in 0..l.out_dim
            {
                let mut acc = l.b[j];
                for p in 0..l.in_dim
                {
                    acc += x[i * l.in_dim + p] * l.w[p * l.out_dim + j];
                }
                y[i * l.out_dim + j] = match l.act
                {
                    Act::Identity => acc,
                    Act::Relu => acc.max(0.0),
                    Act::Silu => acc / (1.0 + (-acc).exp()),
                };
            }
        }
        y
    }

    #[test]
    fn forward_matches_naive() {
        let (batch, din, dout) = (5usize, 6usize, 4usize);
        let x = mk(batch * din, 3.0);
        for &act in &[Act::Identity, Act::Relu, Act::Silu]
        {
            let l = layer(din, dout, act);
            let got = l.forward(&x, batch);
            let want = naive_forward(&l, &x, batch);
            for t in 0..got.len()
            {
                assert!(
                    (got[t] - want[t]).abs() <= 1e-4 * (1.0 + want[t].abs()),
                    "act {act:?} t={t}: {} vs {}",
                    got[t],
                    want[t]
                );
            }
        }
    }

    #[test]
    fn backward_gradcheck() {
        let (batch, din, dout) = (4usize, 5usize, 3usize);
        let x = mk(batch * din, 3.0);
        let seed = mk(batch * dout, 9.0); // = dY
        let h = 1e-3f32;

        for &act in &[Act::Identity, Act::Relu, Act::Silu]
        {
            let l = layer(din, dout, act);
            let g = l.backward(&x, batch, &seed);

            let loss = |ll: &DenseLayer, xx: &[f32]| -> f32 {
                ll.forward(xx, batch)
                    .iter()
                    .zip(&seed)
                    .map(|(a, b)| a * b)
                    .sum()
            };

            // dX
            let mut xb = x.clone();
            for i in 0..x.len()
            {
                let o = xb[i];
                xb[i] = o + h;
                let lp = loss(&l, &xb);
                xb[i] = o - h;
                let lm = loss(&l, &xb);
                xb[i] = o;
                let num = (lp - lm) / (2.0 * h);
                assert!(
                    (g.dx[i] - num).abs() <= 2e-2 * (1.0 + num.abs()),
                    "act {act:?} dX[{i}]: {} vs {}",
                    g.dx[i],
                    num
                );
            }
            // dW
            let mut lw = l.clone();
            for i in 0..l.w.len()
            {
                let o = lw.w[i];
                lw.w[i] = o + h;
                let lp = loss(&lw, &x);
                lw.w[i] = o - h;
                let lm = loss(&lw, &x);
                lw.w[i] = o;
                let num = (lp - lm) / (2.0 * h);
                assert!(
                    (g.dw[i] - num).abs() <= 2e-2 * (1.0 + num.abs()),
                    "act {act:?} dW[{i}]: {} vs {}",
                    g.dw[i],
                    num
                );
            }
        }
    }

    #[test]
    fn sgd_step_reduces_mse() {
        // Une couche linéaire doit voir sa MSE baisser après un pas de SGD sur
        // une cible aléatoire fixe — bout-en-bout forward+backward+update.
        let (batch, din, dout) = (8usize, 4usize, 3usize);
        let x = mk(batch * din, 3.0);
        let target = mk(batch * dout, 11.0);
        let mut l = layer(din, dout, Act::Identity);

        let mse = |pred: &[f32]| -> f32 {
            pred.iter()
                .zip(&target)
                .map(|(p, t)| (p - t) * (p - t))
                .sum::<f32>()
                / pred.len() as f32
        };

        let y0 = l.forward(&x, batch);
        let loss0 = mse(&y0);

        // dL/dY de la MSE = 2/(N)·(pred − target).
        let n = (batch * dout) as f32;
        let dy: Vec<f32> = y0
            .iter()
            .zip(&target)
            .map(|(p, t)| 2.0 / n * (p - t))
            .collect();
        let g = l.backward(&x, batch, &dy);
        l.sgd_step(&g, 0.1);

        let y1 = l.forward(&x, batch);
        let loss1 = mse(&y1);
        assert!(loss1 < loss0, "MSE n'a pas baissé : {loss0} -> {loss1}");
    }

    fn make_mlp() -> Mlp {
        // 4 -> 8 (ReLU) -> 6 (SiLU) -> 3 (Identity).
        Mlp::new(vec![
            DenseLayer::new(4, 8, mk(4 * 8, 1.0), mk(8, 2.0), Act::Relu),
            DenseLayer::new(8, 6, mk(8 * 6, 3.0), mk(6, 4.0), Act::Silu),
            DenseLayer::new(6, 3, mk(6 * 3, 5.0), mk(3, 6.0), Act::Identity),
        ])
    }

    #[test]
    fn mlp_forward_dims_chain() {
        let mlp = make_mlp();
        let batch = 5;
        let x = mk(batch * 4, 7.0);
        let y = mlp.forward(&x, batch);
        assert_eq!(y.len(), batch * 3);
    }

    #[test]
    fn mlp_backprop_gradcheck() {
        // Le backprop CHAÎNÉ à travers la pile doit matcher les différences
        // finies : on vérifie dW de la 1re couche (le gradient a traversé les
        // 3 couches) contre le numérique.
        let mut mlp = make_mlp();
        let batch = 4;
        let x = mk(batch * 4, 7.0);
        let target = mk(batch * 3, 8.0);
        let h = 1e-3f32;

        let (_, grads) = mlp.mse_loss_and_grads(&x, batch, &target);

        let loss = |m: &Mlp| -> f32 {
            let pred = m.forward(&x, batch);
            pred.iter()
                .zip(&target)
                .map(|(p, t)| (p - t) * (p - t))
                .sum::<f32>()
                / pred.len() as f32
        };

        // Échantillon de poids de la couche 0.
        for &i in &[0usize, 5, 11, 23, 31]
        {
            let orig = mlp.layers[0].w[i];
            mlp.layers[0].w[i] = orig + h;
            let lp = loss(&mlp);
            mlp.layers[0].w[i] = orig - h;
            let lm = loss(&mlp);
            mlp.layers[0].w[i] = orig;
            let num = (lp - lm) / (2.0 * h);
            assert!(
                (grads[0].dw[i] - num).abs() <= 3e-2 * (1.0 + num.abs()),
                "dW0[{i}] chaîné : {} vs numérique {}",
                grads[0].dw[i],
                num
            );
        }
    }

    #[test]
    fn mlp_training_reduces_loss_over_epochs() {
        // Entraîne le MLP sur une cible fixe et vérifie que la loss décroît
        // franchement sur plusieurs époques (apprentissage end-to-end).
        let mut mlp = make_mlp();
        let batch = 16;
        let x = mk(batch * 4, 7.0);
        // Cible = sortie d'un MLP "enseignant" fixe (donc atteignable en principe).
        let teacher = {
            let mut t = make_mlp();
            for l in t.layers.iter_mut()
            {
                for w in l.w.iter_mut()
                {
                    *w *= 1.3;
                }
            }
            t
        };
        let target = teacher.forward(&x, batch);

        let loss0 = mlp.mse_loss_and_grads(&x, batch, &target).0;
        let mut last = loss0;
        for _ in 0..300
        {
            last = mlp.train_step_mse(&x, batch, &target, 0.05);
        }
        assert!(last.is_finite(), "loss non finie");
        assert!(
            last < loss0 * 0.5,
            "la loss n'a pas suffisamment décru : {loss0} -> {last}"
        );
    }
}
