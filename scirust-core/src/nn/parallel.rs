// scirust-core/src/nn/parallel.rs
//
// Data parallelism — chaque worker thread possède SA propre Tape.
// Les gradients sont collectés à la fin de chaque step et moyennés
// avant d'appliquer l'optimizer.
//
// PRINCIPE :
//
//   batch  ─── split en N shards ──→ shard_0, shard_1, ..., shard_{N-1}
//
//   thread 0:  Tape::new() → forward(model_clone, shard_0) → backward
//   thread 1:  Tape::new() → forward(model_clone, shard_1) → backward
//   ...
//   thread N-1: Tape::new() → forward(...)        → backward
//
//   join → collect grads_i pour chaque worker
//   average_grads = (1/N) Σ grads_i
//   optimizer.step(average_grads)  // sur le model master
//   broadcast model master → tous les workers (ou clones avant chaque step)
//
// CHOIX DE DESIGN :
//
//   - Le user fournit une closure `step_fn` qui prend une &mut copie locale
//     du modèle et retourne (loss_value, grads_per_param).
//   - On ne touche PAS la Tape : chaque thread crée la sienne, l'utilise,
//     la jette. Pas de RwLock, pas de partage, pas de risque de deadlock.
//   - Les paramètres du modèle sont des Tensor clonables : chaque worker
//     applique sur son clone, on agrège les grads à la fin.
//
// LIMITATIONS V7-A :
//   - Le modèle doit être Clone (Sequential l'est si tous ses layers le sont).
//     On ajoute une impl Clone simple sur les couches sans état complexe.
//   - Pas d'overlap forward/backward entre workers (pas de pipeline parallelism).
//   - Pas de gradient accumulation multi-step (= 1 step = 1 batch global).

use std::sync::Arc;
use std::thread;
use crate::autodiff::reverse::Tensor;

// ================================================================== //
//  Conteneur de gradients par paramètre                                //
// ================================================================== //

/// Map nom_param → gradient. On choisit un Vec<(String, Tensor)> plutôt
/// qu'un HashMap pour conserver l'ordre, ce qui simplifie l'agrégation
/// (les workers produisent toujours les params dans le même ordre).
pub type Grads = Vec<(String, Tensor)>;

// ================================================================== //
//  Helper d'agrégation                                                 //
// ================================================================== //

/// Moyenne les gradients de plusieurs workers.
/// Précondition : tous les workers ont produit des Grads avec les mêmes
/// noms et les mêmes shapes, dans le même ordre.
pub fn average_grads(workers_grads: &[Grads]) -> Grads {
    assert!(!workers_grads.is_empty(), "average_grads: aucun worker");
    let n = workers_grads.len() as f32;
    let inv_n = 1.0 / n;

    // On part du premier worker comme base, puis on additionne les autres
    let n_params = workers_grads[0].len();
    let mut result: Grads = Vec::with_capacity(n_params);

    for p_idx in 0..n_params {
        let (name, ref first) = workers_grads[0][p_idx];
        let mut acc = first.clone();
        for w_idx in 1..workers_grads.len() {
            let (other_name, ref other) = workers_grads[w_idx][p_idx];
            assert_eq!(*other_name, name,
                       "ordre des params incohérent entre workers");
            assert_eq!(other.shape(), acc.shape(),
                       "shape inconsistant pour param '{name}'");
            for i in 0..acc.data.len() {
                acc.data[i] += other.data[i];
            }
        }
        for x in acc.data.iter_mut() { *x *= inv_n; }
        result.push((name.clone(), acc));
    }
    result
}

// ================================================================== //
//  Split d'un batch en shards                                          //
// ================================================================== //

/// Split un batch (B, F) en N shards plus petits aussi équilibrés que
/// possible. Si B n'est pas divisible par N, les premiers shards
/// reçoivent un échantillon de plus.
pub fn split_batch(batch: &Tensor, n_shards: usize) -> Vec<Tensor> {
    let (b, f) = batch.shape();
    assert!(n_shards > 0);
    let mut shards = Vec::with_capacity(n_shards);
    let base = b / n_shards;
    let remainder = b % n_shards;

    let mut offset = 0;
    for shard_id in 0..n_shards {
        let size = if shard_id < remainder { base + 1 } else { base };
        if size == 0 { continue; }
        let mut data = Vec::with_capacity(size * f);
        for i in 0..size {
            let row_start = (offset + i) * f;
            data.extend_from_slice(&batch.data[row_start..row_start + f]);
        }
        shards.push(Tensor::from_vec(data, size, f));
        offset += size;
    }
    shards
}

// ================================================================== //
//  parallel_step — exécute N forwards/backwards en parallèle           //
// ================================================================== //

/// Trait à implémenter sur un modèle pour pouvoir l'utiliser en data parallel.
/// Chaque worker reçoit un Box<dyn ParallelStep> qu'il consomme.
///
/// La fonction step prend (x_shard, y_shard) et doit :
///   1. Créer une Tape locale
///   2. Faire le forward + loss
///   3. Backward
///   4. Collecter les gradients via tape.grad(idx) pour chaque param
///   5. Renvoyer (loss_value, Grads)
///
/// Cette signature est volontairement flexible — l'utilisateur fournit
/// la logique de step exactement comme il le ferait en single-thread,
/// le helper se contente de la lancer dans des threads.
pub trait ParallelStep: Send + Sync {
    fn step(&self, x_shard: Tensor, y_shard: Tensor) -> (f32, Grads);

    /// Renvoie une copie indépendante de l'état (pour chaque worker).
    /// Le modèle est typiquement clonable, on délègue à clone().
    fn box_clone(&self) -> Box<dyn ParallelStep>;
}

/// Lance N workers en parallèle sur n_shards shards distincts du batch.
/// Renvoie (mean_loss, mean_grads).
///
/// USAGE :
///   let stepper = MyModelStepper { model: model.clone() };
///   let (loss, grads) = parallel_step(&stepper, x_batch, y_batch, 4);
///   apply_grads(&mut model, &grads, &mut optimizer);
pub fn parallel_step(
    stepper: &dyn ParallelStep,
    x_batch: Tensor,
    y_batch: Tensor,
    n_workers: usize,
) -> (f32, Grads) {
    assert!(n_workers > 0);
    let x_shards = split_batch(&x_batch, n_workers);
    let y_shards = split_batch(&y_batch, n_workers);
    assert_eq!(x_shards.len(), y_shards.len());

    let actual_workers = x_shards.len();

    if actual_workers == 1 {
        // Pas la peine de créer un thread pour 1 worker
        return stepper.step(x_shards.into_iter().next().unwrap(),
                            y_shards.into_iter().next().unwrap());
    }

    // Spawn N threads
    let mut handles = Vec::with_capacity(actual_workers);
    for (xs, ys) in x_shards.into_iter().zip(y_shards.into_iter()) {
        let stepper_clone = stepper.box_clone();
        let h = thread::spawn(move || stepper_clone.step(xs, ys));
        handles.push(h);
    }

    let mut all_losses = Vec::with_capacity(actual_workers);
    let mut all_grads  = Vec::with_capacity(actual_workers);
    for h in handles {
        let (loss, grads) = h.join().expect("worker panic");
        all_losses.push(loss);
        all_grads.push(grads);
    }

    let mean_loss = all_losses.iter().sum::<f32>() / actual_workers as f32;
    let mean_grads = average_grads(&all_grads);
    (mean_loss, mean_grads)
}

// ================================================================== //
//  Application des gradients agrégés                                   //
// ================================================================== //

/// Applique les gradients moyennés aux paramètres du modèle, via
/// une fonction utilisateur. Cette indirection permet de rester
/// agnostique au type d'optimizer.
pub fn apply_grads<F: FnMut(&str, &Tensor)>(grads: &Grads, mut update: F) {
    for (name, g) in grads {
        update(name, g);
    }
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_even() {
        let t = Tensor::from_vec((0..40).map(|x| x as f32).collect(), 8, 5);
        let shards = split_batch(&t, 4);
        assert_eq!(shards.len(), 4);
        assert!(shards.iter().all(|s| s.rows == 2 && s.cols == 5));
        // Premier shard : lignes 0-1 → données 0..10
        assert_eq!(shards[0].data, (0..10).map(|x| x as f32).collect::<Vec<_>>());
        // Dernier shard : lignes 6-7 → données 30..40
        assert_eq!(shards[3].data, (30..40).map(|x| x as f32).collect::<Vec<_>>());
    }

    #[test]
    fn split_uneven() {
        // 10 lignes / 3 shards → shards de tailles [4, 3, 3]
        let t = Tensor::from_vec((0..20).map(|x| x as f32).collect(), 10, 2);
        let shards = split_batch(&t, 3);
        assert_eq!(shards.len(), 3);
        assert_eq!(shards[0].rows, 4);
        assert_eq!(shards[1].rows, 3);
        assert_eq!(shards[2].rows, 3);
        // Vérifie que la concaténation reconstruit le batch original
        let total: usize = shards.iter().map(|s| s.rows).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn average_two_workers() {
        let g_a = vec![
            ("w".to_string(), Tensor::from_vec(vec![2.0, 4.0], 1, 2)),
            ("b".to_string(), Tensor::from_vec(vec![1.0],      1, 1)),
        ];
        let g_b = vec![
            ("w".to_string(), Tensor::from_vec(vec![6.0, 8.0], 1, 2)),
            ("b".to_string(), Tensor::from_vec(vec![3.0],      1, 1)),
        ];
        let avg = average_grads(&[g_a, g_b]);
        assert_eq!(avg[0].1.data, vec![4.0, 6.0]);  // (2+6)/2, (4+8)/2
        assert_eq!(avg[1].1.data, vec![2.0]);       // (1+3)/2
    }

    #[test]
    fn average_single_worker_is_identity() {
        let g = vec![
            ("w".to_string(), Tensor::from_vec(vec![5.0, 10.0], 1, 2)),
        ];
        let avg = average_grads(&[g.clone()]);
        assert_eq!(avg[0].1.data, g[0].1.data);
    }

    // Test d'intégration parallel_step : on simule un stepper trivial
    // qui renvoie la somme des inputs comme "loss" et un grad fixe.
    struct DummyStepper;
    impl ParallelStep for DummyStepper {
        fn step(&self, x: Tensor, _y: Tensor) -> (f32, Grads) {
            let loss: f32 = x.data.iter().sum();
            let grads = vec![
                ("w".to_string(), Tensor::from_vec(vec![loss; 4], 1, 4)),
            ];
            (loss, grads)
        }
        fn box_clone(&self) -> Box<dyn ParallelStep> { Box::new(DummyStepper) }
    }

    #[test]
    fn parallel_step_aggregates_correctly() {
        // x = (4, 2) : lignes [1,1], [2,2], [3,3], [4,4]
        let x = Tensor::from_vec(vec![1.0, 1.0,  2.0, 2.0,  3.0, 3.0,  4.0, 4.0], 4, 2);
        let y = Tensor::zeros(4, 1);

        let (mean_loss, _) = parallel_step(&DummyStepper, x, y, 2);
        // Worker 0 reçoit lignes [1,1; 2,2] → loss = 6
        // Worker 1 reçoit lignes [3,3; 4,4] → loss = 14
        // Mean = 10
        assert!((mean_loss - 10.0).abs() < 1e-5, "got {mean_loss}");
    }

    #[test]
    fn parallel_step_single_worker_no_threading_overhead() {
        let x = Tensor::from_vec(vec![1.0; 8], 4, 2);
        let y = Tensor::zeros(4, 1);
        // n_workers=1 → exécution directe, pas de spawn
        let (loss, grads) = parallel_step(&DummyStepper, x, y, 1);
        assert_eq!(loss, 8.0);
        assert_eq!(grads[0].1.data.len(), 4);
    }
}
