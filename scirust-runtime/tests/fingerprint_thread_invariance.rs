//! R4 (verrou CI) — le fingerprint 64 bits du forward est bit-identique quel
//! que soit le nombre de threads rayon (1/2/4/8), à modèle et batches fixés.
//!
//! C'est le pendant CI de la mesure « protocole » du rapport technique §6.2
//! (empreinte identique entre appels et entre processus, sur MNIST). Ici :
//! aucune donnée externe — batches synthétiques dérivés d'arithmétique
//! entière uniquement, donc portables et reproductibles partout. La
//! parallélisation du matmul est par lignes indépendantes (l'ordre des
//! additions intra-ligne est fixe) : ce test transforme cette propriété
//! structurelle en régression visible si un futur noyau la casse.

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{KaimingNormal, Linear, Module, PcgEngine, ReLU, Sequential, Zeros};
use scirust_runtime::{fnv_fold_f32, fnv_init};

/// MLP 784-256-10 aux poids déterministes (graine fixe) — même architecture
/// que le binaire d'audit du runtime (`scirust-runtime/src/main.rs`).
fn build_model(seed: u64) -> Sequential {
    let mut rng = PcgEngine::new(seed);
    Sequential::new()
        .add(Linear::new(784, 256, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(256, 10, &KaimingNormal, &Zeros, &mut rng))
}

/// Batches 100 % déterministes : mélangeur entier (constante de Weyl 64 bits)
/// puis 24 bits de poids fort → f32 exact dans [0, 1). Pas de
/// transcendantale, pas d'aléa d'OS — le contenu est identique sur toute
/// plateforme.
fn synthetic_batches(n_batches: usize, rows: usize, cols: usize) -> Vec<Tensor> {
    (0..n_batches)
        .map(|b| {
            let data: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    let h = (i as u64)
                        .wrapping_add((b as u64) << 32)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    ((h >> 40) as f32) / 16_777_216.0
                })
                .collect();
            Tensor::from_vec(data, rows, cols)
        })
        .collect()
}

/// Empreinte FNV des bits de sortie du forward sur tous les batches.
fn fingerprint(model: &mut Sequential, batches: &[Tensor]) -> u64 {
    let mut fp = fnv_init();
    for x in batches
    {
        let tape = Tape::new();
        let v = tape.input(x.clone());
        let logits = model.forward(&tape, v);
        fp = fnv_fold_f32(fp, &tape.value(logits.idx()).data);
    }
    fp
}

#[test]
fn forward_fingerprint_is_thread_count_invariant() {
    let batches = synthetic_batches(4, 64, 784);

    let mut fingerprints: Vec<(usize, u64)> = Vec::new();
    for threads in [1usize, 2, 4, 8]
    {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build();
        assert!(
            pool.is_ok(),
            "création du pool à {threads} threads impossible"
        );
        let Ok(pool) = pool
        else
        {
            return;
        };
        // Le modèle est construit DANS la fermeture : `Sequential` contient
        // des `Box<dyn Module>` non-`Send` et ne doit pas traverser la
        // frontière du pool ; la graine fixe garantit des poids identiques.
        let fp = pool.install(|| {
            let mut model = build_model(42);
            fingerprint(&mut model, &batches)
        });
        fingerprints.push((threads, fp));
    }

    let reference = fingerprints.first().map(|(_, fp)| *fp);
    assert!(reference.is_some(), "aucun fingerprint calculé");
    for (threads, fp) in &fingerprints
    {
        assert_eq!(
            Some(*fp),
            reference,
            "fingerprint divergent à {threads} threads"
        );
    }

    // Stabilité intra-processus : une ré-exécution complète (nouveau modèle,
    // même graine, pool global par défaut) reproduit la même empreinte.
    let mut model = build_model(42);
    assert_eq!(
        Some(fingerprint(&mut model, &batches)),
        reference,
        "le fingerprint n'est pas stable d'une exécution à l'autre"
    );
}
