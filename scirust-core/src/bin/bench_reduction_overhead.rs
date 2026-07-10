//! O1 — banc « coût du déterminisme » : réduction en **ordre de worker figé**
//! (le pattern déterministe de `DataParallelTrainer::train_batch_threaded`)
//! contre une réduction en **ordre d'arrivée** (baseline non déterministe,
//! ce que ferait une accumulation « au fil des terminaisons »).
//!
//! Micro-banc du *pattern de réduction*, pas de l'entraînement complet : le
//! travail par worker (génération d'un gradient synthétique par mélangeur
//! entier) tient lieu de forward/backward ; ce qui est comparé est le coût
//! de collecte + réduction, à travail identique.
//!
//! Les contributions des workers ont des magnitudes hétérogènes (±1e16 et
//! ±1) : l'ordre d'addition est donc numériquement **observable** — la
//! variante figée doit produire une empreinte bit-identique à chaque
//! répétition et à chaque nombre de threads ; la variante « arrivée » peut
//! varier avec l'ordonnanceur (le banc compte les empreintes distinctes
//! observées).
//!
//! Wall-clock ⇒ **hors CI par nature** (protocole) : exécuter en release,
//! sur x86 puis aarch64 (Jetson), et consigner la sortie.
//!
//! ```text
//! cargo run -q --release -p scirust-core --bin bench_reduction_overhead
//! ```

use std::sync::mpsc;
use std::thread;
use std::time::Instant;

/// Dimension du gradient simulé (~ une couche 784×128).
const DIM: usize = 100_352;
/// Répétitions chronométrées par variante et par nombre de threads.
const REPS: usize = 30;
/// Nombres de workers mesurés (mêmes points que les tests d'invariance).
const THREAD_COUNTS: [usize; 4] = [1, 2, 4, 8];

/// Gradient synthétique du worker `w` : contenu déterministe (mélangeur de
/// Weyl 64 bits, 24 bits de poids fort → f32 exact), magnitude ±1e16 pour
/// les workers pairs et ±1 pour les impairs — l'ordre d'addition compte.
fn worker_gradient(w: usize, dim: usize) -> Vec<f32> {
    let scale = if w % 2 == 0 { 1e16 } else { 1.0 };
    (0..dim)
        .map(|i| {
            let h = ((i as u64) ^ ((w as u64) << 32)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let unit = ((h >> 40) as f32) / 16_777_216.0 - 0.5;
            unit * scale
        })
        .collect()
}

/// Empreinte FNV-1a des bits du vecteur réduit (comparaison bit-à-bit).
fn fingerprint(v: &[f32]) -> u64 {
    let mut fp = 0xcbf2_9ce4_8422_2325u64;
    for x in v
    {
        for b in x.to_bits().to_le_bytes()
        {
            fp ^= u64::from(b);
            fp = fp.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fp
}

/// Variante A — **ordre figé** (déterministe) : chaque worker écrit son
/// résultat dans SON slot indexé ; la réduction parcourt ensuite les slots
/// en ordre 0..n, quel que soit l'ordre de terminaison. C'est le pattern
/// exact de `DataParallelTrainer::train_batch_threaded`.
fn reduce_fixed_order(n_workers: usize, dim: usize) -> Vec<f32> {
    let mut slots: Vec<Vec<f32>> = vec![Vec::new(); n_workers];
    thread::scope(|s| {
        for (w, slot) in slots.iter_mut().enumerate()
        {
            s.spawn(move || {
                *slot = worker_gradient(w, dim);
            });
        }
    });
    let mut acc = vec![0.0f32; dim];
    for slot in &slots
    {
        for (a, g) in acc.iter_mut().zip(slot)
        {
            *a += *g;
        }
    }
    let inv = 1.0 / n_workers as f32;
    for a in &mut acc
    {
        *a *= inv;
    }
    acc
}

/// Variante B — **ordre d'arrivée** (baseline non déterministe) : les
/// workers envoient leur gradient sur un canal dès qu'ils terminent ;
/// l'accumulation suit l'ordre de réception, qui dépend de l'ordonnanceur.
fn reduce_arrival_order(n_workers: usize, dim: usize) -> Vec<f32> {
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let mut acc = vec![0.0f32; dim];
    thread::scope(|s| {
        for w in 0..n_workers
        {
            let tx = tx.clone();
            s.spawn(move || {
                // L'échec d'envoi (récepteur fermé) est impossible ici tant
                // que la boucle de réception vit ; on l'ignore sans paniquer.
                let _ = tx.send(worker_gradient(w, dim));
            });
        }
        drop(tx);
        while let Ok(g) = rx.recv()
        {
            for (a, gi) in acc.iter_mut().zip(&g)
            {
                *a += *gi;
            }
        }
    });
    let inv = 1.0 / n_workers as f32;
    for a in &mut acc
    {
        *a *= inv;
    }
    acc
}

/// Médiane (µs) d'une série de durées ; série vide → 0 (n'arrive pas :
/// `REPS > 0`).
fn median_us(mut times: Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.get(times.len() / 2).copied().unwrap_or(0.0)
}

fn main() {
    println!("# O1 — coût du déterminisme : réduction ordre figé vs ordre d'arrivée");
    println!();
    println!(
        "dim = {DIM}, reps = {REPS} ; médianes en µs ; overhead = figé / arrivée. \
         Empreintes : figé doit être unique (bit-identique) ; arrivée = nombre \
         d'empreintes distinctes observées sur {REPS} reps."
    );
    println!();
    println!("| threads | figé (µs) | arrivée (µs) | overhead | fp figé | fp arrivée distincts |");
    println!("|---:|---:|---:|---:|---|---:|");

    for &n in &THREAD_COUNTS
    {
        let mut fixed_times = Vec::with_capacity(REPS);
        let mut arrival_times = Vec::with_capacity(REPS);
        let mut fixed_fps: Vec<u64> = Vec::with_capacity(REPS);
        let mut arrival_fps: Vec<u64> = Vec::with_capacity(REPS);

        for _ in 0..REPS
        {
            let t0 = Instant::now();
            let acc = reduce_fixed_order(n, DIM);
            fixed_times.push(t0.elapsed().as_secs_f64() * 1e6);
            fixed_fps.push(fingerprint(&acc));

            let t0 = Instant::now();
            let acc = reduce_arrival_order(n, DIM);
            arrival_times.push(t0.elapsed().as_secs_f64() * 1e6);
            arrival_fps.push(fingerprint(&acc));
        }

        fixed_fps.sort_unstable();
        fixed_fps.dedup();
        arrival_fps.sort_unstable();
        arrival_fps.dedup();

        // Invariant du banc : la variante figée est bit-identique sur toutes
        // les répétitions (sinon le banc lui-même est faux — signalé, pas
        // masqué).
        let fixed_fp_display = match fixed_fps.as_slice()
        {
            [only] => format!("{only:#018x}"),
            other => format!("NON-DÉTERMINISTE ({} empreintes) — INVALIDE", other.len()),
        };

        let m_fixed = median_us(fixed_times);
        let m_arrival = median_us(arrival_times);
        let overhead = if m_arrival > 0.0
        {
            m_fixed / m_arrival
        }
        else
        {
            f64::NAN
        };
        println!(
            "| {n} | {m_fixed:.0} | {m_arrival:.0} | {overhead:.3}× | {fixed_fp_display} | {} |",
            arrival_fps.len()
        );
    }

    println!();
    println!(
        "Lecture : overhead > 1 = le déterminisme coûte ; ≈ 1 = gratuit ; \
         < 1 = l'ordre figé est même plus rapide (slots sans contention vs \
         canal). L'empreinte « figé » doit être LA MÊME à tous les nombres \
         de threads (l'invariance inter-threads est testée en CI par \
         scirust-core::data_parallel ; ici elle est re-vérifiable à l'œil)."
    );
}
