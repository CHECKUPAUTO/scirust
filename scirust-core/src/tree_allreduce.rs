//! All-reduce **à arbre fixe**, transport-agnostique — le jalon « réduction
//! multi-nœud à arbre fixe » tracé depuis le volet 108 (GROWTH_PLAN).
//!
//! Principe : les rangs 0..n forment un arbre binaire **fixe** (enfants du
//! rang r : 2r+1 et 2r+2). Phase de montée : chaque rang combine son état
//! avec celui de ses enfants **dans un ordre imposé par la topologie**
//! (soi, puis enfant gauche, puis enfant droit) — chaque réception d'enfant
//! est attendue individuellement, donc l'ordre d'ARRIVÉE des messages
//! (gigue réseau, ordonnancement) n'a aucune influence sur le résultat.
//! Phase de descente : le rang 0 diffuse le résultat.
//!
//! La combinaison est un trait ([`Combine`]) ; deux implémentations :
//! - [`FixedOrderSum`] : addition f32 en ordre d'arbre — déterministe et
//!   bit-exacte pour une topologie donnée (l'extension multi-nœud de la
//!   réduction en ordre de rang de `distributed.rs`) ;
//! - [`ExactSum`] : accumulateurs de Kulisch ([`crate::exact_acc`]) —
//!   la somme est **exacte**, donc le résultat est indépendant non
//!   seulement du timing mais AUSSI de la topologie (même bits pour
//!   n = 2, 3, 8, 16…) et correctement arrondi.
//!
//! Le moteur fourni simule les rangs par threads + canaux (mpsc) avec une
//! **gigue adversariale** injectable — la démonstration que le déterminisme
//! vient de la structure, pas de la chance. Pour un déploiement réel, le
//! même code de combinaison se branche sur n'importe quel transport
//! (TCP/MPI/shm) : il suffit que chaque rang attende ses enfants dans
//! l'ordre de l'arbre.

use crate::exact_acc::ExactAcc;
use std::sync::mpsc;

/// Combinaison associée à l'all-reduce (l'ordre d'appel est fixé par
/// l'arbre : soi ⊕ gauche ⊕ droite).
pub trait Combine: Sync {
    /// État transmis entre rangs.
    type State: Send;
    /// État initial d'un rang à partir de sa contribution locale.
    fn leaf(&self, data: &[f32]) -> Self::State;
    /// Absorbe l'état d'un enfant (appelé dans l'ordre de l'arbre).
    fn absorb(&self, acc: &mut Self::State, child: Self::State);
    /// Matérialise le résultat final.
    fn finish(&self, acc: Self::State) -> Vec<f32>;
}

/// Somme f32 en ordre d'arbre fixe : déterministe pour une topologie donnée.
pub struct FixedOrderSum;

impl Combine for FixedOrderSum {
    type State = Vec<f32>;
    fn leaf(&self, data: &[f32]) -> Vec<f32> {
        data.to_vec()
    }
    fn absorb(&self, acc: &mut Vec<f32>, child: Vec<f32>) {
        for (a, c) in acc.iter_mut().zip(child)
        {
            *a += c;
        }
    }
    fn finish(&self, acc: Vec<f32>) -> Vec<f32> {
        acc
    }
}

/// Somme **exacte** par accumulateurs de Kulisch : indépendante du timing ET
/// de la topologie, correctement arrondie (contributions = produits x·1).
pub struct ExactSum;

impl Combine for ExactSum {
    type State = Vec<ExactAcc>;
    fn leaf(&self, data: &[f32]) -> Vec<ExactAcc> {
        data.iter()
            .map(|&x| {
                let mut acc = ExactAcc::new();
                acc.add_product(x, 1.0);
                acc
            })
            .collect()
    }
    fn absorb(&self, acc: &mut Vec<ExactAcc>, child: Vec<ExactAcc>) {
        for (a, c) in acc.iter_mut().zip(child.iter())
        {
            a.merge(c);
        }
    }
    fn finish(&self, acc: Vec<ExactAcc>) -> Vec<f32> {
        acc.iter().map(|a| a.round_f32()).collect()
    }
}

/// All-reduce à arbre fixe sur `inputs[r]` (contribution du rang r), simulé
/// par threads + canaux. `jitter_ms[r]` (optionnel) retarde l'envoi du rang
/// r — l'injection de gigue adversariale des tests. Le résultat ne dépend
/// que de la topologie (et pour [`ExactSum`], même pas d'elle).
pub fn tree_all_reduce<C: Combine>(
    inputs: &[Vec<f32>],
    combine: &C,
    jitter_ms: Option<&[u64]>,
) -> Vec<f32> {
    let n = inputs.len();
    assert!(n > 0, "tree_all_reduce: aucun rang");
    let dim = inputs[0].len();
    assert!(
        inputs.iter().all(|v| v.len() == dim),
        "tree_all_reduce: dimensions hétérogènes"
    );

    // un canal de réception par rang (les enfants y envoient leur état)
    let mut txs = Vec::with_capacity(n);
    let mut rxs = Vec::with_capacity(n);
    for _ in 0..n
    {
        let (tx, rx) = mpsc::channel::<(usize, C::State)>();
        txs.push(tx);
        rxs.push(Some(rx));
    }

    let mut result: Option<Vec<f32>> = None;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for r in (0..n).rev()
        {
            let rx = rxs[r].take().expect("rx unique");
            let parent_tx = if r > 0
            {
                Some(txs[(r - 1) / 2].clone())
            }
            else
            {
                None
            };
            let input = &inputs[r];
            let delay = jitter_ms.map(|j| j[r]).unwrap_or(0);
            handles.push(scope.spawn(move || {
                let mut state = combine.leaf(input);
                // Absorbe chaque enfant DANS L'ORDRE DE L'ARBRE (gauche puis
                // droite). Un message arrivé hors ordre est mis en attente,
                // jamais jeté : l'ordre d'ARRIVÉE est sans effet sur le
                // résultat, seul l'ordre de l'arbre compte.
                let mut pending: Vec<(usize, C::State)> = Vec::new();
                for child in [2 * r + 1, 2 * r + 2]
                {
                    if child < n
                    {
                        let s = if let Some(pos) =
                            pending.iter().position(|(from, _)| *from == child)
                        {
                            pending.swap_remove(pos).1
                        }
                        else
                        {
                            loop
                            {
                                let (from, st) = rx.recv().expect("canal fermé");
                                if from == child
                                {
                                    break st;
                                }
                                pending.push((from, st));
                            }
                        };
                        combine.absorb(&mut state, s);
                    }
                }
                if delay > 0
                {
                    std::thread::sleep(std::time::Duration::from_millis(delay));
                }
                match parent_tx
                {
                    Some(tx) =>
                    {
                        tx.send((r, state)).expect("envoi au parent");
                        None
                    },
                    None => Some(combine.finish(state)),
                }
            }));
        }
        for h in handles
        {
            if let Some(res) = h.join().expect("rang panique")
            {
                result = Some(res);
            }
        }
    });
    result.expect("le rang 0 produit le résultat")
}

// ================================================================== //
//  Transport TCP réel                                                 //
// ================================================================== //

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};

/// État sérialisable pour le transport réseau. Les encodages sont en
/// **little-endian explicite** : les octets sont identiques sur toute
/// plate-forme, la garantie bit-exacte traverse donc le réseau.
pub trait WireState: Sized {
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(b: &[u8]) -> Self;
}

impl WireState for Vec<f32> {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 * self.len());
        for &x in self
        {
            out.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        out
    }
    fn from_bytes(b: &[u8]) -> Self {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_bits(u32::from_le_bytes(*c)))
            .collect()
    }
}

impl WireState for Vec<ExactAcc> {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for acc in self
        {
            for w in acc.to_words()
            {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }
    fn from_bytes(b: &[u8]) -> Self {
        let per = core::mem::size_of::<u64>() * 22; // 2 × 11 mots
        b.chunks_exact(per)
            .map(|chunk| {
                let mut words = [0u64; 22];
                for (i, c) in chunk.as_chunks::<8>().0.iter().enumerate()
                {
                    words[i] = u64::from_le_bytes(*c);
                }
                ExactAcc::from_words(&words)
            })
            .collect()
    }
}

fn send_state(stream: &mut TcpStream, rank: u32, bytes: &[u8]) -> std::io::Result<()> {
    stream.write_all(&rank.to_le_bytes())?;
    stream.write_all(&(bytes.len() as u64).to_le_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()
}

fn recv_state(stream: &mut TcpStream) -> std::io::Result<(u32, Vec<u8>)> {
    let mut hdr = [0u8; 12];
    stream.read_exact(&mut hdr)?;
    let rank = u32::from_le_bytes(hdr[..4].try_into().unwrap());
    let len = u64::from_le_bytes(hdr[4..].try_into().unwrap()) as usize;
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body)?;
    Ok((rank, body))
}

/// Un rang de l'all-reduce à arbre fixe **sur TCP réel** — la version
/// multi-processus/multi-machine du moteur in-process : même topologie,
/// même ordre d'absorption (par enfant, hors-ordre mis en attente), donc
/// mêmes bits. `listener` : socket lié du rang courant (None pour les
/// feuilles) ; `parent` : adresse du parent (None pour le rang 0, qui
/// renvoie `Some(résultat)`).
pub fn tcp_tree_all_reduce_rank<C: Combine>(
    rank: usize,
    n: usize,
    listener: Option<&TcpListener>,
    parent: Option<SocketAddr>,
    input: &[f32],
    combine: &C,
) -> std::io::Result<Option<Vec<f32>>>
where
    C::State: WireState,
{
    let mut state = combine.leaf(input);
    let children: Vec<usize> = [2 * rank + 1, 2 * rank + 2]
        .into_iter()
        .filter(|&c| c < n)
        .collect();
    if !children.is_empty()
    {
        let listener = listener.expect("rang interne sans listener");
        // collecte les états des enfants (l'ordre de CONNEXION est libre)
        let mut pending: Vec<(u32, Vec<u8>)> = Vec::new();
        while pending.len() < children.len()
        {
            let (mut s, _) = listener.accept()?;
            pending.push(recv_state(&mut s)?);
        }
        // absorption DANS L'ORDRE DE L'ARBRE, quel que soit l'ordre d'arrivée
        for &child in &children
        {
            let pos = pending
                .iter()
                .position(|(from, _)| *from == child as u32)
                .expect("enfant manquant");
            let (_, bytes) = pending.swap_remove(pos);
            combine.absorb(&mut state, C::State::from_bytes(&bytes));
        }
    }
    match parent
    {
        Some(addr) =>
        {
            let mut s = TcpStream::connect(addr)?;
            send_state(&mut s, rank as u32, &state.to_bytes())?;
            Ok(None)
        },
        None => Ok(Some(combine.finish(state))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::PcgEngine;
    use crate::philox::Philox4x32;

    fn make_inputs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = PcgEngine::new(seed);
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|i| (rng.float() * 2.0 - 1.0) * 10f32.powi((i % 9) as i32 - 4))
                    .collect()
            })
            .collect()
    }

    /// La propriété centrale : sous gigue adversariale (retards différents à
    /// chaque essai), le résultat est BIT-IDENTIQUE au run sans gigue.
    #[test]
    fn timing_jitter_never_changes_bits() {
        for &n in &[2usize, 3, 5, 8, 16]
        {
            let inputs = make_inputs(n, 64, 42);
            let reference = tree_all_reduce(&inputs, &FixedOrderSum, None);
            let jitter_rng = Philox4x32::new(777);
            for trial in 0..5u32
            {
                let jitter: Vec<u64> = (0..n)
                    .map(|r| (jitter_rng.u32_at(trial, r as u64) % 8) as u64)
                    .collect();
                let jittered = tree_all_reduce(&inputs, &FixedOrderSum, Some(&jitter));
                for i in 0..reference.len()
                {
                    assert_eq!(
                        reference[i].to_bits(),
                        jittered[i].to_bits(),
                        "n={n}, essai {trial}, élément {i}"
                    );
                }
            }
        }
    }

    /// ExactSum : le résultat est indépendant de la TOPOLOGIE elle-même
    /// (même bits pour tout n découpant les mêmes contributions), égal à la
    /// référence correctement arrondie de `reproducible_sum` élément par
    /// élément.
    #[test]
    fn exact_sum_is_topology_independent_and_correctly_rounded() {
        let dim = 32;
        let all = make_inputs(16, dim, 7);
        // référence : somme correctement arrondie des 16 contributions
        let mut expected = vec![0.0f32; dim];
        for (i, e) in expected.iter_mut().enumerate()
        {
            let col: Vec<f32> = all.iter().map(|v| v[i]).collect();
            *e = crate::reproducible::reproducible_sum(&col);
        }
        // arbre complet + gigue adversariale : toujours la référence exacte
        let jitter_rng = Philox4x32::new(313);
        for trial in 0..4u32
        {
            let jitter: Vec<u64> = (0..16)
                .map(|r| (jitter_rng.u32_at(trial, r as u64) % 8) as u64)
                .collect();
            let result = tree_all_reduce(&all, &ExactSum, Some(&jitter));
            for i in 0..dim
            {
                assert_eq!(
                    result[i].to_bits(),
                    expected[i].to_bits(),
                    "essai {trial}, élément {i}"
                );
            }
        }
    }

    /// FixedOrderSum n=1..3 : égal à la somme séquentielle en ordre d'arbre
    /// (rang 0 ⊕ rang 1 ⊕ rang 2) — le contrat de la voie f32.
    #[test]
    fn fixed_order_matches_sequential_tree_order() {
        let inputs = make_inputs(3, 16, 99);
        let result = tree_all_reduce(&inputs, &FixedOrderSum, None);
        for i in 0..16
        {
            let expected = inputs[0][i] + inputs[1][i] + inputs[2][i];
            assert_eq!(result[i].to_bits(), expected.to_bits(), "élément {i}");
        }
    }

    /// TCP réel (sockets 127.0.0.1) : bit-identique au moteur in-process,
    /// sous gigue, pour les deux combinaisons — la garantie traverse le
    /// réseau (encodage little-endian explicite).
    #[test]
    fn tcp_transport_matches_in_process_bitwise() {
        for &n in &[3usize, 8]
        {
            let inputs = make_inputs(n, 32, 2026);
            let expected_fixed = tree_all_reduce(&inputs, &FixedOrderSum, None);
            let expected_exact = tree_all_reduce(&inputs, &ExactSum, None);

            for exact in [false, true]
            {
                // listeners liés d'abord (ports éphémères), adresses connues
                let listeners: Vec<Option<std::net::TcpListener>> = (0..n)
                    .map(|r| {
                        if 2 * r + 1 < n
                        {
                            Some(std::net::TcpListener::bind("127.0.0.1:0").unwrap())
                        }
                        else
                        {
                            None
                        }
                    })
                    .collect();
                let addrs: Vec<Option<std::net::SocketAddr>> = listeners
                    .iter()
                    .map(|l| l.as_ref().map(|l| l.local_addr().unwrap()))
                    .collect();

                let jitter_rng = Philox4x32::new(4242);
                let mut result: Option<Vec<f32>> = None;
                std::thread::scope(|scope| {
                    let mut handles = Vec::new();
                    for r in (0..n).rev()
                    {
                        let listener = listeners[r].as_ref();
                        let parent = if r > 0 { addrs[(r - 1) / 2] } else { None };
                        let input = &inputs[r];
                        let delay = (jitter_rng.u32_at(0, r as u64) % 6) as u64;
                        handles.push(scope.spawn(move || {
                            std::thread::sleep(std::time::Duration::from_millis(delay));
                            if exact
                            {
                                tcp_tree_all_reduce_rank(r, n, listener, parent, input, &ExactSum)
                                    .unwrap()
                            }
                            else
                            {
                                tcp_tree_all_reduce_rank(
                                    r,
                                    n,
                                    listener,
                                    parent,
                                    input,
                                    &FixedOrderSum,
                                )
                                .unwrap()
                            }
                        }));
                    }
                    for h in handles
                    {
                        if let Some(res) = h.join().unwrap()
                        {
                            result = Some(res);
                        }
                    }
                });
                let got = result.expect("rang 0");
                let expected = if exact
                {
                    &expected_exact
                }
                else
                {
                    &expected_fixed
                };
                for i in 0..32
                {
                    assert_eq!(
                        got[i].to_bits(),
                        expected[i].to_bits(),
                        "n={n}, exact={exact}, élément {i}"
                    );
                }
            }
        }
    }

    /// Un seul rang : identité.
    #[test]
    fn single_rank_is_identity() {
        let inputs = make_inputs(1, 8, 5);
        let result = tree_all_reduce(&inputs, &FixedOrderSum, None);
        for i in 0..8
        {
            assert_eq!(result[i].to_bits(), inputs[0][i].to_bits());
        }
    }
}
