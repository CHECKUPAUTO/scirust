//! Preuve de l'all-reduce à arbre fixe sur TCP **réel entre processus
//! séparés** — et, lancé ainsi par l'utilisateur, entre **machines
//! physiques séparées** sur un vrai lien réseau (pas seulement 127.0.0.1).
//! Complète le volet 116-A (`tree_allreduce::tcp_tree_all_reduce_rank`),
//! testé jusqu'ici uniquement en boucle locale.
//!
//! Chaque rang génère sa contribution **localement** (Philox, `seed`+rang :
//! reproductible sur toute machine), participe au protocole TCP réel, puis
//! le rang 0 **recalcule la référence en-process** (mêmes entrées régénérées
//! localement, [`tree_allreduce::tree_all_reduce`]) et compare bit à bit —
//! auto-vérifiant, AUCUNE empreinte à récolter à l'avance. Si les rangs
//! tournent sur des architectures différentes (ex. un rang sur Jetson
//! aarch64, le rang 0 sur Debian x86_64), un `verdict=PASS` prouve le
//! bit-exact inter-architectures **à travers un vrai réseau**.
//!
//! USAGE (un processus par rang — un ou plusieurs par machine) :
//!   proof_tcp_multihost --rank R --n N --seed SEED [--dim D]
//!       [--combine fixed|exact] [--my-addr HOST:PORT] [--parent-addr HOST:PORT]
//!
//! `--my-addr` : requis ssi ce rang a des enfants (2r+1 < n) — adresse de
//! *bind* du listener (ex. `0.0.0.0:9000`), pas nécessairement l'adresse
//! externe. `--parent-addr` : requis ssi rang > 0 — adresse EXTERNE
//! joignable du parent (ex. `192.168.1.10:9000`).
//!
//! Voir `scripts/proof-tcp-multihost.sh` pour un exemple 3 rangs / 2
//! machines complet.

use scirust_core::philox::Philox4x32;
use scirust_core::tree_allreduce::{
    Combine, ExactSum, FixedOrderSum, tcp_tree_all_reduce_rank, tree_all_reduce,
};
use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener};
use std::process::ExitCode;
use std::time::Duration;

fn parse_args() -> HashMap<String, String> {
    let mut m = HashMap::new();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len()
    {
        if let Some(key) = args[i].strip_prefix("--")
        {
            let val = args.get(i + 1).cloned().unwrap_or_default();
            m.insert(key.to_string(), val);
            i += 2;
        }
        else
        {
            i += 1;
        }
    }
    m
}

/// Génère la contribution du rang `r` : déterministe, donc identique sur
/// toute machine pour un `(seed, r, dim)` donné.
fn rank_input(seed: u64, rank: u32, dim: usize) -> Vec<f32> {
    let rng = Philox4x32::new(seed);
    let mut out = vec![0.0f32; dim];
    rng.fill_f32(rank, 0, &mut out);
    out
}

const RETRY_DELAY: Duration = Duration::from_millis(300);
const MAX_RETRIES: u32 = 100; // ~30 s

/// Rang **feuille** (sans enfant) : la connexion au parent est la toute
/// première action du protocole (aucun état accumulé avant), donc réessayer
/// l'appel entier est sûr si le parent n'écoute pas encore.
fn run_leaf<C: Combine>(
    rank: usize,
    n: usize,
    parent: SocketAddr,
    input: &[f32],
    combine: &C,
) -> std::io::Result<()>
where
    C::State: scirust_core::tree_allreduce::WireState,
{
    let mut last_err = None;
    for attempt in 0..MAX_RETRIES
    {
        match tcp_tree_all_reduce_rank(rank, n, None, Some(parent), input, combine)
        {
            Ok(_) => return Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused =>
            {
                if attempt == 0
                {
                    eprintln!("rang {rank} : parent {parent} pas encore prêt, réessai…");
                }
                last_err = Some(e);
                std::thread::sleep(RETRY_DELAY);
            },
            Err(e) => return Err(e),
        }
    }
    Err(last_err.unwrap())
}

/// Paramètres de lancement d'un rang (regroupés pour éviter une signature à
/// rallonge).
struct RunConfig {
    rank: usize,
    n: usize,
    seed: u64,
    dim: usize,
    my_addr: Option<SocketAddr>,
    parent_addr: Option<SocketAddr>,
}

fn run<C: Combine>(cfg: RunConfig, combine: C, combine_name: &str) -> std::io::Result<ExitCode>
where
    C::State: scirust_core::tree_allreduce::WireState,
{
    let RunConfig {
        rank,
        n,
        seed,
        dim,
        my_addr,
        parent_addr,
    } = cfg;
    let input = rank_input(seed, rank as u32, dim);
    let has_children = 2 * rank + 1 < n;

    let result = if has_children
    {
        let addr = my_addr.expect("rang avec enfants : --my-addr requis");
        let listener = TcpListener::bind(addr)?;
        eprintln!(
            "rang {rank}/{n} : écoute sur {addr} ({} enfant(s))",
            [2 * rank + 1, 2 * rank + 2]
                .into_iter()
                .filter(|&c| c < n)
                .count()
        );
        match parent_addr
        {
            Some(paddr) =>
            {
                // rang interne non racine : le blocage sur accept() ci-dessus
                // a déjà fait patienter — un seul essai de connexion suffit.
                tcp_tree_all_reduce_rank(rank, n, Some(&listener), Some(paddr), &input, &combine)?
            },
            None => tcp_tree_all_reduce_rank(rank, n, Some(&listener), None, &input, &combine)?,
        }
    }
    else
    {
        let paddr = parent_addr.expect("rang feuille : --parent-addr requis");
        run_leaf(rank, n, paddr, &input, &combine)?;
        None
    };

    match result
    {
        None =>
        {
            println!("rang {rank} : contribution envoyée (résultat au rang 0)");
            Ok(ExitCode::SUCCESS)
        },
        Some(network_result) =>
        {
            // auto-vérification : régénère les N entrées localement et
            // recalcule la référence EN-PROCESS (aucune empreinte à récolter
            // à l'avance) — un PASS ici couvre TOUT rang ayant réellement
            // participé sur le réseau, quelle que soit son architecture.
            let all_inputs: Vec<Vec<f32>> =
                (0..n as u32).map(|r| rank_input(seed, r, dim)).collect();
            let reference = tree_all_reduce(&all_inputs, &combine, None);
            let ok = network_result
                .iter()
                .zip(&reference)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            println!("PROOF-TCP-MULTIHOST v1");
            println!("config=n={n} dim={dim} seed={seed} combine={combine_name}");
            for (i, (net, refv)) in network_result.iter().zip(&reference).enumerate()
            {
                println!(
                    "result[{i}]=0x{:08x} reference[{i}]=0x{:08x}",
                    net.to_bits(),
                    refv.to_bits()
                );
            }
            println!("verdict={}", if ok { "PASS" } else { "FAIL" });
            Ok(
                if ok
                {
                    ExitCode::SUCCESS
                }
                else
                {
                    ExitCode::FAILURE
                },
            )
        },
    }
}

fn main() -> ExitCode {
    let args = parse_args();
    let get = |k: &str| args.get(k).cloned();
    let rank: usize = match get("rank").and_then(|s| s.parse().ok())
    {
        Some(v) => v,
        None =>
        {
            eprintln!("--rank requis (voir la doc du module)");
            return ExitCode::FAILURE;
        },
    };
    let n: usize = match get("n").and_then(|s| s.parse().ok())
    {
        Some(v) => v,
        None =>
        {
            eprintln!("--n requis");
            return ExitCode::FAILURE;
        },
    };
    let seed: u64 = match get("seed").and_then(|s| s.parse().ok())
    {
        Some(v) => v,
        None =>
        {
            eprintln!("--seed requis");
            return ExitCode::FAILURE;
        },
    };
    let dim: usize = get("dim").and_then(|s| s.parse().ok()).unwrap_or(8);
    let combine_name = get("combine").unwrap_or_else(|| "exact".to_string());
    let my_addr: Option<SocketAddr> = get("my-addr").and_then(|s| s.parse().ok());
    let parent_addr: Option<SocketAddr> = get("parent-addr").and_then(|s| s.parse().ok());

    if rank >= n
    {
        eprintln!("--rank doit être < --n");
        return ExitCode::FAILURE;
    }

    let cfg = RunConfig {
        rank,
        n,
        seed,
        dim,
        my_addr,
        parent_addr,
    };
    let result = match combine_name.as_str()
    {
        "fixed" => run(cfg, FixedOrderSum, "fixed"),
        "exact" => run(cfg, ExactSum, "exact"),
        other =>
        {
            eprintln!("--combine inconnu : {other} (attendu fixed|exact)");
            return ExitCode::FAILURE;
        },
    };

    match result
    {
        Ok(code) => code,
        Err(e) =>
        {
            eprintln!("erreur réseau (rang {rank}) : {e}");
            ExitCode::FAILURE
        },
    }
}
