//! **Contradiction crux: what does a typed `Contradicts` edge recover that similarity can't?** The
//! Q-Page analogue of the call/import crux. A vector retriever ranks by *relatedness*, and
//! relatedness has **no notion of polarity**: a statement that *refutes* a claim and one that
//! *confirms* it are both "about" the claim, so similarity cannot tell them apart — and a lone
//! refutation, swamped by abundant confirmations, is crowded out of the top of the ranking with no
//! flag that it was the one dissent. The Q-Page's `Contradicts` edge carries that polarity by
//! construction: the negative surface is surfaced as such, and [`MemoryGraph::qbelief`] turns it
//! into a `conflict` signal — high precisely when a claim is *contested*, the case a confirm-only
//! retriever is blind to.
//!
//! Fixture: a claim, four confirmations (lexically echoing it), **one** refutation (the minority
//! dissent — shares the claim's subject vocabulary but opposes it), and unrelated decoys. The
//! `Supports`/`Contradicts` edges are exactly what an agent records via
//! `CcosMemory::assert_support` / `assert_contradiction`; we build the graph directly so the
//! lexical baseline has text to embed. We then ask a per-statement TF-IDF retriever to rank the
//! pool by similarity to the claim, and contrast what it can (and cannot) report with the graph.
//!
//! Run: `cargo run --release --example contradiction_crux`

use ccos_core::embeddings::{tokenize, TfidfEmbedder};
use ccos_core::memory::{EdgeType, MemoryGraph, NodeId, NodeType};
use std::collections::BTreeSet;

fn main() {
    let claim = "the scheduler admits every queued job within the deadline";
    // Confirmations: heavy lexical overlap with the claim (a retriever's bread and butter).
    let support: &[&str] = &[
        "the scheduler admits every queued job within the deadline under nominal load",
        "queued jobs are admitted within the deadline by the scheduler in staging",
        "the scheduler met the deadline for all admitted jobs in the soak test",
        "every queued job was admitted within the deadline during the canary run",
    ];
    // The lone refutation: shares the *subject* (scheduler, queued job, deadline) but opposes the
    // claim. Lexically it is squarely "about" the claim — yet its meaning is the opposite.
    let contradiction: &[&str] =
        &["under burst load the scheduler starved a queued job and missed its deadline"];
    // Unrelated distractors.
    let decoys: &[&str] = &[
        "the billing service reconciles outstanding invoices nightly",
        "the cache evicts the least recently used entry when full",
        "the parser tokenizes identifiers in a single linear pass",
    ];

    // ── Build the Q-Page: a claim with two typed evidence surfaces ──────────────────────────────
    let mut g = MemoryGraph::new(0.0, usize::MAX);
    let claim_id: NodeId = "claim".into();
    g.upsert_node(
        claim_id.clone(),
        "claim".into(),
        claim.to_string(),
        NodeType::ContextBlock,
    );
    let mut pool: Vec<(NodeId, String)> = Vec::new();
    for (i, s) in support.iter().enumerate() {
        let id = NodeId(format!("s{i}"));
        g.upsert_node(
            id.clone(),
            id.0.clone(),
            (*s).to_string(),
            NodeType::ContextBlock,
        );
        g.add_edge(id.clone(), claim_id.clone(), 1.0, EdgeType::Supports);
        pool.push((id, (*s).to_string()));
    }
    for (i, x) in contradiction.iter().enumerate() {
        let id = NodeId(format!("x{i}"));
        g.upsert_node(
            id.clone(),
            id.0.clone(),
            (*x).to_string(),
            NodeType::ContextBlock,
        );
        g.add_edge(id.clone(), claim_id.clone(), 1.0, EdgeType::Contradicts);
        pool.push((id, (*x).to_string()));
    }
    for (i, d) in decoys.iter().enumerate() {
        let id = NodeId(format!("d{i}"));
        g.upsert_node(
            id.clone(),
            id.0.clone(),
            (*d).to_string(),
            NodeType::ContextBlock,
        );
        pool.push((id, (*d).to_string()));
    }
    pool.sort_by(|a, b| a.0 .0.cmp(&b.0 .0)); // deterministic

    // Ground-truth polarity, straight from the typed edges.
    let supports: BTreeSet<&NodeId> = g
        .evidence_of(&claim_id, EdgeType::Supports)
        .into_iter()
        .collect();
    let contradicts: BTreeSet<&NodeId> = g
        .evidence_of(&claim_id, EdgeType::Contradicts)
        .into_iter()
        .collect();
    let polarity = |id: &NodeId| -> &'static str {
        if supports.contains(id) {
            "support"
        } else if contradicts.contains(id) {
            "CONTRADICT"
        } else {
            "decoy"
        }
    };

    // ── Lexical baseline: TF-IDF over the pool, ranked by cosine to the claim ───────────────────
    let corpus: Vec<Vec<String>> = pool.iter().map(|(_, t)| tokenize(t)).collect();
    let mut fit_corpus = corpus.clone();
    fit_corpus.push(tokenize(claim));
    let mut tf = TfidfEmbedder::new(128);
    tf.fit(&fit_corpus);
    let qv = tf.embed(&tokenize(claim));
    let mut ranked: Vec<(f64, &NodeId, &String)> = pool
        .iter()
        .zip(&corpus)
        .map(|((id, text), toks)| (TfidfEmbedder::cosine(&qv, &tf.embed(toks)), id, text))
        .collect();
    // Sort by cosine desc, ties broken by id for determinism.
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1 .0.cmp(&b.1 .0)));

    // ── Report ──────────────────────────────────────────────────────────────────────────────────
    println!(
        "# Contradiction crux — what a typed `Contradicts` edge recovers that similarity can't\n"
    );
    println!(
        "claim: \"{claim}\"\nfixture: {} support, {} contradiction, {} decoys ({} in the pool)\n",
        support.len(),
        contradiction.len(),
        decoys.len(),
        pool.len()
    );

    println!("LEXICAL TF-IDF — pool ranked by cosine to the claim (similarity is polarity-blind):");
    println!("  rank  cosine  polarity     text");
    for (i, (cos, id, text)) in ranked.iter().enumerate() {
        let short: String = text.chars().take(58).collect();
        println!(
            "  {:>4}  {:>5.2}  {:<11}  {}",
            i + 1,
            cos,
            polarity(id),
            short
        );
    }

    // Rank of the contradiction, and the support cosine band it falls inside.
    let contra_rank = ranked
        .iter()
        .position(|(_, id, _)| contradicts.contains(id))
        .map(|p| p + 1)
        .unwrap();
    let contra_cos = ranked
        .iter()
        .find(|(_, id, _)| contradicts.contains(id))
        .map(|(c, _, _)| *c)
        .unwrap();
    let support_cos: Vec<f64> = ranked
        .iter()
        .filter(|(_, id, _)| supports.contains(id))
        .map(|(c, _, _)| *c)
        .collect();
    let (min_s, max_s) = (
        support_cos.iter().cloned().fold(f64::INFINITY, f64::min),
        support_cos
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let inside_band = contra_cos >= min_s && contra_cos <= max_s;
    let k = support.len() + contradiction.len();
    let top_k_has_contra = ranked
        .iter()
        .take(k)
        .any(|(_, id, _)| contradicts.contains(id));

    let q = g.qbelief(&claim_id);
    println!(
        "\nThe contradiction ranks #{contra_rank} (cosine {contra_cos:.2}); the support cosines span \
         [{min_s:.2}, {max_s:.2}].\n  → its similarity sits {} the support band, so NO cosine \
         threshold separates support from refutation: similarity cannot label polarity.",
        if inside_band { "INSIDE" } else { "outside" }
    );
    println!(
        "  → top-{k} by similarity {} the refutation — and even when present it carries no flag \
         that it is a refutation.",
        if top_k_has_contra {
            "includes"
        } else {
            "MISSES"
        }
    );
    println!(
        "\nQ-PAGE — the `Contradicts` edge surfaces the dissent by construction, and qbelief makes \
         it a number:\n  belief {:.2}   conflict {:.2}   (support {:.0}, contradiction {:.0})\n  \
         → belief is net-positive (support outweighs the lone dissent) yet conflict {} 0: the claim is \
         *contested*, the exact state a confirm-only retriever never reports.",
        q.belief,
        q.conflict,
        q.support,
        q.contradiction,
        if q.conflict > 0.0 { ">" } else { "==" }
    );
    println!(
        "\nReading: a vector retriever answers \"what is most related to the claim\" — and a \
         refutation is highly related, so it is retrieved without ever being marked as opposition, \
         while the lone dissent competes for top-k slots against a crowd of confirmations. The \
         Q-Page stores polarity as structure: support and contradiction are distinct edges, so the \
         dissent is always surfaced *as* a dissent and `conflict` flags the claim for resolution. \
         This is the contested-knowledge analogue of the import/call crux — structure recovers what \
         a similarity score cannot represent."
    );
}
