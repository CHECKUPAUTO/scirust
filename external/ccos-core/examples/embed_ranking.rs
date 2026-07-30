//! Does the learned LSA embedder beat TF-IDF at its *home turf* — **dense ranking**?
//!
//! The recall benchmark (`recall_eval`) showed LSA hurts CCOS's *entry-selection*
//! recall. But LSA's classic strength is ranking *many* candidates, not picking one
//! entry. This isolates that: for each query with a known target file, rank **all**
//! nodes by cosine in the embedder's space and measure **recall@k** (is the target
//! in the top-k?). Both embedders are fitted on the same corpus in one run, so the
//! comparison is apples-to-apples — no feature flag needed.
//!
//! Run: `cargo run --release --example embed_ranking`

use ccos_core::embeddings::CausalEmbeddings;

const DOMAINS: [&str; 10] = [
    "payment",
    "inventory",
    "shipping",
    "auth",
    "billing",
    "catalog",
    "ledger",
    "checkout",
    "refund",
    "subscription",
];
const SYNONYMS: [&str; 10] = [
    "remittance",
    "stock",
    "dispatch",
    "login",
    "invoicing",
    "listing",
    "accounts",
    "purchase",
    "chargeback",
    "membership",
];

/// `(id, text)` corpus documents.
type Docs = Vec<(String, String)>;
/// `(query, target-file, kind)` evaluation triples.
type Queries = Vec<(String, String, &'static str)>;

/// (id, text) corpus, plus the (query, target-domain) pairs and their kind.
fn build() -> (Docs, Queries) {
    let mut docs: Docs = Vec::new();

    for (d, s) in DOMAINS.iter().zip(SYNONYMS.iter()) {
        // The TARGET file: contains only the domain term (never the synonym).
        docs.push((
            format!("src/{d}.rs"),
            format!(
                "// {d} module: {d} domain logic for the {d} pipeline.\n\
                     pub fn {d}_run() {{}}\npub fn {d}_apply() {{}}\n"
            ),
        ));
        // Several context docs co-occur the domain term with its synonym, building a
        // latent link without ever being a better literal match for the target.
        for j in 0..4 {
            docs.push((
                format!("src/ctx_{d}_{j}.rs"),
                format!("// note {j}: the {d} step (also known as {s}) feeds the next stage; {s} and {d} share state.\n"),
            ));
        }
    }
    // Unrelated filler so the ranking has distractors to push the target past.
    for i in 0..50 {
        docs.push((
            format!("src/filler_{i}.rs"),
            format!("// helper {i}: assorted unrelated utility code path number {i}.\n"),
        ));
    }

    let mut queries: Queries = Vec::new();
    for (d, s) in DOMAINS.iter().zip(SYNONYMS.iter()) {
        // plain: the target's own term (both embedders should ace this).
        queries.push((format!("{d} domain logic"), format!("src/{d}.rs"), "plain"));
        // synonym: a term the target file NEVER contains (TF-IDF should score ~0).
        queries.push((format!("{s} stage state"), format!("src/{d}.rs"), "synonym"));
    }
    (docs, queries)
}

fn recall_at(store: &CausalEmbeddings, queries: &Queries, k: usize, kind: &str) -> f64 {
    let mut hits = 0usize;
    let mut n = 0usize;
    for (q, target, qkind) in queries.iter().filter(|(_, _, kk)| *kk == kind) {
        n += 1;
        let ranked = store.nearest_k(&store.embed_query(q), k);
        // hit if any node of the target file is in the top-k.
        if ranked.iter().any(|(id, _)| id.contains(target.as_str())) {
            hits += 1;
        }
        let _ = qkind;
    }
    if n == 0 {
        0.0
    } else {
        100.0 * hits as f64 / n as f64
    }
}

fn fit_tfidf(docs: &[(String, String)]) -> CausalEmbeddings {
    let mut s = CausalEmbeddings::new();
    s.fit_and_embed(docs.iter().map(|(a, b)| (a.as_str(), b.as_str())));
    s
}
fn fit_lsa(docs: &[(String, String)], rank: usize) -> CausalEmbeddings {
    let mut s = CausalEmbeddings::new();
    s.fit_and_embed_lsa(docs.iter().map(|(a, b)| (a.as_str(), b.as_str())), rank);
    s
}

fn main() {
    let (docs, queries) = build();
    println!(
        "# Dense-ranking recall@k — TF-IDF vs LSA ({} docs, {} queries)\n",
        docs.len(),
        queries.len()
    );

    let embedders: Vec<(String, CausalEmbeddings)> = vec![
        ("tfidf".to_string(), fit_tfidf(&docs)),
        ("lsa-rank16".to_string(), fit_lsa(&docs, 16)),
        ("lsa-rank48".to_string(), fit_lsa(&docs, 48)),
    ];

    for kind in ["plain", "synonym"] {
        println!("## {kind} queries");
        print!("{:<12}", "embedder");
        for k in [1usize, 3, 5, 10] {
            print!("   recall@{k:<3}");
        }
        println!();
        for (name, store) in &embedders {
            print!("{name:<12}");
            for k in [1usize, 3, 5, 10] {
                print!("{:>11.0}%", recall_at(store, &queries, k, kind));
            }
            println!();
        }
        println!();
    }
}
