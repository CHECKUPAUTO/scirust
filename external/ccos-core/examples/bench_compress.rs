use ccos_core::agent_session::AgentSession;
use ccos_core::compressor::CausalCompressor;
use ccos_core::external_memory::Recall;

fn ingest_corpus() -> (AgentSession, Vec<(String, String)>) {
    // Derive the corpus path from the crate root so the example runs from any
    // checkout (it previously hard-coded /root/CCOS/src and panicked elsewhere).
    let root = env!("CARGO_MANIFEST_DIR");
    let src = format!("{root}/src");
    let files: Vec<(String, String)> = std::fs::read_dir(&src)
        .unwrap_or_else(|e| panic!("read_dir {src}: {e}"))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "rs").unwrap_or(false))
        .filter_map(|e| {
            let p = e.path();
            let uri = p
                .strip_prefix(root)
                .map(|x| x.display().to_string())
                .unwrap_or_else(|_| p.display().to_string());
            std::fs::read_to_string(&p).ok().map(|s| (uri, s))
        })
        .collect();
    let mut s = AgentSession::new();
    for (uri, src) in &files {
        s.ingest(uri, src);
    }
    (s, files)
}

fn tokens_of(win: &ccos_core::external_memory::RecallWindow) -> usize {
    win.items
        .iter()
        .map(|i| i.content.chars().count() / 4)
        .sum()
}

fn run(label: &str, recall: Recall, budget: usize) {
    let (mut s, _) = ingest_corpus();
    let raw = s.recall(recall.clone(), budget);
    let raw_tokens = tokens_of(&raw);

    let (mut s2, _) = ingest_corpus();
    let comp = s2.recall_compressed(recall.clone(), budget);
    let comp_tokens = tokens_of(&comp);
    let n_compressed = comp.items.iter().filter(|i| i.ccr_ref.is_some()).count();

    let (mut s3, _) = ingest_corpus();
    let feedback = s3.recall_compressed_with_feedback(recall.clone(), budget, 3);
    let fb_tokens = tokens_of(&feedback);
    let fb_items = feedback.items.len();

    println!("\n=== {label} (budget={budget}) ===");
    println!(
        "RAW:       items={:3} tokens={:5}",
        raw.items.len(),
        raw_tokens
    );
    println!(
        "COMP:      items={:3} tokens={:5}  compressed={}/{}",
        comp.items.len(),
        comp_tokens,
        n_compressed,
        comp.items.len()
    );
    println!(
        "FEEDBACK:  items={:3} tokens={:5}  (budget respected: {})",
        fb_items,
        fb_tokens,
        fb_tokens <= budget
    );
    let ratio_comp = comp_tokens as f64 / raw_tokens.max(1) as f64;
    let ratio_fb = fb_tokens as f64 / raw_tokens.max(1) as f64;
    let extra = fb_items as i64 - comp.items.len() as i64;
    println!(
        "  comp:  {:.0}% reduction ({:.2}x)   feedback: {:.0}% reduction ({:.2}x) +{} extra items",
        (1.0 - ratio_comp) * 100.0,
        1.0 / ratio_comp,
        (1.0 - ratio_fb) * 100.0,
        1.0 / ratio_fb,
        extra
    );
}

fn run_auto_tune() {
    let (mut s, _) = ingest_corpus();
    // Sample = a working_set window at a generous budget.
    let sample_win = s.recall(Recall::working_set(), 8192);
    let owned: Vec<(String, f64, String, String)> = sample_win
        .items
        .iter()
        .map(|i| (i.kind.clone(), i.score, i.uri.clone(), i.content.clone()))
        .collect();
    let sample: Vec<(&str, f64, &str, &str)> = owned
        .iter()
        .map(|(k, s, u, v)| (k.as_str(), *s, u.as_str(), v.as_str()))
        .collect();

    let base = CausalCompressor::new();
    let tuned = base.auto_tune(&sample);
    let base_tokens = CausalCompressor::eval_config(&base.config, &sample);
    let tuned_tokens = CausalCompressor::eval_config(&tuned, &sample);
    println!("\n=== AUTO-TUNE on working_set @8192 sample ===");
    println!("base config:   {} tokens", base_tokens);
    println!(
        "tuned config:  {} tokens  ({:.0}% better)",
        tuned_tokens,
        (1.0 - tuned_tokens as f64 / base_tokens.max(1) as f64) * 100.0
    );
    println!("tuned knobs:");
    println!(
        "  enable_dedup={}  dedup_threshold={:.2}",
        tuned.enable_dedup, tuned.dedup_threshold
    );
    println!(
        "  enable_ast_v2={}  ast_signature_collapse_after={}",
        tuned.enable_ast_v2, tuned.ast_signature_collapse_after
    );
    println!(
        "  enable_prose={}  summary_sentences={}",
        tuned.enable_prose, tuned.summary_sentences
    );
    println!("  min_chars={}", tuned.min_chars);
}

fn main() {
    println!(
        "Corpus: {} Rust files from {}/src",
        ingest_corpus().1.len(),
        env!("CARGO_MANIFEST_DIR")
    );
    run("working_set @2048", Recall::working_set(), 2048);
    run("working_set @8192", Recall::working_set(), 8192);
    run(
        "around parser @4096",
        Recall::around("file:src/parser.rs"),
        4096,
    );
    run(
        "around external_memory @8192",
        Recall::around("file:src/external_memory.rs"),
        8192,
    );
    run(
        "task 'failure propagation' @4096",
        Recall::task("failure propagation"),
        4096,
    );
    run_auto_tune();
}
