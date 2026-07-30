//! **Conflict of Origins** — when two sources disagree about the *same* claim, a flat store has to
//! pick one, average them, or take the most recent. The Q-Page instead **nets them by authority**:
//! the more credible source sets the *direction* of `belief`, while `conflict` reports how contested
//! the claim still is and [`QBelief::is_validated`](ccos_core::memory::QBelief::is_validated) decides
//! whether it is safe to act on. This is the payoff of per-source authority weighting.
//!
//! Fixture: one claim, asserted **+** by a high-authority source A (official docs, authority 0.90)
//! and **−** by a source B (an incident report) whose authority `β` we sweep. The assertions are
//! produced through the [`Extractor`](ccos_core::extractor::Extractor) pipeline (a deterministic
//! [`MockExtractor`] here — the LLM-backed extractor produces the *same* shape from raw text) and
//! recorded via the normal `assert_support` / `assert_contradiction` path, so this is exactly what an
//! ingested document pair would yield.
//!
//! Run: `cargo run --release --example conflict_of_origins`

use ccos_core::external_memory::CcosMemory;
use ccos_core::extractor::{Assertion, Extractor, MockExtractor, Stance};
use ccos_core::memory::{NodeId, QBelief};

const CLAIM: &str = "claim:payment-api-thread-safe";

/// The two opposing assertions: A supports at authority 0.90, B contradicts at authority `beta`.
fn assertions(beta: f64) -> Vec<Assertion> {
    vec![
        Assertion {
            claim: CLAIM.into(),
            source: "src:official-docs".into(),
            stance: Stance::Supports,
            authority: 0.90,
        },
        Assertion {
            claim: CLAIM.into(),
            source: "src:incident-report".into(),
            stance: Stance::Contradicts,
            authority: beta,
        },
    ]
}

/// Distill (mock) → record as assertions → read back the claim's authority-weighted belief.
fn resolve(beta: f64) -> QBelief {
    let mut mem = CcosMemory::new();
    let extractor = MockExtractor::new(assertions(beta));
    for a in extractor.extract("<source documents>").unwrap() {
        match a.stance {
            Stance::Supports => mem.assert_support(&a.source, &a.claim, a.clamped_authority()),
            Stance::Contradicts => {
                mem.assert_contradiction(&a.source, &a.claim, a.clamped_authority())
            }
        };
    }
    mem.graph().qbelief(&NodeId(CLAIM.into()))
}

fn main() {
    let (min_belief, max_conflict) = (0.30, 0.50);
    println!("# Conflict of Origins — authority-weighted resolution of disagreeing sources\n");
    println!("claim: \"the payment API is thread-safe\"");
    println!("  Source A = official-docs   (authority 0.90) → SUPPORTS");
    println!("  Source B = incident-report (authority β)     → CONTRADICTS\n");
    println!("validation gate: belief ≥ {min_belief:.2} AND conflict ≤ {max_conflict:.2}\n");
    println!("    β      belief    conflict   validated?");
    for beta in [0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0] {
        let q = resolve(beta);
        println!(
            "  {beta:>4.2}    {:>+6.2}     {:>5.2}      {}",
            q.belief,
            q.conflict,
            if q.is_validated(min_belief, max_conflict) {
                "yes"
            } else {
                "NO"
            }
        );
    }
    println!(
        "\nReading: the credible source A holds the claim **positive** while the dissent is weak — a\n\
         low-authority incident report does not overturn the official docs, and the claim stays\n\
         *validated*. As B gains authority the two surfaces approach parity: `belief` slides toward 0\n\
         (and past it once B out-weighs A — the more credible origin now wins the direction), while\n\
         `conflict` climbs monotonically because the disagreement is increasingly *real*. The\n\
         validation gate flips to NO as soon as B is credible enough that the claim is no longer a\n\
         confident fact — exactly when an agent *should* stop acting on it and seek resolution.\n\
         A flat or majority store cannot express any of this: it has no notion of *who* said it, so\n\
         it cannot let a trustworthy source outweigh a dubious one, nor report that a leaning claim is\n\
         still contested. Authority weighting + the signed belief / geometric tension make the\n\
         conflict of origins a *computation*, not a coin flip."
    );
}
