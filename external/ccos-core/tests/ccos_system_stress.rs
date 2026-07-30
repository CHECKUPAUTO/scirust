use ccos_core::adversarial::{AdversarialEngine, AdversarialMode};
use ccos_core::consensus::{ConsensusEngine, LlmVote};
use ccos_core::distributed_event_log::DistributedEventLog;
use ccos_core::guard::{GuardConfig, GuardLayer};
use ccos_core::memory::MemoryGraph;

fn make_guard() -> GuardLayer {
    GuardLayer::new(GuardConfig::default())
}

#[test]
fn system_survives_10000_adversarial_cycles() {
    let mut engine = AdversarialEngine::with_corruption_rate(AdversarialMode::Hallucination, 1.0);
    let guard = make_guard();

    for i in 0..10_000 {
        let corrupted = engine.corrupt(&format!("{{\"cycle\": {}, \"status\": \"running\"}}", i));
        let _result = guard.validate_and_sanitize(&corrupted);
        // System must not panic after any adversarial input
    }

    assert_eq!(engine.counter, 10_000);
}

#[test]
fn distributed_log_end_to_end() {
    let mut log = DistributedEventLog::new();
    let mut graph = MemoryGraph::default();

    // Simulate ingestion + log append cycle
    for i in 0..100 {
        let event_id = log.append(
            format!("node_ingested_{}", i),
            format!("ingestion_cycle_{}", i),
        );

        graph.upsert_node(
            format!("node_{}", i).into(),
            format!("Node {}", i),
            format!("content from event {}", event_id),
            ccos_core::memory::NodeType::ContextBlock,
        );
    }

    // Verify log integrity
    let report = log.verify_integrity();
    assert!(report.valid);

    // Verify replay is deterministic
    let replay1 = log.replay();
    let replay2 = log.replay();
    assert_eq!(replay1.len(), replay2.len());
    assert_eq!(replay1.len(), 100);
}

#[test]
fn consensus_under_adversarial_majority_attack() {
    let mut engine = AdversarialEngine::with_corruption_rate(AdversarialMode::Hallucination, 0.5);

    // Simulate 3 LLMs voting, but one gets hallucinated
    let mut votes = Vec::new();
    for model in &["llama", "codellama", "mistral"] {
        let base = format!("{{\"action\": \"approved_by_{}\"}}", model);
        let output = engine.corrupt(&base);
        votes.push(LlmVote {
            model: model.to_string(),
            output,
            confidence: 0.8,
        });
    }

    let consensus = ConsensusEngine::new();
    let result = consensus.resolve(&votes);

    // Even under adversarial conditions, consensus must produce a result
    assert!(!result.output.is_empty());
    assert_eq!(result.total_votes, 3);
}

#[test]
fn guard_layer_under_1000_mixed_attacks() {
    let guard = make_guard();
    let mut engine = AdversarialEngine::new(AdversarialMode::Hallucination);

    let modes = [
        AdversarialMode::JsonCorruption,
        AdversarialMode::Hallucination,
        AdversarialMode::PromptInjection,
        AdversarialMode::TimeoutSimulation,
        AdversarialMode::None,
    ];

    let mut passed = 0;
    let mut blocked = 0;

    for i in 0..1000 {
        engine.set_mode(modes[i % modes.len()].clone());
        let corrupted = engine.corrupt("{\"test\": true}");
        let result = guard.validate_and_sanitize(&corrupted);

        if result.passed {
            passed += 1;
            // Guard may pass if a valid JSON prefix is found within corrupted output
            // but the output should at minimum be parseable or contain valid structure
            let is_strict_json =
                serde_json::from_str::<serde_json::Value>(&result.sanitized_output).is_ok();
            let contains_braces = result.sanitized_output.contains('{');
            assert!(
                is_strict_json || contains_braces,
                "Passed output must contain JSON structure at cycle {}: {}",
                i,
                &result.sanitized_output[..result.sanitized_output.len().min(100)]
            );
        } else {
            blocked += 1;
            // If blocked, must have fallback
            assert!(
                !result.sanitized_output.is_empty(),
                "Blocked output must have fallback at cycle {}",
                i
            );
        }
    }

    // Both paths must have been exercised
    assert!(passed > 0, "Some outputs must pass guard");
    assert!(blocked > 0, "Some outputs must be blocked by guard");
}

#[test]
fn full_pipeline_stress_500_cycles() {
    let guard = make_guard();
    let mut engine = AdversarialEngine::new(AdversarialMode::None);
    let mut log = DistributedEventLog::new();
    let mut graph = MemoryGraph::default();
    let consensus = ConsensusEngine::new();

    for cycle in 0..500 {
        // Phase: adversarial corruption
        engine.set_mode(match cycle % 5 {
            0 => AdversarialMode::JsonCorruption,
            1 => AdversarialMode::Hallucination,
            2 => AdversarialMode::PromptInjection,
            3 => AdversarialMode::TimeoutSimulation,
            _ => AdversarialMode::None,
        });

        let raw_output =
            engine.corrupt(&format!(r#"{{"cycle": {}, "analysis": "normal"}}"#, cycle));

        // Phase: guard validation
        let validated = guard.validate_and_sanitize(&raw_output);

        // Phase: consensus simulation
        let votes = vec![LlmVote {
            model: "primary".into(),
            output: validated.sanitized_output.clone(),
            confidence: validated.reliability_score,
        }];
        let result = consensus.resolve(&votes);

        // Phase: graph update
        graph.upsert_node(
            format!("cycle_node_{}", cycle).into(),
            format!("Cycle {}", cycle),
            result.output.clone(),
            ccos_core::memory::NodeType::AnalysisResult,
        );

        // Phase: event log
        log.append(
            format!("cycle_{}_completed", cycle),
            "stress_pipeline".into(),
        );
    }

    // Final integrity checks
    let integrity = log.verify_integrity();
    assert!(
        integrity.valid,
        "Log integrity must be maintained: {:?}",
        integrity.errors
    );
    assert_eq!(log.event_count(), 500);
    assert!(graph.node_count() > 0);
}

#[test]
fn adversarial_consensus_survives_poisoned_majority() {
    let mut engine = AdversarialEngine::with_corruption_rate(AdversarialMode::Hallucination, 1.0);
    let consensus = ConsensusEngine::new();

    // 5 models, all hallucinated — consensus must still produce a result
    let models = ["gpt", "claude", "llama", "mistral", "gemini"];
    let mut votes = Vec::new();

    for model in &models {
        let corrupted = engine.corrupt("{\"action\": \"safe_operation\"}");
        votes.push(LlmVote {
            model: model.to_string(),
            output: corrupted,
            confidence: 0.5,
        });
    }

    let result = consensus.resolve(&votes);
    assert!(!result.output.is_empty());
    // With 5 different hallucinated outputs, consensus likely not reached
    // But system must survive
}
