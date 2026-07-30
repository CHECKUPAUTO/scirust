use ccos_core::adversarial::{AdversarialEngine, AdversarialMode};
use ccos_core::guard::{GuardConfig, GuardLayer};

fn make_guard() -> GuardLayer {
    GuardLayer::new(GuardConfig::default())
}

/// The guard's core safety guarantee: whatever it emits is always parseable
/// JSON (either validated input or the deterministic fallback).
fn is_valid_json(s: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(s).is_ok()
}

#[test]
fn adversarial_json_corruption_does_not_crash() {
    let guard = make_guard();
    let mut engine = AdversarialEngine::with_corruption_rate(AdversarialMode::JsonCorruption, 1.0);
    for _ in 0..100 {
        let corrupted = engine.corrupt("{\"valid\": true, \"key\": \"value\"}");
        // However the input is mangled, the guard must yield a valid-JSON,
        // non-empty result without panicking.
        let result = guard.validate_and_sanitize(&corrupted);
        assert!(
            is_valid_json(&result.sanitized_output),
            "guard output must always be valid JSON, got: {}",
            &result.sanitized_output[..result.sanitized_output.len().min(80)]
        );
    }
}

#[test]
fn prompt_injection_is_neutralized_by_guard() {
    let mut engine = AdversarialEngine::with_corruption_rate(AdversarialMode::PromptInjection, 1.0);
    let guard = make_guard();

    for _ in 0..20 {
        let corrupted = engine.corrupt("RUN TASK: parse dependencies");
        let guard_result = guard.validate_and_sanitize(&corrupted);

        // Injected prompts are plain text, never valid JSON, so the guard must
        // block them and fall back to a safe, valid-JSON response.
        if !corrupted.is_empty() {
            assert!(
                !guard_result.passed,
                "prompt injection must be blocked, but guard passed: {}",
                &corrupted[..corrupted.len().min(80)]
            );
            assert!(
                is_valid_json(&guard_result.sanitized_output),
                "blocked output must still be valid-JSON fallback"
            );
        }
    }
}

#[test]
fn hallucination_output_is_sanitized() {
    let mut engine = AdversarialEngine::with_corruption_rate(AdversarialMode::Hallucination, 1.0);
    let guard = make_guard();

    for _ in 0..30 {
        let corrupted = engine.corrupt("{\"status\": \"ok\"}");
        let guard_result = guard.validate_and_sanitize(&corrupted);

        // Hallucinated content appended to valid JSON makes the whole payload
        // invalid, so the guard must reject it; either way its emitted output
        // is always valid JSON.
        if corrupted != "{\"status\": \"ok\"}" {
            assert!(
                !guard_result.passed,
                "hallucinated (valid-prefix + trailing text) output must be blocked: {}",
                &corrupted[..corrupted.len().min(100)]
            );
        }
        assert!(
            is_valid_json(&guard_result.sanitized_output),
            "guard output must always be valid JSON"
        );
    }
}

#[test]
fn system_survives_all_adversarial_modes() {
    let guard = make_guard();
    let modes = vec![
        AdversarialMode::JsonCorruption,
        AdversarialMode::Hallucination,
        AdversarialMode::PromptInjection,
        AdversarialMode::TimeoutSimulation,
    ];

    for mode in modes {
        let mut engine = AdversarialEngine::with_corruption_rate(mode.clone(), 0.8);
        for _ in 0..50 {
            let corrupted = engine.corrupt("{\"action\": \"test\"}");
            let _result = guard.validate_and_sanitize(&corrupted);
            // Must not panic
        }
    }
}

#[test]
fn adversarial_engine_counter_accurate() {
    let mut engine = AdversarialEngine::new(AdversarialMode::None);
    for i in 1..=100 {
        engine.corrupt("test");
        assert_eq!(engine.counter, i as u64);
    }
}

#[test]
fn mode_switching_preserves_state() {
    let mut engine = AdversarialEngine::new(AdversarialMode::None);
    engine.corrupt("a");
    engine.set_mode(AdversarialMode::Hallucination);
    engine.corrupt("b");
    engine.set_mode(AdversarialMode::JsonCorruption);
    engine.corrupt("c");
    assert_eq!(engine.counter, 3);
}

#[test]
fn json_corruption_produces_varied_output() {
    let mut engine = AdversarialEngine::with_corruption_rate(AdversarialMode::JsonCorruption, 1.0);
    let mut outputs = std::collections::HashSet::new();
    for _ in 0..50 {
        let corrupted = engine.corrupt("{\"test\": true}");
        outputs.insert(corrupted);
    }
    // With 5 corruption types at 100% rate, we should see variety
    assert!(
        outputs.len() >= 2,
        "JsonCorruption must produce varied corruptions, got {}",
        outputs.len()
    );
}

#[test]
fn adversarial_outputs_never_panic_guard() {
    let guard = make_guard();
    let mut engine = AdversarialEngine::new(AdversarialMode::Hallucination);

    // 500 rapid-fire adversarial attacks — guard must never panic
    for i in 0..500 {
        engine.set_mode(match i % 4 {
            0 => AdversarialMode::JsonCorruption,
            1 => AdversarialMode::Hallucination,
            2 => AdversarialMode::PromptInjection,
            _ => AdversarialMode::TimeoutSimulation,
        });
        let corrupted = engine.corrupt("base");
        let _ = guard.validate_and_sanitize(&corrupted);
        let _ = guard.validate_and_sanitize(""); // edge case
    }
}
