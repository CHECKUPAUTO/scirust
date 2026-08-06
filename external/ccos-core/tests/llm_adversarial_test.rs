use ccos_core::guard::{GuardConfig, GuardLayer};
use serde_json::Value;

fn make_guard() -> GuardLayer {
    GuardLayer::new(GuardConfig::default())
}

// ── Test 1: Invalid JSON (not JSON at all) ──────────────────────
#[test]
fn adversarial_plain_text_rejected() {
    let guard = make_guard();
    let adversarial_inputs = vec![
        "Hello, I am a helpful assistant!",
        "```json\n{\"key\": \"value\"}\n```",
        "The answer is 42.",
        "I think the dependency graph looks like: a -> b -> c",
        "Here is some markdown:\n\n- item 1\n- item 2",
    ];

    for input in &adversarial_inputs {
        let result = guard.validate_and_sanitize(input);
        assert!(
            !result.passed,
            "plain text input must be rejected: '{}'",
            &input[..input.len().min(50)]
        );
        assert!(
            result.reliability_score <= guard.reliability_threshold() + 1e-9,
            "score {:.10} must be <= threshold {}",
            result.reliability_score,
            guard.reliability_threshold()
        );
    }
}

// ── Test 2: Truncated JSON ─────────────────────────────────────
#[test]
fn adversarial_truncated_json_rejected() {
    let guard = make_guard();
    let truncated_inputs = vec![
        r#"{"analysis": {"summary": "incompl"#,
        r#"{"nodes": [{"id": "a"}, {"id": "b"#,
        r#"{"key": "value"#,
        r#"["#,
        r#"{"#,
    ];

    for input in &truncated_inputs {
        let result = guard.validate_and_sanitize(input);
        assert!(
            !result.passed,
            "truncated JSON must be rejected: '{}'",
            &input[..input.len().min(60)]
        );
    }
}

// ── Test 3: Missing required fields ──────────────────────────
#[test]
fn adversarial_missing_fields_blocked() {
    let guard = make_guard();

    // These are valid JSON but semantically incomplete
    let incomplete_inputs = vec![
        r#"{}"#,
        r#"{"analysis": {}}"#,
        r#"{"status": "ok"}"#,
        r#"{"unexpected_field": 123}"#,
    ];

    // The guard validates JSON structure, not semantic content
    // So these should PASS (they are valid JSON)
    for input in &incomplete_inputs {
        let result = guard.validate_and_sanitize(input);
        // Valid JSON should pass structural validation
        assert!(
            result.passed,
            "valid JSON structure must pass guard (content validation is a higher layer): '{}'",
            input
        );
    }
}

// ── Test 4: Hallucinated dependency injection ──────────────────
#[test]
fn adversarial_hallucinated_deps_handled() {
    let guard = make_guard();

    // These simulate LLM hallucinating fake dependencies in valid JSON
    let hallucinated = vec![
        r#"{"analysis": {"dependencies": [{"name": "fake_crate_12345", "version": "99.99.99"}]}}"#,
        r#"{"analysis": {"dependencies": ["does_not_exist::module", "fake::lib"]}}"#,
        r#"{"analysis": {"summary": "ok", "deps": ["ghost_crate"]}}"#,
    ];

    for input in &hallucinated {
        let result = guard.validate_and_sanitize(input);
        // Valid JSON passes the guard layer (content validation is separate)
        assert!(result.passed, "valid JSON structure must pass guard");
        assert!(
            serde_json::from_str::<Value>(input).is_ok(),
            "hallucinated deps in valid JSON must parse correctly"
        );
    }
}

// ── Test 5: Empty response ────────────────────────────────────
#[test]
fn adversarial_empty_response_blocked() {
    let guard = make_guard();

    let result = guard.validate_and_sanitize("");
    assert!(!result.passed, "empty response must be blocked");
    assert_eq!(result.reliability_score, 0.0);

    let whitespace_result = guard.validate_and_sanitize("   \n\t  ");
    assert!(
        !whitespace_result.passed,
        "whitespace-only response must be blocked"
    );
}

// ── Test 6: Control character injection ────────────────────────
#[test]
fn adversarial_control_chars_sanitized() {
    let guard = make_guard();

    let malicious = "{\"key\": \"value\x00\x01\x02\x03\"}";
    let result = guard.validate_and_sanitize(malicious);

    // The sanitized output must not contain control characters
    assert!(
        !result.sanitized_output.contains('\x00'),
        "null bytes must be stripped"
    );
    assert!(
        !result.sanitized_output.contains('\x01'),
        "control chars must be stripped"
    );

    // Either the sanitized version passes or it's blocked
    if result.passed {
        // If it passed, the output must be clean
        for c in result.sanitized_output.chars() {
            assert!(
                !c.is_control() || c == '\n' || c == '\t' || c == '\r',
                "no control chars in sanitized output"
            );
        }
    }
}

// ── Test 7: Extremely deeply nested JSON ──────────────────────
#[test]
fn adversarial_deep_nesting_handled() {
    let guard = make_guard();

    // Build deeply nested JSON
    let mut deep = String::from("{");
    for _ in 0..100 {
        deep.push_str("\"a\":{");
    }
    deep.push_str("\"leaf\": true");
    for _ in 0..100 {
        deep.push('}');
    }

    let result = guard.validate_and_sanitize(&deep);
    // Whether or not it passes, the guard must terminate without panicking and
    // emit valid JSON (the nested payload or the fallback).
    assert!(
        serde_json::from_str::<Value>(&result.sanitized_output).is_ok(),
        "guard output must always be valid JSON"
    );
}

// ── Test 8: Unicode homoglyph attacks ─────────────────────────
#[test]
fn adversarial_unicode_homoglyphs() {
    let guard = make_guard();

    // Unicode lookalike characters that resemble JSON delimiters
    let homoglyph = "｛\"key\": \"value\"｝"; // Fullwidth braces
    let result = guard.validate_and_sanitize(homoglyph);

    // These are NOT valid JSON (invalid braces) so should be blocked
    assert!(!result.passed, "unicode homoglyph JSON must be rejected");
}

// ── Test 9: Very large input ──────────────────────────────────
#[test]
fn adversarial_large_input_truncated() {
    let guard_config = GuardConfig {
        max_output_length: 1024,
        ..GuardConfig::default()
    };
    let guard = GuardLayer::new(guard_config);

    let large = "x".repeat(100_000);
    let result = guard.validate_and_sanitize(&large);

    assert!(!result.passed, "non-JSON large input must be blocked");
    assert!(
        result.sanitized_output.len() <= 1024,
        "fallback output must respect max length"
    );
}

// ── Test 10: Fallback is always valid JSON ────────────────────
#[test]
fn adversarial_fallback_always_valid() {
    let fallback = GuardLayer::fallback_response();
    let parsed: Result<Value, _> = serde_json::from_str(&fallback);
    assert!(
        parsed.is_ok(),
        "fallback response must always be valid JSON"
    );

    let obj = parsed.unwrap();
    assert_eq!(obj["status"], "fallback");
    assert!(obj["message"].as_str().is_some());
    assert!(obj["analysis"]["summary"].as_str().is_some());
}

// ── Test 11: JSON with BOM / weird prefixes ──────────────────
#[test]
fn adversarial_bom_prefix_handled() {
    let guard = make_guard();

    let with_bom = "\u{FEFF}{}\"key\": \"value\"}".to_string();
    let result = guard.validate_and_sanitize(&with_bom);
    // BOM + invalid JSON should still be handled (rejected or sanitized)
    assert!(!result.passed, "BOM-prefixed garbage must be rejected");
}

// ── Test 12: Batch adversarial — rapid successive attacks ─────
#[test]
fn adversarial_rapid_succession() {
    let guard = make_guard();

    let attacks = vec![
        ("", false),
        (r#"{"ok": true}"#, true),
        ("not json", false),
        (r#"[1,2,3]"#, true),
        ("{{{{{{", false),
        (r#"{"a": {"b": {"c": [1,2,3]}}}"#, true),
        ("", false),
        (r#"null"#, true),    // null is valid JSON
        (r#"true"#, true),    // true is valid JSON
        (r#"false"#, true),   // false is valid JSON
        ("undefined", false), // not valid JSON
        (r#"{"incomplete"#, false),
    ];

    for (input, expected_pass) in &attacks {
        let result = guard.validate_and_sanitize(input);
        assert_eq!(
            result.passed,
            *expected_pass,
            "adversarial input '{}': expected pass={}, got pass={}",
            &input[..input.len().min(40)],
            expected_pass,
            result.passed
        );
    }

    // After all attacks, fallback must still be valid
    let fallback = GuardLayer::fallback_response();
    assert!(
        serde_json::from_str::<Value>(&fallback).is_ok(),
        "fallback must remain valid after all attacks"
    );
}
