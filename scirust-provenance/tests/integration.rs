//! End-to-end: transpile a real NumPy snippet with `scirust-transpiler`, sign the
//! emitted Rust with a demo signer, and prove the provenance flow on a genuine
//! artifact — verification passes, survives reformatting, and catches tampering,
//! while the signed artifact stays digest-identical to the unsigned one.

use scirust_license::hashsig::MerkleSigner;
use scirust_provenance::{
    BANNER_PREFIX, Verdict, canonicalize, emit_root, sign_artifact, strip_banner, verify_artifact,
};

fn demo_signer() -> MerkleSigner {
    MerkleSigner::from_seed(&scirust_license::demo_seed(), scirust_license::DEMO_HEIGHT)
}

fn emit_real_artifact() -> String {
    // A small NumPy-subset function the transpiler lowers to deterministic Rust.
    let py = "def f(x, y):\n    return x * y + x\n";
    scirust_transpiler::transpile(py).expect("transpile should succeed")
}

#[test]
fn real_transpiler_artifact_signs_and_verifies() {
    let art = emit_real_artifact();
    assert!(
        art.contains("pub fn f("),
        "unexpected emitter output:\n{art}"
    );

    let signed = sign_artifact(&art, &demo_signer(), 0);
    assert!(signed.contains(BANNER_PREFIX));

    match verify_artifact(&signed, &emit_root())
    {
        Verdict::Verified { leaf } => assert_eq!(leaf, 0),
        other => panic!("expected Verified, got {other:?}"),
    }
}

#[test]
fn signing_is_digest_neutral_on_a_real_artifact() {
    let art = emit_real_artifact();
    let signed = sign_artifact(&art, &demo_signer(), 1);

    // The banner is a comment: the canonical (hashable) form is unchanged, and the
    // only textual difference is the appended provenance line.
    assert_eq!(canonicalize(&signed), canonicalize(&art));
    assert_eq!(strip_banner(&signed).trim(), art.trim());

    // Everything the signer added is a single comment line — Rust-neutral by
    // construction (the lexer discards it before codegen).
    let added: Vec<&str> = signed
        .lines()
        .filter(|l| !art.lines().any(|a| a == *l))
        .collect();
    assert!(
        added.iter().all(|l| l.trim_start().starts_with("//")),
        "signing added non-comment text: {added:?}"
    );
}

#[test]
fn reformatting_the_real_artifact_still_verifies() {
    let art = emit_real_artifact();
    let signed = sign_artifact(&art, &demo_signer(), 2);

    // Whitespace churn + an extra comment must not break verification.
    let reformatted = format!("// vendored copy\n{}", signed.replace('\n', "\n  "));
    assert_eq!(
        verify_artifact(&reformatted, &emit_root()),
        Verdict::Verified { leaf: 2 }
    );
}

#[test]
fn tampering_with_the_real_artifact_is_caught() {
    let art = emit_real_artifact();
    let signed = sign_artifact(&art, &demo_signer(), 3);

    // Flip a `*` to `+` somewhere in the emitted body — a genuine token edit.
    assert!(
        signed.contains('*'),
        "artifact should contain a '*' to flip"
    );
    let tampered = signed.replacen('*', "+", 1);
    assert_eq!(verify_artifact(&tampered, &emit_root()), Verdict::Forged);
}
