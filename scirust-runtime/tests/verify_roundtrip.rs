//! End-to-end du certificat d'inférence : émission puis vérification par
//! le binaire `scirust-verify`, y compris le rejet de toute altération.

use std::process::Command;

use scirust_runtime::quant::{QLayer, QLinear, QModel};

fn tiny_artifact() -> Vec<u8> {
    // Petit modèle int8 déterministe : 4 -> 3, paramètres fixés à la main.
    let layer = QLayer::Linear(QLinear {
        in_f: 4,
        out_f: 3,
        s_in: 0.05,
        relu_after: true,
        scales: vec![0.002, 0.003, 0.004],
        w_q: (0..12).map(|i| (i as i8) - 6).collect(),
        bias_i32: vec![10, -20, 30],
    });
    QModel {
        layers: vec![layer],
    }
    .to_bytes()
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_scirust_verify")
}

#[test]
fn emit_then_verify_roundtrip_and_tamper_detection() {
    let dir = std::env::temp_dir().join(format!("scirust_verify_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.qsr1");
    let proof_path = dir.join("model.proof");

    let artifact = tiny_artifact();
    std::fs::write(&model_path, &artifact).unwrap();

    // 1) emit
    let st = Command::new(bin())
        .args([
            "emit",
            model_path.to_str().unwrap(),
            proof_path.to_str().unwrap(),
            "2",
            "7",
            "8",
        ])
        .status()
        .unwrap();
    assert!(st.success(), "emit must succeed");
    let proof_text = std::fs::read_to_string(&proof_path).unwrap();
    assert!(proof_text.contains("format=SCIRUST-PROOF-1"));
    assert!(proof_text.contains("bundle_sha256="));

    // 2) verify — MATCH attendu (exit 0)
    let st = Command::new(bin())
        .args([
            "verify",
            proof_path.to_str().unwrap(),
            model_path.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(st.success(), "pristine artifact must verify");

    // 3) artefact altéré d'un octet — MISMATCH attendu (exit 1)
    let mut tampered = artifact.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0x01;
    let tampered_path = dir.join("model_tampered.qsr1");
    std::fs::write(&tampered_path, &tampered).unwrap();
    let st = Command::new(bin())
        .args([
            "verify",
            proof_path.to_str().unwrap(),
            tampered_path.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert_eq!(
        st.code(),
        Some(1),
        "tampered artifact must fail verification"
    );

    // 4) bundle altéré (empreinte falsifiée) — MISMATCH attendu
    let fp_line = proof_text
        .lines()
        .find(|l| l.starts_with("vec.0.out_fp="))
        .unwrap();
    let forged = proof_text.replacen(fp_line, "vec.0.out_fp=0000000000000000", 1);
    assert_ne!(forged, proof_text, "forgery must actually change the text");
    let forged_path = dir.join("model_forged.proof");
    std::fs::write(&forged_path, forged).unwrap();
    let st = Command::new(bin())
        .args([
            "verify",
            forged_path.to_str().unwrap(),
            model_path.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert_eq!(st.code(), Some(1), "forged bundle must fail verification");

    // 5) déterminisme : ré-émettre produit un bundle bit-identique
    let proof2 = dir.join("model2.proof");
    let st = Command::new(bin())
        .args([
            "emit",
            model_path.to_str().unwrap(),
            proof2.to_str().unwrap(),
            "2",
            "7",
            "8",
        ])
        .status()
        .unwrap();
    assert!(st.success());
    assert_eq!(
        proof_text,
        std::fs::read_to_string(&proof2).unwrap(),
        "same artifact + seeds ⇒ bit-identical certificate"
    );

    std::fs::remove_dir_all(&dir).ok();
}
