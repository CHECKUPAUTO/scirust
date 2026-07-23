//! The tamper-evident attestation chain: append/verify, determinism, and
//! localized tamper/reorder detection.

use sos_ccos::{CcosChain, CcosError};

#[test]
fn appending_links_and_verifies() {
    let mut chain = CcosChain::new();
    assert!(chain.is_empty());
    let r0 = chain.append(b"input-0", b"output-0");
    let r1 = chain.append(b"input-1", b"output-1");
    assert_eq!(r0.seq, 0);
    assert_eq!(r1.seq, 1);
    assert_eq!(chain.len(), 2);
    assert_eq!(chain.head(), Some(r1.chain_hash));
    // Each entry links to the previous entry's chain hash.
    assert_eq!(chain.entries()[1].prev, chain.entries()[0].chain_hash);
    chain.verify().unwrap();
}

#[test]
fn the_chain_is_deterministic() {
    let mut a = CcosChain::new();
    a.append(b"x", b"y");
    a.append(b"p", b"q");
    let mut b = CcosChain::new();
    b.append(b"x", b"y");
    b.append(b"p", b"q");
    assert_eq!(a, b);
    assert_eq!(a.head(), b.head());
}

#[test]
fn only_hashes_are_retained_not_payloads() {
    let mut chain = CcosChain::new();
    let secret_in = b"a sensitive prompt";
    let secret_out = b"a generated proposal";
    chain.append(secret_in, secret_out);
    // The stored entry commits to the payloads by hash, but does not contain them.
    let json = serde_json::to_string(&chain).unwrap();
    assert!(!json.contains("sensitive prompt"));
    assert!(!json.contains("generated proposal"));
}

#[test]
fn tampering_with_an_entry_is_detected_and_localized() {
    let mut chain = CcosChain::new();
    chain.append(b"in-0", b"out-0");
    chain.append(b"in-1", b"out-1");
    chain.append(b"in-2", b"out-2");
    chain.verify().unwrap();

    // Rewrite entry 1's output hash via the interchange form.
    let mut value = serde_json::to_value(&chain).unwrap();
    value["entries"][1]["output_hash"] = serde_json::Value::String("00".repeat(32));
    let tampered: CcosChain = serde_json::from_value(value).unwrap();

    assert_eq!(
        tampered.verify().unwrap_err(),
        CcosError::ChainBroken { seq: 1 }
    );
}

#[test]
fn reordering_entries_is_detected() {
    let mut chain = CcosChain::new();
    chain.append(b"in-0", b"out-0");
    chain.append(b"in-1", b"out-1");

    // Swap the two entries.
    let mut value = serde_json::to_value(&chain).unwrap();
    let entries = value["entries"].as_array().unwrap().clone();
    value["entries"] = serde_json::Value::Array(vec![entries[1].clone(), entries[0].clone()]);
    let reordered: CcosChain = serde_json::from_value(value).unwrap();

    // The first entry now carries the wrong sequence number / genesis link.
    assert_eq!(
        reordered.verify().unwrap_err(),
        CcosError::ChainBroken { seq: 0 }
    );
}
