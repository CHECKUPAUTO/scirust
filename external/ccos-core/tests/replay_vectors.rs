//! Canonical replay smoke vector shared by native architecture jobs.
use ccos_core::util::sha256_hex;

#[test]
fn canonical_replay_vector_hash_is_stable() {
    let events = "cycle=0\ninput=canonical\nstate=ready\n";
    assert_eq!(
        sha256_hex(events),
        "49213696cd020173ef3cb3c2e669b0e397d0cac814da077e0eee4628c563b4fd"
    );
}
