//! Pure-Rust hash-based digital signatures: Lamport one-time signatures (OTS)
//! authenticated by a Merkle tree, built on SHA-256 only.
//!
//! Why not an elliptic-curve scheme? The rest of SciRust already commits to a
//! pure-`sha2`, "re-derive every field from the bytes" posture (see
//! `scirust-runtime`'s attestation hash chain). A hash chain alone is *not* a
//! signature — it has no secret, so anyone can recompute it (the in-tree
//! `scirust-ids::hashchain` says exactly this). Lamport/Merkle keeps the pure-
//! hash, zero-FFI, deterministic ethos *and* gives genuine asymmetry: the
//! signer holds a secret seed; the verifier holds only a 32-byte Merkle root
//! and cannot forge a signature without inverting SHA-256.
//!
//! ## Construction
//! * A 256-bit message digest is signed bit-by-bit. For bit `i` the signer has
//!   two secrets `sk[i][0]`, `sk[i][1]`; the public images are
//!   `pk[i][b] = H(sk[i][b])`. The signature reveals `sk[i][digestᵢ]` and
//!   carries the *complementary* image `pk[i][1−digestᵢ]`, so the verifier can
//!   rebuild all 512 images, hash them into the leaf commitment, and check the
//!   Merkle path to the root. Forging a different digest needs a secret whose
//!   only public trace is its SHA-256 image — preimage-resistant.
//! * `2^height` Lamport key-pairs are the leaves of a Merkle tree; the root is
//!   the public key. Each leaf signs **one** digest: reusing a leaf for two
//!   different digests leaks secrets and is unsafe (the inherent OTS rule).
//!
//! Everything here is deterministic: same seed + same leaf + same digest →
//! byte-identical signature, on any platform.

use sha2::{Digest, Sha256};

/// Length of every hash in this module.
pub const HASH_LEN: usize = 32;
/// A 32-byte SHA-256 digest.
pub type Hash = [u8; HASH_LEN];

/// Number of message bits signed by one Lamport leaf (a SHA-256 digest).
const MSG_BITS: usize = 256;
/// Bytes of revealed material per signed digest: `MSG_BITS * (secret + image)`.
const REVEAL_BYTES: usize = MSG_BITS * (HASH_LEN + HASH_LEN);

const MIN_TREE_HEIGHT: u32 = 1;
const MAX_TREE_HEIGHT: u32 = 20;

/// Return the supported Merkle-tree height for a caller's request.
///
/// Keeping this normalization in one pure function lets callers describe the
/// effective key capacity without first constructing the (potentially very
/// large) tree. [`MerkleSigner::from_seed`] uses this same function, so the
/// reported metadata cannot drift from the signer's actual height.
pub(crate) const fn effective_height(requested: u32) -> u32 {
    if requested < MIN_TREE_HEIGHT
    {
        MIN_TREE_HEIGHT
    }
    else if requested > MAX_TREE_HEIGHT
    {
        MAX_TREE_HEIGHT
    }
    else
    {
        requested
    }
}

/// Number of leaves created for a requested Merkle-tree height.
pub(crate) const fn capacity_for_height(requested: u32) -> usize {
    1usize << effective_height(requested)
}

fn hash(parts: &[&[u8]]) -> Hash {
    let mut h = Sha256::new();
    for p in parts
    {
        h.update(p);
    }
    h.finalize().into()
}

/// `sk[leaf][i][b]` — a secret derived deterministically from the master seed.
/// Domain-separated so it can never collide with a public image or tree node.
fn derive_secret(master: &Hash, leaf: u32, i: usize, b: u8) -> Hash {
    hash(&[
        b"SRL.sk",
        master,
        &leaf.to_le_bytes(),
        &(i as u16).to_le_bytes(),
        &[b],
    ])
}

/// Public image of a secret: `pk = H(sk)`.
fn image_of(sk: &Hash) -> Hash {
    hash(&[b"SRL.pk", sk])
}

/// Internal Merkle node hash (domain-separated from leaves and secrets).
fn node_hash(left: &Hash, right: &Hash) -> Hash {
    hash(&[b"SRL.node", left, right])
}

/// The `i`-th bit of a 256-bit digest, little-endian within each byte.
fn digest_bit(digest: &Hash, i: usize) -> u8 {
    (digest[i >> 3] >> (i & 7)) & 1
}

/// Commitment to a single Lamport leaf: `H("SRL.leaf" ‖ pk[0][0] ‖ pk[0][1] ‖ …)`.
fn leaf_commitment(master: &Hash, leaf: u32) -> Hash {
    let mut h = Sha256::new();
    h.update(b"SRL.leaf");
    for i in 0..MSG_BITS
    {
        for b in 0..2u8
        {
            let sk = derive_secret(master, leaf, i, b);
            h.update(image_of(&sk));
        }
    }
    h.finalize().into()
}

/// A Lamport+Merkle signature over one 256-bit digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleSig {
    /// Which leaf (one-time key) produced this signature.
    pub leaf: u32,
    /// For each message bit: `(revealed secret sk[i][bitᵢ], image pk[i][1−bitᵢ])`.
    pub reveals: Vec<(Hash, Hash)>,
    /// Sibling hashes from the leaf up to (but excluding) the root.
    pub path: Vec<Hash>,
}

impl MerkleSig {
    /// Pack into a self-describing byte string:
    /// `leaf(u32 LE) ‖ path_len(u8) ‖ path ‖ reveals`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = Vec::with_capacity(5 + self.path.len() * HASH_LEN + REVEAL_BYTES);
        o.extend_from_slice(&self.leaf.to_le_bytes());
        o.push(self.path.len() as u8);
        for p in &self.path
        {
            o.extend_from_slice(p);
        }
        for (s, c) in &self.reveals
        {
            o.extend_from_slice(s);
            o.extend_from_slice(c);
        }
        o
    }

    /// Parse bytes produced by [`MerkleSig::to_bytes`]. Returns `None` on any
    /// malformed input — never panics, since signatures arrive untrusted.
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < 5
        {
            return None;
        }
        let leaf = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let path_len = b[4] as usize;
        let want = 5 + path_len * HASH_LEN + REVEAL_BYTES;
        if b.len() != want
        {
            return None;
        }
        let mut off = 5;
        let mut path = Vec::with_capacity(path_len);
        for _ in 0..path_len
        {
            let mut node = [0u8; HASH_LEN];
            node.copy_from_slice(&b[off..off + HASH_LEN]);
            path.push(node);
            off += HASH_LEN;
        }
        let mut reveals = Vec::with_capacity(MSG_BITS);
        for _ in 0..MSG_BITS
        {
            let mut s = [0u8; HASH_LEN];
            let mut c = [0u8; HASH_LEN];
            s.copy_from_slice(&b[off..off + HASH_LEN]);
            off += HASH_LEN;
            c.copy_from_slice(&b[off..off + HASH_LEN]);
            off += HASH_LEN;
            reveals.push((s, c));
        }
        Some(Self {
            leaf,
            reveals,
            path,
        })
    }

    /// Lower-case hex of [`MerkleSig::to_bytes`] — the on-disk license form.
    pub fn to_hex(&self) -> String {
        let bytes = self.to_bytes();
        let mut s = String::with_capacity(bytes.len() * 2);
        for byte in bytes
        {
            s.push(nibble(byte >> 4));
            s.push(nibble(byte & 0xf));
        }
        s
    }

    /// Inverse of [`MerkleSig::to_hex`]; `None` on non-hex or wrong length.
    pub fn from_hex(s: &str) -> Option<Self> {
        let bytes = hex_decode(s)?;
        Self::from_bytes(&bytes)
    }
}

fn nibble(n: u8) -> char {
    match n
    {
        0..=9 => (b'0' + n) as char,
        _ => (b'a' + (n - 10)) as char,
    }
}

fn unhex(c: u8) -> Option<u8> {
    match c
    {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// Decode a lower/upper-case hex string to bytes; `None` on odd length or any
/// non-hex character.
pub fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let bytes = s.as_bytes();
    if bytes.len() % 2 != 0
    {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0;
    while i < bytes.len()
    {
        let hi = unhex(bytes[i])?;
        let lo = unhex(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Some(out)
}

/// Encode bytes as lower-case hex.
pub fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes
    {
        s.push(nibble(b >> 4));
        s.push(nibble(b & 0xf));
    }
    s
}

/// A Merkle-authenticated Lamport signer: holds the secret master seed and the
/// precomputed tree. The public key is [`MerkleSigner::root`].
pub struct MerkleSigner {
    master: Hash,
    height: u32,
    /// `levels[0]` are the leaf commitments; `levels[height] == [root]`.
    levels: Vec<Vec<Hash>>,
}

impl MerkleSigner {
    /// Build the tree of `2^height` one-time key-pairs from a 32-byte secret
    /// seed. `height` is clamped to `1..=20` (a 20-deep tree already holds a
    /// million licenses and costs a few seconds to build).
    pub fn from_seed(seed: &Hash, height: u32) -> Self {
        let height = effective_height(height);
        let n = capacity_for_height(height);
        let mut leaves = Vec::with_capacity(n);
        for k in 0..n
        {
            leaves.push(leaf_commitment(seed, k as u32));
        }
        let mut levels = vec![leaves];
        while levels.last().map(|l| l.len()).unwrap_or(0) > 1
        {
            let cur = levels.last().unwrap();
            let mut next = Vec::with_capacity(cur.len() / 2);
            let mut i = 0;
            while i < cur.len()
            {
                next.push(node_hash(&cur[i], &cur[i + 1]));
                i += 2;
            }
            levels.push(next);
        }
        Self {
            master: *seed,
            height,
            levels,
        }
    }

    /// The public key: the 32-byte Merkle root to embed in a verifier.
    pub fn root(&self) -> Hash {
        self.levels[self.height as usize][0]
    }

    /// Number of one-time leaves (`2^height`).
    pub fn capacity(&self) -> usize {
        capacity_for_height(self.height)
    }

    /// Sign a 256-bit `digest` with one-time leaf `leaf`.
    ///
    /// # Panics
    /// If `leaf >= capacity()`. Each leaf must sign at most one distinct digest;
    /// callers are responsible for never reusing one (see module docs).
    pub fn sign(&self, leaf: u32, digest: &Hash) -> MerkleSig {
        assert!((leaf as usize) < self.capacity(), "leaf index out of range");
        let mut reveals = Vec::with_capacity(MSG_BITS);
        for i in 0..MSG_BITS
        {
            let bit = digest_bit(digest, i);
            let secret = derive_secret(&self.master, leaf, i, bit);
            let complement = image_of(&derive_secret(&self.master, leaf, i, 1 - bit));
            reveals.push((secret, complement));
        }
        let mut path = Vec::with_capacity(self.height as usize);
        let mut idx = leaf as usize;
        for level in 0..self.height as usize
        {
            path.push(self.levels[level][idx ^ 1]);
            idx >>= 1;
        }
        MerkleSig {
            leaf,
            reveals,
            path,
        }
    }
}

/// Verify `sig` over `digest` against the public Merkle `root`. Total function:
/// any malformed signature simply returns `false`.
pub fn verify(root: &Hash, digest: &Hash, sig: &MerkleSig) -> bool {
    if sig.reveals.len() != MSG_BITS
    {
        return false;
    }
    // Rebuild the leaf commitment from the revealed secrets and complements.
    let mut h = Sha256::new();
    h.update(b"SRL.leaf");
    for (i, (secret, complement)) in sig.reveals.iter().enumerate()
    {
        let revealed_image = image_of(secret);
        let (pk0, pk1) = if digest_bit(digest, i) == 0
        {
            (revealed_image, *complement)
        }
        else
        {
            (*complement, revealed_image)
        };
        h.update(pk0);
        h.update(pk1);
    }
    let mut node: Hash = h.finalize().into();
    // Fold the authentication path up to the root, ordered by the leaf index.
    let mut idx = sig.leaf as usize;
    for sibling in &sig.path
    {
        node = if idx & 1 == 0
        {
            node_hash(&node, sibling)
        }
        else
        {
            node_hash(sibling, &node)
        };
        idx >>= 1;
    }
    &node == root
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(byte: u8) -> Hash {
        [byte; HASH_LEN]
    }

    fn digest(tag: &[u8]) -> Hash {
        hash(&[b"test-digest", tag])
    }

    #[test]
    fn height_normalization_and_capacity_cover_both_clamp_boundaries() {
        assert_eq!(effective_height(0), 1);
        assert_eq!(effective_height(7), 7);
        assert_eq!(effective_height(30), 20);
        assert_eq!(capacity_for_height(0), 2);
        assert_eq!(capacity_for_height(7), 128);
        assert_eq!(capacity_for_height(30), 1_048_576);
    }

    #[test]
    fn sign_then_verify_accepts_a_genuine_signature() {
        let signer = MerkleSigner::from_seed(&seed(7), 4);
        let d = digest(b"license-A");
        let sig = signer.sign(3, &d);
        assert!(verify(&signer.root(), &d, &sig));
    }

    #[test]
    fn every_leaf_authenticates_to_the_same_root() {
        let signer = MerkleSigner::from_seed(&seed(9), 4);
        assert_eq!(signer.capacity(), 16);
        let root = signer.root();
        for leaf in 0..signer.capacity() as u32
        {
            let d = digest(&leaf.to_le_bytes());
            let sig = signer.sign(leaf, &d);
            assert!(verify(&root, &d, &sig), "leaf {leaf} failed to verify");
        }
    }

    #[test]
    fn verifying_a_different_digest_is_rejected() {
        let signer = MerkleSigner::from_seed(&seed(1), 4);
        let sig = signer.sign(0, &digest(b"original"));
        // A signature is bound to its digest: present it against another and it
        // must fail (this is the whole point — the holder cannot relabel it).
        assert!(!verify(&signer.root(), &digest(b"tampered"), &sig));
    }

    #[test]
    fn a_different_root_rejects_the_signature() {
        let signer = MerkleSigner::from_seed(&seed(2), 4);
        let other = MerkleSigner::from_seed(&seed(3), 4);
        let d = digest(b"x");
        let sig = signer.sign(1, &d);
        assert!(verify(&signer.root(), &d, &sig));
        assert!(!verify(&other.root(), &d, &sig));
    }

    #[test]
    fn forging_by_flipping_a_revealed_secret_is_rejected() {
        // Attacker owns a signature and tries to fabricate one for a digest that
        // differs in bit 0. They have pk[0][1−b] but not sk for the new bit, so
        // the best they can do is mutate the bytes — which breaks the leaf hash.
        let signer = MerkleSigner::from_seed(&seed(5), 4);
        let d = digest(b"genuine");
        let mut sig = signer.sign(2, &d);
        // Forge: target digest differs from d. Tamper the revealed secret too.
        let mut forged_digest = d;
        forged_digest[0] ^= 1;
        sig.reveals[0].0[0] ^= 0xAA;
        assert!(!verify(&signer.root(), &forged_digest, &sig));
        assert!(!verify(&signer.root(), &d, &sig));
    }

    #[test]
    fn corrupting_the_auth_path_is_rejected() {
        let signer = MerkleSigner::from_seed(&seed(8), 5);
        let d = digest(b"path");
        let mut sig = signer.sign(10, &d);
        sig.path[0][0] ^= 0x01;
        assert!(!verify(&signer.root(), &d, &sig));
    }

    #[test]
    fn claiming_a_different_leaf_is_rejected() {
        let signer = MerkleSigner::from_seed(&seed(4), 4);
        let d = digest(b"leaf-claim");
        let mut sig = signer.sign(5, &d);
        // Same reveals/path but a lied-about leaf index folds the path wrong.
        sig.leaf = 6;
        assert!(!verify(&signer.root(), &d, &sig));
    }

    #[test]
    fn signing_is_deterministic() {
        let a = MerkleSigner::from_seed(&seed(6), 4);
        let b = MerkleSigner::from_seed(&seed(6), 4);
        let d = digest(b"determinism");
        assert_eq!(a.root(), b.root());
        assert_eq!(a.sign(7, &d), b.sign(7, &d));
    }

    #[test]
    fn signature_survives_a_hex_round_trip() {
        let signer = MerkleSigner::from_seed(&seed(11), 4);
        let d = digest(b"hex");
        let sig = signer.sign(1, &d);
        let hex = sig.to_hex();
        let back = MerkleSig::from_hex(&hex).expect("valid hex");
        assert_eq!(sig, back);
        assert!(verify(&signer.root(), &d, &back));
    }

    #[test]
    fn malformed_signature_bytes_decode_to_none_not_panic() {
        assert!(MerkleSig::from_bytes(&[]).is_none());
        assert!(MerkleSig::from_bytes(&[0, 0, 0, 0]).is_none());
        assert!(MerkleSig::from_hex("zzzz").is_none());
        assert!(MerkleSig::from_hex("abc").is_none()); // odd length
        // Right shape header, wrong total length.
        let mut b = vec![0u8; 5];
        b[4] = 4;
        assert!(MerkleSig::from_bytes(&b).is_none());
    }

    #[test]
    fn hex_helpers_round_trip() {
        let bytes = [0u8, 1, 15, 16, 255, 128, 7];
        assert_eq!(hex_encode(&bytes), "00010f10ff8007");
        assert_eq!(hex_decode("00010f10ff8007").unwrap(), bytes);
        assert!(hex_decode("0g").is_none());
    }
}
