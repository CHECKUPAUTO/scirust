//! Empreintes **SHA-256** de tenseurs et de paramètres de modèle — l'outil de
//! *vérification* de la reproductibilité : deux exécutions (machines, nombres
//! de threads, dates différents) ont produit le même résultat **si et
//! seulement si** leurs empreintes coïncident.
//!
//! Complète les fingerprints FNV-1a 64 bits de [`crate`] (rapides, pour les
//! tests d'invariance) par un condensat cryptographique publiable dans une
//! pièce d'audit. Équivalent fonctionnel de l'utilitaire de hachage de
//! vérification de RepDL (Microsoft, arXiv:2510.09180), réimplémenté
//! indépendamment.
//!
//! L'encodage haché est **indépendant de la plate-forme** : la forme en
//! `u64` little-endian, puis les bits IEEE-754 de chaque valeur en
//! little-endian. Deux tenseurs bit-identiques donnent la même empreinte
//! partout ; un seul bit divergent la change.

use scirust_core::autodiff::reverse::Tensor;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn feed_f32(h: &mut Sha256, data: &[f32]) {
    for &x in data
    {
        h.update(x.to_bits().to_le_bytes());
    }
}

/// Empreinte SHA-256 (hex, 64 caractères) d'un slice `f32` : longueur en
/// `u64` LE puis bits IEEE-754 de chaque valeur en LE.
pub fn sha256_hex_f32(data: &[f32]) -> String {
    let mut h = Sha256::new();
    h.update((data.len() as u64).to_le_bytes());
    feed_f32(&mut h, data);
    hex(&h.finalize())
}

/// Empreinte SHA-256 (hex) d'un tenseur 2-D : forme (`rows`, `cols` en
/// `u64` LE) puis données. Deux tenseurs de même contenu mais de forme
/// différente ont des empreintes différentes.
pub fn sha256_hex_tensor(t: &Tensor) -> String {
    let (rows, cols) = t.shape();
    let mut h = Sha256::new();
    h.update((rows as u64).to_le_bytes());
    h.update((cols as u64).to_le_bytes());
    feed_f32(&mut h, &t.data);
    hex(&h.finalize())
}

/// Empreinte SHA-256 (hex) d'un `state_dict` complet : les clés sont
/// **triées** puis chaque entrée est hachée comme
/// `len(clé) ‖ clé ‖ rows ‖ cols ‖ données` — l'empreinte ne dépend donc pas
/// de l'ordre d'insertion de la map, uniquement de son contenu.
///
/// C'est l'empreinte à publier dans une pièce d'audit pour attester que deux
/// entraînements ont produit **exactement** les mêmes poids.
pub fn sha256_hex_state_dict(sd: &HashMap<String, Tensor>) -> String {
    let mut keys: Vec<&String> = sd.keys().collect();
    keys.sort();
    let mut h = Sha256::new();
    h.update((keys.len() as u64).to_le_bytes());
    for k in keys
    {
        let t = &sd[k];
        let (rows, cols) = t.shape();
        h.update((k.len() as u64).to_le_bytes());
        h.update(k.as_bytes());
        h.update((rows as u64).to_le_bytes());
        h.update((cols as u64).to_le_bytes());
        feed_f32(&mut h, &t.data);
    }
    hex(&h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_digest_is_64_chars_and_deterministic() {
        let a = sha256_hex_f32(&[1.0, 2.0, 3.0]);
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(a, sha256_hex_f32(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn single_bit_flip_changes_hash() {
        let base = [1.0f32, 2.0, 3.0];
        let mut flipped = base;
        flipped[1] = f32::from_bits(flipped[1].to_bits() ^ 1);
        assert_ne!(sha256_hex_f32(&base), sha256_hex_f32(&flipped));
    }

    #[test]
    fn shape_is_part_of_tensor_hash() {
        let flat = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 1, 4);
        let square = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        assert_ne!(sha256_hex_tensor(&flat), sha256_hex_tensor(&square));
    }

    #[test]
    fn state_dict_hash_ignores_insertion_order() {
        let w = Tensor::from_vec(vec![0.5, -0.5], 1, 2);
        let b = Tensor::from_vec(vec![0.1], 1, 1);

        let mut ab = HashMap::new();
        ab.insert("layer.weight".to_string(), w.clone());
        ab.insert("layer.bias".to_string(), b.clone());

        let mut ba = HashMap::new();
        ba.insert("layer.bias".to_string(), b);
        ba.insert("layer.weight".to_string(), w);

        assert_eq!(sha256_hex_state_dict(&ab), sha256_hex_state_dict(&ba));
    }

    #[test]
    fn state_dict_hash_sees_renames_and_value_changes() {
        let w = Tensor::from_vec(vec![0.5, -0.5], 1, 2);

        let mut a = HashMap::new();
        a.insert("w".to_string(), w.clone());
        let mut renamed = HashMap::new();
        renamed.insert("w2".to_string(), w.clone());
        assert_ne!(sha256_hex_state_dict(&a), sha256_hex_state_dict(&renamed));

        let mut changed = HashMap::new();
        changed.insert("w".to_string(), Tensor::from_vec(vec![0.5, -0.25], 1, 2));
        assert_ne!(sha256_hex_state_dict(&a), sha256_hex_state_dict(&changed));
    }
}
