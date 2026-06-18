use serde::{Deserialize, Serialize};

/// Result of an OTA update operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtaResult {
    pub success: bool,
    pub model_hash: String,
    pub signature_valid: bool,
    pub bytes_downloaded: u64,
    pub message: String,
}

/// A cryptographic signing key (simplified — Ed25519-like).
///
/// In production this would use actual Ed25519/ECDSA. Here we use
/// a simple HMAC-SHA256-like hash for demonstration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigningKey {
    /// Key material (32 bytes)
    pub key_bytes: Vec<u8>,
    /// Key identifier
    pub key_id: String,
}

impl SigningKey {
    pub fn new(key_id: &str) -> Self {
        // Deterministic key derivation from key_id (not for production!)
        let key_bytes = key_id
            .bytes()
            .cycle()
            .take(32)
            .map(|b| b.wrapping_mul(7).wrapping_add(0xA5))
            .collect();
        Self {
            key_bytes,
            key_id: key_id.to_string(),
        }
    }

    /// Sign a model artifact hash.
    pub fn sign(&self, model_hash: &[u8]) -> Vec<u8> {
        // Simplified signature: XOR hash with key, then hash again
        let mut sig = vec![0u8; 32];
        for i in 0..32
        {
            let k = self.key_bytes[i % self.key_bytes.len()];
            let m = if i < model_hash.len()
            {
                model_hash[i]
            }
            else
            {
                0
            };
            sig[i] = k.wrapping_add(m).wrapping_mul(0x1B);
        }
        // Hash again
        sig = simple_hash(&sig);
        sig
    }

    /// Verify a signature.
    pub fn verify(&self, model_hash: &[u8], signature: &[u8]) -> bool {
        let expected = self.sign(model_hash);
        expected.len() == signature.len()
            && expected.iter().zip(signature.iter()).all(|(a, b)| a == b)
    }
}

/// Model signature with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSignature {
    pub model_hash: Vec<u8>,
    pub signature: Vec<u8>,
    pub key_id: String,
    pub timestamp: f64,
    pub model_version: String,
}

/// An OTA (Over-The-Air) update package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtaUpdate {
    pub model_version: String,
    pub model_hash: Vec<u8>,
    pub model_bytes: Vec<u8>,
    pub signature: ModelSignature,
    pub target_device_id: String,
    pub rollback_on_failure: bool,
}

impl OtaUpdate {
    /// Create a new OTA update package, signing the model.
    pub fn new(
        model_bytes: Vec<u8>,
        model_version: &str,
        target_device_id: &str,
        signing_key: &SigningKey,
    ) -> Self {
        let model_hash = simple_hash(&model_bytes);
        let signature = signing_key.sign(&model_hash);
        let model_sig = ModelSignature {
            model_hash: model_hash.clone(),
            signature,
            key_id: signing_key.key_id.clone(),
            timestamp: 0.0, // caller can update
            model_version: model_version.to_string(),
        };
        Self {
            model_version: model_version.to_string(),
            model_hash,
            model_bytes,
            signature: model_sig,
            target_device_id: target_device_id.to_string(),
            rollback_on_failure: true,
        }
    }

    /// Verify the OTA package integrity.
    pub fn verify(&self, signing_key: &SigningKey) -> bool {
        // 1. Check model hash matches
        let computed_hash = simple_hash(&self.model_bytes);
        if computed_hash != self.model_hash
        {
            return false;
        }
        // 2. Verify signature
        if !signing_key.verify(&self.model_hash, &self.signature.signature)
        {
            return false;
        }
        // 3. Check key ID matches
        signing_key.key_id == self.signature.key_id
    }

    /// Apply the update to a simulated device.
    pub fn apply(&self, signing_key: &SigningKey) -> OtaResult {
        let hash_hex = self
            .model_hash
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();
        if !self.verify(signing_key)
        {
            return OtaResult {
                success: false,
                model_hash: hash_hex,
                signature_valid: false,
                bytes_downloaded: self.model_bytes.len() as u64,
                message: "Signature verification failed".to_string(),
            };
        }
        OtaResult {
            success: true,
            model_hash: hash_hex,
            signature_valid: true,
            bytes_downloaded: self.model_bytes.len() as u64,
            message: format!("Model {} applied successfully", self.model_version),
        }
    }
}

/// Simple non-cryptographic hash (not for production use).
fn simple_hash(data: &[u8]) -> Vec<u8> {
    let mut state = [0u8; 32];
    for (i, &b) in data.iter().enumerate()
    {
        let idx = i % 32;
        state[idx] = state[idx].wrapping_add(b);
        state[(idx + 1) % 32] = state[(idx + 1) % 32].wrapping_mul(b.wrapping_add(1));
        state[(idx + 2) % 32] ^= b.rotate_left(3);
    }
    // Final mixing
    for i in 0..32
    {
        state[i] = state[i].wrapping_mul(0x5B).wrapping_add(0x9E);
        state[(i + 7) % 32] ^= state[i];
    }
    state.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signing_key_sign_verify() {
        let key = SigningKey::new("test-key-1");
        let data = vec![1, 2, 3, 4, 5];
        let sig = key.sign(&data);
        assert!(key.verify(&data, &sig));
    }

    #[test]
    fn test_signing_key_reject_tampered_data() {
        let key = SigningKey::new("test-key-1");
        let data = vec![1, 2, 3, 4, 5];
        let sig = key.sign(&data);
        let tampered = vec![1, 2, 3, 4, 6];
        assert!(!key.verify(&tampered, &sig));
    }

    #[test]
    fn test_signing_key_reject_wrong_key() {
        let key1 = SigningKey::new("key-1");
        let key2 = SigningKey::new("key-2");
        let data = vec![1, 2, 3, 4, 5];
        let sig = key1.sign(&data);
        assert!(!key2.verify(&data, &sig));
    }

    #[test]
    fn test_ota_create_and_verify() {
        let key = SigningKey::new("production-key");
        let model = vec![0u8; 1024]; // simulated model
        let ota = OtaUpdate::new(model.clone(), "v2.1.0", "device-001", &key);
        assert!(ota.verify(&key));
    }

    #[test]
    fn test_ota_apply_success() {
        let key = SigningKey::new("production-key");
        let model = vec![42u8; 512];
        let ota = OtaUpdate::new(model, "v2.1.0", "device-001", &key);
        let result = ota.apply(&key);
        assert!(result.success);
        assert!(result.signature_valid);
        assert_eq!(result.bytes_downloaded, 512);
    }

    #[test]
    fn test_ota_apply_fails_wrong_key() {
        let key1 = SigningKey::new("prod-key");
        let key2 = SigningKey::new("wrong-key");
        let model = vec![42u8; 512];
        let ota = OtaUpdate::new(model, "v2.1.0", "device-001", &key1);
        let result = ota.apply(&key2);
        assert!(!result.success);
        assert!(!result.signature_valid);
    }

    #[test]
    fn test_ota_apply_fails_tampered_model() {
        let key = SigningKey::new("prod-key");
        let model = vec![42u8; 512];
        let mut ota = OtaUpdate::new(model, "v2.1.0", "device-001", &key);
        // Tamper with model bytes after packaging
        ota.model_bytes[0] = 99;
        let result = ota.apply(&key);
        assert!(!result.success);
    }

    #[test]
    fn test_simple_hash_deterministic() {
        let data = vec![1, 2, 3, 4, 5];
        let h1 = simple_hash(&data);
        let h2 = simple_hash(&data);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 32);
    }
}
