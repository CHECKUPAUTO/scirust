//! # Quantification Native — Pilier 5
//!
//! Ce module fournit les traits et implémentations pour la quantification
//! native des tenseurs: int8, bf16, et int4 packed.
//!
//! ## Traits
//!
//! - [`Quantized`] — trait de base pour les types quantifiés
//! - [`Quantize`] — trait pour quantifier un tenseur fp32
//! - [`Dequantize`] — trait pour déquantifier vers fp32
//!
//! ## Usage
//!
//! ```
//! use scirust_core::quant::*;
//!
//! let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
//!
//! // Quantifier en int8
//! let quantized = Int8Tensor::quantize(&data);
//!
//! // Stocker en mémoire (4× moins de bande passante)
//! let bytes = quantized.to_bytes();
//!
//! // Déquantifier pour le calcul
//! let recovered = Int8Tensor::dequantize(&bytes);
//! ```

mod int8;
mod bf16;
mod int4;

pub use int8::*;
pub use bf16::*;
pub use int4::*;

/// Trait pour les types quantifiés.
///
/// Permet d'abstraire le format de stockage (int8, bf16, int4) tout en
/// fournissant une interface unifiée.
pub trait Quantized {
    /// Type de stockage brut (u8 pour int8/int4, u16 pour bf16).
    type Storage: Copy;

    /// Retourne le format de quantification.
    fn format(&self) -> QuantFormat;

    /// Taille de compression (f32_bytes / packed_bytes).
    fn compression_ratio(&self) -> f32 {
        match self.format() {
            QuantFormat::Fp32 => 1.0,
            QuantFormat::Int8 => 4.0,
            QuantFormat::Bf16 => 2.0,
            QuantFormat::Int4 => 8.0,
        }
    }
}

/// Format de quantification supporté.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    Fp32,
    Int8,
    Bf16,
    Int4,
}

/// Trait pour quantifier un tenseur fp32.
pub trait Quantize<T> {
    fn quantize(data: &[f32]) -> T;
}

/// Trait pour déquantiser vers fp32.
pub trait Dequantize {
    fn dequantize(data: &[u8]) -> Vec<f32>;
}

/// Tensor quantifié avec métadonnées (format, scale).
#[derive(Debug, Clone)]
pub struct QuantTensor {
    /// Données stockées.
    pub storage: Vec<u8>,
    /// Format de quantification.
    pub format: QuantFormat,
    /// Scale global (un scale par canal est aussi supporté).
    pub scale: f32,
    /// Shape d'origine.
    pub shape: Vec<usize>,
}

impl QuantTensor {
    /// Crée un tenseur quantifié int8.
    pub fn quantize_i8(data: &[f32], shape: &[usize]) -> Self {
        let (packed, scale) = quantize_tensor_f32_to_i8(data);
        Self {
            storage: packed.into_iter().map(|b| b as u8).collect(),
            format: QuantFormat::Int8,
            scale,
            shape: shape.to_vec(),
        }
    }

    /// Crée un tenseur quantifié bf16.
    pub fn quantize_bf16(data: &[f32], shape: &[usize]) -> Self {
        let bf16 = quantize_tensor_f32_to_bf16(data);
        // Convertir u16 → 2 × u8 (little-endian)
        let mut storage = Vec::with_capacity(bf16.len() * 2);
        for &v in &bf16 {
            storage.push((v & 0xFF) as u8);
            storage.push(((v >> 8) & 0xFF) as u8);
        }
        Self {
            storage,
            format: QuantFormat::Bf16,
            scale: 1.0, // bf16 n'a pas de scale
            shape: shape.to_vec(),
        }
    }

    /// Crée un tenseur quantifié int4 packed.
    pub fn quantize_i4(data: &[f32], shape: &[usize]) -> Self {
        let (packed, scale) = quantize_tensor_f32_to_i4(data);
        Self {
            storage: packed,
            format: QuantFormat::Int4,
            scale,
            shape: shape.to_vec(),
        }
    }

    /// Déquantifie le tenseur.
    pub fn dequantize(&self) -> Vec<f32> {
        match self.format {
            QuantFormat::Int8 => {
                let i8_data: Vec<i8> = self.storage.iter().map(|&b| b as i8).collect();
                dequantize_i8_to_f32(&i8_data, self.scale)
            }
            QuantFormat::Bf16 => {
                let bf16: Vec<u16> = self
                    .storage
                    .chunks(2)
                    .map(|chunk| (chunk[0] as u16) | ((chunk[1] as u16) << 8))
                    .collect();
                dequantize_bf16_to_f32(&bf16)
            }
            QuantFormat::Int4 => dequantize_i4_to_f32(&self.storage, self.scale),
            QuantFormat::Fp32 => {
                // Convertir u8 → f32
                let bytes = self.storage.chunks(4).map(|chunk| {
                    u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                });
                let result: Vec<f32> = bytes.map(f32::from_bits).collect();
                result
            }
        }
    }

    /// Retourne le nombre de bytes de stockage.
    pub fn bytes(&self) -> usize {
        self.storage.len()
    }

    /// Retourne le nombre d'éléments.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Compression effective.
    pub fn effective_compression(&self) -> f32 {
        (self.numel() * 4) as f32 / self.bytes() as f32
    }
}
