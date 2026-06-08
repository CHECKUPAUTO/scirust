//! # AlignedVec — Vec avec alignement garanti
//!
//! Alternative à `Vec<T>` avec des garanties d'alignement strictes.
//! Utile pour les passes SIMD qui nécessitent des pointeurs alignés.

use super::{align_up, MIN_ALIGN_BYTES};

/// Un buffer de données brutes avec alignement garanti sur ALIGNMENT bytes.
///
/// Contrairement à `Vec<u8>`, la mémoire est allouée avec un alignement
/// suffisant pour stocker n'importe quel type SIMD sur toutes les plateformes cibles.
#[derive(Debug)]
pub struct AlignedVec {
    /// Données brutes (non-Send car utilisé pour transmutes vers T).
    data: Vec<u8>,
    /// Alignement requis (toujours >= 16).
    alignment: usize,
    /// Nombre d'éléments de type T (informationnelle).
    len: usize,
}

unsafe impl Send for AlignedVec {}
unsafe impl Sync for AlignedVec {}

impl AlignedVec {
    /// Crée un nouveau buffer aligné de `len` éléments de type T.
    pub fn new<T>(len: usize) -> Self
    where
        T: Copy,
    {
        let alignment = std::mem::align_of::<T>().max(16).max(MIN_ALIGN_BYTES);
        let byte_len = len * std::mem::size_of::<T>();
        let aligned_len = align_up(byte_len);

        // Allouer avec alignement via vec
        let mut data = vec![0u8; aligned_len];

        Self {
            data,
            alignment,
            len,
        }
    }

    /// Crée un buffer aligné pré-rempli avec une valeur.
    pub fn new_fill<T>(len: usize, val: T) -> Self
    where
        T: Copy,
    {
        let mut vec = Self::new::<T>(len);
        vec.fill(val);
        vec
    }

    /// Retourne un slice mutable de type T, aligné.
    #[inline]
    pub fn as_mut_slice<T>(&mut self) -> &mut [T]
    where
        T: Copy,
    {
        assert!(
            self.alignment >= std::mem::align_of::<T>(),
            "AlignedVec alignment {} < required alignment {}",
            self.alignment,
            std::mem::align_of::<T>()
        );
        let ptr = self.data.as_mut_ptr() as *mut T;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len) }
    }

    /// Retourne un slice immutable de type T.
    #[inline]
    pub fn as_slice<T>(&self) -> &[T]
    where
        T: Copy,
    {
        assert!(
            self.alignment >= std::mem::align_of::<T>(),
            "AlignedVec alignment {} < required alignment {}",
            self.alignment,
            std::mem::align_of::<T>()
        );
        let ptr = self.data.as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    /// Remplit le buffer avec une valeur.
    pub fn fill<T>(&mut self, val: T)
    where
        T: Copy,
    {
        for elem in self.as_mut_slice::<T>().iter_mut() {
            *elem = val;
        }
    }

    /// Retourne le pointeur brut (aligné).
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Retourne un pointeur mutable brut (aligné).
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    /// Retourne l'alignement en bytes.
    #[inline]
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Retourne la longueur en bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Vérifie que le pointeur est aligné sur ALIGNMENT.
    #[inline]
    pub fn is_aligned(&self) -> bool {
        self.data.as_ptr() as usize & (MIN_ALIGN_BYTES - 1) == 0
    }
}

impl<T: Copy> From<AlignedVec> for Vec<T> {
    fn from(vec: AlignedVec) -> Self {
        let ptr = vec.data.as_ptr() as *const T;
        let len = vec.len;
        unsafe { Vec::from_raw_parts(ptr, len, vec.len) }
    }
}

impl<T: Copy> Into<AlignedVec> for Vec<T> {
    fn into(self) -> AlignedVec {
        let alignment = std::mem::align_of::<T>().max(16);
        let data = self.into_bytes();
        AlignedVec {
            data,
            alignment,
            len: self.len(),
        }
    }
}

/// Extension pour `Vec<T>`: convertir en AlignedVec.
pub trait ToAligned<T: Copy>: Sized {
    fn to_aligned(self) -> AlignedVec;
}

impl<T: Copy> ToAligned<T> for Vec<T> {
    fn to_aligned(self) -> AlignedVec {
        let alignment = std::mem::align_of::<T>().max(16);
        let bytes = self.into_bytes();
        AlignedVec {
            data: bytes,
            alignment,
            len: self.len(),
        }
    }
}
