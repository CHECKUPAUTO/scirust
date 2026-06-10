//! # AlignedVec — Vec avec alignement garanti
//!
//! Alternative à `Vec<T>` avec des garanties d'alignement strictes (128 octets,
//! la largeur de ligne de cache / vecteur SIMD sur les plateformes cibles).
//! Le backing est un `Vec<Block>` où `Block` est `#[repr(align(128))]`, ce qui
//! garantit que le pointeur de base est réellement aligné.

use super::MIN_ALIGN_BYTES;

/// Bloc de 128 octets, aligné sur 128 — force l'alignement du buffer sous-jacent.
#[repr(C, align(128))]
#[derive(Clone, Copy)]
struct Block([u8; 128]);

/// Un buffer de données brutes avec alignement garanti sur 128 octets.
#[derive(Debug)]
pub struct AlignedVec {
    /// Backing aligné (chaque `Block` fait 128 octets, aligné sur 128).
    blocks: Vec<Block>,
    /// Alignement requis (toujours >= 16).
    alignment: usize,
    /// Nombre d'éléments de type T.
    len: usize,
    /// Octets effectivement utilisés (`len * size_of::<T>()`).
    byte_len: usize,
}

impl std::fmt::Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Block(128B)")
    }
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
        let n_blocks = byte_len.div_ceil(128).max(1);
        let blocks = vec![Block([0u8; 128]); n_blocks];
        Self {
            blocks,
            alignment,
            len,
            byte_len,
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

    #[inline]
    fn byte_ptr(&self) -> *const u8 {
        self.blocks.as_ptr() as *const u8
    }

    #[inline]
    fn byte_ptr_mut(&mut self) -> *mut u8 {
        self.blocks.as_mut_ptr() as *mut u8
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
        let ptr = self.byte_ptr_mut() as *mut T;
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
        let ptr = self.byte_ptr() as *const T;
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

    /// Pointeur brut (aligné sur 128 octets).
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.byte_ptr()
    }

    /// Pointeur mutable brut (aligné sur 128 octets).
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.byte_ptr_mut()
    }

    /// Alignement en octets.
    #[inline]
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Longueur en octets effectivement utilisés.
    #[inline]
    pub fn len(&self) -> usize {
        self.byte_len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.byte_len == 0
    }

    /// Vérifie que le pointeur est aligné sur 128 octets.
    #[inline]
    pub fn is_aligned(&self) -> bool {
        self.byte_ptr() as usize & (MIN_ALIGN_BYTES - 1) == 0
    }
}

impl<T: Copy> From<AlignedVec> for Vec<T> {
    fn from(vec: AlignedVec) -> Self {
        vec.as_slice::<T>().to_vec()
    }
}

impl<T: Copy> From<Vec<T>> for AlignedVec {
    fn from(v: Vec<T>) -> Self {
        let mut av = AlignedVec::new::<T>(v.len());
        av.as_mut_slice::<T>().copy_from_slice(&v);
        av
    }
}

/// Extension pour `Vec<T>`: convertir en AlignedVec.
#[allow(dead_code)]
pub trait ToAligned<T: Copy>: Sized {
    fn to_aligned(self) -> AlignedVec;
}

impl<T: Copy> ToAligned<T> for Vec<T> {
    fn to_aligned(self) -> AlignedVec {
        AlignedVec::from(self)
    }
}
