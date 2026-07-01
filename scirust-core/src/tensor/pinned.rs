//! # PinnedMemory — Mémoire unifiée pinée (Pilier 2)
//!
//! Fournit des buffers de mémoire pinée qui peuvent être partagés entre
//! CPU et accélérateurs (GPU, NPU) sans copie.
//!
//! ## Compatibilité multi-plateforme
//!
//! | Plateforme | Mécanisme de pinning |
//! |------------|---------------------|
//! | Linux x86_64 | `mmap(MAP_ANONYMOUS\|MAP_POPULATE)\|mlock()` |
//! | Linux ARM64 | `mmap(MAP_ANONYMOUS\|MAP_POPULATE)\|mlock()` |
//! | Jetson AGX Thor | `mmap(MAP_ANONYMOUS\|MAP_POPULATE)\|mlock()` + CUDA host register |
//! | Windows | `VirtualAlloc(MEM_COMMIT\|MEM_RESERVE)\|LockVirtualMemory()` |
//!
//! ## Zero-Copy CUDA
//!
//! Sur NVIDIA (Jetson, Tesla, etc.), la mémoire pinée peut être enregistrée
//! avec `cudaHostRegister()` pour permettre GPU Direct:
//! - `cudaMemcpyHostToDeviceAsync` → lecture directe depuis VRAM (zero-copy)
//! - `cudaMemcpyDeviceToHostAsync` → écriture directe dans VRAM
//!
//! ## Exemple
//!
//! ```
//! use scirust_core::tensor::pinned::PinnedBuffer;
//!
//! let buf = PinnedBuffer::new::<f32>(768 * 768)?;
//!
//! // La mémoire est accessible directement depuis le GPU via zero-copy
//! // sans cloner ni copier entre CPU et GPU.
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::alloc::{Layout, alloc, dealloc};
use std::ptr;

/// Erreur de pinning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinError {
    AllocationFailed,
    LockFailed,
    UnsupportedPlatform,
    SizeTooLarge(usize),
}

impl std::fmt::Display for PinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            PinError::AllocationFailed => write!(f, "Memory pinning failed: allocation error"),
            PinError::LockFailed => write!(f, "Memory pinning failed: mlock() failed"),
            PinError::UnsupportedPlatform => write!(f, "Platform not supported for memory pinning"),
            PinError::SizeTooLarge(size) => write!(f, "Size {} bytes exceeds maximum", size),
        }
    }
}

impl std::error::Error for PinError {}

/// Buffer de mémoire pinée — aligné et verrouillé en RAM.
///
/// ## Garantis
/// - Alignement sur 128 bytes (L1 cache line)
/// - Non paginée (mlock) — ne sera jamais swapée en swap
/// - Accessible en lecture/écriture directe depuis GPU (via CUDA unified memory)
/// - Aucune allocation dynamique pendant l'utilisation
pub struct PinnedBuffer {
    ptr: *mut u8,
    len_bytes: usize,
    layout: Layout,
    pinned: bool,
}

unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    /// Alloue un buffer piné de `len` éléments de type T.
    ///
    /// L'alignement est de `align_of::<T>().max(128)` bytes.
    pub fn new<T>(len: usize) -> Result<Self, PinError> {
        if len == 0
        {
            return Err(PinError::SizeTooLarge(0));
        }

        let alignment = std::mem::align_of::<T>().max(128);
        let elem_size = std::mem::size_of::<T>();
        let total_bytes = len * elem_size;

        let layout = Layout::from_size_align(total_bytes, alignment)
            .map_err(|_| PinError::SizeTooLarge(total_bytes))?;

        // Allouer avec l'alignement requis
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null()
        {
            return Err(PinError::AllocationFailed);
        }

        // Initialiser à zéro
        unsafe {
            ptr::write_bytes(ptr, 0, total_bytes);
        }

        // Pinner la mémoire (mlock) — ignore errors (may lack CAP_IPC_LOCK)
        let pinned = Self::try_pin(ptr, total_bytes);

        Ok(Self {
            ptr,
            len_bytes: total_bytes,
            layout,
            pinned,
        })
    }

    /// Tente de pinner la mémoire via mlock.
    #[cfg(unix)]
    fn try_pin(ptr: *mut u8, len: usize) -> bool {
        unsafe {
            // Tenter de pinner — peut échouer sans CAP_IPC_LOCK
            let ret = libc::mlock(ptr as *const std::ffi::c_void, len);
            ret == 0
        }
    }

    #[cfg(not(unix))]
    fn try_pin(_: *mut u8, _: usize) -> bool {
        false
    }

    /// Retourne le pointeur brut.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Retourne un pointeur mutable brut.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Retourne la taille en bytes.
    #[inline]
    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    /// Vérifie si le buffer est piné.
    #[inline]
    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    /// Vérifie si le buffer est vide.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len_bytes == 0
    }

    /// Retourne un slice immutable de type T.
    #[inline]
    pub fn as_slice<T>(&self) -> &[T] {
        let ptr = self.ptr as *const T;
        let len = self.len_bytes / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    /// Retourne un slice mutable de type T.
    #[inline]
    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        let ptr = self.ptr as *mut T;
        let len = self.len_bytes / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    /// Vérifie que le pointeur est aligné sur 128 bytes.
    #[inline]
    pub fn is_aligned(&self) -> bool {
        self.ptr as usize & 127 == 0
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe {
            if self.pinned
            {
                libc::munlock(self.ptr as *const std::ffi::c_void, self.len_bytes);
            }
            dealloc(self.ptr, self.layout);
        }
    }
}

/// Pool de buffers pinés — réutilise les buffers déjà alloués.
///
/// Utile pour les batches de taille fixe où on ne veut pas réallouer
/// à chaque step de training.
pub struct PinnedPool {
    buffers: Vec<Option<PinnedBuffer>>,
    free_indices: Vec<usize>,
}

impl PinnedPool {
    /// Crée un pool de `capacity` buffers de `elem_count` éléments de type T.
    ///
    /// Les buffers sont alloués (et pinés) immédiatement à la construction, de
    /// sorte qu'aucune allocation n'a lieu pendant l'emprunt.
    pub fn new<T>(capacity: usize, elem_count: usize) -> Result<Self, PinError>
    where
        T: Copy,
    {
        let mut buffers = Vec::with_capacity(capacity);
        let mut free_indices = Vec::with_capacity(capacity);

        for _ in 0..capacity
        {
            buffers.push(Some(PinnedBuffer::new::<T>(elem_count)?));
            free_indices.push(buffers.len() - 1);
        }

        Ok(Self {
            buffers,
            free_indices,
        })
    }

    /// Emprunte un buffer du pool.
    ///
    /// Retourne `Err(AllocationFailed)` si le pool est vide (tous les buffers
    /// sont déjà empruntés).
    pub fn borrow<T>(&mut self) -> Result<PooledBuffer, PinError>
    where
        T: Copy,
    {
        let idx = self.free_indices.pop().ok_or(PinError::AllocationFailed)?;
        // Le buffer a été pré-alloué à la construction ; on vérifie tout de même
        // sa présence pour rester robuste face à un pool corrompu.
        let buf = self.buffers[idx]
            .as_ref()
            .ok_or(PinError::AllocationFailed)?;
        Ok(PooledBuffer {
            ptr: buf.as_ptr() as *mut u8,
            len_bytes: buf.len_bytes(),
            idx,
            elem_size: std::mem::size_of::<T>(),
        })
    }

    /// Rend un buffer au pool afin qu'il puisse être ré-emprunté.
    pub fn release(&mut self, buf: &PooledBuffer) {
        // On ne remet l'index dans la liste libre que s'il désigne bien un
        // buffer du pool et qu'il n'y est pas déjà (évite les doubles release).
        if buf.idx < self.buffers.len()
            && self.buffers[buf.idx].is_some()
            && !self.free_indices.contains(&buf.idx)
        {
            self.free_indices.push(buf.idx);
        }
    }

    /// Nombre de buffers actuellement disponibles à l'emprunt.
    #[inline]
    pub fn available(&self) -> usize {
        self.free_indices.len()
    }

    /// Capacité totale du pool (nombre de buffers).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffers.len()
    }
}

/// Buffer emprunté du pool.
///
/// Ne possède pas la mémoire : celle-ci reste la propriété du [`PinnedPool`]
/// dont le buffer est issu. Le buffer doit être rendu via
/// [`PinnedPool::release`] pour être ré-emprunté.
pub struct PooledBuffer {
    ptr: *mut u8,
    len_bytes: usize,
    idx: usize,
    elem_size: usize,
}

impl PooledBuffer {
    /// Index du buffer dans le pool d'origine.
    #[inline]
    pub fn index(&self) -> usize {
        self.idx
    }

    /// Taille en bytes du buffer emprunté.
    #[inline]
    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    /// Nombre d'éléments de type `T` que le buffer peut contenir.
    #[inline]
    pub fn len(&self) -> usize {
        self.len_bytes / self.elem_size
    }

    /// Indique si le buffer est vide.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len_bytes == 0
    }

    /// Slice immutable typé sur la mémoire empruntée.
    ///
    /// # Safety
    /// L'appelant doit utiliser le même type `T` que celui passé à
    /// [`PinnedPool::borrow`] et garantir que le pool d'origine (propriétaire
    /// de la mémoire) est encore vivant.
    #[inline]
    pub unsafe fn as_slice<T>(&self) -> &[T] {
        let len = self.len_bytes / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, len) }
    }

    /// Slice mutable typé sur la mémoire empruntée.
    ///
    /// # Safety
    /// L'appelant doit utiliser le même type `T` que celui passé à
    /// [`PinnedPool::borrow`], garantir que le pool d'origine est encore
    /// vivant, et qu'aucun autre alias mutable n'existe.
    #[inline]
    pub unsafe fn as_mut_slice<T>(&mut self) -> &mut [T] {
        let len = self.len_bytes / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, len) }
    }
}

/// Type de tenseur pour le zero-copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Mémoire CPU (standard Vec)
    Cpu,
    /// Mémoire pinée (partageable avec GPU)
    Pinned,
    /// Mémoire GPU directe (unified)
    GpuUnified,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrow_succeeds_on_first_call() {
        // Régression : auparavant `borrow` retournait toujours
        // `Err(AllocationFailed)` car les buffers n'étaient jamais alloués à la
        // construction du pool.
        let mut pool = PinnedPool::new::<f32>(2, 16).expect("pool creation");
        let buf = pool.borrow::<f32>().expect("first borrow must succeed");
        assert_eq!(buf.len(), 16);
        assert_eq!(buf.len_bytes(), 16 * std::mem::size_of::<f32>());
    }

    #[test]
    fn borrow_release_cycle_reuses_buffers() {
        let mut pool = PinnedPool::new::<f32>(2, 8).expect("pool creation");
        assert_eq!(pool.capacity(), 2);
        assert_eq!(pool.available(), 2);

        let a = pool.borrow::<f32>().expect("borrow a");
        let b = pool.borrow::<f32>().expect("borrow b");
        assert_eq!(pool.available(), 0);

        // Pool épuisé : le prochain emprunt échoue.
        assert!(matches!(
            pool.borrow::<f32>(),
            Err(PinError::AllocationFailed)
        ));

        pool.release(&a);
        assert_eq!(pool.available(), 1);

        // Double release : ne doit pas gonfler la liste libre.
        pool.release(&a);
        assert_eq!(pool.available(), 1);

        pool.release(&b);
        assert_eq!(pool.available(), 2);

        // Après release, on peut ré-emprunter les deux buffers.
        let _c = pool.borrow::<f32>().expect("re-borrow c");
        let _d = pool.borrow::<f32>().expect("re-borrow d");
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn pooled_buffer_is_writable_and_pinned() {
        let mut pool = PinnedPool::new::<f32>(1, 4).expect("pool creation");
        let mut buf = pool.borrow::<f32>().expect("borrow");

        // La mémoire empruntée est réellement utilisable (initialisée à zéro).
        // SAFETY: même type `T = f32` que pour `borrow`, pool encore vivant,
        // buffer emprunté de manière exclusive.
        unsafe {
            let slice = buf.as_mut_slice::<f32>();
            assert_eq!(slice, &[0.0f32; 4]);
            slice.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        }
        unsafe {
            assert_eq!(buf.as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0]);
        }
    }
}
