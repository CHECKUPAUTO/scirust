//! Tests de validation du pilier 3 (arène).

use crate::PinnedArena;

#[test]
fn test_arena_o1_allocation() {
    // 4 MB holds 1000 allocations of 768 f32 (≈3 MB).
    let mut arena = PinnedArena::new(4 << 20);

    // Mesurer le temps d'allocation
    let start = std::time::Instant::now();
    for _ in 0..1000
    {
        let _slice = arena.alloc_slice_fill::<f32>(768, 0.0).unwrap();
    }
    let elapsed = start.elapsed();

    // Bump allocation is O(1); a generous budget (debug, unoptimised) still
    // catches gross regressions such as a syscall or O(n²) per allocation.
    assert!(
        elapsed.as_millis() < 50,
        "Allocation O(1) trop lente : {} ns pour 1000 allocs",
        elapsed.as_nanos()
    );
}

#[test]
fn test_arena_alignment() {
    let arena = PinnedArena::new(1 << 20);

    // Vérifier que la capacité est alignée sur 128 bytes
    assert_eq!(
        arena.capacity() % 128,
        0,
        "Arena capacity must be aligned on 128 bytes"
    );
}

#[test]
fn test_arena_reset() {
    let mut arena = PinnedArena::new(1 << 20);

    // Allouer
    let _slice1 = arena.alloc_slice_fill::<f32>(768, 0.0).unwrap();
    let count_before = arena.alloc_count();
    assert!(count_before > 0);

    // Reset
    arena.reset();

    // Vérifier que le compteur est remis à zéro
    assert_eq!(arena.alloc_count(), 0, "Reset must clear alloc count");
    assert_eq!(arena.allocated(), 0, "Reset must clear allocated bytes");

    // Réallouer doit réussir (même si l'arène est "pleine")
    let _slice2 = arena.alloc_slice_fill::<f32>(768, 0.0).unwrap();
    assert_eq!(arena.alloc_count(), 1, "Should have 1 alloc after reset");
}

#[test]
fn test_arena_overflow() {
    // Capacity rounds up to a 128-byte multiple (128 here).
    let mut arena = PinnedArena::new(64);

    // Allouer une partie de l'espace
    let slice = arena.alloc_slice_fill::<u8>(60, 0).unwrap();
    assert_eq!(slice.len(), 60);

    // Une allocation qui dépasse la capacité doit échouer.
    assert!(
        arena.alloc_slice_fill::<u8>(200, 0).is_err(),
        "Should overflow"
    );
}

#[test]
fn test_arena_determinism() {
    let mut arena1 = PinnedArena::new(1 << 20);
    let mut arena2 = PinnedArena::new(1 << 20);

    // Même séquence d'allocations
    let s1 = arena1.alloc_slice_fill::<f32>(768, 1.0).unwrap();
    let s2 = arena2.alloc_slice_fill::<f32>(768, 1.0).unwrap();

    // Même taille
    assert_eq!(s1.len(), s2.len());

    // Même valeur
    assert_eq!(s1[0], s2[0]);
    assert_eq!(s1[767], s2[767]);
}

#[test]
fn test_slab() {
    use crate::Slab;

    let mut slab: Slab<f32, 768> = Slab::new(10);

    // Allouer
    let h1 = slab.alloc().unwrap();
    let h2 = slab.alloc().unwrap();
    assert_eq!(slab.count(), 2);

    // Vérifier validité
    assert!(slab.is_valid(h1));
    assert!(slab.is_valid(h2));

    // Libérer
    slab.free(h1);
    assert_eq!(slab.count(), 1);

    // Vérifier invalidité
    assert!(!slab.is_valid(h1));
    assert!(slab.is_valid(h2));

    // Réallouer
    let h3 = slab.alloc().unwrap();
    assert_eq!(slab.count(), 2);

    // Reset
    slab.reset();
    assert_eq!(slab.count(), 0);
    assert!(!slab.is_valid(h3));
}

#[test]
fn test_aligned_vec() {
    use crate::AlignedVec;

    let mut vec = AlignedVec::<f32>::new(100);

    // Vérifier alignement
    assert!(vec.is_aligned(), "AlignedVec must be aligned on 128 bytes");

    // Remplir
    vec.fill(1.0);

    // Vérifier contenu
    let slice = vec.as_slice();
    assert_eq!(slice.len(), 100);
    assert_eq!(slice[0], 1.0);
    assert_eq!(slice[99], 1.0);
}

#[test]
fn test_aligned_vec_new_fill() {
    use crate::AlignedVec;

    let vec = AlignedVec::<f64>::new_fill(32, std::f64::consts::PI);

    assert!(vec.is_aligned());
    let slice = vec.as_slice();
    assert_eq!(slice.len(), 32);
    for &v in slice.iter()
    {
        assert!((v - std::f64::consts::PI).abs() < 1e-15);
    }
}

#[test]
fn test_aligned_vec_as_mut_slice() {
    use crate::AlignedVec;

    let mut vec = AlignedVec::<i32>::new(16);
    {
        let slice = vec.as_mut_slice();
        for (i, v) in slice.iter_mut().enumerate()
        {
            *v = i as i32;
        }
    }
    let slice = vec.as_slice();
    for (i, &v) in slice.iter().enumerate()
    {
        assert_eq!(v, i as i32);
    }
}

#[test]
fn test_arena_new_for_type() {
    let arena = PinnedArena::new_for_type::<f32>(1024);
    // Must be large enough for 1024 f32 (4096 bytes)
    assert!(arena.capacity() >= 4096);
    assert!(arena.capacity().is_multiple_of(128));
}

#[test]
fn test_arena_alloc_with() {
    let mut arena = PinnedArena::new(4096);
    let val: &mut f64 = arena.alloc_with(42.0).unwrap();
    assert_eq!(*val, 42.0);
    // Modify through the reference
    *val = 99.0;
    assert_eq!(*val, 99.0);
}

// A pathological `n` whose byte length (`n * size_of::<T>()`) overflows `usize`
// must be rejected as `Overflow`, NOT wrapped to a small `required` that passes
// the capacity check and then materialises an oversized slice via
// `from_raw_parts_mut` (an out-of-bounds read/write in release builds).
#[test]
fn alloc_slice_rejects_byte_length_overflow_instead_of_going_oob() {
    use crate::ArenaError;
    let mut arena = PinnedArena::new(4096);
    // usize::MAX / 4 + 1 elements of f32 (4 bytes) overflows usize on multiply.
    let n = usize::MAX / 4 + 1;
    let r = arena.alloc_slice::<f32>(n);
    assert_eq!(r.err(), Some(ArenaError::Overflow));
    // The arena must remain usable and consistent after a rejected allocation.
    let ok = arena.alloc_slice_fill::<f32>(8, 1.0).unwrap();
    assert_eq!(ok.len(), 8);
    assert!(ok.iter().all(|&x| x == 1.0));
}

// `u8` allocations make `n * size_of::<T>() == n`, so a near-`usize::MAX` count
// exercises the `checked_add(aligned_offset, byte_len)` guard rather than the
// multiply guard.
#[test]
fn alloc_slice_rejects_offset_plus_len_overflow() {
    use crate::ArenaError;
    let mut arena = PinnedArena::new(4096);
    // Bump the offset a little so aligned_offset > 0.
    let _ = arena.alloc_slice_fill::<u8>(64, 0).unwrap();
    let r = arena.alloc_slice::<u8>(usize::MAX);
    assert_eq!(r.err(), Some(ArenaError::Overflow));
}
