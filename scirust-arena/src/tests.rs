//! Tests de validation du pilier 3 (arène).

#[cfg(test)]
mod tests {
    use super::PinnedArena;

    #[test]
    fn test_arena_o1_allocation() {
        let mut arena = PinnedArena::new::<f32>(1 << 20);

        // Mesurer le temps d'allocation
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _slice = arena.alloc_slice_fill::<f32>(768, 0.0).unwrap();
        }
        let elapsed = start.elapsed();

        // Doit être < 1ms pour 1000 allocations (O(1) garanti)
        assert!(
            elapsed.as_nanos() < 1_000_000,
            "Allocation O(1) trop lente : {} ns pour 1000 allocs (expected < 1ms)",
            elapsed.as_nanos()
        );
    }

    #[test]
    fn test_arena_alignment() {
        let arena = PinnedArena::new::<f32>(1 << 20);

        // Vérifier que la capacité est alignée sur 128 bytes
        assert_eq!(
            arena.capacity() % 128,
            0,
            "Arena capacity must be aligned on 128 bytes"
        );
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = PinnedArena::new::<f32>(1 << 20);

        // Allouer
        let _slice1 = arena.alloc_slice_fill::<f32>(768, 0.0).unwrap();
        let count_before = arena.alloc_count();

        // Reset
        arena.reset();

        // Vérifier que le compteur est remis à zéro
        assert_eq!(arena.alloc_count(), 0, "Reset must clear alloc count");
        assert_eq!(
            arena.allocated(),
            0,
            "Reset must clear allocated bytes"
        );

        // Réallouer doit réussir (même si l'arène est "pleine")
        let _slice2 = arena.alloc_slice_fill::<f32>(768, 0.0).unwrap();
        assert_eq!(arena.alloc_count(), 1, "Should have 1 alloc after reset");
    }

    #[test]
    fn test_arena_overflow() {
        let mut arena = PinnedArena::new::<f32>(64); // Très petit

        // Allouer tout l'espace disponible
        let slice = arena.alloc_slice_fill::<u8>(60, 0).unwrap();
        assert_eq!(slice.len(), 60);

        // Prochain alloc doit échouer
        assert!(
            arena.alloc_slice_fill::<u8>(10, 0).is_err(),
            "Should overflow"
        );
    }

    #[test]
    fn test_arena_determinism() {
        let mut arena1 = PinnedArena::new::<f32>(1 << 20);
        let mut arena2 = PinnedArena::new::<f32>(1 << 20);

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
        use super::Slab;

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
        use super::AlignedVec;

        let mut vec = AlignedVec::new::<f32>(100);

        // Vérifier alignement
        assert!(
            vec.is_aligned(),
            "AlignedVec must be aligned on 128 bytes"
        );

        // Remplir
        vec.fill::<f32>(1.0);

        // Vérifier contenu
        let slice = vec.as_slice::<f32>();
        assert_eq!(slice.len(), 100);
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[99], 1.0);
    }
}
