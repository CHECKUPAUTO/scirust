//! RNG **contre-basé** Philox4x32-10 (Salmon, Moraes, Dror & Shaw,
//! *Parallel Random Numbers: As Easy as 1, 2, 3*, SC'11) — implémentation
//! clean-room depuis le papier, **validée contre les vecteurs de test
//! publiés** de la suite Random123 (voir les tests).
//!
//! Contrairement à un générateur séquentiel ([`crate::nn::PcgEngine`]), la
//! sortie est une **fonction pure** (clé, compteur) → bloc : l'aléa au rang
//! `i` ne dépend pas de qui a tiré les rangs précédents ni dans quel ordre.
//! Conséquence : dropout, initialisations, bruit et shuffles peuvent être
//! calculés en parallèle sur **n'importe quel découpage de threads** en
//! restant bit-identiques — l'« aléa order-independent » (approche JAX)
//! identifié comme trou commun de RepDL et de scirust dans la cartographie
//! (volet 111). Arithmétique entière pure ⇒ **portable par construction**
//! (mêmes bits sur toute plate-forme).

/// Multiplicateurs de rounds (constantes publiées du papier).
const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
/// Constantes de Weyl du key schedule (papier).
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

/// Générateur Philox4x32-10 : une clé de 64 bits, aucun état mutable —
/// chaque bloc de 128 bits est `block(compteur)`.
#[derive(Debug, Clone, Copy)]
pub struct Philox4x32 {
    key: [u32; 2],
}

impl Philox4x32 {
    /// Nouvelle instance ; la graine devient la clé (k0 = poids faibles).
    pub fn new(seed: u64) -> Self {
        Self {
            key: [seed as u32, (seed >> 32) as u32],
        }
    }

    #[inline]
    fn round(c: [u32; 4], k: [u32; 2]) -> [u32; 4] {
        let p0 = (M0 as u64) * (c[0] as u64);
        let p1 = (M1 as u64) * (c[2] as u64);
        let (hi0, lo0) = ((p0 >> 32) as u32, p0 as u32);
        let (hi1, lo1) = ((p1 >> 32) as u32, p1 as u32);
        [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0]
    }

    /// Le bloc de 128 bits associé au compteur (10 rounds, key schedule de
    /// Weyl entre les rounds — structure exacte du papier).
    pub fn block(&self, counter: [u32; 4]) -> [u32; 4] {
        let mut c = counter;
        let mut k = self.key;
        for _ in 0..9
        {
            c = Self::round(c, k);
            k = [k[0].wrapping_add(W0), k[1].wrapping_add(W1)];
        }
        Self::round(c, k)
    }

    /// Le `u32` au rang `index` du flux `stream` — accès direct, sans état :
    /// deux threads qui demandent des rangs différents n'interagissent pas.
    pub fn u32_at(&self, stream: u32, index: u64) -> u32 {
        let i = index / 4;
        let lane = (index % 4) as usize;
        self.block([i as u32, (i >> 32) as u32, stream, 0])[lane]
    }

    /// Le `f32` dans [0, 1) au rang `index` du flux `stream`.
    pub fn f32_at(&self, stream: u32, index: u64) -> f32 {
        // 24 bits de poids fort ⇒ f32 exact dans [0, 1), sans biais d'arrondi
        ((self.u32_at(stream, index) >> 8) as f32) * (1.0 / 16_777_216.0)
    }

    /// Remplit `out` avec les rangs `start..start+len` du flux : le résultat
    /// est bit-identique quel que soit le découpage en appels/threads.
    pub fn fill_f32(&self, stream: u32, start: u64, out: &mut [f32]) {
        for (j, slot) in out.iter_mut().enumerate()
        {
            *slot = self.f32_at(stream, start + j as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vecteurs de test PUBLIÉS (suite Random123, kat_vectors) — reproduits
    /// aussi par notre implémentation Python indépendante (volet 114). C'est
    /// l'ancrage au standard : toute divergence de spec échoue ici.
    #[test]
    fn published_known_answer_vectors() {
        let z = Philox4x32 { key: [0, 0] };
        assert_eq!(
            z.block([0, 0, 0, 0]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        let f = Philox4x32 {
            key: [0xffff_ffff, 0xffff_ffff],
        };
        assert_eq!(
            f.block([0xffff_ffff; 4]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
        // Vecteur « π » du papier
        let p = Philox4x32 {
            key: [0xa409_3822, 0x299f_31d0],
        };
        assert_eq!(
            p.block([0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344]),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    /// La propriété définitoire : le flux ne dépend pas du découpage.
    #[test]
    fn stream_is_chunking_invariant() {
        let rng = Philox4x32::new(2026_0710);
        let mut whole = vec![0.0f32; 257];
        rng.fill_f32(7, 0, &mut whole);

        let mut parts = vec![0.0f32; 257];
        let mut pos = 0usize;
        for chunk in [1usize, 3, 60, 64, 129]
        {
            let (s, e) = (pos, pos + chunk);
            rng.fill_f32(7, s as u64, &mut parts[s..e]);
            pos = e;
        }
        assert_eq!(pos, 257);
        for i in 0..257
        {
            assert_eq!(whole[i].to_bits(), parts[i].to_bits(), "rang {i}");
        }
    }

    /// Threads : quatre workers remplissent des tranches disjointes — le
    /// résultat est bit-identique au remplissage séquentiel.
    #[test]
    fn parallel_fill_matches_sequential_bitwise() {
        let rng = Philox4x32::new(42);
        let mut seq = vec![0.0f32; 1024];
        rng.fill_f32(3, 0, &mut seq);

        let mut par = vec![0.0f32; 1024];
        std::thread::scope(|scope| {
            for (t, chunk) in par.chunks_mut(256).enumerate()
            {
                scope.spawn(move || {
                    rng.fill_f32(3, (t * 256) as u64, chunk);
                });
            }
        });
        for i in 0..1024
        {
            assert_eq!(seq[i].to_bits(), par[i].to_bits(), "rang {i}");
        }
    }

    /// Flux distincts ⇒ contenus distincts ; même flux ⇒ déterministe.
    #[test]
    fn streams_are_independent_and_deterministic() {
        let rng = Philox4x32::new(99);
        assert_ne!(rng.u32_at(0, 0), rng.u32_at(1, 0));
        assert_eq!(rng.u32_at(5, 123), rng.u32_at(5, 123));
        let other = Philox4x32::new(100);
        assert_ne!(rng.u32_at(0, 0), other.u32_at(0, 0));
    }

    /// Statistiques élémentaires du flux f32 : moyenne ≈ 1/2, variance ≈ 1/12.
    #[test]
    fn f32_stream_basic_statistics() {
        let rng = Philox4x32::new(7);
        let n = 65_536;
        let mut sum = 0.0f64;
        let mut sum2 = 0.0f64;
        for i in 0..n
        {
            let v = rng.f32_at(0, i) as f64;
            assert!((0.0..1.0).contains(&v));
            sum += v;
            sum2 += v * v;
        }
        let mean = sum / n as f64;
        let var = sum2 / n as f64 - mean * mean;
        assert!((mean - 0.5).abs() < 0.01, "moyenne {mean}");
        assert!((var - 1.0 / 12.0).abs() < 0.01, "variance {var}");
    }

    /// Contrat de portabilité : empreinte FNV des 4 096 premiers u32.
    #[test]
    fn output_fingerprint_contract() {
        let rng = Philox4x32::new(2026_0710);
        let mut fp = crate::portable_f32::fnv1a_init();
        for i in 0..4096
        {
            fp = crate::portable_f32::fnv1a_fold_bits(fp, rng.u32_at(0, i));
        }
        assert_eq!(fp, 0xf96c_6b6a_eca6_99f5, "empreinte philox : 0x{fp:016x}");
    }
}
