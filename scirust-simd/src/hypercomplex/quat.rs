// scirust-simd/src/hypercomplex/quat.rs
//
// Noyau quaternionique `f32x4` : le cas de base vectorisé de la récursion
// de Cayley-Dickson. Un quaternion q = w + x·i + y·j + z·k est stocké dans
// un registre 128 bits avec le layout de lanes :
//
//   lane :   0    1    2    3
//   q    = [ w ,  x ,  y ,  z ]
//
// Tout ici est `#[inline(always)]` : après inlining dans le produit
// d'octonions/sédénions, ces fonctions se réduisent à des instructions
// shuffle (vpermilps/vshufps sur x86, TBL/EXT/REV sur NEON) et FMA
// (vfmadd231ps / FMLA) sans jamais toucher la mémoire.

use std::simd::{StdFloat, f32x4, simd_swizzle};

/// Masque de conjugaison quaternionique : q̄ = w − x·i − y·j − z·k.
///
/// Une seule multiplication vectorielle par ±1 — le compilateur la lower
/// en un XOR du bit de signe (`vxorps` avec un masque constant), latence 1.
const CONJ_SIGNS: f32x4 = f32x4::from_array([1.0, -1.0, -1.0, -1.0]);

/// Signes du terme en pₓ du produit (voir [`quat_mul`]).
const SIGNS_X: f32x4 = f32x4::from_array([-1.0, 1.0, -1.0, 1.0]);
/// Signes du terme en p_y du produit.
const SIGNS_Y: f32x4 = f32x4::from_array([-1.0, 1.0, 1.0, -1.0]);
/// Signes du terme en p_z du produit.
const SIGNS_Z: f32x4 = f32x4::from_array([-1.0, -1.0, 1.0, 1.0]);

/// Conjugaison quaternionique q ↦ q̄ (négation de la partie vectorielle).
#[inline(always)]
#[must_use]
pub fn quat_conj(q: f32x4) -> f32x4 {
    q * CONJ_SIGNS
}

/// Produit de Hamilton p·q entièrement en registres.
///
/// Décomposition par coefficient de p :
///
/// ```text
///   (p·q)_w = p_w·q_w − p_x·q_x − p_y·q_y − p_z·q_z
///   (p·q)_x = p_w·q_x + p_x·q_w + p_y·q_z − p_z·q_y
///   (p·q)_y = p_w·q_y − p_x·q_z + p_y·q_w + p_z·q_x
///   (p·q)_z = p_w·q_z + p_x·q_y − p_y·q_x + p_z·q_w
/// ```
///
/// soit, colonne par colonne (chaque colonne = 1 broadcast + 1 shuffle
/// + 1 masque de signes + 1 FMA) :
///
/// ```text
///   r  = splat(p_w) · [q_w, q_x, q_y, q_z]                     (mul simple)
///      + splat(p_x) · [q_x, q_w, q_z, q_y] ⊙ (−,+,−,+)         (FMA)
///      + splat(p_y) · [q_y, q_z, q_w, q_x] ⊙ (−,+,+,−)         (FMA)
///      + splat(p_z) · [q_z, q_y, q_x, q_w] ⊙ (−,−,+,+)         (FMA)
/// ```
///
/// Total : 4 shuffles de q, 4 broadcasts de p (eux-mêmes des shuffles
/// intra-registre, jamais un aller-retour mémoire), 3 multiplications de
/// signes et 1 mul + 3 FMA. Aucun spill : tout tient dans ~8 registres
/// vectoriels sur les deux architectures cibles.
#[inline(always)]
#[must_use]
pub fn quat_mul(p: f32x4, q: f32x4) -> f32x4 {
    // Broadcasts de chaque lane de p vers les 4 lanes d'un registre.
    // `simd_swizzle!` avec un index constant est compilé en une seule
    // instruction de permutation (vbroadcastss / DUP lane sur NEON) —
    // on n'extrait jamais le scalaire vers un registre général.
    let pw = simd_swizzle!(p, [0, 0, 0, 0]);
    let px = simd_swizzle!(p, [1, 1, 1, 1]);
    let py = simd_swizzle!(p, [2, 2, 2, 2]);
    let pz = simd_swizzle!(p, [3, 3, 3, 3]);

    // Permutations de q pour aligner chaque produit croisé sur sa lane
    // de destination. Indices constants → vpermilps immédiat (x86) ou
    // REV64/EXT (NEON), 1 cycle de latence.
    let q_x = simd_swizzle!(q, [1, 0, 3, 2]); // [q_x, q_w, q_z, q_y]
    let q_y = simd_swizzle!(q, [2, 3, 0, 1]); // [q_y, q_z, q_w, q_x]
    let q_z = simd_swizzle!(q, [3, 2, 1, 0]); // [q_z, q_y, q_x, q_w]

    // Chaîne de FMA : r = pz·t_z + (py·t_y + (px·t_x + pw·q)).
    // Chaque `mul_add` est un vfmadd/FMLA fusionné (1 seule opération,
    // 1 seul arrondi) — le cœur du débit de la pile hypercomplexe.
    let acc = pw * q;
    let acc = px.mul_add(q_x * SIGNS_X, acc);
    let acc = py.mul_add(q_y * SIGNS_Y, acc);
    pz.mul_add(q_z * SIGNS_Z, acc)
}
