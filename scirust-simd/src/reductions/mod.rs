// scirust-simd/src/reductions/mod.rs
//
// # Réductions SIMD — socle réutilisable
//
// Sommes (rapide / déterministe / Kahan compensée), produit scalaire, normes
// L1/L2/L∞, max/min/argmax/argmin et similarité cosinus, génériques sur le
// scalaire (`f32`, `f64`) via [`SimdReducible`] et bâties sur [`SimdScalar`].
//
// ## Rapide vs Déterministe
//
// La somme flottante n'est pas associative : l'ordre de réduction change le
// dernier bit. Deux régimes sont donc exposés via [`ReductionMode`] :
//
// * [`ReductionMode::Fast`] — accumulateurs SIMD + réduction horizontale
//   matérielle (`reduce_sum`). Débit maximal, mais le résultat dépend de la
//   largeur de vecteur et du matériel.
// * [`ReductionMode::Deterministic`] — accumulation lane-parallèle (chaque lane
//   agrège un sous-ensemble fixe d'indices) puis réduction des lanes dans un
//   **ordre d'indice fixe**, jamais via `reduce_sum`. Le résultat est
//   **reproductible bit à bit** quelle que soit la cible (x86/ARM, largeur de
//   registre) — conforme à la philosophie déterministe de SciRust.
//
// [`sum_kahan`] ajoute une compensation de Neumaier/Kahan lane-parallèle pour
// les sommes longues où l'erreur d'arrondi accumulée compte.
//
// ## Appariement et alignement
//
// Toutes les fonctions découpent via `chunks_exact(WIDTH)` + `from_slice`
// (chargements non alignés) plutôt que `as_simd` : le découpage est alors
// **indépendant de l'alignement**, ce qui est indispensable pour apparier
// correctement deux slices dans `dot`/`cosine_similarity` (leurs préfixes
// d'alignement pourraient différer).

pub mod simd_scalar;

#[cfg(test)]
mod tests;

pub use simd_scalar::SimdScalar;

/// Régime de réduction : compromis débit ⇄ reproductibilité bit-à-bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReductionMode {
    /// Réduction horizontale matérielle. Débit maximal, ordre non spécifié.
    #[default]
    Fast,
    /// Ordre de réduction fixe, reproductible sur toutes les cibles.
    Deterministic,
}

/// Scalaire réductible en SIMD : lie `Self` à son type vectoriel préféré et
/// fournit les opérations scalaires nécessaires aux queues et réductions.
///
/// Implémenté pour `f32` (largeur 8, `f32x8`) et `f64` (largeur 4, `f64x4`) —
/// les largeurs par défaut couvrant 256 bits sur les deux types.
pub trait SimdReducible: Copy + PartialEq {
    /// Type vectoriel associé (`f32x8` pour `f32`, `f64x4` pour `f64`).
    type Simd: SimdScalar<Scalar = Self>;
    /// Nombre de lanes de [`Self::Simd`].
    const WIDTH: usize;
    /// Zéro additif.
    const ZERO: Self;
    /// Élément neutre pour un maximum courant (−∞).
    const NEG_INFINITY: Self;
    /// Élément neutre pour un minimum courant (+∞).
    const INFINITY: Self;

    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    /// Maximum scalaire (sémantique `f{32,64}::max` : ignore un NaN isolé).
    fn max(self, other: Self) -> Self;
    /// Minimum scalaire.
    fn min(self, other: Self) -> Self;
}

macro_rules! impl_simd_reducible {
    ($ty:ty, $simd:ty, $width:literal) => {
        impl SimdReducible for $ty {
            type Simd = $simd;
            const WIDTH: usize = $width;
            const ZERO: Self = 0.0;
            const NEG_INFINITY: Self = <$ty>::NEG_INFINITY;
            const INFINITY: Self = <$ty>::INFINITY;

            #[inline(always)]
            fn add(self, other: Self) -> Self {
                self + other
            }
            #[inline(always)]
            fn sub(self, other: Self) -> Self {
                self - other
            }
            #[inline(always)]
            fn mul(self, other: Self) -> Self {
                self * other
            }
            #[inline(always)]
            fn div(self, other: Self) -> Self {
                self / other
            }
            #[inline(always)]
            fn abs(self) -> Self {
                <$ty>::abs(self)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                <$ty>::sqrt(self)
            }
            #[inline(always)]
            fn max(self, other: Self) -> Self {
                <$ty>::max(self, other)
            }
            #[inline(always)]
            fn min(self, other: Self) -> Self {
                <$ty>::min(self, other)
            }
        }
    };
}

// Largeurs par défaut : 256 bits sur les deux types (bon compromis portable
// AVX2/NEON-jumelé). Les kernels spécialisés peuvent viser d'autres largeurs
// directement via `SimdScalar`.
impl_simd_reducible!(f32, std::simd::f32x8, 8);
impl_simd_reducible!(f64, std::simd::f64x4, 4);

// ------------------------------------------------------------------ //
//  Réduction des lanes d'un accumulateur                              //
// ------------------------------------------------------------------ //

/// Réduit les lanes d'un accumulateur SIMD en un scalaire selon `mode`.
#[inline]
fn reduce_lanes<T: SimdReducible>(acc: T::Simd, mode: ReductionMode) -> T {
    match mode
    {
        ReductionMode::Fast => acc.reduce_sum(),
        ReductionMode::Deterministic =>
        {
            // Ordre d'indice fixe → indépendant du `reduce_sum` matériel.
            let mut total = T::ZERO;
            for i in 0..T::WIDTH
            {
                total = total.add(acc.lane(i));
            }
            total
        },
    }
}

// ------------------------------------------------------------------ //
//  Sommes                                                             //
// ------------------------------------------------------------------ //

/// Somme rapide : 4 accumulateurs SIMD (masquage de latence) + réduction
/// horizontale matérielle. Ordre non spécifié.
#[inline]
#[must_use]
pub fn sum_fast<T: SimdReducible>(data: &[T]) -> T {
    // 4 accumulateurs indépendants pour saturer les unités FMA/ADD.
    let mut acc = [T::Simd::zero(); 4];
    let mut chunks = data.chunks_exact(T::WIDTH);
    for (i, chunk) in chunks.by_ref().enumerate()
    {
        let v = T::Simd::from_slice(chunk);
        acc[i & 3] = acc[i & 3] + v;
    }
    let partial = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    let mut total = partial.reduce_sum();
    for &v in chunks.remainder()
    {
        total = total.add(v);
    }
    total
}

/// Somme déterministe : accumulation lane-parallèle puis réduction des lanes
/// et de la queue dans un ordre fixe. Reproductible bit à bit sur toute cible.
#[inline]
#[must_use]
pub fn sum_deterministic<T: SimdReducible>(data: &[T]) -> T {
    // Un seul accumulateur : lane `k` agrège les indices k, k+W, k+2W, … dans
    // l'ordre — déterministe. (Pas de multi-accumulateur : cela changerait
    // l'affectation lane→indices et donc le résultat.)
    let mut acc = T::Simd::zero();
    let mut chunks = data.chunks_exact(T::WIDTH);
    for chunk in chunks.by_ref()
    {
        acc = acc + T::Simd::from_slice(chunk);
    }
    let mut total = reduce_lanes::<T>(acc, ReductionMode::Deterministic);
    for &v in chunks.remainder()
    {
        total = total.add(v);
    }
    total
}

/// Un pas de Kahan–Neumaier scalaire : `*total += value`, compensé via `*c`.
#[inline(always)]
fn kahan_step<T: SimdReducible>(total: &mut T, c: &mut T, value: T) {
    let y = value.sub(*c);
    let t = total.add(y);
    // (t − total) − y capture l'erreur d'arrondi de `total + y`.
    *c = t.sub(*total).sub(y);
    *total = t;
}

/// Somme compensée de Kahan–Neumaier, lane-parallèle : réduit fortement
/// l'erreur d'arrondi accumulée sur les longues sommes. Déterministe.
#[inline]
#[must_use]
pub fn sum_kahan<T: SimdReducible>(data: &[T]) -> T {
    let mut sum = T::Simd::zero();
    let mut comp = T::Simd::zero(); // compensation courante (par lane)
    let mut chunks = data.chunks_exact(T::WIDTH);
    for chunk in chunks.by_ref()
    {
        let x = T::Simd::from_slice(chunk);
        let y = x - comp;
        let t = sum + y;
        // comp = (t − sum) − y : capture l'erreur d'arrondi de `sum + y`.
        comp = (t - sum) - y;
        sum = t;
    }
    // Réduction finale des lanes + queue en Kahan scalaire, ordre d'indice fixe.
    let mut total = T::ZERO;
    let mut c = T::ZERO;
    for i in 0..T::WIDTH
    {
        kahan_step(&mut total, &mut c, sum.lane(i));
    }
    for &v in chunks.remainder()
    {
        kahan_step(&mut total, &mut c, v);
    }
    total
}

/// Somme selon le régime choisi.
#[inline]
#[must_use]
pub fn sum<T: SimdReducible>(data: &[T], mode: ReductionMode) -> T {
    match mode
    {
        ReductionMode::Fast => sum_fast(data),
        ReductionMode::Deterministic => sum_deterministic(data),
    }
}

// ------------------------------------------------------------------ //
//  Produit scalaire & normes                                         //
// ------------------------------------------------------------------ //

/// Produit scalaire `⟨a, b⟩`. Panique si `a.len() != b.len()`.
///
/// Découpage `chunks_exact` identique sur `a` et `b` → appariement correct
/// quel que soit l'alignement des deux slices.
#[inline]
#[must_use]
pub fn dot<T: SimdReducible>(a: &[T], b: &[T], mode: ReductionMode) -> T {
    assert_eq!(a.len(), b.len(), "dot: longueurs différentes");
    let mut acc = T::Simd::zero();
    let mut ca = a.chunks_exact(T::WIDTH);
    let mut cb = b.chunks_exact(T::WIDTH);
    for (ka, kb) in ca.by_ref().zip(cb.by_ref())
    {
        let va = T::Simd::from_slice(ka);
        let vb = T::Simd::from_slice(kb);
        acc = va.mul_add(vb, acc); // acc += va*vb (FMA fusionnée)
    }
    let mut total = reduce_lanes::<T>(acc, mode);
    for (&x, &y) in ca.remainder().iter().zip(cb.remainder())
    {
        total = total.add(x.mul(y));
    }
    total
}

/// Norme L2 au carré `Σ aᵢ²` (= `dot(a, a)`).
#[inline]
#[must_use]
pub fn l2_norm_sqr<T: SimdReducible>(a: &[T], mode: ReductionMode) -> T {
    let mut acc = T::Simd::zero();
    let mut chunks = a.chunks_exact(T::WIDTH);
    for chunk in chunks.by_ref()
    {
        let v = T::Simd::from_slice(chunk);
        acc = v.mul_add(v, acc);
    }
    let mut total = reduce_lanes::<T>(acc, mode);
    for &v in chunks.remainder()
    {
        total = total.add(v.mul(v));
    }
    total
}

/// Norme L2 (euclidienne) `√Σ aᵢ²`.
#[inline]
#[must_use]
pub fn l2_norm<T: SimdReducible>(a: &[T], mode: ReductionMode) -> T {
    l2_norm_sqr(a, mode).sqrt()
}

/// Norme L1 `Σ |aᵢ|`.
#[inline]
#[must_use]
pub fn l1_norm<T: SimdReducible>(a: &[T], mode: ReductionMode) -> T {
    let mut acc = T::Simd::zero();
    let mut chunks = a.chunks_exact(T::WIDTH);
    for chunk in chunks.by_ref()
    {
        acc = acc + T::Simd::from_slice(chunk).abs();
    }
    let mut total = reduce_lanes::<T>(acc, mode);
    for &v in chunks.remainder()
    {
        total = total.add(v.abs());
    }
    total
}

/// Norme L∞ (maximum absolu) `maxᵢ |aᵢ|`, ou `T::ZERO` si `a` est vide
/// (convention : norme du vecteur nul).
#[inline]
#[must_use]
pub fn linf_norm<T: SimdReducible>(a: &[T]) -> T {
    if a.is_empty()
    {
        return T::ZERO;
    }
    let mut chunks = a.chunks_exact(T::WIDTH);
    let mut acc: Option<T::Simd> = None;
    for chunk in chunks.by_ref()
    {
        let v = T::Simd::from_slice(chunk).abs();
        acc = Some(match acc
        {
            Some(a) => a.simd_max(v),
            None => v,
        });
    }
    let mut best = acc.map(T::Simd::reduce_max).unwrap_or(T::ZERO);
    for &v in chunks.remainder()
    {
        best = best.max(v.abs());
    }
    best
}

/// Similarité cosinus `⟨a, b⟩ / (‖a‖·‖b‖)`. Panique si longueurs différentes.
/// Renvoie `0` si l'une des normes est nulle (vecteur nul → similarité indéfinie,
/// convention pragmatique pour le routage de représentations).
#[inline]
#[must_use]
pub fn cosine_similarity<T: SimdReducible>(a: &[T], b: &[T], mode: ReductionMode) -> T {
    let dot_ab = dot(a, b, mode);
    let na = l2_norm(a, mode);
    let nb = l2_norm(b, mode);
    let denom = na.mul(nb);
    if denom == T::ZERO
    {
        T::ZERO
    }
    else
    {
        dot_ab.div(denom)
    }
}

// ------------------------------------------------------------------ //
//  Extrema                                                            //
// ------------------------------------------------------------------ //

/// Maximum des éléments, ou `None` si vide.
#[inline]
#[must_use]
pub fn reduce_max<T: SimdReducible>(data: &[T]) -> Option<T> {
    if data.is_empty()
    {
        return None;
    }
    let mut chunks = data.chunks_exact(T::WIDTH);
    let mut acc: Option<T::Simd> = None;
    for chunk in chunks.by_ref()
    {
        let v = T::Simd::from_slice(chunk);
        acc = Some(match acc
        {
            Some(a) => a.simd_max(v),
            None => v,
        });
    }
    let mut best = acc.map(|v| v.reduce_max());
    for &v in chunks.remainder()
    {
        best = Some(match best
        {
            Some(m) => m.max(v),
            None => v,
        });
    }
    best
}

/// Minimum des éléments, ou `None` si vide.
#[inline]
#[must_use]
pub fn reduce_min<T: SimdReducible>(data: &[T]) -> Option<T> {
    if data.is_empty()
    {
        return None;
    }
    let mut chunks = data.chunks_exact(T::WIDTH);
    let mut acc: Option<T::Simd> = None;
    for chunk in chunks.by_ref()
    {
        let v = T::Simd::from_slice(chunk);
        acc = Some(match acc
        {
            Some(a) => a.simd_min(v),
            None => v,
        });
    }
    let mut best = acc.map(|v| v.reduce_min());
    for &v in chunks.remainder()
    {
        best = Some(match best
        {
            Some(m) => m.min(v),
            None => v,
        });
    }
    best
}

/// Indice du **premier** maximum, ou `None` si vide.
///
/// Deux passes : maximum SIMD, puis balayage linéaire du premier indice égal.
/// Sémantique NaN volontairement simple (les NaN sont ignorés par `simd_max` ;
/// un tableau tout-NaN renvoie l'indice 0 par le repli du balayage).
#[inline]
#[must_use]
pub fn argmax<T: SimdReducible>(data: &[T]) -> Option<usize> {
    let m = reduce_max(data)?;
    // Premier indice égal au max ; repli sur 0 si aucun (ex. tout-NaN).
    Some(data.iter().position(|&x| x == m).unwrap_or(0))
}

/// Indice du **premier** minimum, ou `None` si vide.
///
/// Même schéma qu'[`argmax`] : minimum SIMD puis balayage linéaire du
/// premier indice égal.
#[inline]
#[must_use]
pub fn argmin<T: SimdReducible>(data: &[T]) -> Option<usize> {
    let m = reduce_min(data)?;
    Some(data.iter().position(|&x| x == m).unwrap_or(0))
}
