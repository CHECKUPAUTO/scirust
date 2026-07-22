// scirust-simd/src/dsp/kalman.rs
//
// # Filtre de Kalman [`KalmanFilter`] — linéaire et étendu (EKF)
//
// Estimateur récursif bayésien de l'état d'un système dynamique bruité, à
// partir d'un modèle de **transition** (comment l'état évolue) et d'un
// modèle de **mesure** (comment l'état se reflète dans les observations).
// Contrairement à [`super::adaptive`] (aucun modèle a priori, coefficients
// appris uniquement à partir de l'erreur instantanée), le filtre de Kalman
// **connaît** ces deux modèles et calcule l'estimateur de variance minimale
// à chaque étape — le « gain de Kalman » que [`super::adaptive::Rls`]
// mentionne déjà dans son commentaire de mise à jour de covariance (RLS
// *est*, historiquement, un cas particulier de filtre de Kalman appliqué à
// un état stationnaire).
//
// **Générique sur le scalaire** comme le reste de `dsp` : la même
// implémentation sert `f32`/`f64` et la virgule fixe déterministe
// (`FixedI32<FRAC>`) — un filtre de Kalman en virgule fixe converge vers les
// **mêmes bits** sur toute architecture, propriété précieuse pour rejouer
// exactement une trajectoire de suivi (robotique, navigation embarquée).
//
// ## Deux étapes : prédiction et mise à jour
//
// * [`KalmanFilter::predict`] (temporel) : `x ← F·x`, `P ← F·P·Fᵀ + Q` — fait
//   avancer l'estimation d'un pas, sans nouvelle observation. `Q` (bruit de
//   processus) grandit l'incertitude : le modèle seul ne peut jamais la
//   réduire.
// * [`KalmanFilter::update`] (mesure) : incorpore une observation `z = H·x +
//   bruit`. Calcule l'innovation `y = z − H·x`, le gain `K = P·Hᵀ·S⁻¹`
//   (`S = H·P·Hᵀ + R` : covariance de l'innovation), corrige `x ← x + K·y` et
//   réduit `P` en conséquence — c'est la seule étape qui **diminue**
//   l'incertitude.
//
// ## EKF — [`KalmanFilter::predict_nonlinear`]/[`KalmanFilter::update_nonlinear`]
//
// Le filtre de Kalman *linéaire* suppose transition et mesure linéaires
// (`F`/`H` constantes). Le filtre de Kalman **étendu** (EKF) généralise aux
// modèles non linéaires (`x ↦ f(x)`, `x ↦ h(x)`) en les **linéarisant** à
// chaque pas autour de l'estimée courante : l'appelant fournit `f`/`h`
// **et** leurs jacobiennes `F = ∂f/∂x`, `H = ∂h/∂x` évaluées en `x` — la
// propagation de covariance utilise alors exactement les mêmes formules que
// le filtre linéaire, avec `F`/`H` en tant que jacobiennes plutôt que
// matrices constantes. [`KalmanFilter::predict`]/[`KalmanFilter::update`] ne
// sont que les cas particuliers `f : x ↦ F·x`, `h : x ↦ H·x` (jacobienne =
// la matrice elle-même) — implémentés en termes des méthodes non linéaires,
// pas dupliqués.
//
// ## Mise à jour de covariance de Joseph
//
// [`KalmanFilter::update_nonlinear`] met à jour `P` par la forme de Joseph
// `P ← (I−K·H)·P·(I−K·H)ᵀ + K·R·Kᵀ` plutôt que la forme simplifiée
// `P ← (I−K·H)·P` (mathématiquement équivalente si `K` est le gain optimal
// exact, mais qui ne garantit plus `P` symétrique semi-définie positive sous
// arrondi/linéarisation approchée — la forme de Joseph le garantit toujours,
// par construction : c'est une somme de deux termes `X·P·Xᵀ` et `K·R·Kᵀ`,
// chacun SDP si `P`/`R` le sont).
//
// ## Inversion de covariance sans pivot
//
// [`KalmanFilter::update_nonlinear`] inverse `S = H·P·Hᵀ + R` via une
// décomposition de Cholesky (`S` garantie symétrique définie positive si `R`
// l'est et `P` semi-définie positive, elle-même maintenue par la mise à jour
// de Joseph ci-dessus) plutôt qu'une élimination de Gauss-Jordan générale :
// aucun choix de pivot à faire (la diagonale de Cholesky reste positive par
// construction pour une entrée réellement SDP), donc déterministe et stable
// pour `f32`/`f64` **et** virgule fixe sans introduire de comparaison de
// magnitude entre lignes.
//
// ## [`UnscentedKalmanFilter`] — transformée non parfumée (UKF)
//
// L'EKF linéarise `f`/`h` par leur jacobienne au premier ordre — approximation
// dégradée si `f`/`h` sont fortement non linéaires, et qui demande à
// l'appelant de fournir cette jacobienne. [`UnscentedKalmanFilter`] évite les
// deux : il propage un petit ensemble de « points sigma » (`2N+1`, choisis
// pour représenter exactement la moyenne et la covariance de l'état courant)
// **directement** à travers `f`/`h` non linéaires (aucune dérivée requise),
// puis reconstruit moyenne/covariance à partir de leurs images pondérées —
// la « transformée non parfumée » (*unscented transform*), exacte au second
// ordre pour toute non-linéarité (l'EKF n'est exact qu'au premier ordre).
//
// Pour un système **linéaire**, la transformée non parfumée reproduit
// *exactement* `F·x`/`F·P·Fᵀ` (vérifié par
// `ukf_matches_linear_kalman_filter_on_linear_system` dans les tests) : l'UKF
// n'est jamais moins précis que le filtre linéaire, seulement plus coûteux
// (`2N+1` évaluations de `f`/`h` par étape contre une seule).
//
// Réutilise directement l'algèbre matricielle et [`cholesky`]/[`invert_spd`]
// ci-dessus (les points sigma sont construits à partir des colonnes du
// facteur de Cholesky de `P`) — aucune nouvelle primitive numérique.

use core::ops::Div;

use crate::fixed::RealScalar;

// ------------------------------------------------------------------ //
//  Petite algèbre matricielle générique (dimensions = const generics)  //
// ------------------------------------------------------------------ //

/// Produit matriciel `A·B` (`A` : `R×K`, `B` : `K×C`, row-major, tableaux
/// imbriqués `[[T; colonnes]; lignes]`).
fn matmul<T: RealScalar, const R: usize, const K: usize, const C: usize>(
    a: &[[T; K]; R],
    b: &[[T; C]; K],
) -> [[T; C]; R] {
    let mut out = [[T::zero(); C]; R];
    for i in 0..R
    {
        for j in 0..C
        {
            let mut acc = T::zero();
            for k in 0..K
            {
                acc = acc + a[i][k] * b[k][j];
            }
            out[i][j] = acc;
        }
    }
    out
}

/// Transposée `Aᵀ`.
fn transpose<T: RealScalar, const R: usize, const C: usize>(a: &[[T; C]; R]) -> [[T; R]; C] {
    let mut out = [[T::zero(); R]; C];
    for i in 0..R
    {
        for j in 0..C
        {
            out[j][i] = a[i][j];
        }
    }
    out
}

/// Produit matrice-vecteur `A·v`.
fn matvec<T: RealScalar, const R: usize, const C: usize>(a: &[[T; C]; R], v: &[T; C]) -> [T; R] {
    let mut out = [T::zero(); R];
    for i in 0..R
    {
        let mut acc = T::zero();
        for j in 0..C
        {
            acc = acc + a[i][j] * v[j];
        }
        out[i] = acc;
    }
    out
}

/// Somme matricielle `A + B`.
fn matadd<T: RealScalar, const R: usize, const C: usize>(
    a: &[[T; C]; R],
    b: &[[T; C]; R],
) -> [[T; C]; R] {
    let mut out = [[T::zero(); C]; R];
    for i in 0..R
    {
        for j in 0..C
        {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

/// Différence matricielle `A − B`.
fn matsub<T: RealScalar, const R: usize, const C: usize>(
    a: &[[T; C]; R],
    b: &[[T; C]; R],
) -> [[T; C]; R] {
    let mut out = [[T::zero(); C]; R];
    for i in 0..R
    {
        for j in 0..C
        {
            out[i][j] = a[i][j] - b[i][j];
        }
    }
    out
}

/// Somme vectorielle `a + b`.
fn vecadd<T: RealScalar, const N: usize>(a: &[T; N], b: &[T; N]) -> [T; N] {
    let mut out = [T::zero(); N];
    for i in 0..N
    {
        out[i] = a[i] + b[i];
    }
    out
}

/// Différence vectorielle `a − b`.
fn vecsub<T: RealScalar, const N: usize>(a: &[T; N], b: &[T; N]) -> [T; N] {
    let mut out = [T::zero(); N];
    for i in 0..N
    {
        out[i] = a[i] - b[i];
    }
    out
}

/// Matrice identité `N×N`.
fn identity<T: RealScalar, const N: usize>() -> [[T; N]; N] {
    let mut out = [[T::zero(); N]; N];
    for (i, row) in out.iter_mut().enumerate()
    {
        row[i] = T::one();
    }
    out
}

/// Décomposition de Cholesky `A = L·Lᵀ` (`L` triangulaire inférieure),
/// `None` si `A` n'est pas symétrique définie positive (pivot `≤ 0`
/// rencontré) — cf. [`invert_spd`].
#[allow(clippy::needless_range_loop)] // `k` indexe deux lignes distinctes (`l[i]`, `l[j]`).
fn cholesky<T: RealScalar + Div<Output = T>, const N: usize>(
    a: &[[T; N]; N],
) -> Option<[[T; N]; N]> {
    let mut l = [[T::zero(); N]; N];
    for i in 0..N
    {
        for j in 0..=i
        {
            let mut sum = T::zero();
            for k in 0..j
            {
                sum = sum + l[i][k] * l[j][k];
            }
            if i == j
            {
                let diag = a[i][i] - sum;
                if diag <= T::zero()
                {
                    return None;
                }
                l[i][j] = diag.sqrt();
            }
            else
            {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    Some(l)
}

/// Inverse d'une matrice triangulaire inférieure à diagonale strictement
/// positive (garantie par [`cholesky`]), par substitution avant.
#[allow(clippy::needless_range_loop)] // `j` indexe à la fois `inv[i][j]` et `l[i][j]`/`inv[k][j]`.
fn invert_lower_triangular<T: RealScalar + Div<Output = T>, const N: usize>(
    l: &[[T; N]; N],
) -> [[T; N]; N] {
    let mut inv = [[T::zero(); N]; N];
    for i in 0..N
    {
        inv[i][i] = T::one() / l[i][i];
        for j in 0..i
        {
            let mut sum = T::zero();
            for k in j..i
            {
                sum = sum + l[i][k] * inv[k][j];
            }
            inv[i][j] = -sum / l[i][i];
        }
    }
    inv
}

/// Inverse `A⁻¹` d'une matrice **symétrique définie positive**, via Cholesky
/// (`A⁻¹ = (L⁻¹)ᵀ·L⁻¹`) — cf. en-tête de module pour la justification de ce
/// choix face à une élimination de Gauss-Jordan générale. `None` si `A`
/// n'est pas SDP.
fn invert_spd<T: RealScalar + Div<Output = T>, const N: usize>(
    a: &[[T; N]; N],
) -> Option<[[T; N]; N]> {
    let l = cholesky(a)?;
    let l_inv = invert_lower_triangular(&l);
    Some(matmul(&transpose(&l_inv), &l_inv))
}

// ------------------------------------------------------------------ //
//  KalmanFilter<T, N, M>                                              //
// ------------------------------------------------------------------ //

/// Filtre de Kalman (état de dimension `N`, mesure de dimension `M`),
/// linéaire et étendu (EKF) — cf. en-tête de module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KalmanFilter<T, const N: usize, const M: usize> {
    x: [T; N],
    p: [[T; N]; N],
}

impl<T: RealScalar + Div<Output = T>, const N: usize, const M: usize> KalmanFilter<T, N, M> {
    /// Construit depuis un état initial `x0` et une covariance initiale `p0`
    /// (typiquement diagonale ; grande si l'état initial est peu fiable).
    #[inline]
    pub fn new(x0: [T; N], p0: [[T; N]; N]) -> Self {
        Self { x: x0, p: p0 }
    }

    /// État estimé courant.
    #[inline]
    pub fn state(&self) -> &[T; N] {
        &self.x
    }

    /// Covariance d'erreur courante.
    #[inline]
    pub fn covariance(&self) -> &[[T; N]; N] {
        &self.p
    }

    /// Prédiction **non linéaire** (EKF) : `x ← transition(x)`,
    /// `P ← J·P·Jᵀ + Q` (`J` = jacobienne de `transition` en `x`, fournie par
    /// l'appelant). [`Self::predict`] est le cas particulier linéaire
    /// (cf. en-tête de module).
    pub fn predict_nonlinear(
        &mut self,
        transition: impl Fn(&[T; N]) -> [T; N],
        jacobian: &[[T; N]; N],
        q: &[[T; N]; N],
    ) {
        self.x = transition(&self.x);
        let jp = matmul(jacobian, &self.p);
        let jpjt = matmul(&jp, &transpose(jacobian));
        self.p = matadd(&jpjt, q);
    }

    /// Prédiction linéaire : `x ← F·x`, `P ← F·P·Fᵀ + Q` (`Q` bruit de
    /// processus, symétrique semi-définie positive).
    #[inline]
    pub fn predict(&mut self, f: &[[T; N]; N], q: &[[T; N]; N]) {
        self.predict_nonlinear(|x| matvec(f, x), f, q);
    }

    /// Mise à jour **non linéaire** (EKF) à partir d'une observation `z` :
    /// innovation `y = z − measurement(x)`, gain `K = P·Jᵀ·S⁻¹`
    /// (`S = J·P·Jᵀ + R`, `J` = jacobienne de `measurement` en `x`), mise à
    /// jour de Joseph de `P` (cf. en-tête de module). [`Self::update`] est le
    /// cas particulier linéaire.
    ///
    /// `None` si `S` n'est pas inversible (mesure dégénérée ou `R` mal
    /// conditionné) — **l'état n'est alors pas modifié**.
    pub fn update_nonlinear(
        &mut self,
        z: &[T; M],
        measurement: impl Fn(&[T; N]) -> [T; M],
        jacobian: &[[T; N]; M],
        r: &[[T; M]; M],
    ) -> Option<()> {
        let innovation = vecsub(z, &measurement(&self.x));
        let p_jt = matmul(&self.p, &transpose(jacobian));
        let jp_jt = matmul(jacobian, &p_jt);
        let s = matadd(&jp_jt, r);
        let s_inv = invert_spd(&s)?;
        let k = matmul(&p_jt, &s_inv);

        self.x = vecadd(&self.x, &matvec(&k, &innovation));

        let kj = matmul(&k, jacobian);
        let i_minus_kj = matsub(&identity::<T, N>(), &kj);
        let joseph = matmul(&matmul(&i_minus_kj, &self.p), &transpose(&i_minus_kj));
        let kr = matmul(&k, r);
        let krkt = matmul(&kr, &transpose(&k));
        self.p = matadd(&joseph, &krkt);
        Some(())
    }

    /// Mise à jour linéaire : mesure `z = H·x + bruit`, `H` matrice
    /// d'observation (`M×N`), `R` bruit de mesure (`M×M`, symétrique définie
    /// positive). `None` si `S = H·P·Hᵀ + R` n'est pas inversible.
    #[inline]
    pub fn update(&mut self, z: &[T; M], h: &[[T; N]; M], r: &[[T; M]; M]) -> Option<()> {
        self.update_nonlinear(z, |x| matvec(h, x), h, r)
    }
}

// ------------------------------------------------------------------ //
//  UnscentedKalmanFilter<T, N, M>                                      //
// ------------------------------------------------------------------ //

/// `2N+1` points sigma représentant exactement la moyenne `x` et la
/// covariance `p` (Julier & Uhlmann) : le point central `x`, puis pour
/// chaque colonne `c` du facteur de Cholesky `L` de `p` (mis à l'échelle par
/// `√(N+λ)`), les deux points `x ± c`. `None` si `p` n'est pas symétrique
/// définie positive.
#[allow(clippy::needless_range_loop)]
fn sigma_points<T: RealScalar + Div<Output = T>, const N: usize>(
    x: &[T; N],
    p: &[[T; N]; N],
    lambda: T,
) -> Option<Vec<[T; N]>> {
    let l = cholesky(p)?;
    let scale = (T::from_i32(N as i32) + lambda).sqrt();

    let mut points = Vec::with_capacity(2 * N + 1);
    points.push(*x);
    for j in 0..N
    {
        let mut plus = *x;
        let mut minus = *x;
        for i in 0..N
        {
            let c = l[i][j] * scale;
            plus[i] = plus[i] + c;
            minus[i] = minus[i] - c;
        }
        points.push(plus);
        points.push(minus);
    }
    Some(points)
}

/// Poids de la transformée non parfumée : `(Wm₀, Wc₀, Wᵢ)` — `Wm₀`/`Wc₀`
/// pour le point sigma central (index `0`), `Wᵢ` commun aux `2N` points
/// restants (`Wm et Wc coïncident pour ceux-là`).
fn ukf_weights<T: RealScalar + Div<Output = T>, const N: usize>(
    alpha: T,
    beta: T,
    lambda: T,
) -> (T, T, T) {
    let n_plus_lambda = T::from_i32(N as i32) + lambda;
    let wm0 = lambda / n_plus_lambda;
    let wc0 = wm0 + (T::one() - alpha * alpha + beta);
    let wi = T::from_i32(2).recip() / n_plus_lambda; // puissance de 2 : recip() exact avant la division.
    (wm0, wc0, wi)
}

/// Filtre de Kalman **non parfumé** (état de dimension `N`, mesure de
/// dimension `M`), sans jacobienne — cf. en-tête de module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnscentedKalmanFilter<T, const N: usize, const M: usize> {
    x: [T; N],
    p: [[T; N]; N],
    alpha: T,
    beta: T,
    kappa: T,
}

impl<T: RealScalar + Div<Output = T>, const N: usize, const M: usize>
    UnscentedKalmanFilter<T, N, M>
{
    /// Construit depuis un état initial `x0`/covariance `p0` et les
    /// paramètres de la transformée non parfumée : `alpha` (étalement des
    /// points sigma, typiquement `1e-3`–`1`), `beta` (connaissance a priori
    /// de la distribution ; `2` pour une gaussienne), `kappa` (mise à
    /// l'échelle secondaire, souvent `0` ou `3−N`).
    #[inline]
    pub fn new(x0: [T; N], p0: [[T; N]; N], alpha: T, beta: T, kappa: T) -> Self {
        Self {
            x: x0,
            p: p0,
            alpha,
            beta,
            kappa,
        }
    }

    /// État estimé courant.
    #[inline]
    pub fn state(&self) -> &[T; N] {
        &self.x
    }

    /// Covariance d'erreur courante.
    #[inline]
    pub fn covariance(&self) -> &[[T; N]; N] {
        &self.p
    }

    #[inline]
    fn lambda(&self) -> T {
        self.alpha * self.alpha * (T::from_i32(N as i32) + self.kappa) - T::from_i32(N as i32)
    }

    /// Prédiction : propage les points sigma directement à travers la
    /// transition non linéaire `transition` (aucune jacobienne), reconstruit
    /// `x`/`P` à partir de leurs images pondérées, ajoute le bruit de
    /// processus `q`. `None` si `P` courante n'est pas SDP (points sigma
    /// indéfinis) — cf. en-tête de module.
    pub fn predict(
        &mut self,
        transition: impl Fn(&[T; N]) -> [T; N],
        q: &[[T; N]; N],
    ) -> Option<()> {
        let lambda = self.lambda();
        let (wm0, wc0, wi) = ukf_weights::<T, N>(self.alpha, self.beta, lambda);
        let points = sigma_points(&self.x, &self.p, lambda)?;
        let propagated: Vec<[T; N]> = points.iter().map(transition).collect();

        let mut x_new = [T::zero(); N];
        for (k, pt) in propagated.iter().enumerate()
        {
            let w = if k == 0 { wm0 } else { wi };
            for i in 0..N
            {
                x_new[i] = x_new[i] + w * pt[i];
            }
        }

        let mut p_new = [[T::zero(); N]; N];
        for (k, pt) in propagated.iter().enumerate()
        {
            let w = if k == 0 { wc0 } else { wi };
            let d = vecsub(pt, &x_new);
            for i in 0..N
            {
                for j in 0..N
                {
                    p_new[i][j] = p_new[i][j] + w * d[i] * d[j];
                }
            }
        }

        self.x = x_new;
        self.p = matadd(&p_new, q);
        Some(())
    }

    /// Mise à jour à partir d'une observation `z` : régénère les points
    /// sigma depuis l'état prédit, les transforme par la mesure non linéaire
    /// `measurement` (aucune jacobienne), calcule la covariance d'innovation
    /// `S`, la covariance croisée état/mesure et le gain `K = Pxz·S⁻¹`.
    ///
    /// `None` si `S` n'est pas inversible ou si `P` courante n'est pas SDP —
    /// **l'état n'est alors pas modifié**.
    pub fn update(
        &mut self,
        z: &[T; M],
        measurement: impl Fn(&[T; N]) -> [T; M],
        r: &[[T; M]; M],
    ) -> Option<()> {
        let lambda = self.lambda();
        let (wm0, wc0, wi) = ukf_weights::<T, N>(self.alpha, self.beta, lambda);
        let points = sigma_points(&self.x, &self.p, lambda)?;
        let measured: Vec<[T; M]> = points.iter().map(measurement).collect();

        let mut z_hat = [T::zero(); M];
        for (k, m) in measured.iter().enumerate()
        {
            let w = if k == 0 { wm0 } else { wi };
            for i in 0..M
            {
                z_hat[i] = z_hat[i] + w * m[i];
            }
        }

        let mut s = [[T::zero(); M]; M];
        let mut pxz = [[T::zero(); M]; N];
        for (k, (pt, m)) in points.iter().zip(&measured).enumerate()
        {
            let w = if k == 0 { wc0 } else { wi };
            let dz = vecsub(m, &z_hat);
            let dx = vecsub(pt, &self.x);
            for i in 0..M
            {
                for j in 0..M
                {
                    s[i][j] = s[i][j] + w * dz[i] * dz[j];
                }
            }
            for i in 0..N
            {
                for j in 0..M
                {
                    pxz[i][j] = pxz[i][j] + w * dx[i] * dz[j];
                }
            }
        }
        let s = matadd(&s, r);
        let s_inv = invert_spd(&s)?;
        let k_gain = matmul(&pxz, &s_inv);

        let innovation = vecsub(z, &z_hat);
        self.x = vecadd(&self.x, &matvec(&k_gain, &innovation));

        let ks_kt = matmul(&matmul(&k_gain, &s), &transpose(&k_gain));
        self.p = matsub(&self.p, &ks_kt);
        Some(())
    }
}
