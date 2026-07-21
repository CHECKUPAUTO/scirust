// scirust-simd/src/geometry/ahrs.rs
//
// # Filtres d'attitude (AHRS) [`MadgwickFilter`] / [`MahonyFilter`]
//
// Fusion de capteurs inertiels (gyroscope + accéléromètre, optionnellement
// magnétomètre) en une estimée d'orientation `Quaternion<T>` — l'algorithme
// standard derrière tout contrôleur de vol de drone, carte AHRS/IMU, ou stack
// d'orientation robotique. Complète [`super::Quaternion`] (représentation et
// composition de rotations, déjà génériques) par la **fusion** proprement
// dite : combiner une intégration gyroscopique (dérive lentement, précise à
// court terme) avec une correction par accéléromètre/magnétomètre (bruitée
// mais sans dérive) pour obtenir une estimée stable à long terme.
//
// ## Convention
//
// `q` représente la rotation **du repère du corps vers le repère monde**
// (même convention que [`Quaternion::rotate_vector`] : `rotate_vector` fait
// passer un vecteur du repère corps au repère monde). L'accéléromètre est
// supposé mesurer, au repos, la direction de la gravité **dans le repère
// corps** (`≈ Rᵀ(q)·[0,0,1]`, `[0,0,1]` = « haut » en repère monde) ; le
// magnétomètre mesure de même le champ magnétique terrestre en repère corps.
//
// ## [`MadgwickFilter`] — descente de gradient (Madgwick, 2010)
//
// Minimise `f(q) = ‖Rᵀ(q)·[0,0,1] − a‖²` (+ un terme magnétique en mode
// MARG) par descente de gradient : `q̇ = q̇_gyro − β·∇f/‖∇f‖`. Le gradient
// analytique de Madgwick (dérivées partielles d'un polynôme en `w,x,y,z`)
// est bien connu mais long et sujet à erreur de transcription pour le cas
// MARG à 6 résidus (accéléromètre + magnétomètre) ; ce module calcule le
// gradient par **différence centrée numérique** sur les 4 composantes brutes
// de `q` (non renormalisées pendant la perturbation — cohérent avec la
// formulation d'origine, qui dérive par rapport aux composantes du
// quaternion, pas de sa version normalisée) : `Rᵀ(q)·v` n'étant qu'une
// expression polynomiale de `w,x,y,z` (aucune racine, aucune branche), elle
// est lisse pour tout `q`, unitaire ou non — la différence finie est donc
// exacte à l'ordre de troncature près, sans risque de transcription d'une
// jacobienne à 6 lignes. Coût : 8 évaluations supplémentaires de `f` par
// échantillon, négligeable aux fréquences IMU typiques (100 Hz – 1 kHz).
//
// ## [`MahonyFilter`] — filtre complémentaire explicite (Mahony et al., 2008)
//
// Corrige directement le gyroscope par un terme d'erreur vectoriel
// `e = a × â` (produit vectoriel entre la mesure et la prédiction, `â` la
// direction de gravité prédite en repère corps), avec rétroaction
// proportionnelle-intégrale (`Kp·e` immédiat, `Ki·∫e·dt` accumulé dans un
// **biais gyroscopique estimé** qui absorbe une dérive constante du
// gyroscope — cf. [`MahonyFilter::bias`]). Entièrement analytique (aucune
// dérivée partielle requise, juste des produits vectoriels), donc à faible
// risque d'implémentation même en mode MARG.
//
// ## IMU (6 ddl) vs MARG (9 ddl) — pourquoi le magnétomètre ?
//
// La gravité seule ne contraint que le roulis/tangage : une rotation pure
// autour de l'axe vertical (lacet) laisse `Rᵀ(q)·[0,0,1]` **inchangé**, donc
// invisible à l'accéléromètre (propriété vérifiée par
// `*_imu_only_cannot_observe_yaw` dans les tests). Le magnétomètre lève cette
// ambiguïté : [`MadgwickFilter::update_marg`]/[`MahonyFilter::update_marg`]
// utilisent une référence magnétique **compensée en inclinaison**
// (composante horizontale ambiguë annulée, seule la verticale — déjà connue
// via l'accéléromètre — et l'horizontale totale sont conservées) calculée à
// partir de l'estimée courante, technique standard partagée par les deux
// algorithmes (cf. [`tilt_compensated_reference`]).

use core::ops::Div;

use crate::fixed::RealScalar;

use super::quaternion::Quaternion;

/// Produit vectoriel `a × b` (fonction libre, cf. [`super::quaternion`]/
/// [`super::dual_quaternion`] : même technique, reproduite localement).
#[inline]
fn cross<T: RealScalar>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalise `v`, `None` si `v` est (numériquement) nul.
#[inline]
fn normalize3<T: RealScalar>(v: [T; 3]) -> Option<[T; 3]> {
    let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if n2 <= T::zero()
    {
        return None;
    }
    let inv = n2.sqrt().recip();
    Some([v[0] * inv, v[1] * inv, v[2] * inv])
}

/// Référence magnétique **compensée en inclinaison** : tourne la mesure de
/// magnétomètre normalisée `mag` (repère corps) en repère monde via
/// l'estimée courante `q`, puis annule l'ambiguïté horizontale (seule la
/// magnitude horizontale totale et la composante verticale sont conservées)
/// — cf. en-tête de module.
#[inline]
fn tilt_compensated_reference<T: RealScalar>(q: Quaternion<T>, mag: [T; 3]) -> [T; 3] {
    let h = q.rotate_vector(mag);
    let bxy = (h[0] * h[0] + h[1] * h[1]).sqrt();
    [bxy, T::zero(), h[2]]
}

// ------------------------------------------------------------------ //
//  MadgwickFilter                                                     //
// ------------------------------------------------------------------ //

/// Filtre d'attitude de Madgwick (descente de gradient) — cf. en-tête de
/// module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MadgwickFilter<T> {
    q: Quaternion<T>,
    beta: T,
}

/// Gradient numérique (différence centrée) du coût scalaire `cost` par
/// rapport aux 4 composantes brutes `w,x,y,z` de `q`.
fn numeric_gradient4<T: RealScalar + Div<Output = T>>(
    cost: impl Fn(Quaternion<T>) -> T,
    q: Quaternion<T>,
    eps: T,
) -> [T; 4] {
    let two_eps = eps + eps;
    let comps = [q.w, q.x, q.y, q.z];
    let mut g = [T::zero(); 4];
    for i in 0..4
    {
        let mut plus = comps;
        let mut minus = comps;
        plus[i] = plus[i] + eps;
        minus[i] = minus[i] - eps;
        let qp = Quaternion::new(plus[0], plus[1], plus[2], plus[3]);
        let qm = Quaternion::new(minus[0], minus[1], minus[2], minus[3]);
        g[i] = (cost(qp) - cost(qm)) / two_eps;
    }
    g
}

/// Normalise un vecteur à 4 composantes, `None` si numériquement nul.
fn normalize4<T: RealScalar>(v: [T; 4]) -> Option<[T; 4]> {
    let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    if n2 <= T::zero()
    {
        return None;
    }
    let inv = n2.sqrt().recip();
    Some([v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv])
}

/// Combine la dérivée gyroscopique `qdot_gyro` avec le pas de correction par
/// descente de gradient (gradient numérique de `cost`, normalisé en
/// direction unitaire) — cf. en-tête de module.
fn gradient_correction<T: RealScalar + Div<Output = T>>(
    qdot_gyro: Quaternion<T>,
    q: Quaternion<T>,
    beta: T,
    cost: impl Fn(Quaternion<T>) -> T,
) -> Quaternion<T> {
    // 1e-3 : au-dessus de la résolution Q16.16 (≈ 1,5e-5) pour éviter qu'une
    // perturbation ±eps sur une composante de q (typiquement dans [-1, 1])
    // ne s'arrondisse à zéro, tout en restant assez petite pour une
    // différence finie fidèle.
    let eps = T::from_i32(1000).recip();
    let grad = numeric_gradient4(cost, q, eps);
    match normalize4(grad)
    {
        Some(g) => Quaternion::new(
            qdot_gyro.w - beta * g[0],
            qdot_gyro.x - beta * g[1],
            qdot_gyro.y - beta * g[2],
            qdot_gyro.z - beta * g[3],
        ),
        None => qdot_gyro,
    }
}

impl<T: RealScalar + Div<Output = T>> MadgwickFilter<T> {
    /// Construit avec l'orientation identité et le gain `beta` (typiquement
    /// `0.03`–`0.1` : plus grand accélère la convergence mais augmente la
    /// sensibilité au bruit de l'accéléromètre/magnétomètre).
    #[inline]
    pub fn new(beta: T) -> Self {
        Self {
            q: Quaternion::identity(),
            beta,
        }
    }

    /// Orientation estimée courante.
    #[inline]
    pub fn orientation(&self) -> Quaternion<T> {
        self.q
    }

    /// Remet l'orientation à l'identité.
    #[inline]
    pub fn reset(&mut self) {
        self.q = Quaternion::identity();
    }

    fn cost_imu(q: Quaternion<T>, accel: [T; 3]) -> T {
        let v = q
            .conjugate()
            .rotate_vector([T::zero(), T::zero(), T::one()]);
        let f = [v[0] - accel[0], v[1] - accel[1], v[2] - accel[2]];
        f[0] * f[0] + f[1] * f[1] + f[2] * f[2]
    }

    fn cost_marg(q: Quaternion<T>, accel: [T; 3], b_ref: [T; 3], mag: [T; 3]) -> T {
        let v = q
            .conjugate()
            .rotate_vector([T::zero(), T::zero(), T::one()]);
        let w = q.conjugate().rotate_vector(b_ref);
        let fg = [v[0] - accel[0], v[1] - accel[1], v[2] - accel[2]];
        let fm = [w[0] - mag[0], w[1] - mag[1], w[2] - mag[2]];
        fg[0] * fg[0]
            + fg[1] * fg[1]
            + fg[2] * fg[2]
            + fm[0] * fm[0]
            + fm[1] * fm[1]
            + fm[2] * fm[2]
    }

    /// Met à jour l'estimée à partir du gyroscope (rad/s) et de
    /// l'accéléromètre (IMU, 6 ddl) — `dt` en secondes. Aucune correction si
    /// `accel` est (numériquement) nul : intégration gyroscopique seule.
    ///
    /// Ne contraint **pas** le lacet (cf. en-tête de module) : une orientation
    /// initiale correcte en roulis/tangage mais fausse en lacet ne sera pas
    /// corrigée par cette méthode seule — utiliser [`Self::update_marg`].
    pub fn update_imu(&mut self, gyro: [T; 3], accel: [T; 3], dt: T) {
        let q = self.q;
        let half = T::from_i32(2).recip(); // puissance de 2 : recip() exact.
        let qdot_gyro = q.mul_quat(Quaternion::from_vector(gyro)).scale(half);

        let qdot = match normalize3(accel)
        {
            Some(a) => gradient_correction(qdot_gyro, q, self.beta, |qp| Self::cost_imu(qp, a)),
            None => qdot_gyro,
        };
        self.q = (q + qdot.scale(dt)).normalize();
    }

    /// Met à jour l'estimée à partir du gyroscope, de l'accéléromètre **et**
    /// du magnétomètre (MARG, 9 ddl) — corrige aussi le lacet (cf. en-tête de
    /// module). Aucune correction magnétique si `mag` est nul (repli sur
    /// [`Self::update_imu`]) ; aucune correction du tout si `accel` **et**
    /// `mag` sont nuls.
    pub fn update_marg(&mut self, gyro: [T; 3], accel: [T; 3], mag: [T; 3], dt: T) {
        let q = self.q;
        let half = T::from_i32(2).recip();
        let qdot_gyro = q.mul_quat(Quaternion::from_vector(gyro)).scale(half);

        let qdot = match (normalize3(accel), normalize3(mag))
        {
            (Some(a), Some(m)) =>
            {
                let b_ref = tilt_compensated_reference(q, m);
                gradient_correction(qdot_gyro, q, self.beta, |qp| {
                    Self::cost_marg(qp, a, b_ref, m)
                })
            },
            (Some(a), None) =>
            {
                gradient_correction(qdot_gyro, q, self.beta, |qp| Self::cost_imu(qp, a))
            },
            (None, _) => qdot_gyro,
        };
        self.q = (q + qdot.scale(dt)).normalize();
    }
}

// ------------------------------------------------------------------ //
//  MahonyFilter                                                       //
// ------------------------------------------------------------------ //

/// Filtre d'attitude de Mahony (complémentaire explicite,
/// proportionnel-intégral) — cf. en-tête de module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MahonyFilter<T> {
    q: Quaternion<T>,
    bias: [T; 3],
    kp: T,
    ki: T,
}

impl<T: RealScalar> MahonyFilter<T> {
    /// Construit avec l'orientation identité, biais nul, gains `kp`
    /// (proportionnel, typiquement `1`–`5`) et `ki` (intégral, typiquement
    /// `0`–`0.3` ; `0` désactive l'estimation de biais).
    #[inline]
    pub fn new(kp: T, ki: T) -> Self {
        Self {
            q: Quaternion::identity(),
            bias: [T::zero(); 3],
            kp,
            ki,
        }
    }

    /// Orientation estimée courante.
    #[inline]
    pub fn orientation(&self) -> Quaternion<T> {
        self.q
    }

    /// Biais gyroscopique estimé courant (rad/s), accumulé par le terme
    /// intégral — converge vers le biais réel du capteur si `ki > 0` et des
    /// corrections d'accéléromètre/magnétomètre suffisamment fréquentes sont
    /// fournies.
    #[inline]
    pub fn bias(&self) -> [T; 3] {
        self.bias
    }

    /// Remet l'orientation à l'identité et le biais estimé à zéro.
    #[inline]
    pub fn reset(&mut self) {
        self.q = Quaternion::identity();
        self.bias = [T::zero(); 3];
    }

    /// Applique la rétroaction proportionnelle-intégrale de l'erreur `error`
    /// (repère corps) au gyroscope, intègre le quaternion et renormalise.
    fn correct(&mut self, error: [T; 3], gyro: [T; 3], dt: T) {
        for (b, &e) in self.bias.iter_mut().zip(&error)
        {
            *b = *b + self.ki * e * dt;
        }
        let omega = [
            gyro[0] + self.kp * error[0] + self.bias[0],
            gyro[1] + self.kp * error[1] + self.bias[1],
            gyro[2] + self.kp * error[2] + self.bias[2],
        ];
        let half = T::from_i32(2).recip();
        let qdot = self.q.mul_quat(Quaternion::from_vector(omega)).scale(half);
        self.q = (self.q + qdot.scale(dt)).normalize();
    }

    /// Met à jour l'estimée à partir du gyroscope (rad/s) et de
    /// l'accéléromètre (IMU, 6 ddl) — `dt` en secondes. Ne contraint pas le
    /// lacet (cf. en-tête de module et [`MadgwickFilter::update_imu`]).
    pub fn update_imu(&mut self, gyro: [T; 3], accel: [T; 3], dt: T) {
        let error = match normalize3(accel)
        {
            Some(a) =>
            {
                let v_hat = self
                    .q
                    .conjugate()
                    .rotate_vector([T::zero(), T::zero(), T::one()]);
                cross(a, v_hat)
            },
            None => [T::zero(); 3],
        };
        self.correct(error, gyro, dt);
    }

    /// Met à jour l'estimée à partir du gyroscope, de l'accéléromètre **et**
    /// du magnétomètre (MARG, 9 ddl) — corrige aussi le lacet.
    pub fn update_marg(&mut self, gyro: [T; 3], accel: [T; 3], mag: [T; 3], dt: T) {
        let mut error = [T::zero(); 3];
        if let Some(a) = normalize3(accel)
        {
            let v_hat = self
                .q
                .conjugate()
                .rotate_vector([T::zero(), T::zero(), T::one()]);
            let e = cross(a, v_hat);
            for i in 0..3
            {
                error[i] = error[i] + e[i];
            }
        }
        if let Some(m) = normalize3(mag)
        {
            let b_ref = tilt_compensated_reference(self.q, m);
            let w_hat = self.q.conjugate().rotate_vector(b_ref);
            let e = cross(m, w_hat);
            for i in 0..3
            {
                error[i] = error[i] + e[i];
            }
        }
        self.correct(error, gyro, dt);
    }
}
