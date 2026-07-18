// scirust-simd/src/fixed/conv2d.rs
//
// # Convolution 2D quantifiée déterministe
//
// [`conv2d`] : extension bidimensionnelle de [`super::conv::conv1d`], pour les
// données à deux dimensions spatiales (images, spectrogrammes temps-fréquence)
// d'un CNN léger. Même technique **im2col + GEMM** : les fenêtres glissantes
// 2D de l'entrée sont dépliées en colonnes contiguës, puis
// [`super::linalg::matmul`] calcule tous les canaux de sortie d'un coup — le
// déterminisme bit-à-bit du GEMM se transmet donc sans travail supplémentaire.
//
// ## Disposition mémoire
//
// * `x` : `in_channels × height × width`, row-major (canal, puis ligne, puis
//   colonne).
// * `weights` : `out_channels × in_channels × kernel_h × kernel_w`, row-major
//   (même convention que PyTorch `Conv2d`).
// * `bias` : `out_channels` éléments, un par canal de sortie.
// * Sortie : `out_channels × height_out × width_out`, row-major, avec
//   `height_out`/`width_out` donnés par [`Conv2dShape::height_out`] /
//   [`Conv2dShape::width_out`] (convolution valide, sans remplissage).
//
// ## im2col
//
// La matrice dépliée a pour forme `(in_channels·kernel_h·kernel_w) ×
// (height_out·width_out)` : la ligne `(ci·kernel_h·kernel_w + kh·kernel_w +
// kw)`, colonne `(oh·width_out + ow)`, vaut `x[ci, oh·stride_h + kh,
// ow·stride_w + kw]` — exactement l'ordre attendu par `weights`.
//
// ## Inférence par lot
//
// [`conv2d_batch`] traite `batch` échantillons en un seul GEMM, résultat
// identique bit-à-bit à `batch` appels de [`conv2d`] concaténés — même
// principe que [`super::conv::conv1d_batch`]/
// [`super::layer::Linear::forward_batch`]. [`super::pool2d`] et
// [`super::activation`] batchent déjà gratuitement (opérations purement par
// canal) en repliant `batch` dans l'axe `channels` de la sortie.
//
// ## Convolution séparable en profondeur (MobileNet)
//
// [`depthwise_conv2d`] applique **un noyau indépendant par canal d'entrée**
// (aucun mélange inter-canaux, contrairement à [`conv2d`]) : `out_channels`
// vaut nécessairement `in_channels`. Ce n'est **pas** un GEMM (chaque canal
// est une réduction indépendante de `kernel_h·kernel_w` éléments via
// [`super::reductions::dot`], pas de somme sur les canaux) — le coût descend
// de `O(in·out·kh·kw)` (convolution dense) à `O(in·kh·kw)` par pixel de
// sortie. [`separable_conv2d`] compose [`depthwise_conv2d`] avec une
// convolution **ponctuelle** `1×1` (mélange inter-canaux, coût `O(in·out)`) :
// c'est exactement [`conv2d`] avec `kernel_h = kernel_w = 1` — aucun code
// neuf pour la partie ponctuelle, seul l'assemblage est nouveau. Coût total
// `O(in·kh·kw + in·out)`, à comparer aux `O(in·out·kh·kw)` d'une convolution
// dense équivalente — l'économie classique de MobileNet.
//
// ## Convolution transposée (suréchantillonnage)
//
// [`conv2d_transpose`] est l'opération **adjointe** de [`conv2d`] : là où
// [`conv2d`] accumule une fenêtre glissante de l'entrée en un seul élément de
// sortie (« gather »), [`conv2d_transpose`] diffuse chaque élément de
// l'entrée sur une fenêtre de sortie pondérée par le noyau (« scatter-add »)
// — l'opération centrale des décodeurs convolutifs et GAN génératifs pour
// suréchantillonner une carte de caractéristiques.
//
// Formellement, en notant `M` la matrice creuse telle que `conv2d(x) = M·x`
// (im2col plus produit par la matrice de poids), [`conv2d_transpose`] avec la
// disposition de poids `in_channels × out_channels × kernel_h × kernel_w`
// calcule exactement `Mᵀ·x` **en réutilisant le même tableau de poids**, sans
// réindexation : `in_channels`/`out_channels` sont simplement inversés par
// rapport à [`conv2d`] (convention PyTorch `ConvTranspose2d`). Cette relation
// d'adjonction — `⟨conv2d(x, W), y⟩ = ⟨x, conv2d_transpose(y, W)⟩` pour tout
// `x`, `y` — est la propriété vérifiée par les tests plutôt qu'une
// réimplémentation indépendante.
//
// Forme de sortie (toujours sans remplissage ni `output_padding`, comme
// [`Conv2dShape`]) : `height_out = (height − 1)·stride_h + kernel_h`,
// symétrique de [`Conv2dShape::height_out`] (`stride` et soustraction du
// noyau échangés — la convolution transposée agrandit au lieu de rétrécir).
//
// ## Convolution dilatée (« à trous »)
//
// [`dilated_conv2d`] espace les prises du noyau de `dilation_h`/`dilation_w`
// (au lieu d'être contiguës) : élargit le champ réceptif sans augmenter le
// nombre de paramètres ni sous-échantillonner (contrairement à empiler des
// `stride > 1`), la technique classique des réseaux à convolution dilatée
// (segmentation sémantique, `WaveNet`). Noyau **effectif**
// `dilation·(kernel−1) + 1` : `dilated_conv2d` avec `dilation = 1` est
// **exactement** [`conv2d`] (même code, dilatation neutre), et avec
// `dilation = d` est **exactement** [`conv2d`] appliqué à un noyau
// « dilaté » où `d−1` zéros sont insérés entre chaque prise — propriété
// vérifiée par les tests plutôt qu'une réimplémentation indépendante (les
// zéros insérés contribuent une somme exactement nulle, l'addition virgule
// fixe étant exacte : les deux formulations coïncident **bit à bit**).
//
// [`DilatedConv2dShape`] est une structure **séparée** de [`Conv2dShape`]
// (même précédent que [`Conv2dTransposeShape`]) plutôt qu'un champ
// `dilation` ajouté à `Conv2dShape` : `Conv2dShape` est construit par
// littéral de structure dans plusieurs dizaines d'appels existants
// (`conv2d`/`conv2d_batch`/`depthwise_conv2d`/`separable_conv2d`, tests,
// bancs d'essai) — lui ajouter un champ obligatoire forcerait leur
// modification mécanique pour une fonctionnalité qu'ils n'utilisent pas.
// Une structure neuve, dont [`DilatedConv2dShape::height_out`]/
// [`DilatedConv2dShape::width_out`] intègrent la taille effective du noyau,
// n'a aucun impact sur le code existant.

use super::reductions::FixedReducible;

/// Dimensions d'une convolution 2D valide (sans remplissage).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dShape {
    /// Nombre de canaux d'entrée.
    pub in_channels: usize,
    /// Hauteur de l'entrée (par canal).
    pub height: usize,
    /// Largeur de l'entrée (par canal).
    pub width: usize,
    /// Nombre de canaux de sortie (filtres).
    pub out_channels: usize,
    /// Hauteur du noyau de convolution.
    pub kernel_h: usize,
    /// Largeur du noyau de convolution.
    pub kernel_w: usize,
    /// Pas de déplacement vertical de la fenêtre glissante.
    pub stride_h: usize,
    /// Pas de déplacement horizontal de la fenêtre glissante.
    pub stride_w: usize,
}

impl Conv2dShape {
    /// Hauteur de sortie `(height − kernel_h) / stride_h + 1`.
    ///
    /// Panique si `stride_h == 0` ou `height < kernel_h`.
    #[must_use]
    pub fn height_out(&self) -> usize {
        assert!(
            self.stride_h >= 1,
            "Conv2dShape::height_out : stride_h doit être ≥ 1"
        );
        assert!(
            self.height >= self.kernel_h,
            "Conv2dShape::height_out : hauteur {} < noyau {}",
            self.height,
            self.kernel_h
        );
        (self.height - self.kernel_h) / self.stride_h + 1
    }

    /// Largeur de sortie `(width − kernel_w) / stride_w + 1`.
    ///
    /// Panique si `stride_w == 0` ou `width < kernel_w`.
    #[must_use]
    pub fn width_out(&self) -> usize {
        assert!(
            self.stride_w >= 1,
            "Conv2dShape::width_out : stride_w doit être ≥ 1"
        );
        assert!(
            self.width >= self.kernel_w,
            "Conv2dShape::width_out : largeur {} < noyau {}",
            self.width,
            self.kernel_w
        );
        (self.width - self.kernel_w) / self.stride_w + 1
    }
}

/// Déplie les fenêtres glissantes 2D de `x` en colonnes contiguës : forme
/// `(in_channels·kernel_h·kernel_w) × (height_out·width_out)`, row-major.
fn im2col2d<T: Copy>(x: &[T], shape: Conv2dShape, height_out: usize, width_out: usize) -> Vec<T> {
    let mut col = Vec::with_capacity(
        shape.in_channels * shape.kernel_h * shape.kernel_w * height_out * width_out,
    );
    for ci in 0..shape.in_channels
    {
        for kh in 0..shape.kernel_h
        {
            for kw in 0..shape.kernel_w
            {
                for oh in 0..height_out
                {
                    for ow in 0..width_out
                    {
                        let h = oh * shape.stride_h + kh;
                        let w = ow * shape.stride_w + kw;
                        col.push(x[ci * (shape.height * shape.width) + h * shape.width + w]);
                    }
                }
            }
        }
    }
    col
}

/// Convolution 2D **valide** (sans remplissage), multi-canaux, déterministe.
///
/// `x` : `shape.in_channels × shape.height × shape.width` ; `weights` :
/// `shape.out_channels × shape.in_channels × shape.kernel_h × shape.kernel_w` ;
/// `bias` : `shape.out_channels`. Retourne `shape.out_channels ×
/// shape.height_out() × shape.width_out()`.
///
/// Panique si les longueurs de slice ne correspondent pas aux dimensions
/// annoncées, ou selon les préconditions de [`Conv2dShape::height_out`] /
/// [`Conv2dShape::width_out`] — incohérence d'appelant.
#[must_use]
pub fn conv2d<T: FixedReducible>(x: &[T], weights: &[T], bias: &[T], shape: Conv2dShape) -> Vec<T> {
    let height_out = shape.height_out();
    let width_out = shape.width_out();
    assert_eq!(
        x.len(),
        shape.in_channels * shape.height * shape.width,
        "conv2d : x de longueur {} ≠ {}×{}×{}",
        x.len(),
        shape.in_channels,
        shape.height,
        shape.width
    );
    assert_eq!(
        weights.len(),
        shape.out_channels * shape.in_channels * shape.kernel_h * shape.kernel_w,
        "conv2d : poids de longueur {} ≠ {}×{}×{}×{}",
        weights.len(),
        shape.out_channels,
        shape.in_channels,
        shape.kernel_h,
        shape.kernel_w
    );
    assert_eq!(
        bias.len(),
        shape.out_channels,
        "conv2d : biais de longueur {} ≠ {}",
        bias.len(),
        shape.out_channels
    );

    let col = im2col2d(x, shape, height_out, width_out);
    let spatial_out = height_out * width_out;
    let mut y = super::linalg::matmul(
        weights,
        &col,
        shape.out_channels,
        shape.in_channels * shape.kernel_h * shape.kernel_w,
        spatial_out,
    );
    for (co, &b) in bias.iter().enumerate()
    {
        for pos in 0..spatial_out
        {
            let idx = co * spatial_out + pos;
            y[idx] = y[idx].wrapping_add(b);
        }
    }
    y
}

/// Déplie les fenêtres glissantes 2D d'un **lot** de `batch` échantillons en
/// colonnes contiguës : forme `(in_channels·kernel_h·kernel_w) ×
/// (batch·height_out·width_out)`, row-major, les colonnes étant groupées par
/// échantillon — le bloc `b` coïncide exactement avec la sortie de
/// [`im2col2d`] pour l'échantillon `b` seul.
fn im2col2d_batch<T: Copy>(
    x: &[T],
    batch: usize,
    shape: Conv2dShape,
    height_out: usize,
    width_out: usize,
) -> Vec<T> {
    let sample_len = shape.in_channels * shape.height * shape.width;
    let spatial_out = height_out * width_out;
    let mut col = Vec::with_capacity(
        shape.in_channels * shape.kernel_h * shape.kernel_w * batch * spatial_out,
    );
    for ci in 0..shape.in_channels
    {
        for kh in 0..shape.kernel_h
        {
            for kw in 0..shape.kernel_w
            {
                for b in 0..batch
                {
                    let x_b = &x[b * sample_len..(b + 1) * sample_len];
                    for oh in 0..height_out
                    {
                        for ow in 0..width_out
                        {
                            let h = oh * shape.stride_h + kh;
                            let w = ow * shape.stride_w + kw;
                            col.push(x_b[ci * (shape.height * shape.width) + h * shape.width + w]);
                        }
                    }
                }
            }
        }
    }
    col
}

/// Convolution 2D **par lot** (cf. [`conv2d`]) : `x` est `batch ×
/// shape.in_channels × shape.height × shape.width` (un échantillon par bloc,
/// contigu) ; retourne `batch × shape.out_channels × shape.height_out() ×
/// shape.width_out()` — même disposition sample-major que l'entrée.
///
/// Résultat **identique bit-à-bit** à `batch` appels de [`conv2d`] concaténés
/// (vérifié par test), en un seul GEMM plutôt que `batch` GEMM plus petits —
/// cf. [`super::conv::conv1d_batch`] pour le détail du raisonnement (même
/// principe, une dimension spatiale de plus). Panique si `x.len() !=
/// batch·in_channels·height·width`, ou selon les préconditions de [`conv2d`].
#[must_use]
pub fn conv2d_batch<T: FixedReducible>(
    x: &[T],
    batch: usize,
    weights: &[T],
    bias: &[T],
    shape: Conv2dShape,
) -> Vec<T> {
    let height_out = shape.height_out();
    let width_out = shape.width_out();
    let spatial_out = height_out * width_out;
    assert_eq!(
        x.len(),
        batch * shape.in_channels * shape.height * shape.width,
        "conv2d_batch : x de longueur {} ≠ {batch}×{}×{}×{}",
        x.len(),
        shape.in_channels,
        shape.height,
        shape.width
    );
    assert_eq!(
        weights.len(),
        shape.out_channels * shape.in_channels * shape.kernel_h * shape.kernel_w,
        "conv2d_batch : poids de longueur {} ≠ {}×{}×{}×{}",
        weights.len(),
        shape.out_channels,
        shape.in_channels,
        shape.kernel_h,
        shape.kernel_w
    );
    assert_eq!(
        bias.len(),
        shape.out_channels,
        "conv2d_batch : biais de longueur {} ≠ {}",
        bias.len(),
        shape.out_channels
    );

    let col = im2col2d_batch(x, batch, shape, height_out, width_out);
    let y_gemm = super::linalg::matmul(
        weights,
        &col,
        shape.out_channels,
        shape.in_channels * shape.kernel_h * shape.kernel_w,
        batch * spatial_out,
    );

    let mut y = vec![T::ZERO; batch * shape.out_channels * spatial_out];
    for (co, &bias_co) in bias.iter().enumerate()
    {
        for b in 0..batch
        {
            let src = co * (batch * spatial_out) + b * spatial_out;
            let dst = b * (shape.out_channels * spatial_out) + co * spatial_out;
            for pos in 0..spatial_out
            {
                y[dst + pos] = y_gemm[src + pos].wrapping_add(bias_co);
            }
        }
    }
    y
}

/// Convolution 2D **profonde** (depthwise), déterministe : un noyau `kernel_h
/// × kernel_w` indépendant par canal d'entrée (cf. en-tête de module).
///
/// `x` : `shape.in_channels × shape.height × shape.width` ; `weights` :
/// `shape.in_channels × shape.kernel_h × shape.kernel_w` (**un** noyau par
/// canal — pas `out_channels × in_channels × kh × kw` comme [`conv2d`]) ;
/// `bias` : `shape.in_channels`. Retourne `shape.in_channels ×
/// shape.height_out() × shape.width_out()`.
///
/// Panique si `shape.out_channels != shape.in_channels` (une convolution
/// profonde ne change pas le nombre de canaux), si les longueurs de slice ne
/// correspondent pas aux dimensions annoncées, ou selon les préconditions de
/// [`Conv2dShape::height_out`]/[`Conv2dShape::width_out`].
#[must_use]
pub fn depthwise_conv2d<T: FixedReducible>(
    x: &[T],
    weights: &[T],
    bias: &[T],
    shape: Conv2dShape,
) -> Vec<T> {
    assert_eq!(
        shape.out_channels, shape.in_channels,
        "depthwise_conv2d : out_channels ({}) doit égaler in_channels ({}) — un noyau par canal",
        shape.out_channels, shape.in_channels
    );
    let height_out = shape.height_out();
    let width_out = shape.width_out();
    let kernel_size = shape.kernel_h * shape.kernel_w;
    let channel_size = shape.height * shape.width;
    assert_eq!(
        x.len(),
        shape.in_channels * channel_size,
        "depthwise_conv2d : x de longueur {} ≠ {}×{}×{}",
        x.len(),
        shape.in_channels,
        shape.height,
        shape.width
    );
    assert_eq!(
        weights.len(),
        shape.in_channels * kernel_size,
        "depthwise_conv2d : poids de longueur {} ≠ {}×{}×{} (un noyau par canal)",
        weights.len(),
        shape.in_channels,
        shape.kernel_h,
        shape.kernel_w
    );
    assert_eq!(
        bias.len(),
        shape.in_channels,
        "depthwise_conv2d : biais de longueur {} ≠ {}",
        bias.len(),
        shape.in_channels
    );

    let spatial_out = height_out * width_out;
    let mut y = vec![T::ZERO; shape.in_channels * spatial_out];
    let mut window = vec![T::ZERO; kernel_size];
    for ci in 0..shape.in_channels
    {
        let kernel = &weights[ci * kernel_size..(ci + 1) * kernel_size];
        let x_ci = &x[ci * channel_size..(ci + 1) * channel_size];
        for oh in 0..height_out
        {
            for ow in 0..width_out
            {
                let mut idx = 0;
                for kh in 0..shape.kernel_h
                {
                    for kw in 0..shape.kernel_w
                    {
                        let h = oh * shape.stride_h + kh;
                        let w = ow * shape.stride_w + kw;
                        window[idx] = x_ci[h * shape.width + w];
                        idx += 1;
                    }
                }
                let acc = super::reductions::dot(kernel, &window);
                y[ci * spatial_out + oh * width_out + ow] = acc.wrapping_add(bias[ci]);
            }
        }
    }
    y
}

/// Convolution séparable en profondeur (MobileNet), déterministe :
/// [`depthwise_conv2d`] suivie d'une convolution ponctuelle `1×1` (mélange
/// inter-canaux, via [`conv2d`] — cf. en-tête de module).
///
/// `x`/`shape` comme [`depthwise_conv2d`] (`shape.out_channels ==
/// shape.in_channels`). `pointwise_weights` : `out_channels ×
/// shape.in_channels × 1 × 1` (soit `out_channels × shape.in_channels`,
/// même convention que [`conv2d`]) ; `pointwise_bias` : `out_channels`.
/// Retourne `out_channels × shape.height_out() × shape.width_out()`.
///
/// Panique selon les préconditions de [`depthwise_conv2d`] et de [`conv2d`]
/// (appliqué à la sortie profonde avec un noyau `1×1`).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn separable_conv2d<T: FixedReducible>(
    x: &[T],
    depthwise_weights: &[T],
    depthwise_bias: &[T],
    pointwise_weights: &[T],
    pointwise_bias: &[T],
    shape: Conv2dShape,
    out_channels: usize,
) -> Vec<T> {
    let depthwise_out = depthwise_conv2d(x, depthwise_weights, depthwise_bias, shape);
    let pointwise_shape = Conv2dShape {
        in_channels: shape.in_channels,
        height: shape.height_out(),
        width: shape.width_out(),
        out_channels,
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
    };
    conv2d(
        &depthwise_out,
        pointwise_weights,
        pointwise_bias,
        pointwise_shape,
    )
}

/// Dimensions d'une convolution transposée 2D, sans remplissage ni
/// `output_padding` (cf. en-tête de module et [`Conv2dShape`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dTransposeShape {
    /// Nombre de canaux d'entrée.
    pub in_channels: usize,
    /// Hauteur de l'entrée (par canal).
    pub height: usize,
    /// Largeur de l'entrée (par canal).
    pub width: usize,
    /// Nombre de canaux de sortie.
    pub out_channels: usize,
    /// Hauteur du noyau de convolution.
    pub kernel_h: usize,
    /// Largeur du noyau de convolution.
    pub kernel_w: usize,
    /// Pas de déplacement vertical entre deux éléments d'entrée consécutifs.
    pub stride_h: usize,
    /// Pas de déplacement horizontal entre deux éléments d'entrée consécutifs.
    pub stride_w: usize,
}

impl Conv2dTransposeShape {
    /// Hauteur de sortie `(height − 1) · stride_h + kernel_h`.
    ///
    /// Panique si `stride_h == 0` ou `height == 0`.
    #[must_use]
    pub fn height_out(&self) -> usize {
        assert!(
            self.stride_h >= 1,
            "Conv2dTransposeShape::height_out : stride_h doit être ≥ 1"
        );
        assert!(
            self.height >= 1,
            "Conv2dTransposeShape::height_out : hauteur doit être ≥ 1"
        );
        (self.height - 1) * self.stride_h + self.kernel_h
    }

    /// Largeur de sortie `(width − 1) · stride_w + kernel_w`.
    ///
    /// Panique si `stride_w == 0` ou `width == 0`.
    #[must_use]
    pub fn width_out(&self) -> usize {
        assert!(
            self.stride_w >= 1,
            "Conv2dTransposeShape::width_out : stride_w doit être ≥ 1"
        );
        assert!(
            self.width >= 1,
            "Conv2dTransposeShape::width_out : largeur doit être ≥ 1"
        );
        (self.width - 1) * self.stride_w + self.kernel_w
    }
}

/// Convolution transposée 2D (déconvolution/suréchantillonnage),
/// déterministe : opération **adjointe** de [`conv2d`] (cf. en-tête de
/// module) — chaque élément d'entrée est diffusé (« scatter-add ») sur une
/// fenêtre de sortie pondérée par le noyau, au lieu d'accumuler une fenêtre
/// d'entrée en un seul élément de sortie.
///
/// `x` : `shape.in_channels × shape.height × shape.width` ; `weights` :
/// `shape.in_channels × shape.out_channels × shape.kernel_h × shape.kernel_w`
/// (axes canaux **inversés** par rapport à [`conv2d`] — convention PyTorch
/// `ConvTranspose2d`, cf. en-tête de module) ; `bias` : `shape.out_channels`.
/// Retourne `shape.out_channels × shape.height_out() × shape.width_out()`.
///
/// Panique si les longueurs de slice ne correspondent pas aux dimensions
/// annoncées, ou selon les préconditions de
/// [`Conv2dTransposeShape::height_out`]/[`Conv2dTransposeShape::width_out`].
#[must_use]
pub fn conv2d_transpose<T: FixedReducible>(
    x: &[T],
    weights: &[T],
    bias: &[T],
    shape: Conv2dTransposeShape,
) -> Vec<T> {
    let height_out = shape.height_out();
    let width_out = shape.width_out();
    assert_eq!(
        x.len(),
        shape.in_channels * shape.height * shape.width,
        "conv2d_transpose : x de longueur {} ≠ {}×{}×{}",
        x.len(),
        shape.in_channels,
        shape.height,
        shape.width
    );
    assert_eq!(
        weights.len(),
        shape.in_channels * shape.out_channels * shape.kernel_h * shape.kernel_w,
        "conv2d_transpose : poids de longueur {} ≠ {}×{}×{}×{}",
        weights.len(),
        shape.in_channels,
        shape.out_channels,
        shape.kernel_h,
        shape.kernel_w
    );
    assert_eq!(
        bias.len(),
        shape.out_channels,
        "conv2d_transpose : biais de longueur {} ≠ {}",
        bias.len(),
        shape.out_channels
    );

    let channel_size = shape.height * shape.width;
    let kernel_size = shape.kernel_h * shape.kernel_w;
    let weights_per_in = shape.out_channels * kernel_size;
    let spatial_out = height_out * width_out;
    let mut y = vec![T::ZERO; shape.out_channels * spatial_out];

    for ci in 0..shape.in_channels
    {
        let x_ci = &x[ci * channel_size..(ci + 1) * channel_size];
        let w_ci = &weights[ci * weights_per_in..(ci + 1) * weights_per_in];
        for ih in 0..shape.height
        {
            for iw in 0..shape.width
            {
                let xv = x_ci[ih * shape.width + iw];
                for co in 0..shape.out_channels
                {
                    let w_co = &w_ci[co * kernel_size..(co + 1) * kernel_size];
                    let y_co = &mut y[co * spatial_out..(co + 1) * spatial_out];
                    for kh in 0..shape.kernel_h
                    {
                        let oh = ih * shape.stride_h + kh;
                        for kw in 0..shape.kernel_w
                        {
                            let ow = iw * shape.stride_w + kw;
                            let contrib = xv.wrapping_mul(w_co[kh * shape.kernel_w + kw]);
                            let idx = oh * width_out + ow;
                            y_co[idx] = y_co[idx].wrapping_add(contrib);
                        }
                    }
                }
            }
        }
    }

    for (co, &b) in bias.iter().enumerate()
    {
        for pos in 0..spatial_out
        {
            let idx = co * spatial_out + pos;
            y[idx] = y[idx].wrapping_add(b);
        }
    }
    y
}

/// Dimensions d'une convolution 2D **dilatée** (« à trous »), valide (sans
/// remplissage) — structure séparée de [`Conv2dShape`] (cf. en-tête de
/// module).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DilatedConv2dShape {
    /// Nombre de canaux d'entrée.
    pub in_channels: usize,
    /// Hauteur de l'entrée (par canal).
    pub height: usize,
    /// Largeur de l'entrée (par canal).
    pub width: usize,
    /// Nombre de canaux de sortie (filtres).
    pub out_channels: usize,
    /// Hauteur du noyau de convolution (prises, avant dilatation).
    pub kernel_h: usize,
    /// Largeur du noyau de convolution (prises, avant dilatation).
    pub kernel_w: usize,
    /// Pas de déplacement vertical de la fenêtre glissante.
    pub stride_h: usize,
    /// Pas de déplacement horizontal de la fenêtre glissante.
    pub stride_w: usize,
    /// Espacement vertical entre deux prises consécutives du noyau
    /// (`1` = convolution ordinaire, contiguë).
    pub dilation_h: usize,
    /// Espacement horizontal entre deux prises consécutives du noyau.
    pub dilation_w: usize,
}

impl DilatedConv2dShape {
    /// Hauteur **effective** du noyau une fois dilaté :
    /// `dilation_h · (kernel_h − 1) + 1`.
    #[must_use]
    pub fn effective_kernel_h(&self) -> usize {
        self.dilation_h * (self.kernel_h - 1) + 1
    }

    /// Largeur **effective** du noyau une fois dilaté :
    /// `dilation_w · (kernel_w − 1) + 1`.
    #[must_use]
    pub fn effective_kernel_w(&self) -> usize {
        self.dilation_w * (self.kernel_w - 1) + 1
    }

    /// Hauteur de sortie `(height − noyau_effectif_h) / stride_h + 1`.
    ///
    /// Panique si `stride_h == 0`, `dilation_h == 0`, ou
    /// `height < noyau_effectif_h`.
    #[must_use]
    pub fn height_out(&self) -> usize {
        assert!(
            self.stride_h >= 1,
            "DilatedConv2dShape::height_out : stride_h doit être ≥ 1"
        );
        assert!(
            self.dilation_h >= 1,
            "DilatedConv2dShape::height_out : dilation_h doit être ≥ 1"
        );
        let eff = self.effective_kernel_h();
        assert!(
            self.height >= eff,
            "DilatedConv2dShape::height_out : hauteur {} < noyau effectif {eff}",
            self.height
        );
        (self.height - eff) / self.stride_h + 1
    }

    /// Largeur de sortie `(width − noyau_effectif_w) / stride_w + 1`.
    ///
    /// Panique si `stride_w == 0`, `dilation_w == 0`, ou
    /// `width < noyau_effectif_w`.
    #[must_use]
    pub fn width_out(&self) -> usize {
        assert!(
            self.stride_w >= 1,
            "DilatedConv2dShape::width_out : stride_w doit être ≥ 1"
        );
        assert!(
            self.dilation_w >= 1,
            "DilatedConv2dShape::width_out : dilation_w doit être ≥ 1"
        );
        let eff = self.effective_kernel_w();
        assert!(
            self.width >= eff,
            "DilatedConv2dShape::width_out : largeur {} < noyau effectif {eff}",
            self.width
        );
        (self.width - eff) / self.stride_w + 1
    }
}

/// Déplie les fenêtres glissantes 2D **dilatées** de `x` en colonnes
/// contiguës — même forme que [`im2col2d`], mais chaque prise du noyau est
/// espacée de `dilation_h`/`dilation_w` plutôt que contiguë.
fn im2col2d_dilated<T: Copy>(
    x: &[T],
    shape: DilatedConv2dShape,
    height_out: usize,
    width_out: usize,
) -> Vec<T> {
    let mut col = Vec::with_capacity(
        shape.in_channels * shape.kernel_h * shape.kernel_w * height_out * width_out,
    );
    for ci in 0..shape.in_channels
    {
        for kh in 0..shape.kernel_h
        {
            for kw in 0..shape.kernel_w
            {
                for oh in 0..height_out
                {
                    for ow in 0..width_out
                    {
                        let h = oh * shape.stride_h + kh * shape.dilation_h;
                        let w = ow * shape.stride_w + kw * shape.dilation_w;
                        col.push(x[ci * (shape.height * shape.width) + h * shape.width + w]);
                    }
                }
            }
        }
    }
    col
}

/// Convolution 2D **dilatée** (« à trous »), valide (sans remplissage),
/// multi-canaux, déterministe (cf. en-tête de module).
///
/// `x` : `shape.in_channels × shape.height × shape.width` ; `weights` :
/// `shape.out_channels × shape.in_channels × shape.kernel_h × shape.kernel_w`
/// (même convention que [`conv2d`], compacte — pas le noyau dilaté avec ses
/// zéros insérés) ; `bias` : `shape.out_channels`. Retourne
/// `shape.out_channels × shape.height_out() × shape.width_out()`.
///
/// `dilation_h = dilation_w = 1` calcule **exactement** [`conv2d`] (même
/// code, dilatation neutre).
///
/// Panique si les longueurs de slice ne correspondent pas aux dimensions
/// annoncées, ou selon les préconditions de
/// [`DilatedConv2dShape::height_out`]/[`DilatedConv2dShape::width_out`].
#[must_use]
pub fn dilated_conv2d<T: FixedReducible>(
    x: &[T],
    weights: &[T],
    bias: &[T],
    shape: DilatedConv2dShape,
) -> Vec<T> {
    let height_out = shape.height_out();
    let width_out = shape.width_out();
    assert_eq!(
        x.len(),
        shape.in_channels * shape.height * shape.width,
        "dilated_conv2d : x de longueur {} ≠ {}×{}×{}",
        x.len(),
        shape.in_channels,
        shape.height,
        shape.width
    );
    assert_eq!(
        weights.len(),
        shape.out_channels * shape.in_channels * shape.kernel_h * shape.kernel_w,
        "dilated_conv2d : poids de longueur {} ≠ {}×{}×{}×{}",
        weights.len(),
        shape.out_channels,
        shape.in_channels,
        shape.kernel_h,
        shape.kernel_w
    );
    assert_eq!(
        bias.len(),
        shape.out_channels,
        "dilated_conv2d : biais de longueur {} ≠ {}",
        bias.len(),
        shape.out_channels
    );

    let col = im2col2d_dilated(x, shape, height_out, width_out);
    let spatial_out = height_out * width_out;
    let mut y = super::linalg::matmul(
        weights,
        &col,
        shape.out_channels,
        shape.in_channels * shape.kernel_h * shape.kernel_w,
        spatial_out,
    );
    for (co, &b) in bias.iter().enumerate()
    {
        for pos in 0..spatial_out
        {
            let idx = co * spatial_out + pos;
            y[idx] = y[idx].wrapping_add(b);
        }
    }
    y
}
