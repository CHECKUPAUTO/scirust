// scirust-core/src/nn/conv_utils.rs
//
// im2col / col2im — fonctions de réorganisation pour les convolutions.
//
// CONVENTIONS DE LAYOUT (toutes en row-major contigu) :
//
//   Input  4D logique : (B, C, H, W)
//   Stockage 2D       : (B, C·H·W) ou (B*C, H·W) — ici on choisit (B, C·H·W)
//                       pour matcher la sortie naturelle d'un Linear sur batch
//
//   Filtre logique    : (out_C, in_C, K, K)
//   Stockage 2D       : (out_C, in_C·K·K)
//
//   im2col output     : (in_C·K·K, B·H_out·W_out)
//                       Chaque colonne est un patch K×K aplati pour un pixel
//                       de sortie d'une image du batch.
//
// FORMULE DES DIMENSIONS DE SORTIE :
//
//   H_out = (H + 2·pad - K) / stride + 1
//   W_out = (W + 2·pad - K) / stride + 1
//
// PADDING :
//   - "valid" : pad = 0
//   - "same"  : pad tel que H_out = H quand stride = 1, soit pad = (K-1)/2
//
// PIÈGE INDEXATION : avec batch B, l'output a B·H_out·W_out colonnes,
// rangées par image puis par pixel (ordre lexicographique
// b → h_out → w_out). C'est crucial pour que matmul(W, im2col_x).reshape
// retombe correctement sur (B, out_C·H_out·W_out).

use crate::autodiff::reverse::Tensor;

#[derive(Clone, Copy, Debug)]
pub enum Padding { Valid, Same }

#[derive(Clone, Copy, Debug)]
pub struct ConvConfig {
    pub batch:    usize,
    pub in_c:     usize,
    pub h:        usize,
    pub w:        usize,
    pub kernel:   usize,
    pub stride:   usize,
    pub padding:  Padding,
    pub out_c:    usize,
}

impl ConvConfig {
    pub fn pad(&self) -> usize {
        match self.padding {
            Padding::Valid => 0,
            Padding::Same  => (self.kernel - 1) / 2,
        }
    }

    pub fn h_out(&self) -> usize {
        (self.h + 2 * self.pad() - self.kernel) / self.stride + 1
    }

    pub fn w_out(&self) -> usize {
        (self.w + 2 * self.pad() - self.kernel) / self.stride + 1
    }

    /// Validation à la construction
    pub fn check(&self) -> Result<(), String> {
        if self.kernel == 0 { return Err("kernel doit être > 0".into()); }
        if self.stride == 0 { return Err("stride doit être > 0".into()); }
        let pad = self.pad();
        if self.h + 2 * pad < self.kernel {
            return Err(format!("input trop petit pour ce kernel : H={}, K={}, pad={}",
                               self.h, self.kernel, pad));
        }
        if self.w + 2 * pad < self.kernel {
            return Err(format!("input trop petit en largeur"));
        }
        Ok(())
    }
}

// ================================================================== //
//  im2col                                                             //
// ================================================================== //
//
// Input  : Tensor de shape (B, C·H·W) row-major, stockage = data[b * C*H*W
//          + c * H*W + h * W + w]
// Output : Tensor de shape (C·K·K, B·H_out·W_out)
//          col[(c*K + kh)*K + kw, b*H_out*W_out + ho*W_out + wo]
//          = input[b, c, ho*stride + kh - pad, wo*stride + kw - pad]
//          (= 0 si hors bornes — padding implicite)

pub fn im2col(input: &Tensor, cfg: &ConvConfig) -> Tensor {
    cfg.check().expect("ConvConfig invalide");

    let (b, c, h, w, k, s) = (cfg.batch, cfg.in_c, cfg.h, cfg.w, cfg.kernel, cfg.stride);
    let pad = cfg.pad();
    let h_out = cfg.h_out();
    let w_out = cfg.w_out();

    let chw  = c * h * w;
    let kk   = k * k;
    let ckk  = c * kk;
    let n_cols = b * h_out * w_out;

    // Vérifie que l'input fait bien (B, C·H·W)
    assert_eq!(input.rows, b, "im2col: rows attendu = B = {b}, got {}", input.rows);
    assert_eq!(input.cols, chw, "im2col: cols attendu = C·H·W = {chw}, got {}", input.cols);

    let mut out = Tensor::zeros(ckk, n_cols);

    // PARCOURS : on itère dans l'ordre (c, kh, kw) pour les lignes,
    // puis (b, ho, wo) pour les colonnes — ordre cohérent avec les
    // formules ci-dessus.
    for c_idx in 0..c {
        for kh in 0..k {
            for kw in 0..k {
                let row = (c_idx * k + kh) * k + kw;
                for bi in 0..b {
                    for ho in 0..h_out {
                        // Position dans l'image source (avec padding)
                        let h_in = ho * s + kh;     // dans le repère paddé
                        for wo in 0..w_out {
                            let w_in = wo * s + kw;
                            let col = bi * h_out * w_out + ho * w_out + wo;
                            // Conversion vers le repère non paddé
                            // h_in_real = h_in - pad ; pareil pour w
                            let in_h_signed = h_in as isize - pad as isize;
                            let in_w_signed = w_in as isize - pad as isize;
                            if in_h_signed >= 0 && in_h_signed < h as isize
                               && in_w_signed >= 0 && in_w_signed < w as isize
                            {
                                let in_h = in_h_signed as usize;
                                let in_w = in_w_signed as usize;
                                let src_idx = bi * chw + c_idx * h * w + in_h * w + in_w;
                                out.data[row * n_cols + col] = input.data[src_idx];
                            }
                            // sinon : pad → 0 (déjà initialisé)
                        }
                    }
                }
            }
        }
    }
    out
}

// ================================================================== //
//  col2im                                                             //
// ================================================================== //
//
// Inverse d'im2col, mais avec un twist crucial : ACCUMULATION.
// Quand stride < kernel, plusieurs colonnes contribuent au même pixel
// d'entrée. col2im somme ces contributions — c'est exactement le
// comportement attendu pour calculer ∂L/∂input à partir de ∂L/∂cols.

pub fn col2im(cols: &Tensor, cfg: &ConvConfig) -> Tensor {
    cfg.check().expect("ConvConfig invalide");

    let (b, c, h, w, k, s) = (cfg.batch, cfg.in_c, cfg.h, cfg.w, cfg.kernel, cfg.stride);
    let pad = cfg.pad();
    let h_out = cfg.h_out();
    let w_out = cfg.w_out();

    let chw  = c * h * w;
    let kk   = k * k;
    let ckk  = c * kk;
    let n_cols = b * h_out * w_out;

    assert_eq!(cols.rows, ckk, "col2im: rows attendu = C·K·K = {ckk}");
    assert_eq!(cols.cols, n_cols, "col2im: cols attendu = B·H_out·W_out = {n_cols}");

    let mut out = Tensor::zeros(b, chw);

    for c_idx in 0..c {
        for kh in 0..k {
            for kw in 0..k {
                let row = (c_idx * k + kh) * k + kw;
                for bi in 0..b {
                    for ho in 0..h_out {
                        let h_in = ho * s + kh;
                        for wo in 0..w_out {
                            let w_in = wo * s + kw;
                            let col = bi * h_out * w_out + ho * w_out + wo;
                            let in_h_signed = h_in as isize - pad as isize;
                            let in_w_signed = w_in as isize - pad as isize;
                            if in_h_signed >= 0 && in_h_signed < h as isize
                               && in_w_signed >= 0 && in_w_signed < w as isize
                            {
                                let in_h = in_h_signed as usize;
                                let in_w = in_w_signed as usize;
                                let dst_idx = bi * chw + c_idx * h * w + in_h * w + in_w;
                                // ACCUMULATION (pas écrasement)
                                out.data[dst_idx] += cols.data[row * n_cols + col];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

// ================================================================== //
//  Tests — point critique, on ne se trompe pas sur les indices        //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;

    fn config_simple() -> ConvConfig {
        ConvConfig {
            batch: 1, in_c: 1, h: 4, w: 4,
            kernel: 3, stride: 1, padding: Padding::Valid, out_c: 1,
        }
    }

    #[test]
    fn h_out_w_out_arithmetic() {
        let cfg = config_simple();
        // (4 + 0 - 3)/1 + 1 = 2
        assert_eq!(cfg.h_out(), 2);
        assert_eq!(cfg.w_out(), 2);

        let cfg_same = ConvConfig { padding: Padding::Same, ..cfg };
        assert_eq!(cfg_same.h_out(), 4);
        assert_eq!(cfg_same.w_out(), 4);
    }

    #[test]
    fn im2col_identity_kernel() {
        // Avec K=1, stride=1, valid : im2col devrait être un simple
        // remappage (C·1·1 lignes × B·H·W colonnes)
        let cfg = ConvConfig {
            batch: 1, in_c: 2, h: 2, w: 2,
            kernel: 1, stride: 1, padding: Padding::Valid, out_c: 1,
        };
        // Input 1×2×2×2 = (1, 8) : 2 channels, chacun avec 4 pixels
        let data = vec![1.0, 2.0, 3.0, 4.0,    // canal 0
                        5.0, 6.0, 7.0, 8.0];   // canal 1
        let input = Tensor::from_vec(data.clone(), 1, 8);
        let cols = im2col(&input, &cfg);

        // Output : (2, 4) — 2 lignes (1 par canal), 4 colonnes (1 par pixel)
        assert_eq!(cols.shape(), (2, 4));
        // Ligne 0 (canal 0) = pixels canal 0 dans l'ordre (b=0, h, w)
        assert_eq!(&cols.data[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&cols.data[4..8], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn im2col_3x3_on_4x4() {
        // Input 4×4 avec un canal, kernel 3×3, stride 1, valid → 2×2 output
        // im2col output : (9, 4)
        let cfg = config_simple();
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let input = Tensor::from_vec(data.clone(), 1, 16);
        let cols = im2col(&input, &cfg);

        assert_eq!(cols.shape(), (9, 4));
        // Première colonne : patch 3×3 en haut-gauche
        // = [1,2,3,5,6,7,9,10,11]
        let col0: Vec<f32> = (0..9).map(|r| cols.data[r * 4 + 0]).collect();
        assert_eq!(col0, vec![1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0]);

        // Quatrième colonne : patch 3×3 en bas-droite
        // = [6,7,8,10,11,12,14,15,16]
        let col3: Vec<f32> = (0..9).map(|r| cols.data[r * 4 + 3]).collect();
        assert_eq!(col3, vec![6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]);
    }

    #[test]
    fn im2col_same_padding_zeros_at_border() {
        let cfg = ConvConfig {
            batch: 1, in_c: 1, h: 3, w: 3,
            kernel: 3, stride: 1, padding: Padding::Same, out_c: 1,
        };
        let data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let input = Tensor::from_vec(data, 1, 9);
        let cols = im2col(&input, &cfg);

        // H_out = W_out = 3 → 9 colonnes
        assert_eq!(cols.shape(), (9, 9));
        // Première colonne : patch centré sur (0, 0). Avec pad=1, le patch
        // 3x3 inclut 4 zéros (les coins paddés) et les vrais pixels (0,0),
        // (0,1), (1,0), (1,1) = 1, 2, 4, 5
        // Layout du patch en colonne : [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
        // Avec pad : la position relative du patch part de (-1, -1)
        // Donc patch = [pad, pad, pad, pad, 1, 2, pad, 4, 5]
        let col0: Vec<f32> = (0..9).map(|r| cols.data[r * 9 + 0]).collect();
        assert_eq!(col0, vec![0.0, 0.0, 0.0,
                              0.0, 1.0, 2.0,
                              0.0, 4.0, 5.0]);
    }

    #[test]
    fn col2im_identity_kernel_round_trip() {
        // K=1, stride=1, valid : im2col puis col2im doit être l'identité
        let cfg = ConvConfig {
            batch: 2, in_c: 3, h: 2, w: 2,
            kernel: 1, stride: 1, padding: Padding::Valid, out_c: 1,
        };
        let data: Vec<f32> = (1..=24).map(|x| x as f32 * 0.5).collect();
        let input = Tensor::from_vec(data.clone(), 2, 12);
        let cols = im2col(&input, &cfg);
        let restored = col2im(&cols, &cfg);
        assert_eq!(restored.shape(), input.shape());
        assert_eq!(restored.data, input.data);
    }

    #[test]
    fn col2im_accumulates_overlapping_contributions() {
        // K=2, stride=1 sur image 3×3 : les patchs se chevauchent.
        // Si on remplit cols avec des 1, le pixel central (1,1) doit
        // recevoir 4 contributions (il est dans 4 patchs).
        let cfg = ConvConfig {
            batch: 1, in_c: 1, h: 3, w: 3,
            kernel: 2, stride: 1, padding: Padding::Valid, out_c: 1,
        };
        // im2col output : (4, 4) — 4 = K*K, 4 = H_out*W_out = 2*2
        let cols = Tensor::from_vec(vec![1.0; 16], 4, 4);
        let restored = col2im(&cols, &cfg);
        assert_eq!(restored.shape(), (1, 9));
        // Coin (0,0) : appartient à 1 patch
        assert_eq!(restored.data[0], 1.0);
        // Coin (1,1) : appartient à 4 patchs
        assert_eq!(restored.data[4], 4.0);
        // Bord (0,1) : appartient à 2 patchs
        assert_eq!(restored.data[1], 2.0);
    }

    #[test]
    fn im2col_batch_dimension_correct() {
        // Vérifie que B > 1 fonctionne et que les images du batch sont
        // bien séparées dans les colonnes.
        let cfg = ConvConfig {
            batch: 2, in_c: 1, h: 2, w: 2,
            kernel: 2, stride: 1, padding: Padding::Valid, out_c: 1,
        };
        // 2 images : la première = [1,2,3,4], la deuxième = [10,20,30,40]
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0,
                                          10.0, 20.0, 30.0, 40.0], 2, 4);
        let cols = im2col(&input, &cfg);
        // Output : (4, 2) — 4 = K*K, 2 = B * H_out * W_out (1 patch par image)
        assert_eq!(cols.shape(), (4, 2));
        // Colonne 0 (image 0) = [1, 2, 3, 4]
        let c0: Vec<f32> = (0..4).map(|r| cols.data[r * 2 + 0]).collect();
        assert_eq!(c0, vec![1.0, 2.0, 3.0, 4.0]);
        // Colonne 1 (image 1) = [10, 20, 30, 40]
        let c1: Vec<f32> = (0..4).map(|r| cols.data[r * 2 + 1]).collect();
        assert_eq!(c1, vec![10.0, 20.0, 30.0, 40.0]);
    }
}
