// scirust-core/src/autodiff/reverse_v6_1.patch.rs
//
// PATCH À APPLIQUER SUR scirust-core/src/autodiff/reverse.rs
//
// Ajoute deux Op majeures :
//   - Op::MaxAxis (axe 0 ou 1, keep_dims=true)
//   - Op::Conv2dForward (op composite avec recompute au backward)
//
// =============================================================== //
// 1) Variantes Op à ajouter :                                     //
// =============================================================== //
//
// Pour MaxAxis :
//     MaxAxis(usize, u8),        // (src, axis)
//
// Pour Conv2dForward, structure plus complexe :
//
//     Conv2dForward {
//         input_idx:  usize,
//         weight_idx: usize,
//         bias_idx:   Option<usize>,
//         config:     crate::nn::conv_utils::ConvConfig,
//     },
//
// Pour MaxPool2d :
//
//     MaxPool2dForward {
//         input_idx: usize,
//         c: usize,
//         h: usize,
//         w: usize,
//         kernel: usize,
//         stride: usize,
//     },
//
// Note : ConvConfig est Copy, donc embedded directement dans l'op.
//
// =============================================================== //
// 2) Méthode Var::max_axis :                                      //
// =============================================================== //

/* COPIER DANS impl<'t> Var<'t> :

    pub fn max_axis(self, axis: u8) -> Var<'t> {
        assert!(axis == 0 || axis == 1);
        let a = self.tape.values.borrow()[self.idx].clone();
        let (rows, cols) = a.shape();
        let out_shape = if axis == 0 { (1, cols) } else { (rows, 1) };
        let mut out = Tensor::zeros(out_shape.0, out_shape.1);
        if axis == 0 {
            for c in 0..cols {
                let mut m = f32::NEG_INFINITY;
                for r in 0..rows {
                    let v = a.data[r * cols + c];
                    if v > m { m = v; }
                }
                out.data[c] = m;
            }
        } else {
            for r in 0..rows {
                let mut m = f32::NEG_INFINITY;
                for c in 0..cols {
                    let v = a.data[r * cols + c];
                    if v > m { m = v; }
                }
                out.data[r] = m;
            }
        }
        let idx = self.tape.push(Op::MaxAxis(self.idx, axis), out);
        Var { tape: self.tape, idx }
    }

*/

// =============================================================== //
// 3) Méthode Var::conv2d_forward :                                //
// =============================================================== //

/* COPIER DANS impl<'t> Var<'t> :

    /// MaxPool2d : input (B, C·H·W) → (B, C·H_out·W_out)
    /// avec H_out = (H - K)/stride + 1
    pub fn max_pool2d(
        self,
        c: usize, h: usize, w: usize,
        kernel: usize, stride: usize,
    ) -> Var<'t> {
        let input_t = self.tape.values.borrow()[self.idx].clone();
        let (b, total) = input_t.shape();
        assert_eq!(total, c * h * w);

        let h_out = (h - kernel) / stride + 1;
        let w_out = (w - kernel) / stride + 1;
        let mut output = Tensor::zeros(b, c * h_out * w_out);

        // Pool canal par canal
        for bi in 0..b {
            for ci in 0..c {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let mut m = f32::NEG_INFINITY;
                        for kh in 0..kernel {
                            for kw in 0..kernel {
                                let h_in = ho * stride + kh;
                                let w_in = wo * stride + kw;
                                let src = bi * c * h * w + ci * h * w + h_in * w + w_in;
                                let v = input_t.data[src];
                                if v > m { m = v; }
                            }
                        }
                        let dst = bi * c * h_out * w_out + ci * h_out * w_out
                                  + ho * w_out + wo;
                        output.data[dst] = m;
                    }
                }
            }
        }

        let idx = self.tape.push(
            Op::MaxPool2dForward {
                input_idx: self.idx,
                c, h, w, kernel, stride,
            },
            output,
        );
        Var { tape: self.tape, idx }
    }

*/

// =============================================================== //
// 3.bis) Méthode Var::conv2d_forward :                            //
// =============================================================== //

/* COPIER DANS impl<'t> Var<'t> :
    /// mais ne stocke QUE le résultat (pas le tenseur im2col intermédiaire).
    /// Le backward reconstruit im2col à partir de l'input.
    ///
    /// Convention : input shape = (B, in_C·H·W) row-major,
    ///              weight shape = (out_C, in_C·K·K),
    ///              bias shape = (1, out_C) ou None,
    ///              output shape = (B, out_C·H_out·W_out).
    pub fn conv2d_forward(
        self,
        weight: Var<'t>,
        bias:   Option<Var<'t>>,
        config: crate::nn::conv_utils::ConvConfig,
    ) -> Var<'t> {
        use crate::nn::conv_utils::im2col;
        let input_t  = self.tape.values.borrow()[self.idx].clone();
        let weight_t = self.tape.values.borrow()[weight.idx].clone();

        // Forward : im2col → matmul → reshape
        let cols = im2col(&input_t, &config);
        // matmul : (out_C, in_C·K·K) @ (in_C·K·K, B·H_out·W_out)
        //        → (out_C, B·H_out·W_out)
        let m = config.out_c;
        let k_dim = config.in_c * config.kernel * config.kernel;
        let n = config.batch * config.h_out() * config.w_out();
        assert_eq!(weight_t.shape(), (m, k_dim));
        assert_eq!(cols.shape(),     (k_dim, n));

        let mut wx = Tensor::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k_dim {
                    acc += weight_t.data[i * k_dim + p] * cols.data[p * n + j];
                }
                wx.data[i * n + j] = acc;
            }
        }

        // Bias broadcast (out_C,) sur les colonnes
        if let Some(b) = bias.as_ref() {
            let bias_t = self.tape.values.borrow()[b.idx].clone();
            assert_eq!(bias_t.shape(), (1, m));
            for i in 0..m {
                for j in 0..n {
                    wx.data[i * n + j] += bias_t.data[i];
                }
            }
        }

        // Reshape de (out_C, B·H_out·W_out) vers (B, out_C·H_out·W_out)
        // : permutation des axes, on fait une copie indexée
        let h_out = config.h_out();
        let w_out = config.w_out();
        let mut output = Tensor::zeros(config.batch, m * h_out * w_out);
        for bi in 0..config.batch {
            for oc in 0..m {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let src = oc * n + bi * h_out * w_out + ho * w_out + wo;
                        let dst = bi * m * h_out * w_out
                                  + oc * h_out * w_out
                                  + ho * w_out + wo;
                        output.data[dst] = wx.data[src];
                    }
                }
            }
        }

        let bias_idx = bias.as_ref().map(|b| b.idx);
        let idx = self.tape.push(
            Op::Conv2dForward {
                input_idx:  self.idx,
                weight_idx: weight.idx,
                bias_idx,
                config,
            },
            output,
        );
        Var { tape: self.tape, idx }
    }

*/

// =============================================================== //
// 4) Match arms backward dans propagate :                         //
// =============================================================== //

/* COPIER DANS LE match op DE propagate :

        Op::MaxAxis(a, axis) => {
            // Tie-breaking : tous les éléments égaux au max reçoivent grad_out.
            // Sécurisé numériquement : on compare aux mêmes octets que ceux
            // copiés au forward (val_a → val_y est une copie binaire stricte).
            let val_a = tape.values.borrow()[a].clone();
            let val_y = tape.values.borrow()[i].clone();
            let shape_a = val_a.shape();
            let mut grad_a = Tensor::zeros(shape_a.0, shape_a.1);
            for r in 0..shape_a.0 {
                for c in 0..shape_a.1 {
                    let yr = if axis == 0 { 0 } else { r };
                    let yc = if axis == 1 { 0 } else { c };
                    let max_val = val_y.data[yr * val_y.cols + yc];
                    if val_a.data[r * shape_a.1 + c] == max_val {
                        grad_a.data[r * shape_a.1 + c] =
                            grad_out.data[yr * grad_out.cols + yc];
                    }
                }
            }
            accumulate(tape, a, &grad_a);
        }

        Op::MaxPool2dForward { input_idx, c, h, w, kernel, stride } => {
            // Backward avec recompute du mask. Tie-breaking : tous les
            // éléments égaux au max d'une fenêtre reçoivent le gradient.
            let input_t = tape.values.borrow()[input_idx].clone();
            let (b, _) = input_t.shape();
            let h_out = (h - kernel) / stride + 1;
            let w_out = (w - kernel) / stride + 1;
            let mut grad_input = Tensor::zeros(b, c * h * w);

            for bi in 0..b {
                for ci in 0..c {
                    for ho in 0..h_out {
                        for wo in 0..w_out {
                            // Recompute le max de cette fenêtre
                            let mut m = f32::NEG_INFINITY;
                            for kh in 0..kernel {
                                for kw in 0..kernel {
                                    let h_in = ho * stride + kh;
                                    let w_in = wo * stride + kw;
                                    let src = bi * c * h * w + ci * h * w + h_in * w + w_in;
                                    let v = input_t.data[src];
                                    if v > m { m = v; }
                                }
                            }
                            // Récupère le gradient de cette cellule de sortie
                            let g_idx = bi * c * h_out * w_out + ci * h_out * w_out
                                        + ho * w_out + wo;
                            let g_val = grad_out.data[g_idx];
                            // Distribue à toutes les cellules égales au max
                            for kh in 0..kernel {
                                for kw in 0..kernel {
                                    let h_in = ho * stride + kh;
                                    let w_in = wo * stride + kw;
                                    let src = bi * c * h * w + ci * h * w + h_in * w + w_in;
                                    if input_t.data[src] == m {
                                        grad_input.data[src] += g_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            accumulate(tape, input_idx, &grad_input);
        }

        Op::Conv2dForward { input_idx, weight_idx, bias_idx, config } => {
            // Backward du Conv2d composite — recompute im2col depuis input.
            //
            // grad_out : (B, out_C·H_out·W_out)
            // 1) reshape grad_out : (out_C, B·H_out·W_out)  (inverse de la
            //    permutation forward)
            // 2) recompute cols = im2col(input)  (B·H_out·W_out, in_C·K·K).T
            // 3) grad_weight = grad_out_reshaped @ cols.T
            // 4) grad_cols   = weight.T @ grad_out_reshaped
            // 5) grad_input  = col2im(grad_cols)
            // 6) grad_bias   = sum(grad_out_reshaped, axis=1)

            use crate::nn::conv_utils::{im2col, col2im};

            let input_t  = tape.values.borrow()[input_idx].clone();
            let weight_t = tape.values.borrow()[weight_idx].clone();

            let m       = config.out_c;
            let k_dim   = config.in_c * config.kernel * config.kernel;
            let h_out   = config.h_out();
            let w_out   = config.w_out();
            let n_cols  = config.batch * h_out * w_out;

            // ---- 1. reshape grad_out (B, out_C·H_out·W_out) → (out_C, B·H_out·W_out)
            let mut g_perm = Tensor::zeros(m, n_cols);
            for bi in 0..config.batch {
                for oc in 0..m {
                    for ho in 0..h_out {
                        for wo in 0..w_out {
                            let src = bi * m * h_out * w_out
                                      + oc * h_out * w_out
                                      + ho * w_out + wo;
                            let dst = oc * n_cols + bi * h_out * w_out
                                      + ho * w_out + wo;
                            g_perm.data[dst] = grad_out.data[src];
                        }
                    }
                }
            }

            // ---- 2. recompute cols
            let cols = im2col(&input_t, &config);   // (k_dim, n_cols)

            // ---- 3. grad_weight = g_perm @ cols.T  →  (out_C, k_dim)
            let mut grad_w = Tensor::zeros(m, k_dim);
            for i_w in 0..m {
                for j_w in 0..k_dim {
                    let mut acc = 0.0f32;
                    for p in 0..n_cols {
                        // g_perm[i_w, p] * cols[j_w, p]  (cols.T donc même indice p)
                        acc += g_perm.data[i_w * n_cols + p]
                             * cols.data[j_w * n_cols + p];
                    }
                    grad_w.data[i_w * k_dim + j_w] = acc;
                }
            }
            accumulate(tape, weight_idx, &grad_w);

            // ---- 4. grad_cols = weight.T @ g_perm  →  (k_dim, n_cols)
            let mut grad_cols = Tensor::zeros(k_dim, n_cols);
            for i_c in 0..k_dim {
                for j_c in 0..n_cols {
                    let mut acc = 0.0f32;
                    for p in 0..m {
                        // weight.T[i_c, p] = weight[p, i_c]
                        acc += weight_t.data[p * k_dim + i_c]
                             * g_perm.data[p * n_cols + j_c];
                    }
                    grad_cols.data[i_c * n_cols + j_c] = acc;
                }
            }

            // ---- 5. grad_input via col2im
            let grad_input = col2im(&grad_cols, &config);
            accumulate(tape, input_idx, &grad_input);

            // ---- 6. grad_bias = somme par ligne de g_perm  (shape (1, out_C))
            if let Some(b_idx) = bias_idx {
                let mut grad_bias = Tensor::zeros(1, m);
                for i_b in 0..m {
                    let mut s = 0.0f32;
                    for j_b in 0..n_cols { s += g_perm.data[i_b * n_cols + j_b]; }
                    grad_bias.data[i_b] = s;
                }
                accumulate(tape, b_idx, &grad_bias);
            }
        }

*/

// =============================================================== //
// 5) Tests à ajouter dans reverse.rs :                            //
// =============================================================== //

/* COPIER DANS #[cfg(test)] mod tests :

    #[test]
    fn test_max_axis_forward() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(
            vec![1.0, 5.0, 3.0,
                 4.0, 2.0, 6.0], 2, 3));
        let m = x.max_axis(1);
        let mt = tape.value(m.idx());
        assert_eq!(mt.shape(), (2, 1));
        assert_eq!(mt.data, vec![5.0, 6.0]);
    }

    #[test]
    fn test_max_axis_grad_routes_to_max_only() {
        // Gradient ne va qu'à l'élément max
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 5.0, 3.0], 1, 3));
        let m = x.max_axis(1).sum();
        m.backward();
        let g = tape.grad(x.idx());
        // Seul l'index 1 (valeur 5.0, le max) reçoit grad = 1
        assert_eq!(g.data, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_max_axis_ties_distribute_grad() {
        // Si plusieurs max → tous reçoivent le gradient (sub-gradient)
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![3.0, 3.0, 1.0], 1, 3));
        let m = x.max_axis(1).sum();
        m.backward();
        let g = tape.grad(x.idx());
        // Les deux 3.0 reçoivent grad = 1
        assert_eq!(g.data, vec![1.0, 1.0, 0.0]);
    }

*/
