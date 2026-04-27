// scirust-core/src/autodiff/reverse_v6.patch.rs
//
// PATCH À APPLIQUER SUR scirust-core/src/autodiff/reverse.rs
// Ajoute Op::SumAxis, Op::Reshape, Op::Reciprocal.
//
// Le script Python du AGENT_INSTRUCTIONS_V6.md applique ces additions
// de façon idempotente.
//
// =============================================================== //
// 1) Variantes Op à ajouter :                                     //
// =============================================================== //
//
//     SumAxis(usize, u8),               // (src, axis) avec axis ∈ {0, 1}
//                                       // keep_dims=true toujours
//     Reshape(usize, usize, usize),     // (src, new_rows, new_cols)
//     Reciprocal(usize),                // 1/x
//
// =============================================================== //
// 2) Méthodes Var à ajouter :                                     //
// =============================================================== //

/* COPIER DANS impl<'t> Var<'t> :

    /// Somme le long d'un axe en gardant la dimension (keep_dims=true).
    /// axis=0 : (M, N) → (1, N)
    /// axis=1 : (M, N) → (M, 1)
    pub fn sum_axis(self, axis: u8) -> Var<'t> {
        assert!(axis == 0 || axis == 1, "sum_axis: axis ∈ {{0, 1}}, got {axis}");
        let a = self.tape.values.borrow()[self.idx].clone();
        let (rows, cols) = a.shape();
        let out_shape = if axis == 0 { (1, cols) } else { (rows, 1) };
        let mut out = Tensor::zeros(out_shape.0, out_shape.1);
        if axis == 0 {
            // somme par colonne
            for c in 0..cols {
                let mut s = 0.0f32;
                for r in 0..rows { s += a.data[r * cols + c]; }
                out.data[c] = s;
            }
        } else {
            // somme par ligne
            for r in 0..rows {
                let mut s = 0.0f32;
                for c in 0..cols { s += a.data[r * cols + c]; }
                out.data[r] = s;
            }
        }
        let idx = self.tape.push(Op::SumAxis(self.idx, axis), out);
        Var { tape: self.tape, idx }
    }

    /// Reshape — change le shape sans modifier les données. Précondition :
    /// rows*cols == new_rows*new_cols. Layout row-major contigu.
    pub fn reshape(self, new_rows: usize, new_cols: usize) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].clone();
        assert_eq!(a.rows * a.cols, new_rows * new_cols,
                   "reshape: nombre d'éléments différent");
        let out = Tensor::from_vec(a.data.clone(), new_rows, new_cols);
        let idx = self.tape.push(Op::Reshape(self.idx, new_rows, new_cols), out);
        Var { tape: self.tape, idx }
    }

    /// 1/x élément par élément. Précondition : x != 0 (clamp à eps en pratique)
    pub fn reciprocal(self) -> Var<'t> {
        let a = self.tape.values.borrow()[self.idx].clone();
        let mut out = a.clone();
        for x in out.data.iter_mut() {
            // Clamp pour éviter division par 0 — attention au signe
            let v = if x.abs() < 1e-12 {
                if *x >= 0.0 { 1e-12 } else { -1e-12 }
            } else { *x };
            *x = 1.0 / v;
        }
        let idx = self.tape.push(Op::Reciprocal(self.idx), out);
        Var { tape: self.tape, idx }
    }

*/

// =============================================================== //
// 3) Match arms backward dans propagate :                         //
// =============================================================== //

/* COPIER DANS LE match op DE propagate :

        Op::SumAxis(a, axis) => {
            // Le backward de sum_axis est un broadcast :
            //   ∂L/∂a[i,j] = ∂L/∂y[broadcast_index(i,j)]
            // Avec keep_dims=true, c'est exactement le motif unbroadcast
            // mais à l'envers : on fait un broadcast manuel.
            let shape_a = tape.nodes.borrow()[a].shape;
            let mut grad_a = Tensor::zeros(shape_a.0, shape_a.1);
            for r in 0..shape_a.0 {
                for c in 0..shape_a.1 {
                    let gr = if axis == 0 { 0 } else { r };
                    let gc = if axis == 1 { 0 } else { c };
                    grad_a.data[r * shape_a.1 + c] =
                        grad_out.data[gr * grad_out.cols + gc];
                }
            }
            accumulate(tape, a, &grad_a);
        }

        Op::Reshape(a, _, _) => {
            // Reshape est sa propre adjointe : reshape le gradient vers
            // la forme originale.
            let shape_a = tape.nodes.borrow()[a].shape;
            let grad_a = Tensor::from_vec(grad_out.data.clone(), shape_a.0, shape_a.1);
            accumulate(tape, a, &grad_a);
        }

        Op::Reciprocal(a) => {
            // d/dx (1/x) = -1/x² = -y² où y = 1/x (réutilise sortie cached)
            let val_y = tape.values.borrow()[i].clone();
            let mut grad_a = grad_out.clone();
            for j in 0..grad_a.data.len() {
                grad_a.data[j] *= -val_y.data[j] * val_y.data[j];
            }
            accumulate(tape, a, &grad_a);
        }

*/

// =============================================================== //
// 4) Tests à ajouter :                                            //
// =============================================================== //

/* COPIER DANS #[cfg(test)] mod tests :

    #[test]
    fn test_sum_axis_0() {
        // (2,3) → (1,3) en sommant par colonne
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3));
        let y = x.sum_axis(0);
        let yt = tape.value(y.idx());
        assert_eq!(yt.shape(), (1, 3));
        // colonne 0 : 1+4=5, colonne 1 : 2+5=7, colonne 2 : 3+6=9
        assert_eq!(yt.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sum_axis_grad_broadcasts_back() {
        // y = sum(sum_axis(x, axis=0)) → toutes les entrées de x doivent avoir gradient 1
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0; 6], 2, 3));
        let s = x.sum_axis(0).sum();
        s.backward();
        let g = tape.grad(x.idx());
        assert!(g.data.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_reshape_round_trip() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3));
        let y = x.reshape(3, 2);
        assert_eq!(tape.value(y.idx()).shape(), (3, 2));
        // Données identiques car row-major
        assert_eq!(tape.value(y.idx()).data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reciprocal() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![2.0, 4.0, 0.5], 1, 3));
        let y = x.reciprocal();
        let yt = tape.value(y.idx());
        assert!((yt.data[0] - 0.5).abs() < 1e-6);
        assert!((yt.data[1] - 0.25).abs() < 1e-6);
        assert!((yt.data[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_reciprocal_grad() {
        // y = 1/x, dy/dx = -1/x². Avec x=2, dy/dx = -0.25
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![2.0], 1, 1));
        let y = x.reciprocal().sum();
        y.backward();
        let g = tape.grad(x.idx());
        assert!((g.data[0] - (-0.25)).abs() < 1e-5);
    }

*/
