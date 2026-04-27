// scirust-core/src/autodiff/reverse_broadcast.patch.rs
//
// PATCH À APPLIQUER SUR scirust-core/src/autodiff/reverse.rs
//
// Ajoute le support du broadcasting dans le tape AD.
// L'agent doit fusionner ces additions dans reverse.rs existant.
//
// =============================================================== //
// 1) Dans l'enum Op, AJOUTER ces 2 variantes :                    //
// =============================================================== //
//
//     BroadcastAdd(usize, usize),  // a + b avec broadcast
//     BroadcastMul(usize, usize),  // a * b (élem) avec broadcast
//
// =============================================================== //
// 2) Dans `impl<'t> Var<'t>`, AJOUTER ces 2 méthodes :            //
// =============================================================== //

/* COPIER À LA FIN DE impl<'t> Var<'t> :

    /// Addition avec broadcasting (ex: matrice + vecteur de biais)
    pub fn add_broadcast(self, other: Var<'t>) -> Var<'t> {
        use crate::tensor::broadcast::broadcast_add;
        let a = self.tape.values.borrow()[self.idx].clone();
        let b = self.tape.values.borrow()[other.idx].clone();
        let out = broadcast_add(&a, &b);
        let idx = self.tape.push(Op::BroadcastAdd(self.idx, other.idx), out);
        Var { tape: self.tape, idx }
    }

    /// Multiplication élémentaire avec broadcasting
    pub fn mul_broadcast(self, other: Var<'t>) -> Var<'t> {
        use crate::tensor::broadcast::broadcast_mul;
        let a = self.tape.values.borrow()[self.idx].clone();
        let b = self.tape.values.borrow()[other.idx].clone();
        let out = broadcast_mul(&a, &b);
        let idx = self.tape.push(Op::BroadcastMul(self.idx, other.idx), out);
        Var { tape: self.tape, idx }
    }

*/

// =============================================================== //
// 3) Dans `fn propagate(...)`, AJOUTER ces deux match arms :     //
// =============================================================== //

/* COPIER DANS LE match op DE propagate (avant l'accolade fermante) :

        Op::BroadcastAdd(a, b) => {
            use crate::tensor::broadcast::unbroadcast;
            let shape_a = tape.nodes.borrow()[a].shape;
            let shape_b = tape.nodes.borrow()[b].shape;
            // d(a+b)/da = 1 (broadcast vers la forme du résultat)
            // → grad_a = unbroadcast(grad_out, shape_a)
            let grad_a = unbroadcast(&grad_out, shape_a);
            let grad_b = unbroadcast(&grad_out, shape_b);
            accumulate(tape, a, &grad_a);
            accumulate(tape, b, &grad_b);
        }

        Op::BroadcastMul(a, b) => {
            use crate::tensor::broadcast::{broadcast_get, unbroadcast};
            let shape_a = tape.nodes.borrow()[a].shape;
            let shape_b = tape.nodes.borrow()[b].shape;
            let val_a = tape.values.borrow()[a].clone();
            let val_b = tape.values.borrow()[b].clone();

            // d(a*b)/da = b, mais b peut avoir une forme broadcastée
            // → on calcule grad_a_pre = grad_out * b (broadcasté), puis unbroadcast
            let mut grad_a_pre = Tensor::zeros(grad_out.rows, grad_out.cols);
            let mut grad_b_pre = Tensor::zeros(grad_out.rows, grad_out.cols);
            for r in 0..grad_out.rows {
                for c in 0..grad_out.cols {
                    let g = grad_out.data[r * grad_out.cols + c];
                    grad_a_pre.data[r * grad_out.cols + c] = g * broadcast_get(&val_b, r, c);
                    grad_b_pre.data[r * grad_out.cols + c] = g * broadcast_get(&val_a, r, c);
                }
            }
            let grad_a = unbroadcast(&grad_a_pre, shape_a);
            let grad_b = unbroadcast(&grad_b_pre, shape_b);
            accumulate(tape, a, &grad_a);
            accumulate(tape, b, &grad_b);
        }

*/

// =============================================================== //
// 4) AJOUTER ces tests à la fin du module tests :                 //
// =============================================================== //

/* COPIER DANS #[cfg(test)] mod tests :

    #[test]
    fn test_broadcast_add_grad() {
        // y = sum(x + b)  où x:(2,3), b:(1,3)
        // dy/dx = 1 (forme x), dy/db = sum(1) sur axe 0 = (1,3) avec 2 partout
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0; 6], 2, 3));
        let b = tape.input(Tensor::from_vec(vec![10.0, 20.0, 30.0], 1, 3));
        let y = x.add_broadcast(b).sum();
        y.backward();

        let grad_b = tape.grad(b.idx());
        assert_eq!(grad_b.shape(), (1, 3));
        // Chaque biais reçoit 2 gradients (un par ligne de x)
        assert!((grad_b.data[0] - 2.0).abs() < 1e-6);
        assert!((grad_b.data[1] - 2.0).abs() < 1e-6);
        assert!((grad_b.data[2] - 2.0).abs() < 1e-6);
    }

*/

// =============================================================== //
// Le script python du AGENT_INSTRUCTIONS_V3.md applique ces       //
// modifications automatiquement (avec des regex idempotentes).    //
// =============================================================== //
