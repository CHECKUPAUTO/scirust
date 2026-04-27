// scirust-core/src/autodiff/reverse_v7a.patch.rs
//
// PATCH MINIMAL pour supporter le data parallelism.
//
// Deux ajouts non-invasifs :
//
//   1. Tape::set_grad(idx, tensor)  — injecte un gradient pré-calculé
//      pour pouvoir utiliser l'optimizer existant après le merge des
//      gradients de plusieurs workers.
//
//   2. impl Clone pour les couches stateless et Sequential.
//      Permet à chaque worker thread d'avoir son propre clone du modèle.
//
// =============================================================== //
// 1) Méthode Tape::set_grad — à ajouter dans impl Tape :         //
// =============================================================== //

/* COPIER DANS impl Tape :

    /// Injecte un gradient pré-calculé pour un node. Utile quand on veut
    /// réutiliser l'API de l'optimizer après une agrégation externe
    /// (par ex. après data parallelism).
    pub fn set_grad(&self, idx: usize, value: Tensor) {
        let mut grads = self.grads.borrow_mut();
        assert!(idx < grads.len(), "set_grad: index hors bornes");
        assert_eq!(grads[idx].shape(), value.shape(),
                   "set_grad: shape mismatch");
        grads[idx] = value;
    }

*/

// =============================================================== //
// 2) impl Clone pour les couches — à ajouter dans nn/module.rs :  //
// =============================================================== //

/* COPIER À LA FIN DE scirust-core/src/nn/module.rs :

impl Clone for Linear {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias:   self.bias.clone(),
            last_w_idx: None,           // les indices dépendent d'une tape
            last_b_idx: None,           // → invalides après clone
            name: self.name.clone(),
        }
    }
}

impl Clone for ReLU {
    fn clone(&self) -> Self { ReLU }
}

impl Clone for Sigmoid {
    fn clone(&self) -> Self { Sigmoid }
}

impl Clone for Sequential {
    fn clone(&self) -> Self {
        // Approche concrète : on reconstruit un Sequential vide et on lui
        // re-pousse des Box<dyn Module> clonés. Pour ça, le trait Module
        // a besoin d'une méthode box_clone(). On l'ajoute aussi.
        let mut out = Sequential::new();
        for layer in &self.layers {
            out.layers.push(layer.box_clone());
        }
        out
    }
}

*/

// =============================================================== //
// 3) Méthode box_clone à ajouter sur Module — modifier le trait : //
// =============================================================== //

/* MODIFIER le trait Module dans nn/module.rs :

pub trait Module {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t>;
    fn parameter_indices(&self) -> Vec<usize>;
    fn sync(&mut self, tape: &Tape);
    fn state_dict(&self) -> Vec<(String, Tensor)>;
    fn load_state_dict(&mut self, dict: &HashMap<String, Tensor>) -> usize;
    fn train(&mut self, _mode: bool) {}

    /// NOUVEAU : permet de cloner un Box<dyn Module>.
    fn box_clone(&self) -> Box<dyn Module>;
}

ET COMPLÉTER chaque impl existant (Linear, ReLU, Sigmoid, Dropout, Sequential,
BatchNorm1d, Conv2d, MaxPool2d) avec :

    fn box_clone(&self) -> Box<dyn Module> {
        Box::new(self.clone())
    }

*/

// =============================================================== //
// 4) impl Clone pour BatchNorm1d / Conv2d / MaxPool2d / Dropout : //
// =============================================================== //

/* AJOUTER dans chaque module concerné :

impl Clone for BatchNorm1d {
    fn clone(&self) -> Self {
        Self {
            gamma: self.gamma.clone(),
            beta:  self.beta.clone(),
            eps: self.eps,
            momentum: self.momentum,
            running_mean: self.running_mean.clone(),
            running_var:  self.running_var.clone(),
            training: self.training,
            last_g_idx: None,
            last_b_idx: None,
            name: self.name.clone(),
        }
    }
}

impl Clone for Conv2d {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias:   self.bias.clone(),
            in_c: self.in_c, out_c: self.out_c,
            kernel: self.kernel, stride: self.stride,
            padding: self.padding,
            last_w_idx: None, last_b_idx: None,
            name: self.name.clone(),
            cached_h: self.cached_h,
            cached_w: self.cached_w,
            cached_batch: self.cached_batch,
        }
    }
}

impl Clone for MaxPool2d {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel,
            stride: self.stride,
            cached_c: self.cached_c,
            cached_h: self.cached_h,
            cached_w: self.cached_w,
        }
    }
}

impl Clone for Dropout {
    fn clone(&self) -> Self {
        // Le RNG est cloné avec le même état → si deux workers exécutent
        // exactement la même séquence d'ops, ils auront le même mask.
        // Pour le data parallelism c'est OK (chaque worker a son shard différent
        // donc des activations différentes), mais à savoir.
        Self {
            p: self.p,
            training: self.training,
            rng: self.rng.clone(),  // → ajouter impl Clone sur PcgEngine
        }
    }
}

impl Clone for PcgEngine {
    fn clone(&self) -> Self {
        Self { state: self.state, inc: self.inc }
    }
}

*/

// =============================================================== //
// IMPORTANT : Padding doit être Clone (déjà Copy normalement)     //
// =============================================================== //
//
// Vérifier que `enum Padding` dans nn/conv_utils.rs dérive bien
// #[derive(Clone, Copy, Debug)] — c'est le cas en v6.1.

// =============================================================== //
// Test de set_grad à ajouter dans reverse.rs                      //
// =============================================================== //

/* COPIER DANS #[cfg(test)] mod tests :

    #[test]
    fn test_set_grad_overrides() {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(vec![1.0, 2.0], 1, 2));
        let custom_grad = Tensor::from_vec(vec![10.0, 20.0], 1, 2);
        tape.set_grad(x.idx(), custom_grad.clone());
        assert_eq!(tape.grad(x.idx()).data, custom_grad.data);
    }

*/
