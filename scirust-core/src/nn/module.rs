// scirust-core/src/nn/module.rs
//
// Trait Module — interface minimale pour tous les modules NN.
//
// Contrat :
//   - forward(tape, input) : push les ops du module sur la tape, retourne
//     la sortie comme une nouvelle Var.
//   - parameter_indices() : retourne les idx des paramètres ENTRAÎNABLES
//     du module sur la tape courante. Doit être appelé APRÈS forward().
//   - sync(tape) : lit les valeurs entraînées depuis la tape pour les
//     persister dans la struct. À appeler après opt.step().
//
// Pour chaîner un Module à travers plusieurs époques, on crée une nouvelle
// Tape à chaque step. Le module garde ses Tensor poids dans sa struct,
// et les ré-injecte sur la nouvelle tape via input() au début de chaque
// forward().
//
// Pour Sequential et autres composeurs, le trait doit être object-safe
// (utilisable derrière `Box<dyn Module>`). Les méthodes ne sont donc pas
// génériques sur le lifetime.

use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::error::Result;
use std::collections::HashMap;

pub struct SteerHook {
    pub target_layer: String,
    pub shift: Tensor,
}

pub trait Module {
    /// Forward pass : push les ops sur la tape, retourne la Var sortie.
    /// Doit aussi enregistrer en interne les indices des paramètres
    /// entraînables (utilisé par parameter_indices ensuite).
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t>;

    fn forward_steered<'t>(
        &mut self,
        tape: &'t Tape,
        input: Var<'t>,
        _hook: Option<&SteerHook>,
    ) -> Var<'t> {
        // Default implementation delegates to forward if no hook or not handled
        self.forward(tape, input)
    }

    /// Indices des paramètres entraînables sur la dernière tape vue.
    /// Ne fonctionne que si forward a été appelé d'abord.
    fn parameter_indices(&self) -> Vec<usize>;

    /// Lit les valeurs courantes des paramètres depuis la tape (après step()).
    /// Met à jour les Tensor stockés dans la struct du module.
    fn sync(&mut self, tape: &Tape);

    /// Return the module's parameters as a map of name -> Tensor.
    /// Used for checkpoint saving via `save_state_dict`.
    fn state_dict(&self) -> HashMap<String, Tensor> {
        HashMap::new()
    }

    /// Load parameters from a state dict produced by `state_dict()`.
    /// Each module implementation should match parameter names and shapes.
    fn load_state_dict(&mut self, _sd: &HashMap<String, Tensor>) -> Result<()> {
        Ok(())
    }
}
