use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::init::Initializer;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::rng::PcgEngine;

/// Variational Autoencoder (VAE) implementation.
pub struct VAE {
    pub encoder_fc: Linear,
    pub fc_mu: Linear,
    pub fc_logvar: Linear,
    pub decoder_fc: Linear,
    pub decoder_out: Linear,
    pub latent_dim: usize,
    pub rng: PcgEngine,
}

impl VAE {
    pub fn new<W: Initializer, B: Initializer>(
        input_dim: usize,
        hidden_dim: usize,
        latent_dim: usize,
        w_init: &W,
        b_init: &B,
        rng: &mut PcgEngine,
    ) -> Self {
        let encoder_fc = Linear::new(input_dim, hidden_dim, w_init, b_init, rng);
        let fc_mu = Linear::new(hidden_dim, latent_dim, w_init, b_init, rng);
        let fc_logvar = Linear::new(hidden_dim, latent_dim, w_init, b_init, rng);
        let decoder_fc = Linear::new(latent_dim, hidden_dim, w_init, b_init, rng);
        let decoder_out = Linear::new(hidden_dim, input_dim, w_init, b_init, rng);

        Self {
            encoder_fc,
            fc_mu,
            fc_logvar,
            decoder_fc,
            decoder_out,
            latent_dim,
            rng: PcgEngine::new(rng.next_u32() as u64),
        }
    }

    pub fn reparameterize<'t>(&mut self, tape: &'t Tape, mu: Var<'t>, logvar: Var<'t>) -> Var<'t> {
        let std = logvar.scale(0.5).exp();
        let (rows, cols) = mu.shape();
        let mut eps_data = vec![0.0f32; rows * cols];
        for e in &mut eps_data
        {
            *e = self.rng.normal(0.0, 1.0);
        }

        let eps = tape.input(Tensor::from_vec(eps_data, rows, cols));
        mu.try_add(std.try_hadamard(eps).unwrap()).unwrap()
    }

    pub fn forward<'t>(&mut self, tape: &'t Tape, x: Var<'t>) -> (Var<'t>, Var<'t>, Var<'t>) {
        let h = self.encoder_fc.forward(tape, x).relu();
        let mu = self.fc_mu.forward(tape, h);
        let logvar = self.fc_logvar.forward(tape, h);
        let z = self.reparameterize(tape, mu, logvar);

        let h_dec = self.decoder_fc.forward(tape, z).relu();
        let recon_x = self.decoder_out.forward(tape, h_dec).sigmoid();

        (recon_x, mu, logvar)
    }

    pub fn kl_loss<'t>(&self, tape: &'t Tape, mu: Var<'t>, logvar: Var<'t>) -> Var<'t> {
        let (rows, cols) = mu.shape();
        let ones = tape.input(Tensor::from_vec(vec![1.0; rows * cols], rows, cols));
        let mu_sq = mu.try_hadamard(mu).unwrap();
        let exp_logvar = logvar.exp();
        let sum_term = ones
            .try_add(logvar)
            .unwrap()
            .try_sub(mu_sq)
            .unwrap()
            .try_sub(exp_logvar)
            .unwrap();
        sum_term.sum().scale(-0.5)
    }
}
