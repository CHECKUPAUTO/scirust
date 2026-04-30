// scirust-core/src/nn/conv2d.rs
//
// Conv2d unifié — remplace les 3 versions précédentes :
//
//   v6.1 : CPU only
//   v7-B : GPU avec descente de l'output en RAM (compatible avec layers
//          CPU qui suivent)
//   v8   : GPU avec output gardé en VRAM (Pattern 3, pour chaînes GPU)
//
// La distinction se fait via une enum ConvBackend :
//
//   let conv = Conv2d::new(...);                       // CpuOnly par défaut
//   let conv = Conv2d::new(...).with_backend(GpuKeepVram(ctx, pipelines));
//
// Le user choisit explicitement son mode. Pas de magie globale,
// pas de duplication de fichier.
//
// MIGRATION DEPUIS V8 :
//   - `Conv2d::on_gpu(ctx, pipelines)` continue de marcher (renvoie GpuKeepVram)
//   - L'ancienne v7-B est accessible via `with_backend(ConvBackend::GpuDescend(...))`

use std::collections::HashMap;
use crate::autodiff::reverse::{Tape, Tensor, Var};
use crate::nn::module::Module;
use crate::nn::init::Initializer;
use crate::nn::rng::PcgEngine;
use crate::nn::conv_utils::{ConvConfig, Padding};
use crate::error::{Result, SciRustError};

#[cfg(feature = "wgpu")]
use scirust_gpu::gpu_tensor::GpuContext;
#[cfg(feature = "wgpu")]
use scirust_gpu::gpu_conv::ConvGpuPipelines;

// ================================================================== //
//  ConvBackend — choix du device pour cette couche                    //
// ================================================================== //

pub enum ConvBackend {
    /// Forward et backward sur CPU. C'est le défaut.
    CpuOnly,

    /// GPU pour le calcul, mais l'output est rapatrié en RAM avant
    /// d'être stocké sur la tape. Adapté quand la couche suivante
    /// est CPU (ex: fin de réseau, BatchNorm CPU, etc.).
    /// Équivalent ergonomique de v7-B.
    #[cfg(feature = "wgpu")]
    GpuDescend {
        ctx:       Arc<GpuContext>,
        pipelines: Arc<ConvGpuPipelines>,
    },

    /// GPU pour le calcul, output reste en VRAM. Pour chaînes GPU
    /// (Conv → ReLU GPU → Conv). Le user doit appeler `to_cpu()`
    /// explicitement pour redescendre.
    /// Équivalent ergonomique de v8.
    #[cfg(feature = "wgpu")]
    GpuKeepVram {
        ctx:       Arc<GpuContext>,
        pipelines: Arc<ConvGpuPipelines>,
    },
}

impl ConvBackend {
    pub fn is_gpu(&self) -> bool {
        match self {
            ConvBackend::CpuOnly => false,
            #[cfg(feature = "wgpu")]
            _ => true,
        }
    }
}

impl Clone for ConvBackend {
    fn clone(&self) -> Self {
        match self {
            ConvBackend::CpuOnly => ConvBackend::CpuOnly,
            #[cfg(feature = "wgpu")]
            ConvBackend::GpuDescend { ctx, pipelines } => ConvBackend::GpuDescend {
                ctx: ctx.clone(), pipelines: pipelines.clone(),
            },
            #[cfg(feature = "wgpu")]
            ConvBackend::GpuKeepVram { ctx, pipelines } => ConvBackend::GpuKeepVram {
                ctx: ctx.clone(), pipelines: pipelines.clone(),
            },
        }
    }
}

// ================================================================== //
//  Conv2d module                                                      //
// ================================================================== //

pub struct Conv2d {
    pub weight:    Tensor,
    pub bias:      Option<Tensor>,
    pub in_c:      usize,
    pub out_c:     usize,
    pub kernel:    usize,
    pub stride:    usize,
    pub padding:   Padding,
    pub backend:   ConvBackend,
    last_w_idx:    Option<usize>,
    last_b_idx:    Option<usize>,
    pub name:      String,
    cached_h:      Option<usize>,
    cached_w:      Option<usize>,
    cached_batch:  Option<usize>,
}

impl Conv2d {
    /// Constructeur principal. Fonction qui PEUT échouer (kernel=0,
    /// in_c=0, etc.) → renvoie Result.
    pub fn try_new<W: Initializer, B: Initializer>(
        in_c:        usize,
        out_c:       usize,
        kernel:      usize,
        stride:      usize,
        padding:     Padding,
        weight_init: &W,
        bias_init:   Option<&B>,
        rng:         &mut PcgEngine,
    ) -> Result<Self> {
        if in_c == 0 || out_c == 0 {
            return Err(SciRustError::InvalidConfig(
                format!("Conv2d: in_c={in_c} out_c={out_c}, doivent être > 0")));
        }
        if kernel == 0 || stride == 0 {
            return Err(SciRustError::InvalidConfig(
                format!("Conv2d: kernel={kernel} stride={stride}, doivent être > 0")));
        }

        let kk = kernel * kernel;
        let mut weight = Tensor::zeros(out_c, in_c * kk);
        weight_init.fill(&mut weight, in_c * kk, out_c, rng);

        let bias = bias_init.map(|init| {
            let mut b = Tensor::zeros(1, out_c);
            init.fill(&mut b, 1, out_c, rng);
            b
        });

        Ok(Self {
            weight, bias, in_c, out_c, kernel, stride, padding,
            backend: ConvBackend::CpuOnly,
            last_w_idx: None, last_b_idx: None,
            name: format!("conv2d_{in_c}_{out_c}_{kernel}"),
            cached_h: None, cached_w: None, cached_batch: None,
        })
    }

    /// Version qui panic en cas d'erreur (compat avec API v6.1).
    /// Préférer `try_new` dans le code nouveau.
    pub fn new<W: Initializer, B: Initializer>(
        in_c:        usize,
        out_c:       usize,
        kernel:      usize,
        stride:      usize,
        padding:     Padding,
        weight_init: &W,
        bias_init:   Option<&B>,
        rng:         &mut PcgEngine,
    ) -> Self {
        Self::try_new(in_c, out_c, kernel, stride, padding,
                      weight_init, bias_init, rng)
            .expect("Conv2d::new failed — utilise try_new pour gérer l'erreur")
    }

    pub fn with_name(mut self, name: &str) -> Self { self.name = name.into(); self }

    pub fn input_dims(mut self, h: usize, w: usize) -> Self {
        self.cached_h = Some(h); self.cached_w = Some(w); self
    }

    /// Configure explicitement le backend de cette couche.
    pub fn with_backend(mut self, backend: ConvBackend) -> Self {
        self.backend = backend; self
    }

    /// Helper pour le mode "descend" (équivalent v7-B).
    #[cfg(feature = "wgpu")]
    pub fn on_gpu_descend(self, ctx: Arc<GpuContext>,
                          pipelines: Arc<ConvGpuPipelines>) -> Self {
        self.with_backend(ConvBackend::GpuDescend { ctx, pipelines })
    }

    /// Helper pour le mode "keep VRAM" (équivalent v8).
    /// Cet alias `on_gpu` reste compatible avec l'API v7-B/v8.
    #[cfg(feature = "wgpu")]
    pub fn on_gpu(self, ctx: Arc<GpuContext>,
                  pipelines: Arc<ConvGpuPipelines>) -> Self {
        self.with_backend(ConvBackend::GpuKeepVram { ctx, pipelines })
    }

    /// Force le retour en mode CPU.
    pub fn on_cpu(mut self) -> Self {
        self.backend = ConvBackend::CpuOnly;
        self
    }

    pub fn uses_gpu(&self) -> bool { self.backend.is_gpu() }
}

impl Module for Conv2d {
    fn forward<'t>(&mut self, tape: &'t Tape, input: Var<'t>) -> Var<'t> {
        let (b, total_features) = input.shape();
        let (h, w) = match (self.cached_h, self.cached_w) {
            (Some(h), Some(w)) => (h, w),
            _ => {
                let per_channel = total_features / self.in_c;
                let side = (per_channel as f64).sqrt() as usize;
                assert_eq!(side * side, per_channel,
                    "Conv2d: utiliser .input_dims(h, w) pour des images non carrées");
                (side, side)
            }
        };
        self.cached_h = Some(h); self.cached_w = Some(w); self.cached_batch = Some(b);

        let cfg = ConvConfig {
            batch: b, in_c: self.in_c, h, w,
            kernel: self.kernel, stride: self.stride,
            padding: self.padding, out_c: self.out_c,
        };
        cfg.check().expect("ConvConfig invalide");

        let weight_v = tape.input(self.weight.clone());
        let bias_v   = self.bias.as_ref().map(|t| tape.input(t.clone()));
        self.last_w_idx = Some(weight_v.idx());
        self.last_b_idx = bias_v.as_ref().map(|v| v.idx());

        match &self.backend {
            ConvBackend::CpuOnly => input.conv2d_forward(weight_v, bias_v, b, self.in_c, h, w, self.out_c, self.kernel, self.stride, cfg.pad()),

            #[cfg(feature = "wgpu")]
            ConvBackend::GpuDescend { ctx, pipelines } => {
                input.conv2d_forward_gpu(weight_v, bias_v, cfg, ctx, pipelines)
            }

            #[cfg(feature = "wgpu")]
            ConvBackend::GpuKeepVram { ctx, pipelines } => {
                if !input.is_gpu() {
                    panic!("Conv2d (GpuKeepVram): input doit être sur GPU. \
                            Utilise input.to_gpu(ctx) avant, ou bascule en GpuDescend.");
                }
                input.conv2d_forward_gpu_keep_vram(weight_v, bias_v, cfg, ctx, pipelines)
            }
        }
    }

    fn parameter_indices(&self) -> Vec<usize> {
        let mut v = Vec::new();
        if let Some(i) = self.last_w_idx { v.push(i); }
        if let Some(i) = self.last_b_idx { v.push(i); }
        v
    }

    fn sync(&mut self, tape: &Tape) {
        if let Some(i) = self.last_w_idx { self.weight = tape.value(i); }
        if let Some(i) = self.last_b_idx { self.bias = Some(tape.value(i)); }
    }

    fn state_dict(&self) -> Vec<(String, Tensor)> {
        let mut v = vec![(format!("{}.weight", self.name), self.weight.clone())];
        if let Some(b) = &self.bias {
            v.push((format!("{}.bias", self.name), b.clone()));
        }
        v
    }

    fn load_state_dict(&mut self, dict: &HashMap<String, Tensor>) -> usize {
        let mut loaded = 0;
        if let Some(t) = dict.get(&format!("{}.weight", self.name)) {
            self.weight = t.clone(); loaded += 1;
        }
        if let Some(t) = dict.get(&format!("{}.bias", self.name)) {
            self.bias = Some(t.clone()); loaded += 1;
        }
        loaded
    }

    fn box_clone(&self) -> Box<dyn Module> { Box::new(self.clone()) }
}

impl Clone for Conv2d {
    fn clone(&self) -> Self {
        Self {
            weight: self.weight.clone(),
            bias:   self.bias.clone(),
            in_c: self.in_c, out_c: self.out_c,
            kernel: self.kernel, stride: self.stride,
            padding: self.padding,
            backend: self.backend.clone(),
            last_w_idx: None, last_b_idx: None,
            name: self.name.clone(),
            cached_h: self.cached_h, cached_w: self.cached_w,
            cached_batch: self.cached_batch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::init::{KaimingNormal, Zeros};

    #[test]
    fn try_new_validates_zero_channels() {
        let mut rng = PcgEngine::new(1);
        let r = Conv2d::try_new(0, 4, 3, 1, Padding::Same,
            &KaimingNormal, Some(&Zeros), &mut rng);
        assert!(matches!(r, Err(SciRustError::InvalidConfig(_))));
    }

    #[test]
    fn try_new_validates_zero_kernel() {
        let mut rng = PcgEngine::new(1);
        let r = Conv2d::try_new(1, 4, 0, 1, Padding::Same,
            &KaimingNormal, Some(&Zeros), &mut rng);
        assert!(matches!(r, Err(SciRustError::InvalidConfig(_))));
    }

    #[test]
    fn default_backend_is_cpu() {
        let mut rng = PcgEngine::new(1);
        let conv = Conv2d::new(1, 4, 3, 1, Padding::Same,
            &KaimingNormal, Some(&Zeros), &mut rng);
        assert!(!conv.uses_gpu());
        assert!(matches!(conv.backend, ConvBackend::CpuOnly));
    }

    #[test]
    fn with_backend_switches_mode() {
        let mut rng = PcgEngine::new(1);
        let conv = Conv2d::new(1, 4, 3, 1, Padding::Same,
            &KaimingNormal, Some(&Zeros), &mut rng);
        let conv = conv.on_cpu();   // explicit CPU
        assert!(matches!(conv.backend, ConvBackend::CpuOnly));
    }

    #[test]
    fn forward_cpu_works() {
        let mut rng = PcgEngine::new(1);
        let mut conv = Conv2d::new(1, 2, 3, 1, Padding::Valid,
            &KaimingNormal, Some(&Zeros), &mut rng).input_dims(5, 5);
        let tape = Tape::new();
        let x = tape.input(Tensor::zeros(1, 25));
        let y = conv.forward(&tape, x);
        assert_eq!(y.shape(), (1, 18));   // (5-3+1)² × 2
    }

    #[test]
    fn clone_preserves_backend() {
        let mut rng = PcgEngine::new(1);
        let conv1 = Conv2d::new(1, 4, 3, 1, Padding::Same,
            &KaimingNormal, Some(&Zeros), &mut rng);
        let conv2 = conv1.clone();
        assert_eq!(conv1.uses_gpu(), conv2.uses_gpu());
        // Indices de tape réinitialisés sur le clone
        assert_eq!(conv2.last_w_idx, None);
    }
}
