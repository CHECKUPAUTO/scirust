// scirust-gpu/src/cuda_backend.rs
//
// CudaBackend — implémentation SimdBackend déléguant aux GPUs NVIDIA via `cust`.
//
// Activation : --features cuda
// Sans la feature : le module compile en stub qui retombe sur ScalarBackend.
//
// Modèle d'exécution :
//   - Les buffers CPU sont copiés sur le GPU à chaque appel (DeviceBuffer)
//   - Les kernels sont compilés une fois au démarrage et cachés
//   - Pour une utilisation réelle, on voudra :
//       * un Tensor "GPU-resident" qui évite les copies aller-retour
//       * un cache de DeviceBuffer pour réutiliser la VRAM
//       * un stream CUDA partagé pour overlap copy/compute
//
// Cette première version est un point d'ancrage : la structure est correcte,
// les kernels PTX inline sont prêts, l'API SimdBackend est respectée.

#![cfg(feature = "cuda")]

use cust::launch;
use cust::prelude::*;
use std::sync::OnceLock;

use scirust_core::matrix::backend::SimdBackend;
use scirust_core::matrix::view::{MatrixShape, MatrixView, MatrixViewMut};

// ------------------------------------------------------------------ //
//  PTX inline — kernels compilés à part dans cuda/kernels/*.cu        //
//  Pour un build automatique : utiliser `nvcc --ptx` dans build.rs    //
// ------------------------------------------------------------------ //

const SAXPY_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry saxpy(
    .param .f32 alpha,
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u32 n
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<5>;
    .reg .f32   %f<5>;
    .reg .b64   %rd<8>;

    ld.param.f32  %f1, [alpha];
    ld.param.u64  %rd1, [x_ptr];
    ld.param.u64  %rd2, [y_ptr];
    ld.param.u32  %r2, [n];

    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %ntid.x;
    mad.lo.s32    %r1, %r3, %r4, %tid.x;
    setp.ge.s32   %p1, %r1, %r2;
    @%p1 bra      DONE;

    mul.wide.s32  %rd3, %r1, 4;
    add.s64       %rd4, %rd1, %rd3;
    add.s64       %rd5, %rd2, %rd3;
    ld.global.f32 %f2, [%rd4];
    ld.global.f32 %f3, [%rd5];
    fma.rn.f32    %f4, %f1, %f2, %f3;
    st.global.f32 [%rd5], %f4;

DONE:
    ret;
}
"#;

// ------------------------------------------------------------------ //
//  Contexte CUDA — initialisé paresseusement                          //
// ------------------------------------------------------------------ //

struct CudaCtx {
    _ctx: Context,
    module_saxpy: Module,
    stream: Stream,
}

static CTX: OnceLock<CudaCtx> = OnceLock::new();

fn ctx() -> &'static CudaCtx {
    CTX.get_or_init(|| {
        cust::init(CudaFlags::empty()).expect("cust init failed");
        let device = Device::get_device(0).expect("aucun GPU CUDA détecté");
        let ctx = Context::new(device).expect("CUDA context");
        let module = Module::from_ptx(SAXPY_PTX, &[]).expect("PTX saxpy");
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("stream");
        CudaCtx {
            _ctx: ctx,
            module_saxpy: module,
            stream,
        }
    })
}

// ------------------------------------------------------------------ //
//  CudaBackend                                                        //
// ------------------------------------------------------------------ //

pub struct CudaBackend;

impl CudaBackend {
    /// Vérifie qu'un GPU CUDA est disponible. Retourne None sinon
    /// (l'appelant peut alors retomber sur un backend CPU).
    pub fn try_init() -> Option<Self> {
        cust::init(CudaFlags::empty()).ok()?;
        Device::get_device(0).ok()?;
        Some(Self)
    }
}

impl SimdBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn saxpy_f32(&self, alpha: f32, x: &[f32], y: &mut [f32]) {
        let ctx = ctx();
        let n = x.len() as u32;
        let x_dev = DeviceBuffer::from_slice(x).expect("upload x");
        let mut y_dev = DeviceBuffer::from_slice(y).expect("upload y");

        let func = ctx.module_saxpy.get_function("saxpy").expect("saxpy fn");
        let block = 256u32;
        let grid = (n + block - 1) / block;

        unsafe {
            launch!(func<<<grid, block, 0, ctx.stream>>>(
                alpha,
                x_dev.as_device_ptr(),
                y_dev.as_device_ptr(),
                n
            ))
            .expect("launch saxpy");
        }
        ctx.stream.synchronize().expect("sync");
        y_dev.copy_to(y).expect("download y");
    }

    fn daxpy_f64(&self, alpha: f64, x: &[f64], y: &mut [f64]) {
        // TODO : kernel f64 (PTX similaire avec .f64). Fallback CPU pour l'instant.
        for (yi, xi) in y.iter_mut().zip(x.iter())
        {
            *yi += alpha * xi;
        }
    }

    fn sdot_f32(&self, x: &[f32], y: &[f32]) -> f32 {
        // TODO : kernel reduction (cublas SDOT idéal, ou impl maison à 2 passes).
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn ddot_f64(&self, x: &[f64], y: &[f64]) -> f64 {
        x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
    }

    fn sgemv_f32(&self, alpha: f32, a: MatrixView<f32>, x: &[f32], beta: f32, y: &mut [f32]) {
        // TODO : cuBLAS SGEMV ou kernel custom (1 thread = 1 row de A)
        let (m, k) = a.shape();
        for i in 0..m
        {
            let row = a.row_slice(i).expect("row_slice");
            let dot: f32 = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            y[i] = alpha * dot + beta * y[i];
        }
    }

    fn sgemm_f32(
        &self,
        alpha: f32,
        a: MatrixView<f32>,
        b: MatrixView<f32>,
        beta: f32,
        mut c: MatrixViewMut<f32>,
    ) {
        // TODO : cuBLAS SGEMM (le bon choix en pratique) ou kernel tiled custom
        // (algorithme naïf 1 thread = 1 cell C[i,j])
        let (m, k) = a.shape();
        let (_, n) = b.shape();
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = 0.0f32;
                for p in 0..k
                {
                    acc += a[(i, p)] * b[(p, j)];
                }
                c[(i, j)] = alpha * acc + beta * c[(i, j)];
            }
        }
    }

    fn relu_f32(&self, v: &mut [f32]) {
        // TODO : kernel max(x, 0) (très simple en CUDA)
        for x in v.iter_mut()
        {
            *x = x.max(0.0);
        }
    }
}

// ------------------------------------------------------------------ //
//  Stub sans la feature cuda                                           //
// ------------------------------------------------------------------ //

#[cfg(not(feature = "cuda"))]
pub struct CudaBackend;

#[cfg(not(feature = "cuda"))]
impl CudaBackend {
    pub fn try_init() -> Option<Self> {
        None
    }
}

// ------------------------------------------------------------------ //
//  Tests (skip si pas de GPU)                                          //
// ------------------------------------------------------------------ //
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn cuda_init_or_skip() {
        match CudaBackend::try_init()
        {
            Some(b) =>
            {
                let x = vec![1.0f32, 2.0, 3.0, 4.0];
                let mut y = vec![0.0f32; 4];
                b.saxpy_f32(2.0, &x, &mut y);
                assert_eq!(y, vec![2.0, 4.0, 6.0, 8.0]);
            },
            None =>
            {
                eprintln!("[skip] aucun GPU CUDA disponible");
            },
        }
    }
}
