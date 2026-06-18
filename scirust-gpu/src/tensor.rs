//! VRAM-resident `GpuTensor` — the foundation for device-resident autograd.
//!
//! A `GpuTensor` holds an `f32` row-major matrix in GPU storage (a wgpu buffer)
//! plus shape metadata. The tensor **stays in VRAM** across operations; only
//! explicit `download` pulls data back to CPU.
//!
//! This is the building block for a `DeviceTensor::Gpu` variant in the autograd
//! tape — the tape's value storage can become a `DeviceTensor::Gpu(Arc<GpuTensor>)`
//! and ops that detect a GPU target dispatch through wgpu rather than CPU.

use crate::wgpu_backend::WgpuContext;
use crate::{BackendError, BackendResult};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// A float tensor resident in GPU memory (a wgpu storage buffer + shape).
#[derive(Debug, Clone)]
pub struct GpuTensor {
    pub(crate) buf: Arc<wgpu::Buffer>,
    pub rows: usize,
    pub cols: usize,
    /// Total element count: rows * cols
    pub elems: usize,
}

impl GpuTensor {
    /// Create a new GPU-resident tensor from CPU data.
    pub fn upload(ctx: &WgpuContext, data: &[f32], rows: usize, cols: usize) -> Self {
        let elems = rows * cols;
        let _bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;

        let buf = if data.is_empty()
        {
            ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu-tensor-empty"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        }
        else
        {
            ctx.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gpu-tensor"),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
        };

        Self {
            buf: Arc::new(buf),
            rows,
            cols,
            elems,
        }
    }

    /// Download from GPU to CPU `Vec<f32>`.
    pub fn download(&self, ctx: &WgpuContext) -> BackendResult<Vec<f32>> {
        if self.elems == 0
        {
            return Ok(Vec::new());
        }
        let bytes = (self.elems * std::mem::size_of::<f32>()) as u64;
        ctx.download_buffer(&self.buf, self.elems, bytes)
    }

    /// Row-major GEMM: `C = op(self) · op(other)`.
    /// Result stays resident. `ta`/`tb` request transpose of self/other.
    pub fn matmul(
        &self,
        other: &GpuTensor,
        ctx: &WgpuContext,
        ta: bool,
        tb: bool,
    ) -> BackendResult<GpuTensor> {
        let m = if ta { self.cols } else { self.rows };
        let k = if ta { self.rows } else { self.cols };
        let n = if tb { other.rows } else { other.cols };
        let kb = if tb { other.cols } else { other.rows };
        if k != kb
        {
            return Err(BackendError::ShapeMismatch(format!(
                "inner dims: op(A) {}×{}, op(B) {}×{}",
                m, k, kb, n
            )));
        }
        let elems = m * n;
        let bytes = (elems.max(1) * std::mem::size_of::<f32>()) as u64;
        let c_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-gemm-c"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if m != 0 && n != 0 && k != 0
        {
            ctx.encode_gemm(&self.buf, &other.buf, &c_buf, m, k, n, ta, tb, 1.0, 0.0);
        }
        Ok(GpuTensor {
            buf: Arc::new(c_buf),
            rows: m,
            cols: n,
            elems,
        })
    }

    /// Elementwise add: `self + other` (same shape), result resident.
    pub fn add(&self, other: &GpuTensor, ctx: &WgpuContext) -> BackendResult<GpuTensor> {
        ctx.ew_resident_tensor(self, other, 0)
    }

    /// Elementwise multiply: `self * other` (same shape), result resident.
    pub fn mul(&self, other: &GpuTensor, ctx: &WgpuContext) -> BackendResult<GpuTensor> {
        ctx.ew_resident_tensor(self, other, 1)
    }

    /// Elementwise ReLU, result resident.
    pub fn relu(&self, ctx: &WgpuContext) -> BackendResult<GpuTensor> {
        ctx.ew_resident_tensor(self, self, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgpu_backend::WgpuContext;

    fn get_ctx() -> Option<WgpuContext> {
        WgpuContext::new().ok()
    }

    #[test]
    fn test_gpu_tensor_upload_download() {
        let Some(ctx) = get_ctx()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = GpuTensor::upload(&ctx, &data, 2, 3);
        assert_eq!(t.rows, 2);
        assert_eq!(t.cols, 3);
        assert_eq!(t.elems, 6);
        let downloaded = t.download(&ctx).unwrap();
        assert_eq!(downloaded, data);
    }

    #[test]
    fn test_gpu_tensor_matmul() {
        let Some(ctx) = get_ctx()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let a = GpuTensor::upload(&ctx, &[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = GpuTensor::upload(&ctx, &[5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = a.matmul(&b, &ctx, false, false).unwrap();
        assert_eq!((c.rows, c.cols), (2, 2));
        let out = c.download(&ctx).unwrap();
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
        assert!((out[0] - 19.0).abs() < 1e-4);
        assert!((out[1] - 22.0).abs() < 1e-4);
    }

    #[test]
    fn test_gpu_tensor_add() {
        let Some(ctx) = get_ctx()
        else
        {
            eprintln!("wgpu: no adapter, skipping");
            return;
        };
        let a = GpuTensor::upload(&ctx, &[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = GpuTensor::upload(&ctx, &[5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = a.add(&b, &ctx).unwrap();
        let out = c.download(&ctx).unwrap();
        assert_eq!(out, vec![6.0, 8.0, 10.0, 12.0]);
    }
}
