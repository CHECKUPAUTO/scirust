//! Fail-closed CUDA availability preflight for the SciAgent Thor gate.
//!
//! Ordinary CUDA-enabled development builds may run on machines without a CUDA
//! runtime/device, so this test keeps the repository's established skip behavior
//! by default. Setting `SCIRUST_REQUIRE_CUDA` turns that absence into a hard
//! failure. The self-hosted Thor gate sets it to `1`.

#![cfg(feature = "cuda")]

use scirust_cuda::CudaRawRuntime;

#[test]
fn required_cuda_device_can_be_acquired() {
    match CudaRawRuntime::new(0)
    {
        Ok(runtime) =>
        {
            let info = runtime.device_info();
            eprintln!(
                "cuda device {}: {} sm_{}{}",
                info.ordinal, info.name, info.compute_capability.0, info.compute_capability.1
            );
        },
        Err(error) =>
        {
            assert!(
                std::env::var_os("SCIRUST_REQUIRE_CUDA").is_none(),
                "SCIRUST_REQUIRE_CUDA is set, so a real CUDA device is mandatory, but device 0 \
                 could not be acquired: {error}"
            );
            eprintln!("cuda: no device, skipping required-device preflight ({error})");
        },
    }
}
