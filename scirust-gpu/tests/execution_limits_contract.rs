#[cfg(any(feature = "wgpu", feature = "cuda"))]
use scirust_compute::ComputeBackend;

#[cfg(feature = "wgpu")]
use scirust_gpu::WgpuComputeAdapter;

#[cfg(feature = "cuda")]
use scirust_gpu::CudaComputeAdapter;

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_execution_limits_match_reported_workgroup_facts() {
    let adapter = match WgpuComputeAdapter::new()
    {
        Ok(adapter) => adapter,
        Err(error) if std::env::var_os("SCIRUST_REQUIRE_WGPU").is_some() =>
        {
            panic!("SCIRUST_REQUIRE_WGPU is set but WGPU acquisition failed: {error}")
        },
        Err(error) =>
        {
            eprintln!("wgpu: {error}; skipping execution-limit contract test");
            return;
        },
    };

    let expected = adapter.capabilities().max_workgroup_size.map(Some);
    let limits = ComputeBackend::execution_limits(&adapter);
    assert_eq!(limits.max_workgroup_size, expected);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_execution_limits_match_reported_workgroup_facts() {
    let adapter = match CudaComputeAdapter::new()
    {
        Ok(adapter) => adapter,
        Err(error) if std::env::var_os("SCIRUST_REQUIRE_CUDA").is_some() =>
        {
            panic!("SCIRUST_REQUIRE_CUDA is set but CUDA acquisition failed: {error}")
        },
        Err(error) =>
        {
            eprintln!("cuda: {error}; skipping execution-limit contract test");
            return;
        },
    };

    let expected = adapter.capabilities().max_workgroup_size.map(Some);
    let limits = ComputeBackend::execution_limits(&adapter);
    assert_eq!(limits.max_workgroup_size, expected);
}
