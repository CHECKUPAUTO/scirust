extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::{ComputeError, ComputeResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KernelFormat {
    Reference,
    Wgsl,
    SpirV,
    Ptx,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelModule {
    pub format: KernelFormat,
    pub entry_point: String,
    pub code: Vec<u8>,
}

impl KernelModule {
    pub fn new(
        format: KernelFormat,
        entry_point: impl Into<String>,
        code: impl Into<Vec<u8>>,
    ) -> ComputeResult<Self> {
        let entry_point = entry_point.into();
        let code = code.into();

        if entry_point.is_empty()
        {
            return Err(ComputeError::InvalidArgument(
                "kernel entry point must not be empty",
            ));
        }

        if code.is_empty()
        {
            return Err(ComputeError::InvalidArgument(
                "kernel module code must not be empty",
            ));
        }

        Ok(Self {
            format,
            entry_point,
            code,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_kernel_module_is_accepted() {
        let module = KernelModule::new(KernelFormat::Wgsl, "main", b"kernel".to_vec()).unwrap();

        assert_eq!(module.entry_point, "main");
        assert_eq!(module.format, KernelFormat::Wgsl);
    }

    #[test]
    fn empty_kernel_fields_are_rejected() {
        assert!(KernelModule::new(KernelFormat::Wgsl, "", b"kernel".to_vec()).is_err());

        assert!(KernelModule::new(KernelFormat::Wgsl, "main", Vec::new()).is_err());
    }
}
