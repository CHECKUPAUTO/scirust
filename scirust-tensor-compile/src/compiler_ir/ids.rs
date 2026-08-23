macro_rules! compiler_ir_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u32);

        impl $name {
            pub const fn new(value: u32) -> Self {
                Self(value)
            }

            pub const fn get(self) -> u32 {
                self.0
            }
        }
    };
}

compiler_ir_id!(IrValueId);
compiler_ir_id!(IrOperationId);
compiler_ir_id!(IrBlockId);
compiler_ir_id!(IrRegionId);
