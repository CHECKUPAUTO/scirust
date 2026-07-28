macro_rules! resource_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(u64);

        impl $name {
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            pub const fn get(self) -> u64 {
                self.0
            }
        }
    };
}

resource_id!(BufferId);
resource_id!(KernelId);
resource_id!(StreamId);
resource_id!(EventId);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_identifiers_preserve_values() {
        assert_eq!(BufferId::new(1).get(), 1);
        assert_eq!(KernelId::new(2).get(), 2);
        assert_eq!(StreamId::new(3).get(), 3);
        assert_eq!(EventId::new(4).get(), 4);
    }
}
