//! The [`module_gate!`] macro for defining a per-crate entitlement gate.
//!
//! Every licensed crate gates access the same way: a zero-sized capability token
//! whose only constructor requires verified [`Entitlements`](crate::Entitlements)
//! covering one [`Module`](crate::Module). That token + `unlock` boilerplate is
//! identical across crates, so it lives here as a macro; each crate then hangs
//! its gated entry points off the generated token in a normal `impl` block.

/// Define a capability-token gate for a licensable [`Module`](crate::Module).
///
/// Generates a `Copy` zero-sized token type whose **only** constructor,
/// `unlock(&Entitlements)`, calls [`Entitlements::require`](crate::Entitlements::require)
/// for the given module variant and returns the token. The token's field is
/// private, so no value of the type can exist without having passed the
/// entitlement check. The generated type also gets a `pub const MODULE`.
///
/// The crate adds its gated entry points in a separate `impl` block.
///
/// ```
/// scirust_license::module_gate! {
///     /// Access token for the Core module.
///     pub CoreAccess => Core
/// }
///
/// # fn main() {
/// use scirust_license::{Vendor, License, Module, verify_license};
/// let vendor = Vendor::from_seed(&[1u8; 32], 4);
/// let signed = vendor.issue_with_leaf(
///     License::new("Acme", "L-1", [Module::Core], 0, None),
///     0,
/// );
/// let ent = verify_license(&signed, &vendor.root(), 1).unwrap();
/// assert!(CoreAccess::unlock(&ent).is_ok());
/// assert_eq!(CoreAccess::MODULE, Module::Core);
/// # }
/// ```
#[macro_export]
macro_rules! module_gate {
    ($(#[$meta:meta])* $vis:vis $name:ident => $variant:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy)]
        $vis struct $name {
            // Private: no value can be built without going through `unlock`.
            _sealed: (),
        }

        impl $name {
            /// The licensable module this token unlocks.
            pub const MODULE: $crate::Module = $crate::Module::$variant;

            /// Unlock this module against a verified entitlement set: returns the
            /// token if the entitlements cover `Self::MODULE`, otherwise
            /// `LicenseError::NotEntitled`. Holding the `Entitlements` already
            /// proves the signature and validity window checked out (see
            /// `verify_license`).
            pub fn unlock(
                entitlements: &$crate::Entitlements,
            ) -> ::core::result::Result<Self, $crate::LicenseError> {
                entitlements.require(Self::MODULE)?;
                ::core::result::Result::Ok(Self { _sealed: () })
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{License, Module, Vendor, verify_license};

    // Define a gate with the macro at module scope, exactly as a crate would.
    crate::module_gate! {
        /// Test token for the Core module.
        pub CoreGate => Core
    }

    fn vendor() -> Vendor {
        Vendor::from_seed(&[3u8; 32], 4)
    }

    #[test]
    fn the_generated_const_names_the_right_module() {
        assert_eq!(CoreGate::MODULE, Module::Core);
    }

    #[test]
    fn unlock_succeeds_only_when_the_module_is_entitled() {
        let v = vendor();
        let core = v.issue_with_leaf(License::new("X", "L1", [Module::Core], 0, None), 0);
        let ent = verify_license(&core, &v.root(), 1).expect("valid");
        assert!(CoreGate::unlock(&ent).is_ok());

        let other = v.issue_with_leaf(License::new("X", "L2", [Module::Water], 0, None), 1);
        let ent2 = verify_license(&other, &v.root(), 1).expect("valid");
        assert_eq!(
            CoreGate::unlock(&ent2).err(),
            Some(crate::LicenseError::NotEntitled(Module::Core)),
        );
    }
}
