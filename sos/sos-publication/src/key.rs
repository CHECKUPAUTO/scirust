//! Stable intra-publication handles: [`ClaimKey`], [`FigureKey`], [`TableKey`],
//! [`RefKey`], [`SectionId`].
//!
//! These are the short, human-authored names prose uses to point at a registry
//! entry ("as Figure `fig-fit` shows", "claim `C1` is supported by …"). They are
//! **not** content addresses — a claim's content address is its
//! [`content_id`](crate::claim::Claim::content_id), an object's is its
//! [`ObjectId`](sos_core::ObjectId). Keeping the two apart is deliberate: a key
//! is an editorial label the author controls; an address is a hash the author
//! cannot forge. Distinct newtypes stop a figure key being passed where a claim
//! key is meant.

use serde::{Deserialize, Serialize};
use sos_core::canonical::{Canonical, CanonicalEncoder};

/// Define a `String`-newtype key with the usual conveniences and a canonical
/// encoding (so it can sit inside a hashed [`Body`](sos_core::Body)).
macro_rules! string_key
{
    ($(#[$meta:meta])* $name:ident) =>
    {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name(pub String);

        impl $name
        {
            #[doc = concat!("Construct a [`", stringify!($name), "`] from any string-like value.")]
            #[must_use]
            pub fn new(value: impl Into<String>) -> Self
            {
                Self(value.into())
            }

            /// Borrow the key's string value.
            #[must_use]
            pub fn as_str(&self) -> &str
            {
                &self.0
            }
        }

        impl Canonical for $name
        {
            fn encode(&self, enc: &mut CanonicalEncoder)
            {
                enc.str(&self.0);
            }
        }

        impl core::fmt::Display for $name
        {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
            {
                f.write_str(&self.0)
            }
        }

        impl From<&str> for $name
        {
            fn from(value: &str) -> Self
            {
                Self(value.to_owned())
            }
        }

        impl From<String> for $name
        {
            fn from(value: String) -> Self
            {
                Self(value)
            }
        }
    };
}

string_key! {
    /// A claim's stable handle within a publication (e.g. `"C1"`).
    ClaimKey
}
string_key! {
    /// A figure's stable handle within a publication (e.g. `"fig-fit"`).
    FigureKey
}
string_key! {
    /// A table's stable handle within a publication (e.g. `"tbl-residuals"`).
    TableKey
}
string_key! {
    /// A bibliography entry's stable handle within a publication (e.g. `"ref-kepler"`).
    RefKey
}
string_key! {
    /// A section's stable handle within a publication (e.g. `"methods"`).
    SectionId
}
