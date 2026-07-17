//! The reversible outer structure (spec §11–§13): the exact v0.1 round function,
//! the balanced Feistel shell, and the deliberately-weakened control variants.

pub mod controls;
pub mod feistel;
pub mod round;

pub use controls::Variant;
pub use feistel::{State, forward, inverse};
pub use round::{RoundLayers, f_round, f_round_traced, g_pre_rotation};
