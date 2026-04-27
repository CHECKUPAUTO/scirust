// scirust-core/src/io/mod.rs
//
// Module io — sérialisation / désérialisation des tenseurs et modèles.

pub mod safetensors;

pub use safetensors::{
    save_safetensors, load_safetensors,
    serialize, deserialize,
};
