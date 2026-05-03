// scirust-core/src/io/mod.rs
//
// Module io — sérialisation / désérialisation des tenseurs et modèles.

pub mod safetensors;

pub use safetensors::{
    save_safetensors, load_safetensors,
    serialize, deserialize,
    serialize_with_metadata, deserialize_with_metadata,
    serialize_state_dict, deserialize_state_dict,
    save_state_dict, load_state_dict,
};
