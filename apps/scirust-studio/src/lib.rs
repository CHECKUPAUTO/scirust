//! SciRust Studio's interface.
//!
//! A Dioxus 0.7 **Web** application compiled to WebAssembly and loaded by the
//! Tauri shell from bundled static assets. It is not a website: there is no
//! server, no network access and no browser tab — the shell hands the
//! operating system's WebView a set of local files and grants the resulting
//! page exactly one privileged capability, the typed command bridge in
//! [`backend`].
//!
//! # How this crate is arranged, and why
//!
//! Everything that can be reasoned about without a DOM lives in a module
//! that compiles on the host:
//!
//! * [`model`] — the application state and the pure `(state, message) ->
//!   state` reducers. The lifecycle a user experiences is tested here.
//! * [`chart`] — the geometry that turns a result's real coordinates into
//!   SVG paths, including the reduction that must never hide a peak.
//! * [`actions`] — one registry driving the command palette, the keyboard
//!   shortcuts and the activity console.
//! * [`i18n`] — the string table, with a test that every key is genuinely
//!   translated.
//! * [`backend`] — the bridge types and the trait; the real implementation
//!   is `wasm32`-only, the mock is feature-gated and cannot ship.
//!
//! Only [`ui`] needs a browser, and it is compiled for `wasm32` alone. The
//! practical effect is that `cargo test` exercises the application's logic
//! in seconds, and `cargo check --target wasm32-unknown-unknown` type-checks
//! the components against it.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod actions;
pub mod backend;
pub mod chart;
pub mod i18n;
pub mod model;

#[cfg(target_arch = "wasm32")]
pub mod ui;

/// This interface's version, from the crate manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
