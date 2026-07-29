//! The interface's entry point.
//!
//! On `wasm32` this launches the Dioxus application into the WebView the
//! Tauri shell created. On any other target it exits non-zero with an
//! explanation, because a Web frontend has nothing to do on a host: building
//! it is `dx build --platform web`, and a silent success here would let a
//! packaging script believe it had produced a frontend when it had produced
//! a command-line program that does nothing.

#![forbid(unsafe_code)]

#[cfg(target_arch = "wasm32")]
fn main() {
    scirust_studio_ui::ui::launch();
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    eprintln!(
        "scirust-studio-ui is a WebAssembly frontend and does not run on the host.\n\
         \n\
         Build it with:\n    dx build --release --platform web\n\
         \n\
         Then build the desktop application, which bundles the result:\n    \
         cargo tauri build\n\
         \n\
         See docs/studio/DESKTOP_BUILD.md."
    );
    std::process::ExitCode::FAILURE
}
