//! SciRust Studio desktop binary.

// On Windows a GUI build must not open a console window behind the
// application. `windows_subsystem = "windows"` suppresses it — but only in a
// release build, so `cargo run` during development keeps its diagnostics.
#![cfg_attr(all(not(debug_assertions), windows), windows_subsystem = "windows")]
#![forbid(unsafe_code)]

fn main() {
    std::process::exit(scirust_studio_desktop_lib::run());
}
