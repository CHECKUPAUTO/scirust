//! A configuration audit that fails the build if the shell's security
//! posture regresses.
//!
//! Written as a test rather than a review checklist because the failure mode
//! is silent: adding `shell:allow-execute` to make a feature work does not
//! break anything visible, and a CSP relaxed to `unsafe-inline` to clear a
//! console warning looks like a fix. Each assertion below is a property that
//! must survive every future change, with the reason attached.

const CONFIG: &str = include_str!("../tauri.conf.json");
const MAIN_WINDOW_CAPABILITY: &str = include_str!("../capabilities/main-window.json");

fn config() -> serde_json::Value {
    serde_json::from_str(CONFIG).expect("tauri.conf.json is valid JSON")
}

fn capability() -> serde_json::Value {
    serde_json::from_str(MAIN_WINDOW_CAPABILITY).expect("the capability file is valid JSON")
}

fn csp() -> String {
    config()["app"]["security"]["csp"]
        .as_str()
        .expect("a CSP must be configured")
        .to_string()
}

#[test]
fn a_content_security_policy_is_configured() {
    let value = &config()["app"]["security"]["csp"];
    assert!(
        !value.is_null(),
        "CSP must not be null: a null policy disables it entirely"
    );
    assert!(!csp().trim().is_empty());
}

/// The whole point of bundling assets: nothing is fetched from a network.
#[test]
fn the_policy_permits_no_remote_origins() {
    let csp = csp();
    for forbidden in ["http://", "https://", "//cdn", "unpkg", "jsdelivr"]
    {
        // `http://ipc.localhost` is Tauri's own IPC origin on Windows, not a
        // network destination; everything else is refused.
        let offending: Vec<&str> = csp
            .split(';')
            .filter(|d| d.contains(forbidden) && !d.contains("ipc.localhost"))
            .collect();
        assert!(
            offending.is_empty(),
            "`{forbidden}` appears in the CSP: {offending:?}"
        );
    }
}

#[test]
fn the_policy_uses_no_wildcards() {
    let csp = csp();
    assert!(
        !csp.contains(" *") && !csp.contains("*;") && !csp.contains("'*'"),
        "a wildcard source defeats the policy: {csp}"
    );
}

/// `unsafe-inline` is the reflex fix for a console warning and it undoes
/// most of what a CSP is for.
#[test]
fn the_policy_allows_no_inline_or_evaluated_script() {
    let csp = csp();
    assert!(!csp.contains("unsafe-inline"), "{csp}");
    // `wasm-unsafe-eval` is required to instantiate a WebAssembly module and
    // is *not* `unsafe-eval`: it permits WASM compilation only, not
    // `eval()` of JavaScript. It is the minimum a Dioxus/WASM frontend needs.
    assert!(
        !csp.contains("'unsafe-eval'"),
        "unsafe-eval permits arbitrary JavaScript evaluation: {csp}"
    );
    assert!(
        csp.contains("'wasm-unsafe-eval'"),
        "the WASM frontend cannot instantiate without this: {csp}"
    );
}

#[test]
fn the_policy_forbids_framing_and_object_embedding() {
    let csp = csp();
    for directive in [
        "object-src 'none'",
        "frame-src 'none'",
        "frame-ancestors 'none'",
    ]
    {
        assert!(csp.contains(directive), "missing `{directive}`: {csp}");
    }
}

#[test]
fn the_default_source_is_self() {
    assert!(csp().starts_with("default-src 'self'"), "{}", csp());
}

/// The frontend must not be able to reach any of these. File dialogs, the
/// sidecar and every filesystem access run in native Rust, behind typed
/// commands that exchange *contents* rather than paths.
#[test]
fn the_main_window_has_no_dangerous_permission() {
    let text = MAIN_WINDOW_CAPABILITY;
    for forbidden in [
        "shell:",
        "http:",
        "fs:",
        "process:",
        "os:allow-exec",
        "updater:",
        "deep-link:",
        // The shell links `tauri-plugin-dialog` (and, transitively,
        // `tauri-plugin-fs`) so `studio_open_scenario` can show a native
        // picker from Rust. Granting the webview `dialog:*` would let it open
        // its own picker and hold the resulting path — which is the one thing
        // that command's design exists to prevent.
        "dialog:",
    ]
    {
        assert!(
            !text.contains(forbidden),
            "the main window must not be granted `{forbidden}` — the frontend \
             has no feature that needs it, and granting it would let the \
             webview act as the user"
        );
    }
}

/// Specifically the two that would let the webview run programs.
#[test]
fn the_frontend_can_neither_spawn_nor_execute() {
    let text = MAIN_WINDOW_CAPABILITY;
    assert!(!text.contains("allow-spawn"), "{text}");
    assert!(!text.contains("allow-execute"), "{text}");
}

#[test]
fn the_capability_is_scoped_to_the_main_window_only() {
    let capability = capability();
    let windows = capability["windows"]
        .as_array()
        .expect("a capability must name its windows");
    assert_eq!(
        windows.len(),
        1,
        "a future window must get its own capability rather than inheriting this one"
    );
    assert_eq!(windows[0], "main");
}

#[test]
fn the_asset_protocol_is_disabled() {
    // The asset protocol exposes arbitrary local files to the webview by
    // path. Nothing here reads files from the frontend.
    assert_eq!(
        config()["app"]["security"]["assetProtocol"]["enable"],
        serde_json::Value::Bool(false)
    );
}

#[test]
fn csp_modification_is_not_disabled() {
    assert_eq!(
        config()["app"]["security"]["dangerousDisableAssetCspModification"],
        serde_json::Value::Bool(false),
        "disabling this lets bundled assets escape the policy"
    );
}

#[test]
fn the_prototype_is_frozen() {
    assert_eq!(
        config()["app"]["security"]["freezePrototype"],
        serde_json::Value::Bool(true)
    );
}

/// Production assets come from the bundle, never from a server.
#[test]
fn the_frontend_is_served_from_bundled_files() {
    let config = config();
    let dist = config["build"]["frontendDist"]
        .as_str()
        .expect("frontendDist must be configured");
    assert!(
        !dist.starts_with("http"),
        "a production build must not point at a server: {dist}"
    );
    assert!(dist.starts_with("../"), "{dist}");
}

#[test]
fn the_window_matches_the_specified_geometry() {
    let config = config();
    let window = &config["app"]["windows"][0];
    assert_eq!(window["title"], "SciRust Studio");
    assert_eq!(window["label"], "main");
    assert_eq!(window["width"], 1440);
    assert_eq!(window["height"], 900);
    assert_eq!(window["minWidth"], 1100);
    assert_eq!(window["minHeight"], 700);
    assert_eq!(window["resizable"], serde_json::Value::Bool(true));
}

/// The worker ships with the application; it is never downloaded.
#[test]
fn the_worker_is_bundled_as_an_external_binary() {
    let config = config();
    let bins = config["bundle"]["externalBin"]
        .as_array()
        .expect("externalBin must list the worker");
    assert_eq!(bins.len(), 1);
    assert_eq!(bins[0], "binaries/scirust-studio-worker");
}

/// No updater endpoint may be configured: signing and updating are a later
/// phase, and a half-configured updater is worse than none.
#[test]
fn no_updater_is_configured() {
    let config = config();
    assert!(
        config["plugins"].get("updater").is_none(),
        "the updater is deliberately not part of this phase"
    );
    assert!(!CONFIG.contains("\"endpoints\""), "{CONFIG}");
    assert!(!CONFIG.contains("pubkey"), "{CONFIG}");
}

/// Every icon the configuration names must exist, and a Windows resource
/// icon must be among them.
///
/// This is here because the failure it prevents is invisible on the
/// platform most of this is developed on. `tauri-build` only demands
/// `icons/icon.ico` when it is building a Windows resource, so a missing
/// one compiles perfectly on Linux and macOS and then fails on a Windows
/// runner with `` `icons/icon.ico` not found ``. It cost a red CI job once;
/// asserting it here means the next time it fails in a second, locally, on
/// whatever platform the change was written on.
#[test]
fn every_configured_icon_exists_including_the_windows_resource() {
    let icons_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let config = config();
    let icons = config["bundle"]["icon"]
        .as_array()
        .expect("the bundle must name its icons");
    assert!(!icons.is_empty(), "an application needs an icon");

    for entry in icons
    {
        let relative = entry.as_str().expect("an icon path is a string");
        let path = icons_dir.join(relative);
        let metadata = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("{relative} is configured but missing: {e}"));
        // A one-pixel placeholder satisfies "exists" and satisfies nobody
        // else; the smallest real icon here is several hundred bytes.
        assert!(
            metadata.len() > 256,
            "{relative} is {} bytes — that is a placeholder, not an icon. \
             Regenerate with scripts/studio/generate-icons.py",
            metadata.len()
        );
    }

    let ico = icons
        .iter()
        .filter_map(|i| i.as_str())
        .find(|i| i.ends_with(".ico"));
    assert!(
        ico.is_some(),
        "tauri-build requires a .ico to embed as the Windows executable's \
         resource; without one the Windows build fails and no other platform \
         notices"
    );
}

/// The Windows resource icon must be a real multi-size ICO.
///
/// Checked by reading the directory rather than by trusting the extension:
/// a PNG renamed to `.ico` passes an existence check and fails the actual
/// Windows build.
#[test]
fn the_windows_icon_is_a_real_multi_size_ico() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("icons/icon.ico");
    let bytes = std::fs::read(&path).expect("icons/icon.ico");

    assert!(
        bytes.len() > 22,
        "an ICO has a header and at least one entry"
    );
    assert_eq!(&bytes[0..2], &[0, 0], "reserved field");
    assert_eq!(&bytes[2..4], &[1, 0], "type 1 = icon");

    let count = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
    assert!(
        count >= 4,
        "an application icon needs several sizes; found {count}"
    );

    // Windows picks 16 for the title bar and 32 for the taskbar; an icon
    // without them is upscaled and looks it.
    let mut sizes: Vec<u32> = Vec::new();
    for index in 0..count
    {
        let entry = 6 + index * 16;
        let width = bytes[entry];
        sizes.push(if width == 0 { 256 } else { u32::from(width) });
    }
    for required in [16, 32]
    {
        assert!(
            sizes.contains(&required),
            "icon.ico has {sizes:?} but needs {required}"
        );
    }
}
