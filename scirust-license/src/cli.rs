//! Command logic for the `license-tool` binary, factored out of `main` so it is
//! deterministic and unit-testable: [`run`] takes the argument vector and the
//! current time explicitly and returns captured output plus an exit code. The
//! binary is a thin wrapper that supplies `std::env::args()` and the real clock.

use crate::hashsig::{Hash, hex_decode, hex_encode};
use crate::{
    DEMO_HEIGHT, Entitlements, License, LicenseError, Module, SignedLicense, Vendor, demo_root,
    demo_seed, verify_license, verify_license_on_node,
};

/// Captured result of a CLI invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliResult {
    /// Text to print to stdout.
    pub stdout: String,
    /// Process exit code (0 ok, 1 verification failed, 2 usage error).
    pub exit: i32,
}

impl CliResult {
    fn ok(stdout: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            exit: 0,
        }
    }

    fn fail(stdout: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            exit: 1,
        }
    }

    fn usage(stdout: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            exit: 2,
        }
    }
}

const USAGE: &str = "\
license-tool — SciRust module licensing

USAGE:
    license-tool modules
    license-tool keygen [--seed-hex HEX] [--height N]
    license-tool issue --licensee NAME --id ID --modules a,b,c --leaf N \
[--expires UNIX] [--node MACHINE_ID] [--seed-hex HEX] [--height N]
    license-tool inspect <file> [--node MACHINE_ID] [--root-hex HEX] [--now UNIX]
    license-tool check <file> --module M [--node MACHINE_ID] [--root-hex HEX] [--now UNIX]

Notes:
    * With no --seed-hex/--root-hex, the bundled demo key is used.
    * A real vendor keeps its seed offline; verifiers embed only the root.
    * --node binds a license to one machine (issue) and presents this machine's
      id when verifying (inspect/check). A node-locked license fails to verify
      without the matching --node.";

/// Run a `license-tool` invocation. `args` excludes the program name; `now` is
/// the current Unix time (injected for determinism).
pub fn run(args: &[String], now: u64) -> CliResult {
    let Some((cmd, rest)) = args.split_first()
    else
    {
        return CliResult::usage(USAGE.to_string());
    };
    match cmd.as_str()
    {
        "modules" => cmd_modules(),
        "keygen" => cmd_keygen(rest),
        "issue" => cmd_issue(rest, now),
        "inspect" => cmd_inspect(rest, now),
        "check" => cmd_check(rest, now),
        "help" | "--help" | "-h" => CliResult::ok(USAGE.to_string()),
        other => CliResult::usage(format!("unknown command '{other}'\n\n{USAGE}")),
    }
}

fn cmd_modules() -> CliResult {
    let mut out = String::from("Licensable modules:\n");
    let mut all: Vec<Module> = Module::ALL.to_vec();
    all.push(Module::Industrial);
    all.sort_by_key(|m| m.code());
    for m in all
    {
        out.push_str(&format!("  {:<24} {}\n", m.as_str(), m.description()));
    }
    CliResult::ok(out)
}

fn cmd_keygen(rest: &[String]) -> CliResult {
    let flags = match Flags::parse(rest)
    {
        Ok(f) => f,
        Err(e) => return CliResult::usage(e),
    };
    let seed = match flags.seed(demo_seed())
    {
        Ok(s) => s,
        Err(e) => return CliResult::usage(e),
    };
    let height = match flags.height(DEMO_HEIGHT)
    {
        Ok(h) => h,
        Err(e) => return CliResult::usage(e),
    };
    let vendor = Vendor::from_seed(&seed, height);
    // `from_seed` clamps the requested height into a supported range, so report
    // the *effective* height (capacity is 2^effective_height) rather than the
    // possibly-clamped request, which would otherwise disagree with capacity.
    let effective_height = vendor.capacity().trailing_zeros();
    CliResult::ok(format!(
        "root (public key): {}\ncapacity: {} one-time licenses (height {})\n",
        hex_encode(&vendor.root()),
        vendor.capacity(),
        effective_height
    ))
}

fn cmd_issue(rest: &[String], now: u64) -> CliResult {
    let flags = match Flags::parse(rest)
    {
        Ok(f) => f,
        Err(e) => return CliResult::usage(e),
    };
    let (Some(licensee), Some(id), Some(modules_str)) =
        (flags.get("licensee"), flags.get("id"), flags.get("modules"))
    else
    {
        return CliResult::usage(format!(
            "issue requires --licensee, --id and --modules\n\n{USAGE}"
        ));
    };
    let modules = match parse_modules(modules_str)
    {
        Ok(m) => m,
        Err(e) => return CliResult::usage(e),
    };
    let expires = match flags.get("expires").map(|s| parse_u64(s, "expires"))
    {
        Some(Ok(v)) => Some(v),
        Some(Err(e)) => return CliResult::usage(e),
        None => None,
    };
    // Each one-time leaf may sign at most one distinct license (reuse leaks
    // Lamport secrets), so the caller must own leaf allocation explicitly: there
    // is no safe default. Defaulting to 0 would make two `issue` runs without
    // --leaf reuse the same leaf. Require it.
    let leaf = match flags.get("leaf").map(|s| parse_u64(s, "leaf"))
    {
        Some(Ok(v)) => v as u32,
        Some(Err(e)) => return CliResult::usage(e),
        None =>
        {
            return CliResult::usage(format!(
                "issue requires --leaf N (each leaf signs at most one license; \
                 reusing a leaf is cryptographically unsafe)\n\n{USAGE}"
            ));
        },
    };
    let seed = match flags.seed(demo_seed())
    {
        Ok(s) => s,
        Err(e) => return CliResult::usage(e),
    };
    let height = match flags.height(DEMO_HEIGHT)
    {
        Ok(h) => h,
        Err(e) => return CliResult::usage(e),
    };
    let vendor = Vendor::from_seed(&seed, height);
    if leaf as usize >= vendor.capacity()
    {
        return CliResult::usage(format!(
            "leaf {leaf} out of range for height {height} (capacity {})",
            vendor.capacity()
        ));
    }
    let mut license = License::new(licensee, id, modules, now, expires);
    if let Some(machine_id) = flags.get("node")
    {
        license = license.with_node_lock(machine_id);
    }
    let signed = vendor.issue_with_leaf(license, leaf);
    CliResult::ok(format!("{}\n", signed.to_json()))
}

fn cmd_inspect(rest: &[String], now: u64) -> CliResult {
    let flags = match Flags::parse(rest)
    {
        Ok(f) => f,
        Err(e) => return CliResult::usage(e),
    };
    let Some(path) = flags.positional.first()
    else
    {
        return CliResult::usage(format!("inspect requires a <file>\n\n{USAGE}"));
    };
    let signed = match load_license(path)
    {
        Ok(s) => s,
        Err(e) => return CliResult::fail(e),
    };
    let root = match flags.root(demo_root())
    {
        Ok(r) => r,
        Err(e) => return CliResult::usage(e),
    };
    let now = match flags.get("now").map(|s| parse_u64(s, "now"))
    {
        Some(Ok(v)) => v,
        Some(Err(e)) => return CliResult::usage(e),
        None => now,
    };
    match verify_node_aware(&signed, &root, now, flags.get("node"))
    {
        Ok(ent) =>
        {
            let mods: Vec<&str> = ent.modules().iter().map(|m| m.as_str()).collect();
            let expiry = match ent.expires_at()
            {
                Some(t) => t.to_string(),
                None => "never".to_string(),
            };
            CliResult::ok(format!(
                "VALID\n  licensee: {}\n  id:       {}\n  modules:  {}\n  expires:  {}\n  node:     {}\n  digest:   {}\n",
                ent.licensee(),
                ent.license_id(),
                mods.join(", "),
                expiry,
                node_status(&signed),
                signed.license.digest_hex(),
            ))
        },
        Err(e) => CliResult::fail(format!("INVALID: {e}\n")),
    }
}

fn cmd_check(rest: &[String], now: u64) -> CliResult {
    let flags = match Flags::parse(rest)
    {
        Ok(f) => f,
        Err(e) => return CliResult::usage(e),
    };
    let Some(path) = flags.positional.first()
    else
    {
        return CliResult::usage(format!("check requires a <file>\n\n{USAGE}"));
    };
    let Some(module_str) = flags.get("module")
    else
    {
        return CliResult::usage(format!("check requires --module\n\n{USAGE}"));
    };
    let Some(module) = Module::from_id(module_str)
    else
    {
        return CliResult::usage(format!("unknown module '{module_str}'"));
    };
    let signed = match load_license(path)
    {
        Ok(s) => s,
        Err(e) => return CliResult::fail(e),
    };
    let root = match flags.root(demo_root())
    {
        Ok(r) => r,
        Err(e) => return CliResult::usage(e),
    };
    let now = match flags.get("now").map(|s| parse_u64(s, "now"))
    {
        Some(Ok(v)) => v,
        Some(Err(e)) => return CliResult::usage(e),
        None => now,
    };
    match verify_node_aware(&signed, &root, now, flags.get("node"))
    {
        Ok(ent) => match ent.require(module)
        {
            Ok(()) => CliResult::ok(format!("GRANTED: {module}\n")),
            Err(e) => CliResult::fail(format!("DENIED: {e}\n")),
        },
        Err(e) => CliResult::fail(format!("DENIED: {e}\n")),
    }
}

fn load_license(path: &str) -> Result<SignedLicense, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    SignedLicense::from_json(&text).map_err(|e| format!("cannot parse {path}: {e}"))
}

/// Verify, choosing the node-aware path when a `--node` id is supplied. Without
/// `--node`, a node-locked license is refused with `NodeRequired`.
fn verify_node_aware(
    signed: &SignedLicense,
    root: &Hash,
    now: u64,
    node: Option<&str>,
) -> Result<Entitlements, LicenseError> {
    match node
    {
        Some(id) => verify_license_on_node(signed, root, now, id),
        None => verify_license(signed, root, now),
    }
}

/// One-line description of a license's node binding, for `inspect`.
fn node_status(signed: &SignedLicense) -> String {
    match signed.license.node_lock
    {
        None => "floating (any machine)".to_string(),
        Some(fp) => format!("locked ({}…)", &hex_encode(&fp)[..16]),
    }
}

fn parse_modules(s: &str) -> Result<Vec<Module>, String> {
    let mut out = Vec::new();
    for part in s.split(',')
    {
        let id = part.trim();
        if id.is_empty()
        {
            continue;
        }
        match Module::from_id(id)
        {
            Some(m) => out.push(m),
            None => return Err(format!("unknown module '{id}'")),
        }
    }
    if out.is_empty()
    {
        return Err("no modules specified".to_string());
    }
    Ok(out)
}

fn parse_u64(s: &str, field: &str) -> Result<u64, String> {
    s.parse::<u64>()
        .map_err(|_| format!("invalid {field}: '{s}' is not a non-negative integer"))
}

fn parse_root_hex(s: &str) -> Result<Hash, String> {
    let bytes = hex_decode(s).ok_or_else(|| format!("invalid hex: '{s}'"))?;
    if bytes.len() != 32
    {
        return Err(format!(
            "a root must be 32 bytes (64 hex chars), got {}",
            bytes.len()
        ));
    }
    let mut root = [0u8; 32];
    root.copy_from_slice(&bytes);
    Ok(root)
}

/// Minimal `--key value` flag parser with positionals. Unknown bare tokens are
/// positionals; a `--key` consumes the following token as its value.
struct Flags {
    map: Vec<(String, String)>,
    positional: Vec<String>,
}

impl Flags {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut map = Vec::new();
        let mut positional = Vec::new();
        let mut i = 0;
        while i < args.len()
        {
            let a = &args[i];
            if let Some(key) = a.strip_prefix("--")
            {
                let Some(val) = args.get(i + 1)
                else
                {
                    return Err(format!("flag --{key} needs a value"));
                };
                map.push((key.to_string(), val.clone()));
                i += 2;
            }
            else
            {
                positional.push(a.clone());
                i += 1;
            }
        }
        Ok(Self { map, positional })
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.map
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn seed(&self, default: Hash) -> Result<Hash, String> {
        match self.get("seed-hex")
        {
            Some(s) => parse_root_hex(s),
            None => Ok(default),
        }
    }

    fn root(&self, default: Hash) -> Result<Hash, String> {
        match self.get("root-hex")
        {
            Some(s) => parse_root_hex(s),
            None => Ok(default),
        }
    }

    fn height(&self, default: u32) -> Result<u32, String> {
        match self.get("height")
        {
            Some(s) => s
                .parse::<u32>()
                .map_err(|_| format!("invalid height: '{s}'")),
            None => Ok(default),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("scirust_license_cli_{name}"))
    }

    #[test]
    fn no_args_prints_usage_with_exit_2() {
        let r = run(&[], 100);
        assert_eq!(r.exit, 2);
        assert!(r.stdout.contains("USAGE"));
    }

    #[test]
    fn unknown_command_is_a_usage_error() {
        let r = run(&argv(&["frobnicate"]), 100);
        assert_eq!(r.exit, 2);
        assert!(r.stdout.contains("unknown command"));
    }

    #[test]
    fn modules_lists_the_whole_catalogue() {
        let r = run(&argv(&["modules"]), 100);
        assert_eq!(r.exit, 0);
        assert!(r.stdout.contains("navigation"));
        assert!(r.stdout.contains("industrial"));
        // One line per module (32 in ALL + Industrial) plus the header.
        assert_eq!(r.stdout.lines().count(), Module::ALL.len() + 1 + 1);
    }

    #[test]
    fn keygen_on_the_demo_seed_prints_the_embedded_root() {
        let r = run(&argv(&["keygen"]), 100);
        assert_eq!(r.exit, 0);
        assert!(r.stdout.contains(crate::DEMO_ROOT_HEX));
    }

    #[test]
    fn issue_then_inspect_round_trips_through_a_file() {
        let issued = run(
            &argv(&[
                "issue",
                "--licensee",
                "Acme",
                "--id",
                "L-1",
                "--modules",
                "navigation,control",
                "--expires",
                "2000",
                "--leaf",
                "0",
            ]),
            1_000,
        );
        assert_eq!(issued.exit, 0, "issue failed: {}", issued.stdout);

        let path = temp_path("roundtrip.json");
        std::fs::write(&path, issued.stdout.trim()).unwrap();
        let p = path.to_string_lossy().to_string();

        let inspected = run(&argv(&["inspect", &p, "--now", "1500"]), 0);
        assert_eq!(inspected.exit, 0, "inspect failed: {}", inspected.stdout);
        assert!(inspected.stdout.contains("VALID"));
        assert!(inspected.stdout.contains("navigation, control"));

        // A check against an entitled module is granted; an unlisted one denied.
        let granted = run(
            &argv(&["check", &p, "--module", "navigation", "--now", "1500"]),
            0,
        );
        assert_eq!(granted.exit, 0);
        assert!(granted.stdout.contains("GRANTED"));

        let denied = run(
            &argv(&["check", &p, "--module", "water", "--now", "1500"]),
            0,
        );
        assert_eq!(denied.exit, 1);
        assert!(denied.stdout.contains("DENIED"));

        // Past expiry, even an entitled module is denied.
        let expired = run(
            &argv(&["check", &p, "--module", "navigation", "--now", "9999"]),
            0,
        );
        assert_eq!(expired.exit, 1);
        assert!(expired.stdout.contains("expired"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn node_locked_license_only_verifies_on_the_bound_machine() {
        // Issue a license bound to a specific machine via --node.
        let issued = run(
            &argv(&[
                "issue",
                "--licensee",
                "Acme",
                "--id",
                "L-NODE",
                "--modules",
                "navigation",
                "--node",
                "press-line-07",
                "--leaf",
                "0",
            ]),
            1_000,
        );
        assert_eq!(issued.exit, 0, "issue failed: {}", issued.stdout);
        // The raw machine id must not appear in the signed license file.
        assert!(
            !issued.stdout.contains("press-line-07"),
            "raw machine id leaked into the license: {}",
            issued.stdout
        );

        let path = temp_path("nodelock.json");
        std::fs::write(&path, issued.stdout.trim()).unwrap();
        let p = path.to_string_lossy().to_string();

        // Without --node, a node-locked license cannot be verified.
        let blind = run(&argv(&["inspect", &p, "--now", "1500"]), 0);
        assert_eq!(blind.exit, 1, "blind inspect should fail: {}", blind.stdout);
        assert!(blind.stdout.contains("node-locked"));

        // With the matching --node it is VALID and reported as locked.
        let right = run(
            &argv(&["inspect", &p, "--node", "press-line-07", "--now", "1500"]),
            0,
        );
        assert_eq!(right.exit, 0, "right-node inspect failed: {}", right.stdout);
        assert!(right.stdout.contains("VALID"));
        assert!(right.stdout.contains("node:") && right.stdout.contains("locked"));

        // With a different --node it is rejected.
        let wrong = run(
            &argv(&["inspect", &p, "--node", "press-line-99", "--now", "1500"]),
            0,
        );
        assert_eq!(wrong.exit, 1);
        assert!(wrong.stdout.contains("different machine"));

        // check follows the same rule: granted on the right node, denied elsewhere.
        let granted = run(
            &argv(&[
                "check",
                &p,
                "--module",
                "navigation",
                "--node",
                "press-line-07",
                "--now",
                "1500",
            ]),
            0,
        );
        assert_eq!(granted.exit, 0);
        assert!(granted.stdout.contains("GRANTED"));

        let denied = run(
            &argv(&[
                "check",
                &p,
                "--module",
                "navigation",
                "--node",
                "press-line-99",
                "--now",
                "1500",
            ]),
            0,
        );
        assert_eq!(denied.exit, 1);
        assert!(denied.stdout.contains("DENIED"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn issue_rejects_an_unknown_module() {
        let r = run(
            &argv(&[
                "issue",
                "--licensee",
                "A",
                "--id",
                "X",
                "--modules",
                "teleport",
            ]),
            10,
        );
        assert_eq!(r.exit, 2);
        assert!(r.stdout.contains("unknown module 'teleport'"));
    }

    #[test]
    fn issue_requires_its_mandatory_flags() {
        let r = run(&argv(&["issue", "--licensee", "A"]), 10);
        assert_eq!(r.exit, 2);
        assert!(r.stdout.contains("--licensee, --id and --modules"));
    }

    #[test]
    fn inspect_reports_a_tampered_license_as_invalid() {
        let issued = run(
            &argv(&[
                "issue",
                "--licensee",
                "Acme",
                "--id",
                "L-1",
                "--modules",
                "water",
                "--leaf",
                "0",
            ]),
            1_000,
        );
        let mut signed = SignedLicense::from_json(issued.stdout.trim()).unwrap();
        signed.license.modules.push(Module::Grid); // self-grant attempt
        let path = temp_path("tampered.json");
        std::fs::write(&path, signed.to_json()).unwrap();
        let p = path.to_string_lossy().to_string();

        let r = run(&argv(&["inspect", &p]), 1_000);
        assert_eq!(r.exit, 1);
        assert!(r.stdout.contains("INVALID"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_flag_without_a_value_is_a_usage_error() {
        let r = run(&argv(&["keygen", "--height"]), 10);
        assert_eq!(r.exit, 2);
        assert!(r.stdout.contains("needs a value"));
    }

    #[test]
    fn inspect_on_a_missing_file_fails_cleanly() {
        let r = run(&argv(&["inspect", "/no/such/license.json"]), 10);
        assert_eq!(r.exit, 1);
        assert!(r.stdout.contains("cannot read"));
    }

    // Regression: a requested --height beyond the signer's clamped range
    // (1..=20) must report the *effective* height, consistent with capacity,
    // not the unclamped request. Before the fix this printed "(height 30)"
    // beside a capacity of 2^20.
    #[test]
    fn keygen_reports_the_effective_clamped_height_not_the_request() {
        let r = run(&argv(&["keygen", "--height", "30"]), 100);
        assert_eq!(r.exit, 0, "keygen failed: {}", r.stdout);
        // Clamp ceiling is 20 → 2^20 == 1048576 one-time licenses.
        assert!(
            r.stdout
                .contains("capacity: 1048576 one-time licenses (height 20)"),
            "expected effective height 20 next to its capacity, got: {}",
            r.stdout
        );
        assert!(
            !r.stdout.contains("(height 30)"),
            "leaked the unclamped requested height: {}",
            r.stdout
        );
    }

    // Regression: `issue` must not silently default --leaf to 0. Two licenses
    // issued without --leaf would otherwise both sign with leaf 0, reusing a
    // one-time Lamport leaf (cryptographically unsafe). It is now required.
    #[test]
    fn issue_requires_an_explicit_leaf() {
        let r = run(
            &argv(&[
                "issue",
                "--licensee",
                "Acme",
                "--id",
                "L-1",
                "--modules",
                "navigation",
            ]),
            1_000,
        );
        assert_eq!(r.exit, 2, "expected a usage error, got: {}", r.stdout);
        assert!(
            r.stdout.contains("--leaf"),
            "usage error should mention --leaf: {}",
            r.stdout
        );
    }

    // With an explicit --leaf, issuing still succeeds (guards against the fix
    // over-rejecting valid input).
    #[test]
    fn issue_with_an_explicit_leaf_succeeds() {
        let r = run(
            &argv(&[
                "issue",
                "--licensee",
                "Acme",
                "--id",
                "L-1",
                "--modules",
                "navigation",
                "--leaf",
                "3",
            ]),
            1_000,
        );
        assert_eq!(r.exit, 0, "issue with --leaf failed: {}", r.stdout);
    }
}
