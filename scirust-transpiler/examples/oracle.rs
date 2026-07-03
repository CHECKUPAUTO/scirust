//! Differential oracle: prove each transpiled port against **real NumPy**.
//!
//! For every case we:
//!   1. generate `TRIALS` seeded input sets, formatting each `f64` as a
//!      round-trippable decimal so Python and Rust receive *bit-identical*
//!      inputs;
//!   2. transpile the Python source to Rust, compile it **once** with `rustc`
//!      (std only, no cargo) and feed all trials through stdin;
//!   3. run the original Python source under CPython + NumPy on the same inputs;
//!   4. compare the two outputs line-by-line within a declared tolerance.
//!
//! A wrong transpilation differs by orders of magnitude more than the
//! floating-point tolerance, so this is a genuine correctness gate. Requires
//! `python3` (+numpy) and `rustc` on PATH.
//!
//! Run with:  `cargo run -p scirust-transpiler --example oracle`

use scirust_transpiler::Ty;
use std::process::Command;

const ATOL: f64 = 1e-7;
const RTOL: f64 = 1e-9;
const TRIALS: usize = 200;

#[derive(Clone)]
enum ArgSpec {
    Scalar { lo: f64, hi: f64 },
    Array { n: usize, lo: f64, hi: f64 },
}

struct Case {
    name: &'static str,
    call: &'static str,
    src: &'static str,
    args: Vec<ArgSpec>,
}

fn cases() -> Vec<Case> {
    use ArgSpec::*;
    vec![
        Case {
            name: "rk4_step (scalar ODE step)",
            call: "rk4_step",
            src: "def rk4_step(y, k, h):\n    k1 = -k * y\n    k2 = -k * (y + 0.5 * h * k1)\n    k3 = -k * (y + 0.5 * h * k2)\n    k4 = -k * (y + h * k3)\n    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)\n",
            args: vec![
                Scalar { lo: -2.0, hi: 2.0 },
                Scalar { lo: 0.1, hi: 2.0 },
                Scalar { lo: 0.01, hi: 0.2 },
            ],
        },
        Case {
            name: "dot (vector dot product)",
            call: "d",
            src: "def d(a: np.ndarray, b: np.ndarray):\n    return np.dot(a, b)\n",
            args: vec![
                Array {
                    n: 7,
                    lo: -2.0,
                    hi: 2.0,
                },
                Array {
                    n: 7,
                    lo: -2.0,
                    hi: 2.0,
                },
            ],
        },
        Case {
            name: "norm (euclidean)",
            call: "norm",
            src: "def norm(x: np.ndarray):\n    return np.sqrt(np.sum(x * x))\n",
            args: vec![Array {
                n: 7,
                lo: -2.0,
                hi: 2.0,
            }],
        },
        Case {
            name: "weighted_mean",
            call: "wmean",
            src: "def wmean(x: np.ndarray, w: np.ndarray):\n    return np.sum(x * w) / np.sum(w)\n",
            args: vec![
                Array {
                    n: 6,
                    lo: -3.0,
                    hi: 3.0,
                },
                Array {
                    n: 6,
                    lo: 0.1,
                    hi: 2.0,
                },
            ],
        },
        Case {
            name: "cumsum (loop + array out)",
            call: "cumsum",
            src: "def cumsum(x: np.ndarray):\n    y = np.zeros(len(x))\n    acc = 0.0\n    for i in range(len(x)):\n        acc = acc + x[i]\n        y[i] = acc\n    return y\n",
            args: vec![Array {
                n: 8,
                lo: -2.0,
                hi: 2.0,
            }],
        },
        Case {
            name: "saxpy (a*x + y)",
            call: "saxpy",
            src: "def saxpy(a, x: np.ndarray, y: np.ndarray):\n    return a * x + y\n",
            args: vec![
                Scalar { lo: -2.0, hi: 2.0 },
                Array {
                    n: 6,
                    lo: -2.0,
                    hi: 2.0,
                },
                Array {
                    n: 6,
                    lo: -2.0,
                    hi: 2.0,
                },
            ],
        },
        Case {
            name: "tanh_activation",
            call: "act",
            src: "def act(x: np.ndarray):\n    return np.tanh(x)\n",
            args: vec![Array {
                n: 6,
                lo: -3.0,
                hi: 3.0,
            }],
        },
        // Control flow (Phase 1): if without else.
        Case {
            name: "relu (if, scalar)",
            call: "relu",
            src: "def relu(x):\n    if x > 0.0:\n        return x\n    return 0.0\n",
            args: vec![Scalar { lo: -3.0, hi: 3.0 }],
        },
        // Two sequential ifs — clamp to [lo, hi].
        Case {
            name: "clamp (two ifs)",
            call: "clamp",
            src: "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n",
            args: vec![
                Scalar { lo: -3.0, hi: 3.0 },
                Scalar { lo: -1.0, hi: 0.0 },
                Scalar { lo: 0.5, hi: 1.5 },
            ],
        },
        // if / elif / else — piecewise sign.
        Case {
            name: "sign (if/elif/else)",
            call: "sign",
            src: "def sign(x):\n    if x > 0.0:\n        return 1.0\n    elif x < 0.0:\n        return -1.0\n    else:\n        return 0.0\n",
            args: vec![Scalar { lo: -2.0, hi: 2.0 }],
        },
    ]
}

// ---- deterministic PRNG (SplitMix64) --------------------------------------

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }
}

/// Round-trippable decimal for an f64 (parses identically in Rust and Python).
fn lit(v: f64) -> String {
    format!("{:?}", v)
}

enum Val {
    Scalar(f64),
    Array(Vec<f64>),
}

fn gen_args(specs: &[ArgSpec], rng: &mut Rng) -> Vec<Val> {
    specs
        .iter()
        .map(|s| match s
        {
            ArgSpec::Scalar { lo, hi } => Val::Scalar(rng.uniform(*lo, *hi)),
            ArgSpec::Array { n, lo, hi } =>
            {
                Val::Array((0..*n).map(|_| rng.uniform(*lo, *hi)).collect())
            },
        })
        .collect()
}

/// Flatten one trial's args to a single whitespace line (Rust reads this).
fn flat_line(args: &[Val]) -> String {
    let mut parts = Vec::new();
    for a in args
    {
        match a
        {
            Val::Scalar(x) => parts.push(lit(*x)),
            Val::Array(xs) => parts.extend(xs.iter().map(|x| lit(*x))),
        }
    }
    parts.join(" ")
}

/// One trial's args as a Python tuple literal.
fn py_tuple(args: &[Val]) -> String {
    let items: Vec<String> = args
        .iter()
        .map(|a| match a
        {
            Val::Scalar(x) => lit(*x),
            Val::Array(xs) => format!(
                "np.array([{}])",
                xs.iter().map(|x| lit(*x)).collect::<Vec<_>>().join(", ")
            ),
        })
        .collect();
    format!("({},)", items.join(", "))
}

fn parse_line(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .collect()
}

fn ret_ty(case: &Case) -> Ty {
    let sir = scirust_transpiler::transpile_to_sir(case.src).expect("transpile to sir");
    sir.funcs
        .iter()
        .find(|f| f.name == case.call)
        .map(|f| f.ret)
        .unwrap_or(Ty::Scalar)
}

/// Generate the arg-binding lines for the Rust `main` from the arg specs.
fn rust_bindings(specs: &[ArgSpec]) -> (String, String) {
    let mut binds = String::new();
    let mut call = Vec::new();
    binds.push_str("        let mut off = 0usize;\n");
    for (i, s) in specs.iter().enumerate()
    {
        match s
        {
            ArgSpec::Scalar { .. } =>
            {
                binds.push_str(&format!("        let a{i} = nums[off]; off += 1;\n"));
            },
            ArgSpec::Array { n, .. } =>
            {
                binds.push_str(&format!(
                    "        let a{i} = &nums[off..off + {n}]; off += {n};\n"
                ));
            },
        }
        call.push(format!("a{i}"));
    }
    binds.push_str("        let _ = off;\n");
    (binds, call.join(", "))
}

fn run_rust_batch(
    case: &Case,
    rust_fn: &str,
    trials: &[Vec<Val>],
    ret: Ty,
    tmp: &std::path::Path,
) -> Result<Vec<Vec<f64>>, String> {
    let (binds, call) = rust_bindings(&case.args);
    let emit = match ret
    {
        Ty::Array => "for v in r.iter() { print!(\"{:.17e} \", v); } println!();",
        _ => "println!(\"{:.17e}\", r);",
    };
    let program = format!(
        "{fns}\nuse std::io::BufRead;\nfn main() {{\n    let stdin = std::io::stdin();\n    for line in stdin.lock().lines() {{\n        let line = line.unwrap();\n        if line.trim().is_empty() {{ continue; }}\n        let nums: Vec<f64> = line.split_whitespace().map(|t| t.parse().unwrap()).collect();\n{binds}        let r = {call_name}({call});\n        {emit}\n    }}\n}}\n",
        fns = rust_fn,
        binds = binds,
        call_name = case.call,
        call = call,
        emit = emit,
    );
    let src_path = tmp.join(format!("case_{}.rs", case.call));
    let bin_path = tmp.join(format!("case_{}.bin", case.call));
    std::fs::write(&src_path, program).map_err(|e| e.to_string())?;
    let out = Command::new("rustc")
        .args(["-O", "--edition", "2021", "-A", "warnings", "-o"])
        .arg(&bin_path)
        .arg(&src_path)
        .output()
        .map_err(|e| format!("rustc spawn failed: {}", e))?;
    if !out.status.success()
    {
        return Err(format!(
            "rustc failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let stdin_data: String = trials
        .iter()
        .map(|t| flat_line(t))
        .collect::<Vec<_>>()
        .join("\n");
    let output = pipe_stdin(&bin_path.to_string_lossy(), &[], &stdin_data)?;
    Ok(output.lines().map(parse_line).collect())
}

fn run_python_batch(
    case: &Case,
    trials: &[Vec<Val>],
    tmp: &std::path::Path,
) -> Result<Vec<Vec<f64>>, String> {
    let tuples: Vec<String> = trials.iter().map(|t| py_tuple(t)).collect();
    let script = format!(
        "import numpy as np\n{src}\ninputs = [\n{rows}\n]\nout = []\nfor args in inputs:\n    r = {call}(*args)\n    arr = np.atleast_1d(np.asarray(r, dtype=float)).ravel()\n    out.append(' '.join('%.17e' % v for v in arr))\nimport sys\nsys.stdout.write('\\n'.join(out))\n",
        src = case.src,
        rows = tuples
            .iter()
            .map(|t| format!("    {},", t))
            .collect::<Vec<_>>()
            .join("\n"),
        call = case.call,
    );
    let path = tmp.join(format!("case_{}.py", case.call));
    std::fs::write(&path, script).map_err(|e| e.to_string())?;
    let out = Command::new("python3")
        .arg(&path)
        .output()
        .map_err(|e| format!("python3 spawn failed: {}", e))?;
    if !out.status.success()
    {
        return Err(format!(
            "python failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(parse_line)
        .collect())
}

fn pipe_stdin(bin: &str, args: &[&str], data: &str) -> Result<String, String> {
    use std::io::Write;
    let mut child = Command::new(bin)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn {} failed: {}", bin, e))?;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(data.as_bytes())
        .map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    if !out.status.success()
    {
        return Err(format!(
            "binary crashed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn approx_eq(a: &[f64], b: &[f64]) -> Option<String> {
    if a.len() != b.len()
    {
        return Some(format!(
            "length mismatch: rust {} vs py {}",
            a.len(),
            b.len()
        ));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate()
    {
        let tol = ATOL + RTOL * y.abs();
        if (x - y).abs() > tol
        {
            return Some(format!(
                "index {}: rust {:.17e} vs py {:.17e} (|Δ|={:.3e} > tol {:.3e})",
                i,
                x,
                y,
                (x - y).abs(),
                tol
            ));
        }
    }
    None
}

fn main() {
    for (bin, probe) in [("rustc", "--version"), ("python3", "--version")]
    {
        if Command::new(bin).arg(probe).output().is_err()
        {
            eprintln!("oracle requires `{}` on PATH — skipping", bin);
            std::process::exit(2);
        }
    }
    if Command::new("python3")
        .args(["-c", "import numpy"])
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("oracle requires numpy (`pip install numpy`) — skipping");
        std::process::exit(2);
    }

    let tmp = std::env::temp_dir().join(format!("scirust_oracle_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let mut total = 0usize;
    let mut failures = 0usize;
    println!("SciRust transpiler — differential oracle vs NumPy");
    println!(
        "tolerance: |Δ| ≤ {:.0e} + {:.0e}·|numpy|, {} trials/case\n",
        ATOL, RTOL, TRIALS
    );

    for case in cases()
    {
        total += 1;
        let rust_fn = match scirust_transpiler::transpile(case.src)
        {
            Ok(s) => s,
            Err(e) =>
            {
                println!("  ✗ {:<28} TRANSPILE ERROR: {}", case.name, e);
                failures += 1;
                continue;
            },
        };
        let ret = ret_ty(&case);
        let mut rng = Rng(0xC0FFEE ^ hash_name(case.call));
        let trials: Vec<Vec<Val>> = (0..TRIALS)
            .map(|_| gen_args(&case.args, &mut rng))
            .collect();

        let rust_out = run_rust_batch(&case, &rust_fn, &trials, ret, &tmp);
        let py_out = run_python_batch(&case, &trials, &tmp);
        match (rust_out, py_out)
        {
            (Ok(rv), Ok(pv)) =>
            {
                let mut fail = 0usize;
                let mut first = String::new();
                if rv.len() != pv.len()
                {
                    first = format!("trial count mismatch: rust {} vs py {}", rv.len(), pv.len());
                    fail = TRIALS;
                }
                else
                {
                    for (r, p) in rv.iter().zip(pv.iter())
                    {
                        if let Some(msg) = approx_eq(r, p)
                        {
                            fail += 1;
                            if first.is_empty()
                            {
                                first = msg;
                            }
                        }
                    }
                }
                if fail == 0
                {
                    println!("  ✓ {:<28} {}/{} trials match", case.name, TRIALS, TRIALS);
                }
                else
                {
                    failures += 1;
                    println!(
                        "  ✗ {:<28} {}/{} FAILED — first: {}",
                        case.name, fail, TRIALS, first
                    );
                }
            },
            (Err(e), _) | (_, Err(e)) =>
            {
                failures += 1;
                println!("  ✗ {:<28} HARNESS ERROR: {}", case.name, e);
            },
        }
    }

    let _ = std::fs::remove_dir_all(&tmp);
    println!();
    if failures == 0
    {
        println!(
            "ORACLE GREEN — {}/{} cases match NumPy within tolerance",
            total, total
        );
    }
    else
    {
        println!("ORACLE RED — {}/{} cases failed", failures, total);
        std::process::exit(1);
    }
}

fn hash_name(s: &str) -> u64 {
    let mut h = 1469598103934665603u64;
    for b in s.bytes()
    {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}
