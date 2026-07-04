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
    Scalar {
        lo: f64,
        hi: f64,
    },
    Array {
        n: usize,
        lo: f64,
        hi: f64,
    },
    /// A square n×n matrix, generated strictly diagonally dominant so it is
    /// well-conditioned and non-singular (a fair, stable input for a solver).
    Matrix {
        n: usize,
    },
    /// A symmetric n×n matrix with well-separated eigenvalues (a stable input
    /// for a symmetric eigensolver — distinct eigenvalues make the two
    /// implementations agree to high precision).
    SymMatrix {
        n: usize,
    },
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
        // while — Newton's method for sqrt, fixed iteration count.
        Case {
            name: "newton_sqrt (while, fixed)",
            call: "newton_sqrt",
            src: "def newton_sqrt(a):\n    x = a\n    i = 0\n    while i < 20:\n        x = 0.5 * (x + a / x)\n        i = i + 1\n    return x\n",
            args: vec![Scalar { lo: 0.1, hi: 5.0 }],
        },
        // while — Newton's method with a convergence condition (data-dependent
        // iteration count; bit-identical ops => same count in Rust and NumPy).
        Case {
            name: "newton_conv (while, converge)",
            call: "newton_conv",
            src: "def newton_conv(a, tol):\n    x = a\n    d = x * x - a\n    while abs(d) > tol:\n        x = x - d / (2.0 * x)\n        d = x * x - a\n    return x\n",
            args: vec![
                Scalar { lo: 0.1, hi: 5.0 },
                Scalar {
                    lo: 1e-10,
                    hi: 1e-8,
                },
            ],
        },
        // Routing (Phase 1): np.linalg.solve -> scirust-solvers LU solver.
        // A is a 5×5 diagonally-dominant matrix; compare the solution vector.
        Case {
            name: "linalg.solve -> scirust-solvers",
            call: "solve",
            src: "def solve(A, b):\n    return np.linalg.solve(A, b)\n",
            args: vec![
                Matrix { n: 5 },
                Array {
                    n: 5,
                    lo: -3.0,
                    hi: 3.0,
                },
            ],
        },
        // Routing (Phase 1): A @ b matrix-vector product -> scirust-solvers.
        Case {
            name: "matvec (A @ b) -> scirust-solvers",
            call: "mv",
            src: "def mv(A, b):\n    return A @ b\n",
            args: vec![
                Matrix { n: 5 },
                Array {
                    n: 5,
                    lo: -3.0,
                    hi: 3.0,
                },
            ],
        },
        // Routing (Phase 1): np.linalg.inv -> scirust-solvers. Returns a MATRIX
        // (flattened row-major vs numpy.linalg.inv).
        Case {
            name: "linalg.inv -> scirust-solvers (matrix out)",
            call: "inv",
            src: "def inv(A):\n    return np.linalg.inv(A)\n",
            args: vec![Matrix { n: 4 }],
        },
        // Routing (Phase 1): np.linalg.det -> scirust-solvers (LU determinant).
        // A is a 4×4 diagonally-dominant matrix; compare the scalar determinant.
        Case {
            name: "linalg.det -> scirust-solvers",
            call: "det",
            src: "def det(A):\n    return np.linalg.det(A)\n",
            args: vec![Matrix { n: 4 }],
        },
        // Routing (Phase 1): np.linalg.eigvalsh -> scirust-solvers symmetric
        // eigensolver. A is a 5×5 symmetric matrix; compare the (ascending)
        // eigenvalue vector.
        Case {
            name: "linalg.eigvalsh -> scirust-solvers",
            call: "eigvals",
            src: "def eigvals(A):\n    return np.linalg.eigvalsh(A)\n",
            args: vec![SymMatrix { n: 5 }],
        },
        // Routing (Phase 1): np.fft.fft -> scirust-signal. Real input, COMPLEX
        // output (compared re/im interleaved vs numpy.fft.fft). n = 8 (radix-2).
        Case {
            name: "fft.fft -> scirust-signal (complex out)",
            call: "spec",
            src: "def spec(x: np.ndarray):\n    return np.fft.fft(x)\n",
            args: vec![Array {
                n: 8,
                lo: -1.0,
                hi: 1.0,
            }],
        },
        // np.abs(np.fft.fft(x)) -> real magnitude spectrum.
        Case {
            name: "abs(fft) magnitude spectrum",
            call: "mag",
            src: "def mag(x: np.ndarray):\n    return np.abs(np.fft.fft(x))\n",
            args: vec![Array {
                n: 8,
                lo: -1.0,
                hi: 1.0,
            }],
        },
        // np.fft.rfft -> half spectrum (N/2+1 complex bins) vs numpy.fft.rfft.
        Case {
            name: "fft.rfft -> scirust-signal (half spectrum)",
            call: "rf",
            src: "def rf(x: np.ndarray):\n    return np.fft.rfft(x)\n",
            args: vec![Array {
                n: 8,
                lo: -1.0,
                hi: 1.0,
            }],
        },
        // np.fft.ifft(np.fft.fft(x)) -> round-trip, should recover x (complex).
        Case {
            name: "ifft(fft) round-trip",
            call: "rt",
            src: "def rt(x: np.ndarray):\n    return np.fft.ifft(np.fft.fft(x))\n",
            args: vec![Array {
                n: 8,
                lo: -1.0,
                hi: 1.0,
            }],
        },
        // ---- intrinsic coverage: every supported math intrinsic & operator ----
        // sin, cos, abs (scalar).
        Case {
            name: "sin/cos/abs (scalar)",
            call: "trig",
            src: "def trig(x):\n    return np.sin(x) + np.cos(x) + np.abs(x)\n",
            args: vec![Scalar { lo: -3.0, hi: 3.0 }],
        },
        // exp (scalar).
        Case {
            name: "exp (scalar)",
            call: "es",
            src: "def es(x):\n    return np.exp(x)\n",
            args: vec![Scalar { lo: -2.0, hi: 2.0 }],
        },
        // ** power operator (integer and via powf path).
        Case {
            name: "power ** (poly)",
            call: "poly",
            src: "def poly(x):\n    return x ** 3 - 2.0 * x ** 2 + 1.0\n",
            args: vec![Scalar { lo: -2.0, hi: 2.0 }],
        },
        // np.ones + len, array return.
        Case {
            name: "ones + len (array out)",
            call: "ov",
            src: "def ov(x: np.ndarray):\n    return np.ones(len(x))\n",
            args: vec![Array {
                n: 5,
                lo: -1.0,
                hi: 1.0,
            }],
        },
        // elementwise exp over an array (ArrayUnaryFn path).
        Case {
            name: "exp (elementwise array)",
            call: "ea",
            src: "def ea(x: np.ndarray):\n    return np.exp(x)\n",
            args: vec![Array {
                n: 6,
                lo: -2.0,
                hi: 2.0,
            }],
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
    /// Row-major square matrix.
    Matrix(Vec<Vec<f64>>),
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
            ArgSpec::Matrix { n } =>
            {
                let n = *n;
                let mut rows = Vec::with_capacity(n);
                for i in 0..n
                {
                    let mut row: Vec<f64> = (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect();
                    // Strict diagonal dominance -> non-singular, well-conditioned.
                    let off: f64 = row
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, v)| v.abs())
                        .sum();
                    row[i] = off + rng.uniform(1.0, 2.0);
                    rows.push(row);
                }
                Val::Matrix(rows)
            },
            ArgSpec::SymMatrix { n } =>
            {
                let n = *n;
                // Start from a random matrix, symmetrise (A = (B+Bᵀ)/2), then
                // add a distinct diagonal shift so eigenvalues are well
                // separated (both eigensolvers then agree to high precision).
                let b: Vec<Vec<f64>> = (0..n)
                    .map(|_| (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect())
                    .collect();
                let mut a = vec![vec![0.0f64; n]; n];
                for i in 0..n
                {
                    for j in 0..n
                    {
                        a[i][j] = 0.5 * (b[i][j] + b[j][i]);
                    }
                    a[i][i] += 3.0 * (i as f64);
                }
                Val::Matrix(a)
            },
        })
        .collect()
}

/// Flatten one trial's args to a single whitespace line (Rust reads this).
/// Matrices are flattened row-major.
fn flat_line(args: &[Val]) -> String {
    let mut parts = Vec::new();
    for a in args
    {
        match a
        {
            Val::Scalar(x) => parts.push(lit(*x)),
            Val::Array(xs) => parts.extend(xs.iter().map(|x| lit(*x))),
            Val::Matrix(rows) => parts.extend(rows.iter().flat_map(|r| r.iter().map(|x| lit(*x)))),
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
            Val::Matrix(rows) =>
            {
                let rs: Vec<String> = rows
                    .iter()
                    .map(|r| {
                        format!(
                            "[{}]",
                            r.iter().map(|x| lit(*x)).collect::<Vec<_>>().join(", ")
                        )
                    })
                    .collect();
                format!("np.array([{}])", rs.join(", "))
            },
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
            ArgSpec::Matrix { n } | ArgSpec::SymMatrix { n } =>
            {
                let nn = n * n;
                binds.push_str(&format!(
                    "        let a{i} = &nums[off..off + {nn}]; off += {nn};\n"
                ));
            },
        }
        call.push(format!("a{i}"));
    }
    binds.push_str("        let _ = off;\n");
    (binds, call.join(", "))
}

#[allow(clippy::too_many_arguments)]
fn run_rust_batch(
    case: &Case,
    rust_fn: &str,
    deps: &[&str],
    trials: &[Vec<Val>],
    ret: Ty,
    tmp: &std::path::Path,
    workspace_root: &std::path::Path,
    shared_target: &std::path::Path,
) -> Result<Vec<Vec<f64>>, String> {
    let (binds, call) = rust_bindings(&case.args);
    let emit = match ret
    {
        Ty::Array => "for v in r.iter() { print!(\"{:.17e} \", v); } println!();",
        // Complex arrays are serialised as interleaved (re, im) — the Python
        // side does the same, so the vectors line up for comparison.
        Ty::ComplexArray =>
        {
            "for c in r.iter() { print!(\"{:.17e} {:.17e} \", c.re, c.im); } println!();"
        },
        // A produced matrix: flatten row-major (Python side ravels the same way).
        Ty::MatrixVal =>
        {
            "for __i in 0..r.rows() { for __v in r.row(__i).iter() { print!(\"{:.17e} \", __v); } } println!();"
        },
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

    // Std-only cases compile with bare `rustc` (fast); routed cases (which use
    // `scirust-*` kernels) compile as a tiny standalone cargo project with the
    // needed path deps, sharing one target dir so the dep tree builds once.
    let bin_path = if deps.is_empty()
    {
        let src_path = tmp.join(format!("case_{}.rs", case.call));
        let bin_path = tmp.join(format!("case_{}.bin", case.call));
        std::fs::write(&src_path, &program).map_err(|e| e.to_string())?;
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
        bin_path
    }
    else
    {
        compile_cargo(case, &program, deps, tmp, workspace_root, shared_target)?
    };

    let stdin_data: String = trials
        .iter()
        .map(|t| flat_line(t))
        .collect::<Vec<_>>()
        .join("\n");
    let output = pipe_stdin(&bin_path.to_string_lossy(), &[], &stdin_data)?;
    Ok(output.lines().map(parse_line).collect())
}

/// Compile a routed case as a standalone cargo project depending on the given
/// `scirust-*` crates (by path), returning the built binary's path.
fn compile_cargo(
    case: &Case,
    program: &str,
    deps: &[&str],
    tmp: &std::path::Path,
    workspace_root: &std::path::Path,
    shared_target: &std::path::Path,
) -> Result<std::path::PathBuf, String> {
    let pkg = format!("oc_{}", case.call);
    let proj = tmp.join(format!("proj_{}", case.call));
    std::fs::create_dir_all(proj.join("src")).map_err(|e| e.to_string())?;
    let mut toml = format!(
        "[package]\nname = \"{pkg}\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[dependencies]\n"
    );
    for d in deps
    {
        toml.push_str(&format!(
            "{d} = {{ path = \"{}/{d}\" }}\n",
            workspace_root.display()
        ));
    }
    std::fs::write(proj.join("Cargo.toml"), toml).map_err(|e| e.to_string())?;
    std::fs::write(proj.join("src/main.rs"), program).map_err(|e| e.to_string())?;
    let out = Command::new("cargo")
        .args(["build", "--release", "--quiet", "--manifest-path"])
        .arg(proj.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", shared_target)
        .output()
        .map_err(|e| format!("cargo spawn failed: {}", e))?;
    if !out.status.success()
    {
        return Err(format!(
            "cargo build failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(shared_target.join("release").join(&pkg))
}

fn run_python_batch(
    case: &Case,
    trials: &[Vec<Val>],
    tmp: &std::path::Path,
) -> Result<Vec<Vec<f64>>, String> {
    let tuples: Vec<String> = trials.iter().map(|t| py_tuple(t)).collect();
    // `_ser` serialises a result to a flat float vector — complex arrays become
    // interleaved (re, im), matching the Rust driver.
    let script = format!(
        "import numpy as np\n{src}\ndef _ser(r):\n    a = np.atleast_1d(np.asarray(r)).ravel()\n    if np.iscomplexobj(a):\n        o = np.empty(a.size * 2)\n        o[0::2] = a.real\n        o[1::2] = a.imag\n        return o\n    return a.astype(float)\ninputs = [\n{rows}\n]\nout = []\nfor args in inputs:\n    r = {call}(*args)\n    arr = _ser(r)\n    out.append(' '.join('%.17e' % v for v in arr))\nimport sys\nsys.stdout.write('\\n'.join(out))\n",
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
    // Workspace root = parent of this crate's dir; shared cargo target dir so
    // routed cases build the `scirust-*` dep tree only once.
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let shared_target = tmp.join("cargo-target");

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
        let sir = scirust_transpiler::transpile_to_sir(case.src).expect("transpile to sir");
        let ret = ret_ty(&case);
        let deps = scirust_transpiler::required_crates(&sir);
        let mut rng = Rng(0xC0FFEE ^ hash_name(case.call));
        let trials: Vec<Vec<Val>> = (0..TRIALS)
            .map(|_| gen_args(&case.args, &mut rng))
            .collect();

        let rust_out = run_rust_batch(
            &case,
            &rust_fn,
            &deps,
            &trials,
            ret,
            &tmp,
            &workspace_root,
            &shared_target,
        );
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
