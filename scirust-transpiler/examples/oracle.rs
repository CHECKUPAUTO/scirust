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

use scirust_transpiler::{RetTy, Ty};
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

/// Which source language a case is written in (drives the transpiler entry
/// point and which reference runtime the output is proven against).
#[derive(Clone, Copy, PartialEq)]
enum Lang {
    /// Proven against real CPython + NumPy.
    Python,
    /// Proven against real Octave.
    Matlab,
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
        // Transpose A.T -> matrix out.
        Case {
            name: "transpose (A.T) -> matrix out",
            call: "tp",
            src: "def tp(A):\n    return A.T\n",
            args: vec![Matrix { n: 4 }],
        },
        // Matrix-matrix product with chaining: A @ A.T (Gram matrix).
        Case {
            name: "matmul A @ A.T -> scirust-solvers",
            call: "gram",
            src: "def gram(A):\n    return A @ A.T\n",
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
        // Routing (Phase 2): np.linalg.svd -> scirust-solvers (thin SVD) via
        // TUPLE UNPACKING. (a) singular values S are unique and descending, so
        // compared directly to numpy. U and Vh are computed but unused here.
        Case {
            name: "svd singular values (tuple unpack)",
            call: "singvals",
            src: "def singvals(A):\n    U, S, Vh = np.linalg.svd(A)\n    return S\n",
            args: vec![Matrix { n: 4 }],
        },
        // (b) reconstruction U @ diag(S) @ Vh ≈ A — gauge-invariant, so it
        // exercises U and V (whose individual signs are ambiguous) too.
        Case {
            name: "svd reconstruction U@diag(S)@Vh",
            call: "recon",
            src: "def recon(A):\n    U, S, Vh = np.linalg.svd(A)\n    return U @ np.diag(S) @ Vh\n",
            args: vec![Matrix { n: 4 }],
        },
        // Routing (Phase 2): np.linalg.qr -> scirust-solvers (Householder QR)
        // via tuple unpacking. Q/R signs are gauge-dependent, so we prove the
        // gauge-invariant reconstruction Q @ R ≈ A (square A => reduced == full).
        Case {
            name: "qr reconstruction Q@R (tuple unpack)",
            call: "qr_rec",
            src: "def qr_rec(A):\n    Q, R = np.linalg.qr(A)\n    return Q @ R\n",
            args: vec![Matrix { n: 4 }],
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
        // ---- expanded intrinsic vocabulary (Phase 2) ----
        // log / log10 (positive domain), elementwise over an array.
        Case {
            name: "log + log10 (array)",
            call: "logs",
            src: "def logs(x: np.ndarray):\n    return np.log(x) + np.log10(x)\n",
            args: vec![Array {
                n: 6,
                lo: 0.1,
                hi: 5.0,
            }],
        },
        // floor / ceil, elementwise.
        Case {
            name: "floor + ceil (array)",
            call: "rounding",
            src: "def rounding(x: np.ndarray):\n    return np.floor(x) + np.ceil(x)\n",
            args: vec![Array {
                n: 6,
                lo: -3.0,
                hi: 3.0,
            }],
        },
        // sinh / cosh / arctan (scalar).
        Case {
            name: "sinh/cosh/arctan (scalar)",
            call: "hyp",
            src: "def hyp(x):\n    return np.sinh(x) + np.cosh(x) + np.arctan(x)\n",
            args: vec![Scalar { lo: -2.0, hi: 2.0 }],
        },
        // max / min / mean reductions in one expression.
        Case {
            name: "max - min + mean (reductions)",
            call: "stats",
            src: "def stats(x: np.ndarray):\n    return np.max(x) - np.min(x) + np.mean(x)\n",
            args: vec![Array {
                n: 8,
                lo: -3.0,
                hi: 3.0,
            }],
        },
        // prod reduction (well-conditioned range so the product stays O(1)).
        Case {
            name: "prod (reduction)",
            call: "prodf",
            src: "def prodf(x: np.ndarray):\n    return np.prod(x)\n",
            args: vec![Array {
                n: 5,
                lo: 0.5,
                hi: 1.5,
            }],
        },
        // ---- Python élargi (Phase 2): user-defined function calls + lists ----
        // Scalar composition: sumsq calls sq twice.
        Case {
            name: "user call: sumsq (scalar compose)",
            call: "sumsq",
            src: "def sq(x):\n    return x * x\ndef sumsq(a, b):\n    return sq(a) + sq(b)\n",
            args: vec![Scalar { lo: -3.0, hi: 3.0 }, Scalar { lo: -3.0, hi: 3.0 }],
        },
        // Array composition, hint-free: `x` is inferred as an array purely from
        // `dbl`'s (array) parameter type — no annotation needed.
        Case {
            name: "user call: sumdbl (array compose, hint-free)",
            call: "sumdbl",
            src: "def dbl(v: np.ndarray):\n    return 2.0 * v\ndef sumdbl(x):\n    return np.sum(dbl(x))\n",
            args: vec![Array {
                n: 6,
                lo: -2.0,
                hi: 2.0,
            }],
        },
        // Three-level chain: chain -> twice -> inc.
        Case {
            name: "user call: chain (3-level)",
            call: "chain",
            src: "def inc(x):\n    return x + 1.0\ndef twice(x):\n    return inc(inc(x))\ndef chain(x):\n    return twice(x) * 2.0\n",
            args: vec![Scalar { lo: -3.0, hi: 3.0 }],
        },
        // List literal as a weight vector, consumed by np.dot (len must match x).
        Case {
            name: "list literal: weighted average",
            call: "wavg",
            src: "def wavg(x: np.ndarray):\n    w = [0.5, 0.3, 0.2]\n    return np.dot(x, w)\n",
            args: vec![Array {
                n: 3,
                lo: -3.0,
                hi: 3.0,
            }],
        },
        // ---- general tuple returns (Phase 2) ----
        // `return a, b` — two scalars out, compared element-wise.
        Case {
            name: "tuple return: addsub (a+b, a-b)",
            call: "addsub",
            src: "def addsub(a, b):\n    return a + b, a - b\n",
            args: vec![Scalar { lo: -3.0, hi: 3.0 }, Scalar { lo: -3.0, hi: 3.0 }],
        },
        // Array -> (scalar, scalar) tuple via reductions.
        Case {
            name: "tuple return: minmax (min, max)",
            call: "minmax",
            src: "def minmax(x: np.ndarray):\n    return np.min(x), np.max(x)\n",
            args: vec![Array {
                n: 7,
                lo: -3.0,
                hi: 3.0,
            }],
        },
        // Three-element tuple return.
        Case {
            name: "tuple return: stats3 (min, mean, max)",
            call: "stats3",
            src: "def stats3(x: np.ndarray):\n    return np.min(x), np.mean(x), np.max(x)\n",
            args: vec![Array {
                n: 8,
                lo: -3.0,
                hi: 3.0,
            }],
        },
    ]
}

/// MATLAB/Octave cases — the *same* SIR → Rust pipeline, entered through the
/// MATLAB front-end and proven against real Octave (not NumPy). These exercise
/// the MATLAB-specific semantics: 1-based indexing, inclusive `for` ranges,
/// element-wise `.*`/`./`, output-variable return, and branch-assigned
/// (hoisted) locals.
fn matlab_cases() -> Vec<Case> {
    use ArgSpec::*;
    vec![
        // Array -> scalar via `.*`, `sum`, `sqrt` (euclidean norm).
        Case {
            name: "M: norm2 (.* + sum + sqrt)",
            call: "norm2",
            src: "function y = norm2(x)\n  y = sqrt(sum(x .* x));\nend\n",
            args: vec![Array {
                n: 7,
                lo: -2.0,
                hi: 2.0,
            }],
        },
        // Two arrays -> scalar: sum(a .* b) (dot product).
        Case {
            name: "M: dot (sum(a .* b))",
            call: "dot_m",
            src: "function s = dot_m(a, b)\n  s = sum(a .* b);\nend\n",
            args: vec![
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
        // Scalar if/else with a branch-assigned (hoisted) output var.
        Case {
            name: "M: relu (if/else, hoisted)",
            call: "relu",
            src: "function y = relu(x)\n  if x > 0.0\n    y = x;\n  else\n    y = 0.0;\n  end\nend\n",
            args: vec![Scalar { lo: -3.0, hi: 3.0 }],
        },
        // if/elseif/else (piecewise sign), all branches assign the output.
        Case {
            name: "M: sign (if/elseif/else)",
            call: "sign_m",
            src: "function y = sign_m(x)\n  if x > 0.0\n    y = 1.0;\n  elseif x < 0.0\n    y = -1.0;\n  else\n    y = 0.0;\n  end\nend\n",
            args: vec![Scalar { lo: -2.0, hi: 2.0 }],
        },
        // Sequential ifs mutating the output var (clamp to [lo, hi]).
        Case {
            name: "M: clamp (sequential ifs)",
            call: "clamp_m",
            src: "function y = clamp_m(x, lo, hi)\n  y = x;\n  if y < lo\n    y = lo;\n  end\n  if y > hi\n    y = hi;\n  end\nend\n",
            args: vec![
                Scalar { lo: -3.0, hi: 3.0 },
                Scalar { lo: -1.0, hi: 0.0 },
                Scalar { lo: 0.5, hi: 1.5 },
            ],
        },
        // Scalar power `^` (polynomial).
        Case {
            name: "M: poly (^ power)",
            call: "poly_m",
            src: "function y = poly_m(x)\n  y = x^3 - 2.0 * x^2 + 1.0;\nend\n",
            args: vec![Scalar { lo: -2.0, hi: 2.0 }],
        },
        // for-loop, 1-based indexing, `length`, scalar accumulator.
        Case {
            name: "M: mysum (for, 1-based idx)",
            call: "mysum",
            src: "function s = mysum(x)\n  s = 0.0;\n  for i = 1:length(x)\n    s = s + x(i);\n  end\nend\n",
            args: vec![Array {
                n: 8,
                lo: -2.0,
                hi: 2.0,
            }],
        },
        // while-loop with a fixed iteration count (Newton's method for sqrt).
        Case {
            name: "M: newton_sqrt (while)",
            call: "newton_m",
            src: "function x = newton_m(a)\n  x = a;\n  i = 0;\n  while i < 20\n    x = 0.5 * (x + a / x);\n    i = i + 1;\n  end\nend\n",
            args: vec![Scalar { lo: 0.1, hi: 5.0 }],
        },
        // Element-wise array output: x .* w + x.
        Case {
            name: "M: ew_scale (array out)",
            call: "ew_scale",
            src: "function y = ew_scale(x, w)\n  y = x .* w + x;\nend\n",
            args: vec![
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
        // ---- MATLAB multi-output `[a, b] = f(...)` (Phase 2) ----
        // Two scalar outputs from two scalar inputs.
        Case {
            name: "M: sumdiff [s,d] (multi-output)",
            call: "sumdiff",
            src: "function [s, d] = sumdiff(a, b)\n  s = a + b;\n  d = a - b;\nend\n",
            args: vec![Scalar { lo: -3.0, hi: 3.0 }, Scalar { lo: -3.0, hi: 3.0 }],
        },
        // Array -> two scalar outputs: norm and its square.
        Case {
            name: "M: normstats [n,ss] (multi-output)",
            call: "normstats",
            src: "function [n, ss] = normstats(x)\n  ss = sum(x .* x);\n  n = sqrt(ss);\nend\n",
            args: vec![Array {
                n: 6,
                lo: -2.0,
                hi: 2.0,
            }],
        },
        // Array -> three scalar outputs exercising the NEW MATLAB reductions
        // (min/mean/max) together with multi-output.
        Case {
            name: "M: stats3 [lo,mu,hi] (min/mean/max)",
            call: "stats3",
            src: "function [lo, mu, hi] = stats3(x)\n  lo = min(x);\n  mu = mean(x);\n  hi = max(x);\nend\n",
            args: vec![Array {
                n: 8,
                lo: -3.0,
                hi: 3.0,
            }],
        },
        // Expanded MATLAB math intrinsics (single output): log/floor/atan.
        Case {
            name: "M: mathx (log/floor/atan)",
            call: "mathx",
            src: "function y = mathx(x)\n  y = log(x) + floor(x) + atan(x);\nend\n",
            args: vec![Scalar { lo: 0.5, hi: 5.0 }],
        },
        // ---- MATLAB linear algebra routed to scirust-solvers (Phase 2) ----
        // det(A) — A inferred as a matrix from the intrinsic; scalar out.
        Case {
            name: "M: det(A) -> scirust-solvers",
            call: "mdet",
            src: "function d = mdet(A)\n  d = det(A);\nend\n",
            args: vec![Matrix { n: 4 }],
        },
        // inv(A) — matrix out (row-major flatten vs Octave inv(A)).
        Case {
            name: "M: inv(A) -> scirust-solvers (matrix out)",
            call: "minv",
            src: "function B = minv(A)\n  B = inv(A);\nend\n",
            args: vec![Matrix { n: 4 }],
        },
        // A \ b (backslash left-division / solve) -> verified LU solver.
        Case {
            name: "M: A \\ b (solve) -> scirust-solvers",
            call: "msolve",
            src: "function x = msolve(A, b)\n  x = A \\ b;\nend\n",
            args: vec![
                Matrix { n: 5 },
                Array {
                    n: 5,
                    lo: -3.0,
                    hi: 3.0,
                },
            ],
        },
        // norm(v) — Euclidean 2-norm of a vector (sqrt(sum(v .* v))).
        Case {
            name: "M: norm(v) (vector 2-norm)",
            call: "mnorm",
            src: "function y = mnorm(v)\n  y = norm(v);\nend\n",
            args: vec![Array {
                n: 7,
                lo: -2.0,
                hi: 2.0,
            }],
        },
        // dot(a, b) — MATLAB inner-product intrinsic (fixed-order reduction).
        Case {
            name: "M: dot(a, b) (inner product)",
            call: "mdot",
            src: "function s = mdot(a, b)\n  s = dot(a, b);\nend\n",
            args: vec![
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
        // eig(A) — eigenvalues (ascending) of a symmetric matrix, routed to the
        // verified symmetric eigensolver. Octave's `eig` returns ascending real
        // eigenvalues for a symmetric input, so this is proven on SymMatrix.
        Case {
            name: "M: eig(A) -> scirust-solvers (symmetric)",
            call: "meig",
            src: "function e = meig(A)\n  e = eig(A);\nend\n",
            args: vec![SymMatrix { n: 5 }],
        },
        // ---- MATLAB rounding / modular scalar functions (Phase 2) ----
        // round (half away from zero) + fix (truncate toward zero).
        Case {
            name: "M: round + fix (rounding)",
            call: "mround",
            src: "function y = mround(x)\n  y = round(x) + fix(x);\nend\n",
            args: vec![Scalar { lo: -4.0, hi: 4.0 }],
        },
        // mod(a, b) — result follows the divisor sign.
        Case {
            name: "M: mod(a, b) (modulo)",
            call: "mmod",
            src: "function y = mmod(a, b)\n  y = mod(a, b);\nend\n",
            args: vec![Scalar { lo: -5.0, hi: 5.0 }, Scalar { lo: 1.0, hi: 4.0 }],
        },
        // rem(a, b) — result follows the dividend sign.
        Case {
            name: "M: rem(a, b) (remainder)",
            call: "mrem",
            src: "function y = mrem(a, b)\n  y = rem(a, b);\nend\n",
            args: vec![Scalar { lo: -5.0, hi: 5.0 }, Scalar { lo: 1.0, hi: 4.0 }],
        },
        // sign(x) — -1 / 0 / +1 (sign(0) == 0).
        Case {
            name: "M: sign(x) (-1/0/+1)",
            call: "msign",
            src: "function y = msign(x)\n  y = sign(x);\nend\n",
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

/// Transpile a case's source to Rust through the front-end for its language.
fn transpile_case(lang: Lang, src: &str) -> Result<String, String> {
    match lang
    {
        Lang::Python => scirust_transpiler::transpile(src),
        Lang::Matlab => scirust_transpiler::transpile_matlab(src),
    }
}

/// Lower a case's source to SIR through the front-end for its language.
fn sir_case(lang: Lang, src: &str) -> Result<scirust_transpiler::SirModule, String> {
    match lang
    {
        Lang::Python => scirust_transpiler::transpile_to_sir(src),
        Lang::Matlab => scirust_transpiler::transpile_matlab_to_sir(src),
    }
}

fn ret_ty(case: &Case, lang: Lang) -> RetTy {
    let sir = sir_case(lang, case.src).expect("transpile to sir");
    sir.funcs
        .iter()
        .find(|f| f.name == case.call)
        .map(|f| f.ret.clone())
        .unwrap_or(RetTy::Single(Ty::Scalar))
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
    ret: &RetTy,
    tmp: &std::path::Path,
    workspace_root: &std::path::Path,
    shared_target: &std::path::Path,
) -> Result<Vec<Vec<f64>>, String> {
    let (binds, call) = rust_bindings(&case.args);
    let emit: String = match ret
    {
        RetTy::Single(Ty::Array) =>
        {
            "for v in r.iter() { print!(\"{:.17e} \", v); } println!();".to_string()
        },
        // Complex arrays are serialised as interleaved (re, im) — the Python
        // side does the same, so the vectors line up for comparison.
        RetTy::Single(Ty::ComplexArray) =>
        {
            "for c in r.iter() { print!(\"{:.17e} {:.17e} \", c.re, c.im); } println!();"
                .to_string()
        },
        // A produced matrix: flatten row-major (Python side ravels the same way).
        RetTy::Single(Ty::MatrixVal) =>
        {
            "for __i in 0..r.rows() { for __v in r.row(__i).iter() { print!(\"{:.17e} \", __v); } } println!();"
                .to_string()
        },
        // A tuple of scalars: print each field in order.
        RetTy::Tuple(ts) =>
        {
            let mut s = String::new();
            for i in 0..ts.len()
            {
                s.push_str(&format!("print!(\"{{:.17e}} \", r.{}); ", i));
            }
            s.push_str("println!();");
            s
        },
        _ => "println!(\"{:.17e}\", r);".to_string(),
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

/// One trial's args as an Octave cell literal `{ a1, a2, ... }`.
/// Arrays become **column** vectors (so `A \ b` is conformant), matrices become
/// `[r1; r2; ...]`. Reductions/element-wise ops are orientation-agnostic in the
/// values they produce, so a column orientation is safe for every case.
fn octave_tuple(args: &[Val]) -> String {
    let items: Vec<String> = args
        .iter()
        .map(|a| match a
        {
            Val::Scalar(x) => lit(*x),
            Val::Array(xs) => format!(
                "[{}]",
                xs.iter().map(|x| lit(*x)).collect::<Vec<_>>().join("; ")
            ),
            Val::Matrix(rows) =>
            {
                let rs: Vec<String> = rows
                    .iter()
                    .map(|r| r.iter().map(|x| lit(*x)).collect::<Vec<_>>().join(", "))
                    .collect();
                format!("[{}]", rs.join("; "))
            },
        })
        .collect();
    format!("{{ {} }}", items.join(", "))
}

/// Run the MATLAB source under **real Octave** over all trials, serialising each
/// result to a flat float line (column-major `r(:)`, which matches row/scalar
/// outputs and the Rust driver's ordering for 1-D results).
fn run_octave_batch(
    case: &Case,
    trials: &[Vec<Val>],
    ret: &RetTy,
    tmp: &std::path::Path,
) -> Result<Vec<Vec<f64>>, String> {
    let rows: Vec<String> = trials.iter().map(|t| octave_tuple(t)).collect();
    // Single-output: `__r = f(args)`. Multi-output: `[__o1, …] = f(args)` and
    // serialise each output in order (matching the Rust tuple print order).
    let call_and_ser = match ret
    {
        RetTy::Tuple(ts) =>
        {
            let names: Vec<String> = (0..ts.len()).map(|i| format!("__o{}", i)).collect();
            let sers: Vec<String> = names.iter().map(|n| format!("__ser({})", n)).collect();
            format!(
                "[{}] = {call}(__args{{:}});\n  printf('%s\\n', [{}]);",
                names.join(", "),
                sers.join(", "),
                call = case.call,
            )
        },
        _ => format!(
            "__r = {call}(__args{{:}});\n  printf('%s\\n', __ser(__r));",
            call = case.call
        ),
    };
    let script = format!(
        "1;\n{src}\nfunction __s = __ser(r)\n  __t = r.';\n  a = __t(:);\n  __s = sprintf('%.17e ', a);\nend\ninputs = {{\n{rows}\n}};\nfor __k = 1:numel(inputs)\n  __args = inputs{{__k}};\n  {call_and_ser}\nend\n",
        src = case.src,
        rows = rows
            .iter()
            .map(|r| format!("  {},", r))
            .collect::<Vec<_>>()
            .join("\n"),
        call_and_ser = call_and_ser,
    );
    let path = tmp.join(format!("mcase_{}.m", case.call));
    std::fs::write(&path, script).map_err(|e| e.to_string())?;
    let out = Command::new("octave")
        .args(["--norc", "--quiet", "--no-gui"])
        .arg(&path)
        .output()
        .map_err(|e| format!("octave spawn failed: {}", e))?;
    if !out.status.success()
    {
        return Err(format!(
            "octave failed:\n{}",
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
            "length mismatch: rust {} vs ref {}",
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
                "index {}: rust {:.17e} vs ref {:.17e} (|Δ|={:.3e} > tol {:.3e})",
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

    // Octave is only required if MATLAB cases are present; if it is missing we
    // skip those (and say so) rather than failing the whole Python suite.
    let have_octave = Command::new("octave")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

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
    let mut skipped = 0usize;
    println!("SciRust transpiler — differential oracle vs real runtimes");
    println!(
        "tolerance: |Δ| ≤ {:.0e} + {:.0e}·|ref|, {} trials/case",
        ATOL, RTOL, TRIALS
    );
    println!(
        "  Python cases → NumPy · MATLAB cases → Octave{}\n",
        if have_octave
        {
            ""
        }
        else
        {
            " (MISSING — skipped)"
        }
    );

    // Run every Python case first, then every MATLAB case, tagging each with the
    // reference runtime it is proven against.
    let all: Vec<(Case, Lang)> = cases()
        .into_iter()
        .map(|c| (c, Lang::Python))
        .chain(matlab_cases().into_iter().map(|c| (c, Lang::Matlab)))
        .collect();

    for (case, lang) in all
    {
        total += 1;
        if lang == Lang::Matlab && !have_octave
        {
            skipped += 1;
            println!("  · {:<32} SKIPPED (octave not on PATH)", case.name);
            continue;
        }
        let rust_fn = match transpile_case(lang, case.src)
        {
            Ok(s) => s,
            Err(e) =>
            {
                println!("  ✗ {:<32} TRANSPILE ERROR: {}", case.name, e);
                failures += 1;
                continue;
            },
        };
        let sir = sir_case(lang, case.src).expect("transpile to sir");
        let ret = ret_ty(&case, lang);
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
            &ret,
            &tmp,
            &workspace_root,
            &shared_target,
        );
        let ref_out = match lang
        {
            Lang::Python => run_python_batch(&case, &trials, &tmp),
            Lang::Matlab => run_octave_batch(&case, &trials, &ret, &tmp),
        };
        match (rust_out, ref_out)
        {
            (Ok(rv), Ok(pv)) =>
            {
                let refname = match lang
                {
                    Lang::Python => "numpy",
                    Lang::Matlab => "octave",
                };
                let mut fail = 0usize;
                let mut first = String::new();
                if rv.len() != pv.len()
                {
                    first = format!(
                        "trial count mismatch: rust {} vs {} {}",
                        rv.len(),
                        refname,
                        pv.len()
                    );
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
                    println!(
                        "  ✓ {:<32} {}/{} trials match ({})",
                        case.name, TRIALS, TRIALS, refname
                    );
                }
                else
                {
                    failures += 1;
                    println!(
                        "  ✗ {:<32} {}/{} FAILED — first: {}",
                        case.name, fail, TRIALS, first
                    );
                }
            },
            (Err(e), _) | (_, Err(e)) =>
            {
                failures += 1;
                println!("  ✗ {:<32} HARNESS ERROR: {}", case.name, e);
            },
        }
    }

    let _ = std::fs::remove_dir_all(&tmp);
    println!();
    let proven = total - failures - skipped;
    if failures == 0
    {
        println!(
            "ORACLE GREEN — {}/{} cases match their reference runtime within tolerance{}",
            proven,
            total,
            if skipped > 0
            {
                format!(" ({} skipped)", skipped)
            }
            else
            {
                String::new()
            }
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
