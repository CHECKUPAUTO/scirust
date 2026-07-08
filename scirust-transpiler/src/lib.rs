//! # scirust-transpiler
//!
//! An **inbound** scientific transpiler (Phase 0 MVP): it converts a
//! *contractual subset* of Python/NumPy into deterministic, safe, std-only
//! Rust. See [`docs/TRANSPILER_DESIGN.md`](https://github.com/CHECKUPAUTO/scirust/blob/master/docs/TRANSPILER_DESIGN.md)
//! for the full architecture and roadmap.
//!
//! Pipeline: source text → [`front_python`] (lex/parse) → [`sir`] (typed IR,
//! via [`lower`]) → [`emit`] (Rust source). Every emitted reduction pins its
//! order, so the output is bit-reproducible regardless of parallelism.
//!
//! ## Supported subset (contract)
//!
//! * top-level `def`s with `float` / `int` / `np.ndarray` parameters (hints
//!   optional — array-ness is inferred from indexing / `np.sum` / `np.dot` /
//!   `len` usage);
//! * scalars (`f64`) and 1-D arrays (`Vec<f64>` / `&[f64]`);
//! * arithmetic `+ - * / **`, unary minus, elementwise array ops and
//!   scalar/array broadcasting;
//! * intrinsics: reductions `np.sum/prod/mean/max/min`, `np.dot`; builders
//!   `np.zeros/ones/diag`, `len`; elementwise/scalar math
//!   `np.sqrt/exp/log/log10/sin/cos/sinh/cosh/tanh/abs/floor/ceil/arctan`;
//! * `for i in range(...)` loops, indexing `a[i]`, index-assignment `a[i] = …`,
//!   `return`;
//! * list literals `[a, b, c]` (→ `Vec<f64>`), tuple unpacking
//!   `U, S, Vh = np.linalg.svd(A)` / `Q, R = np.linalg.qr(A)`, tuple returns
//!   `return a, b` (scalar elements), and calls to **other user functions**
//!   defined earlier in the module (define-before-use).
//!
//! Anything outside the subset is **refused with a diagnostic**, never guessed.
//! Correctness of any given port is established by the differential oracle
//! (`tests/oracle.rs`), which compiles the emitted Rust and checks it against
//! real NumPy on seeded random inputs.

pub mod emit;
pub mod front_matlab;
pub mod front_python;
pub mod lower;
pub mod lower_matlab;
pub mod sir;

pub use sir::{RetTy, SirFunc, SirModule, Ty, required_crates};

/// Transpile a Python/NumPy subset source string into a Rust module string.
///
/// Returns the emitted Rust (prelude + one `pub fn` per input `def`) or a
/// human-readable diagnostic if the source falls outside the supported subset.
pub fn transpile(python_src: &str) -> Result<String, String> {
    let ast = front_python::parse_python(python_src)?;
    let sir = lower::lower_module(&ast)?;
    Ok(emit::emit_module(&sir))
}

/// Transpile and return the typed SIR (useful for inspection / tests).
pub fn transpile_to_sir(python_src: &str) -> Result<SirModule, String> {
    let ast = front_python::parse_python(python_src)?;
    lower::lower_module(&ast)
}

/// Transpile a MATLAB/Octave subset source string into a Rust module string.
///
/// A second source front-end onto the *same* SIR + emitter as the Python path,
/// so the MATLAB output inherits the same determinism and the same
/// oracle-validated kernels.
pub fn transpile_matlab(matlab_src: &str) -> Result<String, String> {
    let ast = front_matlab::parse_matlab(matlab_src)?;
    let sir = lower_matlab::lower_module(&ast)?;
    Ok(emit::emit_module(&sir))
}

/// Transpile MATLAB and return the typed SIR (used for cross-front-end
/// equivalence tests against the NumPy-proven Python path).
pub fn transpile_matlab_to_sir(matlab_src: &str) -> Result<SirModule, String> {
    let ast = front_matlab::parse_matlab(matlab_src)?;
    lower_matlab::lower_module(&ast)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig_of(rust: &str, name: &str) -> String {
        // Match the top-level `pub fn NAME(` (column 0) — prelude helpers live
        // indented inside `pub mod np { … }`, so a same-named builtin (e.g.
        // `np::cumsum`) does not shadow the user's function here.
        rust.lines()
            .find(|l| l.starts_with(&format!("pub fn {}(", name)))
            .unwrap_or("")
            .trim()
            .to_string()
    }

    #[test]
    fn scalar_function_signature_and_body() {
        let src = "def f(x, y):\n    return x * y + 1.0\n";
        let rust = transpile(src).unwrap();
        assert_eq!(sig_of(&rust, "f"), "pub fn f(x: f64, y: f64) -> f64 {");
        assert!(rust.contains("return ((x * y) + 1.0f64);"));
    }

    #[test]
    fn array_param_inferred_from_sum() {
        // `x` is only used via np.sum -> must be inferred as an array.
        let src = "def total(x):\n    return np.sum(x)\n";
        let rust = transpile(src).unwrap();
        assert_eq!(sig_of(&rust, "total"), "pub fn total(x: &[f64]) -> f64 {");
        assert!(rust.contains("np::sum(x)"));
    }

    #[test]
    fn norm_uses_ew_and_sqrt() {
        // `x` is only combined elementwise (x*x) then summed — not a *direct*
        // array-consumer usage — so the subset contract requires a hint here.
        let src = "def norm(x: np.ndarray):\n    return np.sqrt(np.sum(x * x))\n";
        let rust = transpile(src).unwrap();
        assert!(rust.contains("np::sum(&(np::ew2(x, x, |x, y| x * y)))"));
        assert!(rust.contains(").sqrt()"));
    }

    #[test]
    fn hint_forces_array_type() {
        let src = "def f(a: np.ndarray, k: float):\n    return k * np.sum(a)\n";
        let rust = transpile(src).unwrap();
        assert_eq!(sig_of(&rust, "f"), "pub fn f(a: &[f64], k: f64) -> f64 {");
    }

    #[test]
    fn zeros_and_index_assignment_and_loop() {
        let src = "def cumsum(x: np.ndarray):\n    y = np.zeros(len(x))\n    acc = 0.0\n    for i in range(len(x)):\n        acc = acc + x[i]\n        y[i] = acc\n    return y\n";
        let rust = transpile(src).unwrap();
        assert_eq!(
            sig_of(&rust, "cumsum"),
            "pub fn cumsum(x: &[f64]) -> Vec<f64> {"
        );
        assert!(rust.contains("let mut y: Vec<f64> = np::zeros(x.len());"));
        assert!(rust.contains("let mut acc: f64 = 0.0f64;"));
        assert!(rust.contains("for i in (0usize)..(x.len()) {"));
        assert!(rust.contains("y[i] = acc;"));
    }

    #[test]
    fn power_uses_powi_for_integer_exponent() {
        let src = "def sq(x):\n    return x ** 2\n";
        let rust = transpile(src).unwrap();
        assert!(rust.contains(".powi(2)"));
    }

    #[test]
    fn scalar_broadcast_order_is_preserved() {
        // 1.0 - a  (scalar on the left, non-commutative)
        let src = "def f(a: np.ndarray):\n    return 1.0 - a\n";
        let rust = transpile(src).unwrap();
        assert!(rust.contains("np::map1(a, |x| 1.0f64 - x)"));
    }

    #[test]
    fn rejects_unsupported_call() {
        let src = "def f(x):\n    return np.fft(x)\n";
        let err = transpile(src).unwrap_err();
        assert!(err.contains("unsupported function"));
    }

    #[test]
    fn rejects_assignment_inside_loop_without_init() {
        let src = "def f(x: np.ndarray):\n    for i in range(len(x)):\n        acc = x[i]\n    return acc\n";
        let err = transpile(src).unwrap_err();
        assert!(err.contains("initialise it before the loop"));
    }

    #[test]
    fn emitted_module_has_prelude() {
        let rust = transpile("def f(x):\n    return x\n").unwrap();
        assert!(rust.contains("pub mod np"));
        assert!(rust.contains("pub fn sum(a: &[f64]) -> f64"));
    }

    #[test]
    fn if_without_else() {
        let src = "def relu(x):\n    if x > 0.0:\n        return x\n    return 0.0\n";
        let rust = transpile(src).unwrap();
        // Scope the check to the emitted function (the prelude legitimately
        // contains `} else {` in its min-length helpers).
        let func = &rust[rust.find("pub fn relu").unwrap()..];
        assert!(func.contains("if (x > 0.0f64) {"));
        assert!(func.contains("return x;"));
        assert!(func.contains("return 0.0f64;"));
        assert!(!func.contains("else"));
    }

    #[test]
    fn if_elif_else_desugars() {
        let src = "def sign(x):\n    if x > 0.0:\n        return 1.0\n    elif x < 0.0:\n        return -1.0\n    else:\n        return 0.0\n";
        let rust = transpile(src).unwrap();
        assert!(rust.contains("if (x > 0.0f64) {"));
        // elif becomes a nested `if` inside the else branch.
        assert!(rust.contains("} else {"));
        assert!(rust.contains("if (x < 0.0f64) {"));
    }

    #[test]
    fn comparison_outside_condition_is_rejected() {
        // A comparison is only allowed as an `if`/`elif` condition, never as a
        // value; the parser rejects `y = x > 0.0`.
        let src = "def f(x):\n    y = x > 0.0\n    return y\n";
        assert!(transpile(src).is_err());
    }

    #[test]
    fn while_loop_emits() {
        let src = "def newton(a):\n    x = a\n    i = 0\n    while i < 20:\n        x = 0.5 * (x + a / x)\n        i = i + 1\n    return x\n";
        let rust = transpile(src).unwrap();
        let func = &rust[rust.find("pub fn newton").unwrap()..];
        assert!(func.contains("while (i < 20"));
        assert!(func.contains("x = (0.5f64 * (x + (a / x)));"));
        assert!(func.contains("i = (i + 1"));
    }

    #[test]
    fn linalg_solve_routes_to_solvers() {
        let src = "def solve(A, b):\n    return np.linalg.solve(A, b)\n";
        let rust = transpile(src).unwrap();
        // A inferred as a matrix (flat &[f64]); b as a vector.
        assert!(rust.contains("pub fn solve(A: &[f64], b: &[f64]) -> Vec<f64>"));
        // Routed to the verified LU solver, not re-derived in std Rust.
        assert!(rust.contains("scirust_solvers::linalg::solve"));
        assert!(rust.contains("Matrix::from_row_major"));

        // And the module is flagged as needing the scirust-solvers crate.
        let sir = transpile_to_sir(src).unwrap();
        assert_eq!(required_crates(&sir), vec!["scirust-solvers"]);
    }

    #[test]
    fn linalg_det_routes_to_solvers() {
        let src = "def det(A):\n    return np.linalg.det(A)\n";
        let rust = transpile(src).unwrap();
        // A inferred as a matrix (flat &[f64]); scalar return.
        assert!(rust.contains("pub fn det(A: &[f64]) -> f64"));
        assert!(rust.contains(".determinant()"));
        assert!(rust.contains("Matrix::from_row_major"));

        let sir = transpile_to_sir(src).unwrap();
        assert_eq!(required_crates(&sir), vec!["scirust-solvers"]);
    }

    #[test]
    fn linalg_eigvalsh_routes_to_solvers() {
        let src = "def eig(A):\n    return np.linalg.eigvalsh(A)\n";
        let rust = transpile(src).unwrap();
        assert!(rust.contains("pub fn eig(A: &[f64]) -> Vec<f64>"));
        assert!(rust.contains("scirust_solvers::linalg::eigen_symmetric"));
        assert!(rust.contains(".eigenvalues"));

        let sir = transpile_to_sir(src).unwrap();
        assert_eq!(required_crates(&sir), vec!["scirust-solvers"]);
    }

    #[test]
    fn fft_routes_to_signal_with_complex_return() {
        let src = "def spec(x: np.ndarray):\n    return np.fft.fft(x)\n";
        let rust = transpile(src).unwrap();
        assert!(rust.contains("pub fn spec(x: &[f64]) -> Vec<scirust_signal::complex::Complex>"));
        // Full complex DFT (all N bins) via the verified in-place FFT.
        assert!(rust.contains("scirust_signal::fft::fft(&mut __b)"));
        assert!(rust.contains("scirust_signal::complex::Complex::new(__v, 0.0)"));

        let sir = transpile_to_sir(src).unwrap();
        assert_eq!(required_crates(&sir), vec!["scirust-signal"]);
    }

    #[test]
    fn abs_of_fft_is_a_real_magnitude_array() {
        let src = "def mag(x: np.ndarray):\n    return np.abs(np.fft.fft(x))\n";
        let rust = transpile(src).unwrap();
        assert!(rust.contains("pub fn mag(x: &[f64]) -> Vec<f64>"));
        assert!(rust.contains(".iter().map(|c| c.mag())"));
    }

    #[test]
    fn rfft_and_ifft_route_to_signal() {
        let rf = transpile("def rf(x: np.ndarray):\n    return np.fft.rfft(x)\n").unwrap();
        assert!(rf.contains("scirust_signal::fft::fft_real(x)"));

        let rt =
            transpile("def rt(x: np.ndarray):\n    return np.fft.ifft(np.fft.fft(x))\n").unwrap();
        assert!(rt.contains("scirust_signal::fft::ifft(&mut __ib)"));
        assert!(rt.contains("scirust_signal::fft::fft(&mut __b)"));
    }

    #[test]
    fn matmul_operator_routes_to_matvec() {
        let src = "def mv(A, b):\n    return A @ b\n";
        let rust = transpile(src).unwrap();
        assert!(rust.contains("pub fn mv(A: &[f64], b: &[f64]) -> Vec<f64>"));
        assert!(rust.contains(".matvec(__b)"));

        let sir = transpile_to_sir(src).unwrap();
        assert_eq!(required_crates(&sir), vec!["scirust-solvers"]);
    }

    #[test]
    fn linalg_inv_returns_a_matrix_value() {
        let src = "def inv(A):\n    return np.linalg.inv(A)\n";
        let rust = transpile(src).unwrap();
        assert!(rust.contains("pub fn inv(A: &[f64]) -> scirust_solvers::Matrix"));
        assert!(rust.contains(".inverse()"));

        let sir = transpile_to_sir(src).unwrap();
        assert_eq!(required_crates(&sir), vec!["scirust-solvers"]);
    }

    #[test]
    fn transpose_and_matmul_route_to_solvers() {
        let tp = transpile("def tp(A):\n    return A.T\n").unwrap();
        assert!(tp.contains("pub fn tp(A: &[f64]) -> scirust_solvers::Matrix"));
        assert!(tp.contains(".transpose()"));

        // A @ A.T chains transpose (MatrixVal) into matmul.
        let g = transpile("def gram(A):\n    return A @ A.T\n").unwrap();
        assert!(g.contains(".matmul(&"));
        assert!(g.contains(".transpose()"));
        assert_eq!(
            required_crates(&transpile_to_sir("def gram(A):\n    return A @ A.T\n").unwrap()),
            vec!["scirust-solvers"]
        );
    }

    #[test]
    fn std_only_module_needs_no_external_crates() {
        let sir = transpile_to_sir("def f(x):\n    return x + 1.0\n").unwrap();
        assert!(required_crates(&sir).is_empty());
    }

    // ---- MATLAB front-end -------------------------------------------------

    #[test]
    fn matlab_norm2_infers_array_from_elementwise() {
        // `.*` operands are array evidence, so `x` is inferred as an array;
        // the body lowers to the same sqrt/sum/ew core as the Python `norm`.
        let src = "function y = norm2(x)\n  y = sqrt(sum(x .* x));\nend\n";
        let rust = transpile_matlab(src).unwrap();
        assert_eq!(sig_of(&rust, "norm2"), "pub fn norm2(x: &[f64]) -> f64 {");
        assert!(rust.contains("np::sum(&(np::ew2(x, x, |x, y| x * y)))"));
        assert!(rust.contains(").sqrt()"));
    }

    #[test]
    fn matlab_relu_hoists_branch_assigned_output() {
        // `y` is first assigned inside the `if`/`else`, so it must be hoisted to
        // an uninitialised `let mut y: f64;` and written in every branch.
        let src =
            "function y = relu(x)\n  if x > 0.0\n    y = x;\n  else\n    y = 0.0;\n  end\nend\n";
        let rust = transpile_matlab(src).unwrap();
        let func = &rust[rust.find("pub fn relu").unwrap()..];
        assert!(func.contains("let mut y: f64;"));
        assert!(func.contains("if (x > 0.0f64) {"));
        assert!(func.contains("} else {"));
        assert!(func.contains("return y;"));
    }

    #[test]
    fn matlab_for_loop_is_one_based_and_inclusive() {
        // `for i = 1:length(x)` -> `for i in 1..(len+1)`; `x(i)` -> `x[i-1]`.
        let src = "function s = mysum(x)\n  s = 0.0;\n  for i = 1:length(x)\n    s = s + x(i);\n  end\nend\n";
        let rust = transpile_matlab(src).unwrap();
        assert!(rust.contains("for i in (1usize)..((x.len() + 1usize)) {"));
        assert!(rust.contains("x[(i - 1usize)]"));
    }

    #[test]
    fn matlab_elementwise_array_output() {
        let src = "function y = ew_scale(x, w)\n  y = x .* w + x;\nend\n";
        let rust = transpile_matlab(src).unwrap();
        assert_eq!(
            sig_of(&rust, "ew_scale"),
            "pub fn ew_scale(x: &[f64], w: &[f64]) -> Vec<f64> {"
        );
    }

    #[test]
    fn matlab_sequential_ifs_reassign_output() {
        // `y` first bound at top level (`let mut y: f64 = x;`), then mutated by
        // two independent `if`s — no hoisting needed here.
        let src = "function y = clamp_m(x, lo, hi)\n  y = x;\n  if y < lo\n    y = lo;\n  end\n  if y > hi\n    y = hi;\n  end\nend\n";
        let rust = transpile_matlab(src).unwrap();
        let func = &rust[rust.find("pub fn clamp_m").unwrap()..];
        assert!(func.contains("let mut y: f64 = x;"));
        assert!(func.contains("if (y < lo) {"));
        assert!(func.contains("y = lo;"));
        assert!(func.contains("if (y > hi) {"));
    }

    #[test]
    fn matlab_rejects_unknown_intrinsic() {
        let src = "function y = f(x)\n  y = frobnicate(x);\nend\n";
        let err = transpile_matlab(src).unwrap_err();
        assert!(err.contains("unknown function or variable"));
    }

    #[test]
    fn matlab_scalar_only_needs_no_external_crates() {
        let sir =
            transpile_matlab_to_sir("function y = poly_m(x)\n  y = x^3 - 2.0 * x^2 + 1.0;\nend\n")
                .unwrap();
        assert!(required_crates(&sir).is_empty());
    }

    #[test]
    fn matlab_multi_output_returns_a_tuple() {
        let src = "function [s, d] = sumdiff(a, b)\n  s = a + b;\n  d = a - b;\nend\n";
        let rust = transpile_matlab(src).unwrap();
        assert_eq!(
            sig_of(&rust, "sumdiff"),
            "pub fn sumdiff(a: f64, b: f64) -> (f64, f64) {"
        );
        assert!(rust.contains("return (s, d);"));
    }

    #[test]
    fn matlab_single_output_unchanged() {
        let rust = transpile_matlab("function y = sq(x)\n  y = x * x;\nend\n").unwrap();
        assert_eq!(sig_of(&rust, "sq"), "pub fn sq(x: f64) -> f64 {");
        assert!(rust.contains("return y;"));
    }

    #[test]
    fn matlab_three_outputs_with_new_reductions() {
        // Exercises `[a, b, c] = …` plus the MATLAB min/mean/max reductions.
        let src = "function [lo, mu, hi] = stats3(x)\n  lo = min(x);\n  mu = mean(x);\n  hi = max(x);\nend\n";
        let rust = transpile_matlab(src).unwrap();
        assert_eq!(
            sig_of(&rust, "stats3"),
            "pub fn stats3(x: &[f64]) -> (f64, f64, f64) {"
        );
        assert!(rust.contains("np::min(x)"));
        assert!(rust.contains("np::max(x)"));
        assert!(rust.contains("np::sum(x)") && rust.contains("x.len()"));
        assert!(rust.contains("return (lo, mu, hi);"));
    }

    #[test]
    fn matlab_new_math_intrinsics() {
        let rust =
            transpile_matlab("function y = mathx(x)\n  y = log(x) + floor(x) + atan(x);\nend\n")
                .unwrap();
        assert!(rust.contains("(x).ln()"));
        assert!(rust.contains("(x).floor()"));
        assert!(rust.contains("(x).atan()"));
    }

    #[test]
    fn matlab_det_and_inv_route_to_solvers() {
        // `A` is inferred as a matrix purely from `det(A)` / `inv(A)`.
        let d = transpile_matlab("function d = mdet(A)\n  d = det(A);\nend\n").unwrap();
        assert_eq!(sig_of(&d, "mdet"), "pub fn mdet(A: &[f64]) -> f64 {");
        assert!(d.contains(".determinant()"));
        let sir = transpile_matlab_to_sir("function d = mdet(A)\n  d = det(A);\nend\n").unwrap();
        assert_eq!(required_crates(&sir), vec!["scirust-solvers"]);

        let inv = transpile_matlab("function B = minv(A)\n  B = inv(A);\nend\n").unwrap();
        assert_eq!(
            sig_of(&inv, "minv"),
            "pub fn minv(A: &[f64]) -> scirust_solvers::Matrix {"
        );
        assert!(inv.contains(".inverse()"));
    }

    #[test]
    fn matlab_backslash_routes_to_lu_solver() {
        // `A \ b` -> LU solve; `A` inferred matrix, `b` inferred vector.
        let src = "function x = msolve(A, b)\n  x = A \\ b;\nend\n";
        let rust = transpile_matlab(src).unwrap();
        assert_eq!(
            sig_of(&rust, "msolve"),
            "pub fn msolve(A: &[f64], b: &[f64]) -> Vec<f64> {"
        );
        assert!(rust.contains("scirust_solvers::linalg::solve"));
        assert_eq!(
            required_crates(&transpile_matlab_to_sir(src).unwrap()),
            vec!["scirust-solvers"]
        );
    }

    #[test]
    fn matlab_backslash_needs_a_matrix_on_the_left() {
        // A scalar on the left of `\` (scalar left-division) is outside the
        // subset and rejected — only `matrix \ vector` (solve) is supported.
        let src = "function x = f(b)\n  s = 2.0;\n  x = s \\ b;\nend\n";
        let err = transpile_matlab(src).unwrap_err();
        assert!(err.contains("expects a matrix on the left"));
    }

    #[test]
    fn matlab_norm_is_euclidean_and_needs_a_vector() {
        // `norm(v)` -> sqrt(sum(v .* v)); `v` inferred a vector purely from the
        // intrinsic, and the emitted code is std-only (no external crate).
        let src = "function y = mnorm(v)\n  y = norm(v);\nend\n";
        let rust = transpile_matlab(src).unwrap();
        assert_eq!(sig_of(&rust, "mnorm"), "pub fn mnorm(v: &[f64]) -> f64 {");
        assert!(rust.contains("np::sum"));
        assert!(rust.contains(".sqrt()"));
        assert!(required_crates(&transpile_matlab_to_sir(src).unwrap()).is_empty());
        // A scalar argument (here a scalar expression) is rejected — `norm` of a
        // vector only.
        let bad = transpile_matlab("function y = f(x)\n  y = norm(x * 2.0);\nend\n");
        assert!(bad.is_err());
    }

    #[test]
    fn matlab_dot_routes_to_fixed_order_reduction() {
        // `dot(a, b)` -> the fixed-order `np::dot`; BOTH operands infer as
        // vectors (the second argument is evidence too).
        let src = "function s = mdot(a, b)\n  s = dot(a, b);\nend\n";
        let rust = transpile_matlab(src).unwrap();
        assert_eq!(
            sig_of(&rust, "mdot"),
            "pub fn mdot(a: &[f64], b: &[f64]) -> f64 {"
        );
        assert!(rust.contains("np::dot"));
    }

    #[test]
    fn matlab_eig_routes_to_symmetric_eigensolver() {
        // `eig(A)` -> ascending eigenvalues via the verified symmetric
        // eigensolver; `A` inferred a matrix, result an (array) vector.
        let src = "function e = meig(A)\n  e = eig(A);\nend\n";
        let rust = transpile_matlab(src).unwrap();
        assert_eq!(
            sig_of(&rust, "meig"),
            "pub fn meig(A: &[f64]) -> Vec<f64> {"
        );
        assert!(rust.contains("eigen_symmetric"));
        assert_eq!(
            required_crates(&transpile_matlab_to_sir(src).unwrap()),
            vec!["scirust-solvers"]
        );
        // `eig` on a non-matrix (scalar expression) argument is rejected.
        let bad = transpile_matlab("function e = f(x)\n  e = eig(x * 2.0);\nend\n");
        assert!(bad.is_err());
    }

    #[test]
    fn matlab_round_and_fix_map_to_f64_methods() {
        // `round` (half away from zero) and `fix` (truncate) map to the f64
        // methods; both work elementwise, so a bare scalar stays std-only.
        let rust = transpile_matlab("function y = f(x)\n  y = round(x) + fix(x);\nend\n").unwrap();
        assert!(rust.contains("(x).round()"));
        assert!(rust.contains("(x).trunc()"));
    }

    #[test]
    fn matlab_mod_and_rem_compose_from_floor_and_trunc() {
        // mod(a,b) = a - b*floor(a/b); rem(a,b) = a - b*fix(a/b). Both std-only.
        let m = transpile_matlab("function y = f(a, b)\n  y = mod(a, b);\nend\n").unwrap();
        assert!(m.contains(".floor()"));
        assert!(!m.contains(".trunc()"));
        let r = transpile_matlab("function y = f(a, b)\n  y = rem(a, b);\nend\n").unwrap();
        assert!(r.contains(".trunc()"));
        assert!(!r.contains(".floor()"));
        assert!(
            required_crates(
                &transpile_matlab_to_sir("function y = f(a, b)\n  y = mod(a, b);\nend\n").unwrap()
            )
            .is_empty()
        );
    }

    #[test]
    fn matlab_sign_is_minus_one_zero_plus_one() {
        // sign(x) -> -1/0/+1 with sign(0) == 0 (NOT f64::signum); argument bound
        // once.
        let rust = transpile_matlab("function y = f(x)\n  y = sign(x);\nend\n").unwrap();
        assert_eq!(sig_of(&rust, "f"), "pub fn f(x: f64) -> f64 {");
        assert!(rust.contains("if __x > 0.0"));
        assert!(rust.contains("else if __x < 0.0"));
        assert!(!rust.contains("signum"));
        // `sign` on an array (a vector) is refused in this subset.
        let bad = transpile_matlab("function y = f(v)\n  y = sign(v) + sum(v);\nend\n");
        assert!(bad.is_err());
    }

    #[test]
    fn matlab_atan2_and_hypot_map_to_binary_f64_methods() {
        // atan2(y, x) / hypot(a, b) -> `(l).method(r)`, argument order preserved.
        let a = transpile_matlab("function r = f(y, x)\n  r = atan2(y, x);\nend\n").unwrap();
        assert_eq!(sig_of(&a, "f"), "pub fn f(y: f64, x: f64) -> f64 {");
        assert!(a.contains("(y).atan2(x)"));
        let h = transpile_matlab("function r = f(a, b)\n  r = hypot(a, b);\nend\n").unwrap();
        assert!(h.contains("(a).hypot(b)"));
        // Both are std-only (no external crate).
        assert!(
            required_crates(
                &transpile_matlab_to_sir("function r = f(y, x)\n  r = atan2(y, x);\nend\n")
                    .unwrap()
            )
            .is_empty()
        );
        // A vector argument is refused (scalar-only in this subset).
        let bad = transpile_matlab("function r = f(v)\n  r = hypot(sum(v), v);\nend\n");
        assert!(bad.is_err());
    }

    #[test]
    fn matlab_two_arg_max_min_vs_one_arg_reduction() {
        // Two args -> f64::max/min on two scalars; the operands stay scalar
        // (NOT inferred as arrays by the reduction rule).
        let two =
            transpile_matlab("function s = f(a, b)\n  s = max(a, b) - min(a, b);\nend\n").unwrap();
        assert_eq!(sig_of(&two, "f"), "pub fn f(a: f64, b: f64) -> f64 {");
        assert!(two.contains("(a).max(b)"));
        assert!(two.contains("(a).min(b)"));
        // One arg -> the reduction over a vector is still available.
        let one = transpile_matlab("function m = f(v)\n  m = max(v);\nend\n").unwrap();
        assert_eq!(sig_of(&one, "f"), "pub fn f(v: &[f64]) -> f64 {");
        assert!(one.contains("np::max(v)"));
    }

    #[test]
    fn matlab_power_is_the_functional_form_of_pow() {
        // power(a, b) shares the `^` lowering; an integer exponent folds to powi.
        let rust = transpile_matlab("function y = f(a)\n  y = power(a, 3.0);\nend\n").unwrap();
        assert!(rust.contains(".powi(3)"));
        let rust2 = transpile_matlab("function y = f(a, b)\n  y = power(a, b);\nend\n").unwrap();
        assert!(rust2.contains(".powf(b)"));
    }

    #[test]
    fn matlab_elementwise_power_broadcasts_and_maps() {
        // v .^ 2 — array base, scalar exponent -> map1 broadcast, array out.
        let sq = transpile_matlab("function y = f(v)\n  y = v .^ 2;\nend\n").unwrap();
        assert_eq!(sig_of(&sq, "f"), "pub fn f(v: &[f64]) -> Vec<f64> {");
        assert!(sq.contains("np::map1(v, |x| x.powf("));
        // a .^ b — two arrays -> elementwise ew2.
        let ew = transpile_matlab("function y = f(a, b)\n  y = a .^ b;\nend\n").unwrap();
        assert!(ew.contains("np::ew2(a, b, |x, y| x.powf(y))"));
        // 2 .^ v — scalar base, array exponent -> the scalar is on the left.
        let bc = transpile_matlab("function y = f(v)\n  y = 2.0 .^ v;\nend\n").unwrap();
        assert!(bc.contains("np::map1(v, |x| (2.0f64).powf(x))"));
    }

    #[test]
    fn matlab_matrix_power_operator_on_arrays_is_rejected() {
        // `^` (matrix power) on a vector is refused; `.^` is the elementwise form.
        let err = transpile_matlab("function y = f(v)\n  y = v ^ 2 + sum(v);\nend\n").unwrap_err();
        assert!(err.contains("matrix power"));
    }

    #[test]
    fn matlab_vector_builtins_infer_array_and_route_to_prelude() {
        // cumsum / cumprod / cummax / cummin / diff / sort / flip take a vector
        // and return a vector; the argument is inferred as an array purely from
        // the builtin.
        for (call, helper) in [
            ("cumsum", "np::cumsum"),
            ("cumprod", "np::cumprod"),
            ("cummax", "np::cummax"),
            ("cummin", "np::cummin"),
            ("diff", "np::diff"),
            ("sort", "np::sort"),
            ("flip", "np::flip"),
        ]
        {
            let src = format!("function y = f(v)\n  y = {}(v);\nend\n", call);
            let rust = transpile_matlab(&src).unwrap();
            assert_eq!(sig_of(&rust, "f"), "pub fn f(v: &[f64]) -> Vec<f64> {");
            assert!(rust.contains(helper), "{} should emit {}", call, helper);
        }
        // The prelude helpers are present and std-only.
        let rust = transpile_matlab("function y = f(v)\n  y = cumsum(v);\nend\n").unwrap();
        assert!(rust.contains("pub fn cumsum(a: &[f64]) -> Vec<f64>"));
        assert!(
            required_crates(
                &transpile_matlab_to_sir("function y = f(v)\n  y = sort(v);\nend\n").unwrap()
            )
            .is_empty()
        );
        // A scalar argument (scalar expression) is rejected.
        let bad = transpile_matlab("function y = f(x)\n  y = diff(x * 2.0);\nend\n");
        assert!(bad.is_err());
    }

    #[test]
    fn matlab_reduction_stats_infer_array_and_return_scalar() {
        // var / std / median take a vector and return a scalar; the argument is
        // inferred as an array from the reduction.
        for (call, helper) in [
            ("var", "np::var"),
            ("std", "np::std"),
            ("median", "np::median"),
        ]
        {
            let src = format!("function y = f(v)\n  y = {}(v);\nend\n", call);
            let rust = transpile_matlab(&src).unwrap();
            assert_eq!(sig_of(&rust, "f"), "pub fn f(v: &[f64]) -> f64 {");
            assert!(rust.contains(helper), "{} should emit {}", call, helper);
        }
        // `var` uses the sample (N-1) normalisation, std-only.
        let rust = transpile_matlab("function y = f(v)\n  y = var(v);\nend\n").unwrap();
        assert!(rust.contains("ss / (n as f64 - 1.0)"));
        assert!(
            required_crates(
                &transpile_matlab_to_sir("function y = f(v)\n  y = std(v);\nend\n").unwrap()
            )
            .is_empty()
        );
    }

    #[test]
    fn matlab_linspace_constructs_a_vector_from_scalars() {
        // linspace(a, b, n) builds a vector: scalar a/b params, integer count.
        let rust =
            transpile_matlab("function y = f(a, b)\n  y = linspace(a, b, 5);\nend\n").unwrap();
        assert_eq!(sig_of(&rust, "f"), "pub fn f(a: f64, b: f64) -> Vec<f64> {");
        assert!(rust.contains("np::linspace(a, b, 5usize)"));
        assert!(rust.contains("pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64>"));
        // The count may be a `length(x)`, not just a literal.
        let dyn_n =
            transpile_matlab("function y = f(a, b, x)\n  y = linspace(a, b, length(x));\nend\n")
                .unwrap();
        assert!(dyn_n.contains("np::linspace(a, b, x.len())"));
    }

    #[test]
    fn matlab_star_routes_matrix_products() {
        // matrix * vector -> matvec (A a matrix from `\`, x the vector solution).
        let mv =
            transpile_matlab("function r = f(A, b)\n  x = A \\ b;\n  r = A * x;\nend\n").unwrap();
        assert_eq!(
            sig_of(&mv, "f"),
            "pub fn f(A: &[f64], b: &[f64]) -> Vec<f64> {"
        );
        assert!(mv.contains(".matvec("));
        // matrix * matrix -> matmul (A a matrix from `inv`, B a produced matrix).
        let mm = transpile_matlab("function C = f(A)\n  B = inv(A);\n  C = A * B;\nend\n").unwrap();
        assert_eq!(
            sig_of(&mm, "f"),
            "pub fn f(A: &[f64]) -> scirust_solvers::Matrix {"
        );
        assert!(mm.contains(".matmul("));
        assert_eq!(
            required_crates(
                &transpile_matlab_to_sir(
                    "function r = f(A, b)\n  x = A \\ b;\n  r = A * x;\nend\n"
                )
                .unwrap()
            ),
            vec!["scirust-solvers"]
        );
        // Scalar * array is still broadcast, not a matrix product.
        let bc = transpile_matlab("function y = f(x)\n  y = 2.0 * cumsum(x);\nend\n").unwrap();
        assert!(bc.contains("np::map1"));
        assert!(!bc.contains(".matvec("));
    }

    #[test]
    fn matlab_transpose_operator_routes_to_transpose() {
        // `A'` -> Transpose (matrix out); `A` inferred a matrix from the operator.
        let t = transpile_matlab("function B = f(A)\n  B = A';\nend\n").unwrap();
        assert_eq!(
            sig_of(&t, "f"),
            "pub fn f(A: &[f64]) -> scirust_solvers::Matrix {"
        );
        assert!(t.contains(".transpose()"));
        // `.'` (non-conjugate transpose) parses the same for real matrices.
        assert!(transpile_matlab("function B = f(A)\n  B = A.';\nend\n").is_ok());
        // Composes with `*`: A' * A -> transpose then matmul (Gram matrix).
        let g = transpile_matlab("function C = f(A)\n  C = A' * A;\nend\n").unwrap();
        assert!(g.contains(".transpose()"));
        assert!(g.contains(".matmul("));
        // Transposing a scalar is rejected (only matrices transpose).
        let bad = transpile_matlab("function y = f(x)\n  y = (x + 1.0)';\nend\n");
        assert!(bad.is_err());
    }

    #[test]
    fn matlab_trace_and_cross_route_to_prelude() {
        // trace(A) -> scalar; `A` inferred a matrix from the intrinsic.
        let t = transpile_matlab("function t = f(A)\n  t = trace(A);\nend\n").unwrap();
        assert_eq!(sig_of(&t, "f"), "pub fn f(A: &[f64]) -> f64 {");
        assert!(t.contains("np::trace(A)"));
        // cross(a, b) -> vector; BOTH operands inferred vectors, std-only.
        let c = transpile_matlab("function c = f(a, b)\n  c = cross(a, b);\nend\n").unwrap();
        assert_eq!(
            sig_of(&c, "f"),
            "pub fn f(a: &[f64], b: &[f64]) -> Vec<f64> {"
        );
        assert!(c.contains("np::cross(a, b)"));
        assert!(
            required_crates(
                &transpile_matlab_to_sir("function c = f(a, b)\n  c = cross(a, b);\nend\n")
                    .unwrap()
            )
            .is_empty()
        );
        // trace of a non-matrix (scalar expression) is rejected.
        let bad = transpile_matlab("function t = f(x)\n  t = trace(x * 2.0);\nend\n");
        assert!(bad.is_err());
    }

    #[test]
    fn matlab_diag_dispatches_on_operand_type() {
        // diag(matrix) -> EXTRACT the diagonal (vector out).
        let ex = transpile_matlab("function d = f(A)\n  d = diag(A' * A);\nend\n").unwrap();
        assert_eq!(sig_of(&ex, "f"), "pub fn f(A: &[f64]) -> Vec<f64> {");
        assert!(ex.contains(".row(__i)[__i]"));
        // diag(vector) -> CONSTRUCT a diagonal matrix (matrix out).
        let ct = transpile_matlab("function M = f(v)\n  M = diag(cumsum(v));\nend\n").unwrap();
        assert_eq!(
            sig_of(&ct, "f"),
            "pub fn f(v: &[f64]) -> scirust_solvers::Matrix {"
        );
        assert!(ct.contains("from_fn"));
    }

    #[test]
    fn matlab_trapz_integrates_and_is_std_only() {
        // trapz(v) -> scalar via the fixed 0.5*(v[i-1]+v[i]) trapezoid rule.
        let rust = transpile_matlab("function t = f(v)\n  t = trapz(v);\nend\n").unwrap();
        assert_eq!(sig_of(&rust, "f"), "pub fn f(v: &[f64]) -> f64 {");
        assert!(rust.contains("np::trapz(v)"));
        assert!(rust.contains("0.5 * (a[i - 1] + a[i])"));
        assert!(
            required_crates(
                &transpile_matlab_to_sir("function t = f(v)\n  t = trapz(v);\nend\n").unwrap()
            )
            .is_empty()
        );
    }

    // ---- tuples / SVD -----------------------------------------------------

    #[test]
    fn svd_unpacks_and_routes_to_solvers() {
        let src = "def singvals(A):\n    U, S, Vh = np.linalg.svd(A)\n    return S\n";
        let rust = transpile(src).unwrap();
        // A inferred as a matrix; S is the (array) singular-value vector.
        assert!(rust.contains("pub fn singvals(A: &[f64]) -> Vec<f64>"));
        // Tuple destructure with the three element types.
        assert!(rust.contains(
            "let (U, S, Vh): (scirust_solvers::Matrix, Vec<f64>, scirust_solvers::Matrix) ="
        ));
        // Routed to the verified SVD kernel; Vh = Vᵀ (matches numpy).
        assert!(rust.contains("scirust_solvers::linalg::svd"));
        assert!(rust.contains("__svd.v.transpose()"));
        assert!(rust.contains("return S;"));

        let sir = transpile_to_sir(src).unwrap();
        assert_eq!(required_crates(&sir), vec!["scirust-solvers"]);
    }

    #[test]
    fn svd_reconstruction_chains_diag_and_matmul() {
        let src =
            "def recon(A):\n    U, S, Vh = np.linalg.svd(A)\n    return U @ np.diag(S) @ Vh\n";
        let rust = transpile(src).unwrap();
        assert!(rust.contains("pub fn recon(A: &[f64]) -> scirust_solvers::Matrix"));
        // np.diag(S) -> square diagonal matrix, chained through matmul.
        assert!(rust.contains("scirust_solvers::Matrix::from_fn"));
        assert!(rust.contains(".matmul(&"));
    }

    #[test]
    fn svd_as_scalar_value_is_rejected() {
        let src = "def f(A):\n    x = np.linalg.svd(A)\n    return x\n";
        let err = transpile(src).unwrap_err();
        assert!(err.contains("returns a tuple"));
    }

    #[test]
    fn qr_unpacks_and_routes_to_solvers() {
        let src = "def qr_rec(A):\n    Q, R = np.linalg.qr(A)\n    return Q @ R\n";
        let rust = transpile(src).unwrap();
        assert!(rust.contains("pub fn qr_rec(A: &[f64]) -> scirust_solvers::Matrix"));
        assert!(rust.contains("let (Q, R): (scirust_solvers::Matrix, scirust_solvers::Matrix) ="));
        assert!(rust.contains("scirust_solvers::linalg::qr_decompose"));
        assert!(rust.contains("__qr.q()") && rust.contains("__qr.r()"));

        let sir = transpile_to_sir(src).unwrap();
        assert_eq!(required_crates(&sir), vec!["scirust-solvers"]);
    }

    #[test]
    fn qr_as_scalar_value_is_rejected() {
        let src = "def f(A):\n    x = np.linalg.qr(A)\n    return x\n";
        let err = transpile(src).unwrap_err();
        assert!(err.contains("returns a tuple"));
    }

    // ---- expanded intrinsic vocabulary ------------------------------------

    #[test]
    fn new_unary_math_intrinsics_map_to_f64_methods() {
        let rust = transpile("def f(x):\n    return np.log(x) + np.arctan(x)\n").unwrap();
        assert!(rust.contains("(x).ln()"));
        assert!(rust.contains("(x).atan()"));
        // Elementwise over an array uses map1 with the same method.
        let ra = transpile("def f(x: np.ndarray):\n    return np.floor(x)\n").unwrap();
        assert!(ra.contains("np::map1(x, |x| x.floor())"));
    }

    #[test]
    fn reductions_prod_max_min_and_mean() {
        let rust = transpile(
            "def f(x: np.ndarray):\n    return np.prod(x) + np.max(x) - np.min(x) + np.mean(x)\n",
        )
        .unwrap();
        assert!(rust.contains("np::prod(x)"));
        assert!(rust.contains("np::max(x)"));
        assert!(rust.contains("np::min(x)"));
        // mean desugars to sum(x) / len(x).
        assert!(rust.contains("np::sum(x)") && rust.contains("x.len()"));
    }

    #[test]
    fn reduction_infers_array_param() {
        // `x` is used only via np.max -> inferred as an array.
        let rust = transpile("def f(x):\n    return np.max(x)\n").unwrap();
        assert_eq!(sig_of(&rust, "f"), "pub fn f(x: &[f64]) -> f64 {");
    }

    #[test]
    fn tuple_unpack_arity_mismatch_is_rejected() {
        let src = "def f(A):\n    U, S = np.linalg.svd(A)\n    return S\n";
        let err = transpile(src).unwrap_err();
        assert!(err.contains("expects 3 names"));
    }

    #[test]
    fn tuple_unpack_of_unsupported_rhs_is_rejected() {
        let src = "def f(x: np.ndarray):\n    a, b = np.sum(x)\n    return a\n";
        let err = transpile(src).unwrap_err();
        assert!(err.contains("multi-output kernel"));
    }

    // ---- Python élargi: user-defined function calls + lists ---------------

    #[test]
    fn user_call_scalar_composition() {
        let src = "def sq(x):\n    return x * x\ndef sumsq(a, b):\n    return sq(a) + sq(b)\n";
        let rust = transpile(src).unwrap();
        assert_eq!(
            sig_of(&rust, "sumsq"),
            "pub fn sumsq(a: f64, b: f64) -> f64 {"
        );
        // Direct Rust calls to the earlier function.
        assert!(rust.contains("return (sq(a) + sq(b));"));
    }

    #[test]
    fn user_call_infers_array_param_from_callee_signature() {
        // `x` is never indexed/summed directly here — its array-ness comes
        // solely from `dbl`'s (hinted) array parameter.
        let src = "def dbl(v: np.ndarray):\n    return 2.0 * v\ndef sumdbl(x):\n    return np.sum(dbl(x))\n";
        let rust = transpile(src).unwrap();
        assert_eq!(sig_of(&rust, "sumdbl"), "pub fn sumdbl(x: &[f64]) -> f64 {");
        assert!(rust.contains("np::sum(&(dbl(x)))"));
    }

    #[test]
    fn list_literal_is_a_vec() {
        let src = "def wavg(x: np.ndarray):\n    w = [0.5, 0.3, 0.2]\n    return np.dot(x, w)\n";
        let rust = transpile(src).unwrap();
        assert!(rust.contains("let mut w: Vec<f64> = vec![0.5f64, 0.3f64, 0.2f64];"));
        assert!(rust.contains("np::dot(x, &(w))"));
    }

    #[test]
    fn call_to_undefined_function_is_rejected() {
        let src = "def f(x):\n    return g(x)\n";
        let err = transpile(src).unwrap_err();
        assert!(err.contains("unsupported function"));
    }

    #[test]
    fn user_call_arity_mismatch_is_rejected() {
        let src = "def sq(x):\n    return x * x\ndef f(a):\n    return sq(a, a)\n";
        let err = transpile(src).unwrap_err();
        assert!(err.contains("expects 1 argument"));
    }

    #[test]
    fn user_call_arg_type_mismatch_is_rejected() {
        // `g` expects an array, but `a` is pinned to a scalar by its `float`
        // hint, so passing it to `g` must be rejected (not silently coerced).
        let src =
            "def g(v: np.ndarray):\n    return np.sum(v)\ndef f(a: float):\n    return g(a)\n";
        let err = transpile(src).unwrap_err();
        assert!(err.contains("expected") && err.contains("Array"));
    }

    #[test]
    fn define_before_use_is_required() {
        // `sumsq` calls `sq`, but `sq` is defined *after* it -> unknown.
        let src = "def sumsq(a, b):\n    return sq(a) + sq(b)\ndef sq(x):\n    return x * x\n";
        let err = transpile(src).unwrap_err();
        assert!(err.contains("unsupported function"));
    }

    // ---- general tuple returns --------------------------------------------

    #[test]
    fn tuple_return_two_scalars() {
        let rust = transpile("def addsub(a, b):\n    return a + b, a - b\n").unwrap();
        assert_eq!(
            sig_of(&rust, "addsub"),
            "pub fn addsub(a: f64, b: f64) -> (f64, f64) {"
        );
        assert!(rust.contains("return ((a + b), (a - b));"));
    }

    #[test]
    fn tuple_return_three_scalars_from_reductions() {
        let rust =
            transpile("def s(x: np.ndarray):\n    return np.min(x), np.mean(x), np.max(x)\n")
                .unwrap();
        assert_eq!(
            sig_of(&rust, "s"),
            "pub fn s(x: &[f64]) -> (f64, f64, f64) {"
        );
        assert!(
            rust.contains("return (np::min(x), (np::sum(x) / ((x.len()) as f64)), np::max(x));")
        );
    }

    #[test]
    fn tuple_return_of_arrays_is_rejected() {
        // Tuple elements must be scalars in this subset.
        let src = "def f(x: np.ndarray):\n    return x, x\n";
        let err = transpile(src).unwrap_err();
        assert!(err.contains("tuple return elements must be scalars"));
    }

    #[test]
    fn inconsistent_single_vs_tuple_return_is_rejected() {
        let src = "def f(x):\n    if x > 0.0:\n        return x, x\n    return x\n";
        let err = transpile(src).unwrap_err();
        assert!(err.contains("inconsistent types"));
    }

    #[test]
    fn calling_a_tuple_returning_function_is_rejected() {
        // `pair` returns a tuple; using it as a value in `g` is unsupported.
        let src = "def pair(x):\n    return x, x\ndef g(x):\n    return pair(x)\n";
        let err = transpile(src).unwrap_err();
        assert!(err.contains("returns a tuple"));
    }
}
