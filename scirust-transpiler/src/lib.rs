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
//! * intrinsics: `np.sum`, `np.dot`, `np.zeros`, `np.ones`, `len`,
//!   `np.sqrt/exp/sin/cos/abs/tanh` (scalar or elementwise);
//! * `for i in range(...)` loops, indexing `a[i]`, index-assignment `a[i] = …`,
//!   `return`.
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

pub use sir::{SirFunc, SirModule, Ty, required_crates};

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
        rust.lines()
            .find(|l| l.contains(&format!("pub fn {}(", name)))
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
}
