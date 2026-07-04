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
pub mod front_python;
pub mod lower;
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
    fn std_only_module_needs_no_external_crates() {
        let sir = transpile_to_sir("def f(x):\n    return x + 1.0\n").unwrap();
        assert!(required_crates(&sir).is_empty());
    }
}
