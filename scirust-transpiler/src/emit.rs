//! Emit deterministic, std-only Rust source from the [`crate::sir`] IR.
//!
//! Design choices that make the output *trustworthy*:
//! * every reduction (`sum`, `dot`) runs in a fixed ascending index order, so
//!   the result is independent of any parallelism — bit-reproducible;
//! * only `std` is used, so the differential oracle can compile the output with
//!   `rustc` alone (no cargo, no external crates);
//! * a small, fixed `np` prelude of helper functions is emitted once and reused,
//!   rather than open-coding numerics at each call site.

use crate::sir::*;
use std::collections::HashSet;

/// Emit a full Rust module (prelude + all functions) from the SIR.
pub fn emit_module(m: &SirModule) -> String {
    let mut out = String::new();
    out.push_str(PRELUDE);
    out.push('\n');
    for f in &m.funcs
    {
        out.push_str(&emit_func(f));
        out.push('\n');
    }
    out
}

/// The deterministic numeric prelude. `sum`/`dot` pin the reduction order.
pub const PRELUDE: &str = r#"#[allow(dead_code)]
pub mod np {
    /// Fixed ascending-order sum (associativity pinned -> reproducible).
    pub fn sum(a: &[f64]) -> f64 {
        let mut s = 0.0f64;
        for i in 0..a.len() {
            s += a[i];
        }
        s
    }
    /// Fixed ascending-order dot product over the common prefix length.
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        let n = if a.len() < b.len() { a.len() } else { b.len() };
        let mut s = 0.0f64;
        for i in 0..n {
            s += a[i] * b[i];
        }
        s
    }
    pub fn zeros(n: usize) -> Vec<f64> {
        vec![0.0f64; n]
    }
    pub fn ones(n: usize) -> Vec<f64> {
        vec![1.0f64; n]
    }
    /// Elementwise map over one array.
    pub fn map1<F: Fn(f64) -> f64>(a: &[f64], f: F) -> Vec<f64> {
        let mut o = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            o.push(f(a[i]));
        }
        o
    }
    /// Elementwise binary op over the common prefix length.
    pub fn ew2<F: Fn(f64, f64) -> f64>(a: &[f64], b: &[f64], f: F) -> Vec<f64> {
        let n = if a.len() < b.len() { a.len() } else { b.len() };
        let mut o = Vec::with_capacity(n);
        for i in 0..n {
            o.push(f(a[i], b[i]));
        }
        o
    }
}
"#;

struct Ctx {
    params: HashSet<String>,
}

/// An emitted fragment together with its static type and whether the code text
/// is already a `&[f64]` (true only for array *parameters*, which are borrowed).
struct Frag {
    code: String,
    ty: Ty,
    borrowed: bool,
}

fn emit_func(f: &SirFunc) -> String {
    let params: HashSet<String> = f
        .params
        .iter()
        .filter(|(_, t)| *t == Ty::Array || *t == Ty::Matrix)
        .map(|(n, _)| n.clone())
        .collect();
    let ctx = Ctx { params };

    let sig_params: Vec<String> = f
        .params
        .iter()
        .map(|(n, t)| format!("{}: {}", n, param_ty(*t)))
        .collect();
    let ret = ret_ty(f.ret);

    let mut s = format!(
        "pub fn {}({}) -> {} {{\n",
        f.name,
        sig_params.join(", "),
        ret
    );
    for st in &f.body
    {
        emit_stmt(st, &ctx, 1, &mut s);
    }
    s.push_str("}\n");
    s
}

fn param_ty(t: Ty) -> &'static str {
    match t
    {
        Ty::Scalar => "f64",
        Ty::Array => "&[f64]",
        Ty::Matrix => "&[f64]",
        Ty::Int => "usize",
        Ty::Bool => "bool",
        Ty::ComplexArray => "Vec<scirust_signal::complex::Complex>",
        Ty::MatrixVal => "scirust_solvers::Matrix",
    }
}

fn ret_ty(t: Ty) -> &'static str {
    match t
    {
        Ty::Array => "Vec<f64>",
        Ty::ComplexArray => "Vec<scirust_signal::complex::Complex>",
        Ty::MatrixVal => "scirust_solvers::Matrix",
        _ => "f64",
    }
}

/// Type string for a hoisted local `let mut name: T;` — always an *owned* type
/// (locals are values, not borrowed parameters).
fn local_ty(t: Ty) -> &'static str {
    match t
    {
        Ty::Array => "Vec<f64>",
        Ty::ComplexArray => "Vec<scirust_signal::complex::Complex>",
        Ty::MatrixVal => "scirust_solvers::Matrix",
        Ty::Int => "usize",
        Ty::Bool => "bool",
        _ => "f64",
    }
}

fn indent(n: usize) -> String {
    "    ".repeat(n)
}

fn emit_stmt(st: &SirStmt, ctx: &Ctx, ind: usize, out: &mut String) {
    let pad = indent(ind);
    match st
    {
        SirStmt::Declare { name, ty } =>
        {
            // Uninitialised hoisted binding; Rust's definite-assignment analysis
            // guarantees it is written before any read.
            out.push_str(&format!("{}let mut {}: {};\n", pad, name, local_ty(*ty)));
        },
        SirStmt::LetTuple { names, value } =>
        {
            // Destructuring bind of a multi-output kernel:
            // `let (n0, n1, …): (T0, T1, …) = <tuple>;`
            let tup = emit_tuple(value, ctx);
            let name_list = names
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let ty_list = names
                .iter()
                .map(|(_, t)| local_ty(*t))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!(
                "{}let ({}): ({}) = {};\n",
                pad, name_list, ty_list, tup
            ));
        },
        SirStmt::Let { name, ty, value } =>
        {
            let v = emit(value, ctx);
            match ty
            {
                Ty::Array => out.push_str(&format!(
                    "{}let mut {}: Vec<f64> = {};\n",
                    pad,
                    name,
                    owned_array(v)
                )),
                Ty::ComplexArray => out.push_str(&format!(
                    "{}let mut {}: Vec<scirust_signal::complex::Complex> = {};\n",
                    pad, name, v.code
                )),
                Ty::MatrixVal => out.push_str(&format!(
                    "{}let mut {}: scirust_solvers::Matrix = {};\n",
                    pad, name, v.code
                )),
                _ => out.push_str(&format!(
                    "{}let mut {}: f64 = {};\n",
                    pad,
                    name,
                    scalar_of(v)
                )),
            }
        },
        SirStmt::Reassign { name, value } =>
        {
            let v = emit(value, ctx);
            match v.ty
            {
                Ty::Array => out.push_str(&format!("{}{} = {};\n", pad, name, owned_array(v))),
                Ty::ComplexArray | Ty::MatrixVal =>
                {
                    out.push_str(&format!("{}{} = {};\n", pad, name, v.code))
                },
                _ => out.push_str(&format!("{}{} = {};\n", pad, name, scalar_of(v))),
            }
        },
        SirStmt::SetIndex { name, index, value } =>
        {
            let idx = emit(index, ctx);
            let v = emit(value, ctx);
            out.push_str(&format!(
                "{}{}[{}] = {};\n",
                pad,
                name,
                idx.code,
                scalar_of(v)
            ));
        },
        SirStmt::For {
            var,
            start,
            end,
            body,
        } =>
        {
            let s = emit(start, ctx);
            let e = emit(end, ctx);
            out.push_str(&format!(
                "{}for {} in ({})..({}) {{\n",
                pad, var, s.code, e.code
            ));
            for b in body
            {
                emit_stmt(b, ctx, ind + 1, out);
            }
            out.push_str(&format!("{}}}\n", pad));
        },
        SirStmt::If { cond, then, els } =>
        {
            let c = emit(cond, ctx);
            out.push_str(&format!("{}if {} {{\n", pad, c.code));
            for b in then
            {
                emit_stmt(b, ctx, ind + 1, out);
            }
            if els.is_empty()
            {
                out.push_str(&format!("{}}}\n", pad));
            }
            else
            {
                out.push_str(&format!("{}}} else {{\n", pad));
                for b in els
                {
                    emit_stmt(b, ctx, ind + 1, out);
                }
                out.push_str(&format!("{}}}\n", pad));
            }
        },
        SirStmt::While { cond, body } =>
        {
            let c = emit(cond, ctx);
            out.push_str(&format!("{}while {} {{\n", pad, c.code));
            for b in body
            {
                emit_stmt(b, ctx, ind + 1, out);
            }
            out.push_str(&format!("{}}}\n", pad));
        },
        SirStmt::Return(e) =>
        {
            let v = emit(e, ctx);
            match v.ty
            {
                Ty::Array => out.push_str(&format!("{}return {};\n", pad, owned_array(v))),
                Ty::ComplexArray | Ty::MatrixVal =>
                {
                    out.push_str(&format!("{}return {};\n", pad, v.code))
                },
                _ => out.push_str(&format!("{}return {};\n", pad, scalar_of(v))),
            }
        },
    }
}

/// Coerce a fragment to an `f64` scalar (wrapping integer fragments).
fn scalar_of(f: Frag) -> String {
    match f.ty
    {
        Ty::Int => format!("(({}) as f64)", f.code),
        _ => f.code,
    }
}

/// A `&[f64]` view of an array fragment (borrow non-parameter Vecs).
fn slice_of(f: &Frag) -> String {
    if f.borrowed
    {
        f.code.clone()
    }
    else
    {
        format!("&({})", f.code)
    }
}

/// An owned `Vec<f64>` from an array fragment (clone borrowed parameters).
fn owned_array(f: Frag) -> String {
    if f.borrowed
    {
        format!("({}).to_vec()", f.code)
    }
    else
    {
        f.code
    }
}

fn op_sym(op: Op) -> &'static str {
    match op
    {
        Op::Add => "+",
        Op::Sub => "-",
        Op::Mul => "*",
        Op::Div => "/",
    }
}

fn emit(e: &SirExpr, ctx: &Ctx) -> Frag {
    match e
    {
        SirExpr::ScalarLit(v) => Frag {
            code: fmt_f64(*v),
            ty: Ty::Scalar,
            borrowed: false,
        },
        SirExpr::IntLit(v) => Frag {
            code: format!("{}usize", v),
            ty: Ty::Int,
            borrowed: false,
        },
        SirExpr::Var { name, ty } => Frag {
            code: name.clone(),
            ty: *ty,
            borrowed: (*ty == Ty::Array || *ty == Ty::Matrix) && ctx.params.contains(name),
        },
        SirExpr::ScalarBin { op, l, r } =>
        {
            let l = emit(l, ctx);
            let r = emit(r, ctx);
            Frag {
                code: format!("({} {} {})", scalar_of(l), op_sym(*op), scalar_of(r)),
                ty: Ty::Scalar,
                borrowed: false,
            }
        },
        SirExpr::IntBin { op, l, r } =>
        {
            let l = emit(l, ctx);
            let r = emit(r, ctx);
            Frag {
                code: format!("({} {} {})", l.code, op_sym(*op), r.code),
                ty: Ty::Int,
                borrowed: false,
            }
        },
        SirExpr::ScalarNeg(x) =>
        {
            let x = emit(x, ctx);
            Frag {
                code: format!("(-({}))", scalar_of(x)),
                ty: Ty::Scalar,
                borrowed: false,
            }
        },
        SirExpr::ScalarPow { base, exp } =>
        {
            let b = emit(base, ctx);
            let bcode = scalar_of(b);
            // Integer exponent -> powi (exact, faster); else powf.
            let code = if let SirExpr::ScalarLit(v) = exp.as_ref()
            {
                if v.fract() == 0.0 && v.abs() < 1e9
                {
                    format!("({}).powi({})", bcode, *v as i64)
                }
                else
                {
                    format!("({}).powf({})", bcode, fmt_f64(*v))
                }
            }
            else
            {
                let e = emit(exp, ctx);
                format!("({}).powf({})", bcode, scalar_of(e))
            };
            Frag {
                code,
                ty: Ty::Scalar,
                borrowed: false,
            }
        },
        SirExpr::Index { base, idx } =>
        {
            let b = emit(base, ctx);
            let i = emit(idx, ctx);
            Frag {
                code: format!("{}[{}]", b.code, i.code),
                ty: Ty::Scalar,
                borrowed: false,
            }
        },
        SirExpr::ScalarUnaryFn { func, arg } =>
        {
            let a = emit(arg, ctx);
            Frag {
                code: format!("({}).{}()", scalar_of(a), func.rust_method()),
                ty: Ty::Scalar,
                borrowed: false,
            }
        },
        SirExpr::Len(a) =>
        {
            let a = emit(a, ctx);
            Frag {
                code: format!("{}.len()", slice_of(&a)),
                ty: Ty::Int,
                borrowed: false,
            }
        },
        SirExpr::Sum(a) =>
        {
            let a = emit(a, ctx);
            Frag {
                code: format!("np::sum({})", slice_of(&a)),
                ty: Ty::Scalar,
                borrowed: false,
            }
        },
        SirExpr::Dot(a, b) =>
        {
            let a = emit(a, ctx);
            let b = emit(b, ctx);
            Frag {
                code: format!("np::dot({}, {})", slice_of(&a), slice_of(&b)),
                ty: Ty::Scalar,
                borrowed: false,
            }
        },
        SirExpr::Zeros(n) =>
        {
            let n = emit(n, ctx);
            Frag {
                code: format!("np::zeros({})", n.code),
                ty: Ty::Array,
                borrowed: false,
            }
        },
        SirExpr::Ones(n) =>
        {
            let n = emit(n, ctx);
            Frag {
                code: format!("np::ones({})", n.code),
                ty: Ty::Array,
                borrowed: false,
            }
        },
        SirExpr::EwBin { op, l, r } =>
        {
            let l = emit(l, ctx);
            let r = emit(r, ctx);
            Frag {
                code: format!(
                    "np::ew2({}, {}, |x, y| x {} y)",
                    slice_of(&l),
                    slice_of(&r),
                    op_sym(*op)
                ),
                ty: Ty::Array,
                borrowed: false,
            }
        },
        SirExpr::ScalarBroadcast {
            op,
            scalar,
            arr,
            arr_is_left,
        } =>
        {
            let s = emit(scalar, ctx);
            let a = emit(arr, ctx);
            let scode = scalar_of(s);
            let body = if *arr_is_left
            {
                format!("|x| x {} {}", op_sym(*op), scode)
            }
            else
            {
                format!("|x| {} {} x", scode, op_sym(*op))
            };
            Frag {
                code: format!("np::map1({}, {})", slice_of(&a), body),
                ty: Ty::Array,
                borrowed: false,
            }
        },
        SirExpr::ArrayUnaryFn { func, arg } =>
        {
            let a = emit(arg, ctx);
            Frag {
                code: format!("np::map1({}, |x| x.{}())", slice_of(&a), func.rust_method()),
                ty: Ty::Array,
                borrowed: false,
            }
        },
        SirExpr::Cmp { op, l, r } =>
        {
            let l = emit(l, ctx);
            let r = emit(r, ctx);
            Frag {
                code: format!("({} {} {})", scalar_of(l), op.rust_sym(), scalar_of(r)),
                ty: Ty::Bool,
                borrowed: false,
            }
        },
        SirExpr::LinSolve { a, b } =>
        {
            // Route to the verified LU solver; A is flat row-major, n = b.len().
            let a = emit(a, ctx);
            let b = emit(b, ctx);
            let code = format!(
                "{{ let __b: &[f64] = {bs}; let __n = __b.len(); \
                 scirust_solvers::linalg::solve(\
                 scirust_solvers::Matrix::from_row_major(__n, __n, ({amat}).to_vec()), __b)\
                 .expect(\"scirust-transpiler: linear solve failed (singular matrix?)\") }}",
                bs = slice_of(&b),
                amat = a.code,
            );
            Frag {
                code,
                ty: Ty::Array,
                borrowed: false,
            }
        },
        SirExpr::Det(a) =>
        {
            // Route to the verified LU-based determinant; A is flat row-major,
            // n = isqrt(A.len()).
            let a = emit(a, ctx);
            let code = format!(
                "{{ let __a: &[f64] = {amat}; let __n = (__a.len() as f64).sqrt() as usize; \
                 scirust_solvers::Matrix::from_row_major(__n, __n, __a.to_vec())\
                 .determinant()\
                 .expect(\"scirust-transpiler: determinant failed\") }}",
                amat = slice_of(&a),
            );
            Frag {
                code,
                ty: Ty::Scalar,
                borrowed: false,
            }
        },
        SirExpr::Eigvalsh(a) =>
        {
            // Route to the verified symmetric eigensolver; eigenvalues come back
            // sorted ascending (matching numpy.linalg.eigvalsh). A is flat
            // row-major, n = isqrt(A.len()).
            let a = emit(a, ctx);
            let code = format!(
                "{{ let __a: &[f64] = {amat}; let __n = (__a.len() as f64).sqrt() as usize; \
                 scirust_solvers::linalg::eigen_symmetric(\
                 &scirust_solvers::Matrix::from_row_major(__n, __n, __a.to_vec()))\
                 .expect(\"scirust-transpiler: symmetric eigendecomposition failed\")\
                 .eigenvalues }}",
                amat = slice_of(&a),
            );
            Frag {
                code,
                ty: Ty::Array,
                borrowed: false,
            }
        },
        SirExpr::Inv(a) =>
        {
            // Matrix inverse routed to the verified kernel; A is flat row-major,
            // n = isqrt(A.len()). Returns a `Matrix` value (carries its shape).
            let a = emit(a, ctx);
            let code = format!(
                "{{ let __a: &[f64] = {amat}; let __n = (__a.len() as f64).sqrt() as usize; \
                 scirust_solvers::Matrix::from_row_major(__n, __n, __a.to_vec())\
                 .inverse().expect(\"scirust-transpiler: matrix inverse failed (singular?)\") }}",
                amat = slice_of(&a),
            );
            Frag {
                code,
                ty: Ty::MatrixVal,
                borrowed: false,
            }
        },
        SirExpr::Matmul { a, b } =>
        {
            // Matrix-matrix product routed to the verified kernel; accepts flat
            // matrix params and produced `Matrix` values alike.
            let a = emit(a, ctx);
            let b = emit(b, ctx);
            let code = format!(
                "{}.matmul(&{}).expect(\"scirust-transpiler: matmul dimension mismatch\")",
                as_matrix(&a),
                as_matrix(&b),
            );
            Frag {
                code,
                ty: Ty::MatrixVal,
                borrowed: false,
            }
        },
        SirExpr::Transpose(a) =>
        {
            let a = emit(a, ctx);
            Frag {
                code: format!("{}.transpose()", as_matrix(&a)),
                ty: Ty::MatrixVal,
                borrowed: false,
            }
        },
        SirExpr::Diag(a) =>
        {
            // 1-D array -> square diagonal matrix (for SVD reconstruction).
            let a = emit(a, ctx);
            let code = format!(
                "{{ let __d: &[f64] = {ds}; scirust_solvers::Matrix::from_fn(__d.len(), __d.len(), \
                 |__i, __j| if __i == __j {{ __d[__i] }} else {{ 0.0 }}) }}",
                ds = slice_of(&a),
            );
            Frag {
                code,
                ty: Ty::MatrixVal,
                borrowed: false,
            }
        },
        SirExpr::Matvec { a, b } =>
        {
            // Matrix-vector product routed to the verified kernel. A is flat
            // row-major with cols = b.len(), rows = A.len() / cols.
            let a = emit(a, ctx);
            let b = emit(b, ctx);
            let code = format!(
                "{{ let __b: &[f64] = {bs}; let __c = __b.len(); let __r = {amat}.len() / __c; \
                 scirust_solvers::Matrix::from_row_major(__r, __c, ({amat}).to_vec())\
                 .matvec(__b).expect(\"scirust-transpiler: matvec dimension mismatch\") }}",
                bs = slice_of(&b),
                amat = a.code,
            );
            Frag {
                code,
                ty: Ty::Array,
                borrowed: false,
            }
        },
        SirExpr::Fft(a) =>
        {
            // Full complex DFT of a real signal (all N bins, matching
            // numpy.fft.fft): lift to complex, run the verified in-place FFT.
            let a = emit(a, ctx);
            let code = format!(
                "{{ let mut __b: Vec<scirust_signal::complex::Complex> = {}.iter()\
                 .map(|&__v| scirust_signal::complex::Complex::new(__v, 0.0)).collect(); \
                 scirust_signal::fft::fft(&mut __b); __b }}",
                a.code,
            );
            Frag {
                code,
                ty: Ty::ComplexArray,
                borrowed: false,
            }
        },
        SirExpr::Rfft(a) =>
        {
            // Real FFT (positive-frequency half spectrum).
            let a = emit(a, ctx);
            Frag {
                code: format!("scirust_signal::fft::fft_real({})", slice_of(&a)),
                ty: Ty::ComplexArray,
                borrowed: false,
            }
        },
        SirExpr::Ifft(a) =>
        {
            // Inverse DFT (1/N normalised) of a complex spectrum, in place.
            let a = emit(a, ctx);
            let code = format!(
                "{{ let mut __ib: Vec<scirust_signal::complex::Complex> = {}; \
                 scirust_signal::fft::ifft(&mut __ib); __ib }}",
                a.code,
            );
            Frag {
                code,
                ty: Ty::ComplexArray,
                borrowed: false,
            }
        },
        SirExpr::ComplexAbs(a) =>
        {
            // |z| over a complex array -> real magnitude array.
            let a = emit(a, ctx);
            Frag {
                code: format!("{}.iter().map(|c| c.mag()).collect::<Vec<f64>>()", a.code),
                ty: Ty::Array,
                borrowed: false,
            }
        },
    }
}

/// Emit a Rust tuple expression for a multi-output kernel.
fn emit_tuple(t: &TupleExpr, ctx: &Ctx) -> String {
    match t
    {
        TupleExpr::Svd(a) =>
        {
            // Thin SVD via the verified kernel; `Vh = Vᵀ` matches numpy's third
            // return. On a square matrix, thin == full, so the shapes line up
            // with `numpy.linalg.svd(A)`.
            let a = emit(a, ctx);
            format!(
                "{{ let __svd = scirust_solvers::linalg::svd(&{m})\
                 .expect(\"scirust-transpiler: SVD failed\"); \
                 (__svd.u, __svd.s, __svd.v.transpose()) }}",
                m = as_matrix(&a),
            )
        },
    }
}

/// Produce a `scirust_solvers::Matrix` value from a matrix-typed fragment —
/// either a flat `&[f64]` parameter (assumed square) or an already-built
/// `Matrix` value (which carries its own shape).
fn as_matrix(f: &Frag) -> String {
    match f.ty
    {
        Ty::MatrixVal => f.code.clone(),
        Ty::Matrix =>
        {
            let slice = if f.borrowed
            {
                f.code.clone()
            }
            else
            {
                format!("&({})", f.code)
            };
            format!(
                "{{ let __m: &[f64] = {slice}; let __n = (__m.len() as f64).sqrt() as usize; \
                 scirust_solvers::Matrix::from_row_major(__n, __n, __m.to_vec()) }}"
            )
        },
        _ => f.code.clone(),
    }
}

/// Format an `f64` as a round-trippable Rust literal.
fn fmt_f64(v: f64) -> String {
    if v == 0.0
    {
        // Normalise -0.0 and 0.0 to a clean literal.
        return "0.0f64".to_string();
    }
    format!("{:?}f64", v)
}
