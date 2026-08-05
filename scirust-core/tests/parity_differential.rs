// tests/parity_differential.rs
//
// SciRust Tensor Parity Profile 1.0 — harness différentiel Rust-only.
//
// Charge les fixtures générées HORS-LIGNE contre le baseline figé
// (PyTorch 2.13.0, commit cf30153c4c131c8164ee7798e5022d810682e2cb —
// voir docs/tensor-parity/provenance/generate_fixtures.py), exécute les
// noyaux de parité de scirust_core::tensor::parity, compare les sorties
// selon la tolérance (atol + rtol*|ref|), et vérifie les gradients
// (gout * dérivée == gx de la fixture).
//
// Aucune dépendance Python/PyTorch en CI : uniquement les fichiers de
// fixture commités.

use scirust_core::error::SciRustError;
use scirust_core::tensor::parity;
use scirust_core::tensor::tensor_nd::TensorND;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;

const FIXTURES: &str = "../tests/parity/fixtures";

#[derive(Debug, Deserialize)]
struct Fixture {
    op: String,
    kind: String,
    dtype: String,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    kind: String,
    shape: Vec<usize>,
    #[serde(default)]
    scalar: Option<f32>,
    #[serde(default)]
    axis: Option<usize>,
    #[serde(default)]
    start: Option<usize>,
    #[serde(default)]
    end: Option<usize>,
    #[serde(default)]
    dims: Option<Vec<usize>>,
    #[serde(default)]
    eps: Option<f32>,
    #[serde(default)]
    new_shape: Option<Vec<usize>>,
    #[serde(default)]
    bcast_to: Option<Vec<usize>>,
    #[serde(default)]
    out_shape: Option<Vec<usize>>,
    #[serde(default)]
    x: Vec<f32>,
    #[serde(default)]
    a: Vec<f32>,
    #[serde(default)]
    b: Vec<f32>,
    #[serde(default)]
    w: Vec<f32>,
    #[serde(default)]
    target: Vec<f32>,
    #[serde(default)]
    indices: Vec<usize>,
    y: Vec<f32>,
    #[serde(default)]
    gout: Vec<f32>,
    #[serde(default)]
    gx: Vec<f32>,
    #[serde(default)]
    gb: Vec<f32>,
    #[serde(default)]
    gw: Vec<f32>,
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURES)
}

fn load_fixture(family: &str, op: &str) -> Fixture {
    let path = fixtures_dir().join(family).join(format!("{op}.json"));
    let txt =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read fixture {path:?}: {e}"));
    serde_json::from_str(&txt).unwrap_or_else(|e| panic!("invalid fixture {path:?}: {e}"))
}

fn tensor(data: &[f32], shape: &[usize]) -> TensorND {
    TensorND::new(data.to_vec(), shape.to_vec())
}

/// Index linéaire row-major (avec vérification de bounds explicite pour les
/// chemins de test).
fn linear_index(shape: &[usize], coords: &[usize], data: &[f32]) -> usize {
    let mut off = 0usize;
    let mut stride = 1usize;
    for (i, &dim) in shape.iter().enumerate().rev()
    {
        assert!(
            coords[i] < dim,
            "coord {} out of bounds for dim {}",
            coords[i],
            dim
        );
        off += coords[i] * stride;
        stride *= dim;
    }
    assert!(
        off < data.len(),
        "linear index {off} out of range {}",
        data.len()
    );
    off
}

fn close(a: f32, b: f32, atol: f32, rtol: f32) -> bool {
    let diff = (a - b).abs();
    let scale = b.abs();
    diff <= atol + rtol * scale
}

fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: length mismatch {} vs {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate()
    {
        assert!(
            close(*g, *w, atol, rtol),
            "{name}[{i}]: got {g}, want {w} (atol {atol}, rtol {rtol})"
        );
    }
}

fn run_unary(
    family: &str,
    op: &str,
    atol: f32,
    rtol: f32,
    f: impl Fn(&TensorND) -> Result<TensorND, SciRustError>,
) {
    run_kind("unary", family, op, atol, rtol, f)
}

fn run_kind(
    kind: &str,
    family: &str,
    op: &str,
    atol: f32,
    rtol: f32,
    f: impl Fn(&TensorND) -> Result<TensorND, SciRustError>,
) {
    let fx = load_fixture(family, op);
    assert_eq!(fx.kind, kind, "{op}: unexpected fixture kind {}", fx.kind);
    assert_eq!(
        fx.dtype, "f32",
        "{op}: unexpected fixture dtype {}",
        fx.dtype
    );
    assert_eq!(fx.op, op, "fixture file/op name mismatch");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        assert_eq!(c.kind, kind, "{op} c{ci}: case kind drift");
        assert_eq!(
            c.scalar.is_some(),
            kind == "unary_scalar",
            "{op} c{ci}: scalar drift"
        );
        let t = tensor(&c.x, &c.shape);
        let out = f(&t).unwrap_or_else(|e| panic!("{op} case {ci}: {e}"));
        assert_close(
            &format!("{op} forward c{ci}"),
            out.data.as_ref(),
            &c.y,
            atol,
            rtol,
        );
        let g_expected: Vec<f32> = c
            .gout
            .iter()
            .zip(c.x.iter())
            .map(|(g, &x)| g * parity::d_unary(op, x).unwrap())
            .collect();
        assert_close(&format!("{op} grad c{ci}"), &g_expected, &c.gx, atol, rtol);
    }
}

type UnaryOp = fn(&TensorND) -> Result<TensorND, SciRustError>;

#[test]
fn parity_elementwise_unary() {
    let table: Vec<(&str, UnaryOp)> = vec![
        ("neg", parity::neg),
        ("reciprocal", parity::reciprocal),
        ("exp", parity::exp),
        ("log", parity::log),
        ("log10", parity::log10),
        ("sqrt", parity::sqrt),
        ("sin", parity::sin),
        ("cos", parity::cos),
        ("tan", parity::tan),
        ("asin", parity::asin),
        ("acos", parity::acos),
        ("atan", parity::atan),
        ("sinh", parity::sinh),
        ("cosh", parity::cosh),
        ("tanh", parity::tanh),
        ("sigmoid", parity::sigmoid),
        ("relu", parity::relu),
        ("silu", parity::silu),
        ("gelu", parity::gelu),
    ];
    for (op, f) in table
    {
        run_unary("elementwise", op, 1e-5, 1e-5, f);
    }
}

#[test]
fn parity_elementwise_pow_scalar() {
    run_kind("unary_scalar", "elementwise", "pow", 1e-5, 1e-5, |t| {
        parity::pow_scalar(t, 2.0)
    });
}

#[test]
fn parity_elementwise_binary() {
    for (op, f) in [
        (
            "add",
            parity::add as fn(&TensorND, &TensorND) -> Result<TensorND, SciRustError>,
        ),
        ("sub", parity::sub),
        ("mul", parity::mul),
        ("div", parity::div),
        ("atan2", parity::atan2),
    ]
    {
        let fx = load_fixture("elementwise", op);
        assert_eq!(
            fx.kind, "binary",
            "{op}: unexpected fixture kind {}",
            fx.kind
        );
        for (ci, c) in fx.cases.iter().enumerate()
        {
            let a = tensor(&c.x, &c.shape);
            let b = tensor(&c.b, &c.shape);
            let out = f(&a, &b).unwrap_or_else(|e| panic!("{op} case {ci}: {e}"));
            assert_close(
                &format!("{op} forward c{ci}"),
                out.data.as_ref(),
                &c.y,
                1e-5,
                1e-5,
            );
            let gx_expected: Vec<f32> = c
                .gout
                .iter()
                .zip(c.x.iter().zip(c.b.iter()))
                .map(|(g, (&x, &y))| g * parity::d_binary(op, x, y).unwrap().0)
                .collect();
            assert_close(
                &format!("{op} grad-x c{ci}"),
                &gx_expected,
                &c.gx,
                1e-5,
                1e-5,
            );
            let gb_expected: Vec<f32> = c
                .gout
                .iter()
                .zip(c.x.iter().zip(c.b.iter()))
                .map(|(g, (&x, &y))| g * parity::d_binary(op, x, y).unwrap().1)
                .collect();
            assert_close(
                &format!("{op} grad-b c{ci}"),
                &gb_expected,
                &c.gb,
                1e-5,
                1e-5,
            );
        }
    }
}

/// Gradcheck pour softmax/log_softmax (gradients couplés le long de la
/// dernière dimension) :
///   softmax:     gx_i = y_i * (gout_i - Σ_j gout_j·y_j)
///   log_softmax: gx_i = gout_i - exp(ls_i)·Σ_j gout_j
#[test]
fn parity_normalization() {
    for (op, f) in [
        (
            "softmax",
            parity::softmax_last as fn(&TensorND) -> Result<TensorND, SciRustError>,
        ),
        ("log_softmax", parity::log_softmax_last),
    ]
    {
        let fx = load_fixture("normalization", op);
        assert_eq!(
            fx.kind, "unary",
            "{op}: unexpected fixture kind {}",
            fx.kind
        );
        for (ci, c) in fx.cases.iter().enumerate()
        {
            let t = tensor(&c.x, &c.shape);
            let out = f(&t).unwrap_or_else(|e| panic!("{op} case {ci}: {e}"));
            assert_close(
                &format!("{op} forward c{ci}"),
                out.data.as_ref(),
                &c.y,
                1e-5,
                1e-5,
            );
            let axis_len = *c.shape.last().unwrap();
            let mut g_expected = vec![0.0f32; c.x.len()];
            for base in (0..c.x.len()).step_by(axis_len)
            {
                let gout_sum: f32 = (0..axis_len).map(|j| c.gout[base + j]).sum();
                for j in 0..axis_len
                {
                    let idx = base + j;
                    let g = if op == "softmax"
                    {
                        let y = out.data[idx];
                        let dot = (0..axis_len)
                            .map(|k| c.gout[base + k] * out.data[base + k])
                            .sum::<f32>();
                        y * (c.gout[idx] - dot)
                    }
                    else
                    {
                        c.gout[idx] - out.data[idx].exp() * gout_sum
                    };
                    g_expected[idx] = g;
                }
            }
            assert_close(&format!("{op} grad c{ci}"), &g_expected, &c.gx, 1e-5, 1e-5);
        }
    }
}

/// Gradcheck layer_norm / rms_norm (affine, normalized_shape = dernière dim) :
/// compare y, gx, gw, gb contre les fixtures torch 2.13.0 (tol 1e-4).
#[test]
fn parity_normalization_affine() {
    let fx = load_fixture("normalization", "layer_norm");
    assert_eq!(fx.kind, "normalization");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let dims = c.dims.as_ref().expect("layer_norm case must have dims");
        let eps = c.eps.expect("layer_norm case must have eps");
        let x = tensor(&c.x, &c.shape);
        let w = tensor(&c.w, dims);
        let b = tensor(&c.b, dims);
        let out = parity::layer_norm(&x, &w, &b, dims, eps)
            .unwrap_or_else(|e| panic!("layer_norm case {ci}: {e}"));
        assert_close(
            &format!("layer_norm fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
        let gout = tensor(&c.gout, &c.shape);
        let (gx, gw, gb) = parity::d_layer_norm(&gout, &x, &w, dims, eps)
            .unwrap_or_else(|e| panic!("d_layer_norm case {ci}: {e}"));
        assert_close(
            &format!("layer_norm gx c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("layer_norm gw c{ci}"),
            gw.data.as_ref(),
            &c.gw,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("layer_norm gb c{ci}"),
            gb.data.as_ref(),
            &c.gb,
            1e-4,
            1e-4,
        );
    }

    let fx = load_fixture("normalization", "rms_norm");
    assert_eq!(fx.kind, "normalization");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let dims = c.dims.as_ref().expect("rms_norm case must have dims");
        let eps = c.eps.expect("rms_norm case must have eps");
        let x = tensor(&c.x, &c.shape);
        let w = tensor(&c.w, dims);
        let out = parity::rms_norm(&x, &w, dims, eps)
            .unwrap_or_else(|e| panic!("rms_norm case {ci}: {e}"));
        assert_close(
            &format!("rms_norm fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
        let gout = tensor(&c.gout, &c.shape);
        let (gx, gw) = parity::d_rms_norm(&gout, &x, &w, dims, eps)
            .unwrap_or_else(|e| panic!("d_rms_norm case {ci}: {e}"));
        assert_close(
            &format!("rms_norm gx c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("rms_norm gw c{ci}"),
            gw.data.as_ref(),
            &c.gw,
            1e-4,
            1e-4,
        );
    }
}

fn run_reduction(
    op: &str,
    atol: f32,
    rtol: f32,
    f: impl Fn(&TensorND, usize) -> Result<TensorND, SciRustError>,
) {
    let fx = load_fixture("reductions", op);
    assert_eq!(
        fx.kind, "reduction",
        "{op}: unexpected fixture kind {}",
        fx.kind
    );
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let axis = c.axis.expect("reduction case must have axis");
        let t = tensor(&c.x, &c.shape);
        let out = f(&t, axis).unwrap_or_else(|e| panic!("{op} case {ci} axis {axis}: {e}"));
        let out_shape = c.out_shape.as_ref().unwrap();
        assert_eq!(
            out.shape(),
            out_shape,
            "{op} c{ci}: shape {:?} != {:?}",
            out.shape(),
            out_shape
        );
        assert_close(
            &format!("{op} forward c{ci}"),
            out.data.as_ref(),
            &c.y,
            atol,
            rtol,
        );

        let axis_len = c.shape[axis];
        let n_outer = c.y.len();
        let out_shape = out_shape.clone();
        let mut g_expected = vec![0.0f32; c.x.len()];
        let mut coords = vec![0usize; c.shape.len()];
        for outer in 0..n_outer
        {
            let gout = c.gout[outer];
            // décodage row-major de l'index de sortie en coordonnées d'entrée
            let mut rem = outer;
            for k in (0..out_shape.len()).rev()
            {
                let pos = if k < axis { k } else { k + 1 };
                let dim = out_shape[k];
                coords[pos] = rem % dim;
                rem /= dim;
            }
            debug_assert_eq!(rem, 0);
            coords[axis] = 0;
            // ligne le long de l'axe (pour mean/var)
            let mut line = Vec::with_capacity(axis_len);
            for j in 0..axis_len
            {
                coords[axis] = j;
                line.push(c.x[linear_index(c.shape.as_slice(), &coords, &c.x)]);
            }
            let mean = line.iter().sum::<f32>() / axis_len as f32;
            for j in 0..axis_len
            {
                coords[axis] = j;
                let idx = linear_index(c.shape.as_slice(), &coords, &c.x);
                let d = match op
                {
                    "sum" => parity::d_sum(gout),
                    "mean" => parity::d_mean(gout, axis_len as f32),
                    "var" => parity::d_var(c.x[idx], mean, gout, axis_len as f32),
                    _ => panic!("unknown reduction {op}"),
                };
                g_expected[idx] = d;
            }
        }
        assert_close(&format!("{op} grad c{ci}"), &g_expected, &c.gx, atol, rtol);
    }
}

#[test]
fn parity_reductions() {
    run_reduction("sum", 1e-4, 1e-4, parity::sum_axis);
    run_reduction("mean", 1e-4, 1e-4, parity::mean_axis);
    run_reduction("var", 1e-4, 1e-4, parity::var_axis);
}

#[test]
fn parity_error_paths_are_structured() {
    let t = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let e = parity::sum_axis(&t, 5).unwrap_err();
    assert!(matches!(e, SciRustError::AxisOutOfBounds { .. }));
    let e = parity::softmax_last(&tensor(&[1.0], &[])).unwrap_err();
    assert!(matches!(e, SciRustError::RankMismatch { .. }));
    let a = tensor(&[1.0, 2.0], &[2]);
    let b = tensor(&[1.0, 2.0, 3.0], &[3]);
    let e = parity::add(&a, &b).unwrap_err();
    assert!(matches!(e, SciRustError::ShapeMismatch { .. }));
}

#[test]
fn parity_fixtures_manifest_is_valid() {
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(fixtures_dir().join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(
        manifest["pytorch"]["source_commit"],
        "cf30153c4c131c8164ee7798e5022d810682e2cb"
    );
    let version = manifest["pytorch"]["version"].as_str().unwrap();
    assert!(
        version.starts_with("2.13.0"),
        "fixtures were not generated from the frozen baseline: {version}"
    );
    let files = manifest["files"].as_object().unwrap();
    assert!(!files.is_empty(), "fixture manifest must list files");
    for (rel, want) in files
    {
        let p = fixtures_dir().join(rel);
        assert!(p.exists(), "fixture listed in manifest but missing: {p:?}");
        let digest = Sha256::digest(fs::read(&p).unwrap());
        let got = format!("{digest:x}");
        let want = want.as_str().unwrap();
        assert_eq!(
            got, want,
            "fixture {rel}: sha256 drift — regenerate with docs/tensor-parity/provenance/generate_fixtures.py"
        );
    }
}

// ------------------------------------------------------------------ //
//  Famille shape : reshape, transpose, permute, broadcast_to, slice,
//  flatten (kinds "shape"; grads exacts selon torch autograd).
// ------------------------------------------------------------------ //

#[test]
fn parity_shape_family() {
    // (op, tolerance) — forward/grad exacts (reorders/copies), 1e-6 pour les
    // grads accumulés (broadcast/slice).
    let cases: Vec<(&str, f32)> = vec![
        ("reshape", 0.0),
        ("transpose", 0.0),
        ("permute", 0.0),
        ("broadcast_to", 1e-6),
        ("slice", 1e-6),
        ("flatten", 0.0),
    ];
    for (op, tol) in cases
    {
        let fx = load_fixture("shape", op);
        assert_eq!(
            fx.kind, "shape",
            "{op}: unexpected fixture kind {}",
            fx.kind
        );
        for (ci, c) in fx.cases.iter().enumerate()
        {
            let t = tensor(&c.x, &c.shape);
            let out: TensorND = match op
            {
                "reshape" => parity::reshape(&t, c.new_shape.as_ref().unwrap()),
                "transpose" => parity::transpose2(&t, 0, 1),
                "permute" => parity::permute(&t, c.dims.as_ref().unwrap()),
                "broadcast_to" => parity::broadcast_to(&t, c.bcast_to.as_ref().unwrap()),
                "slice" =>
                {
                    parity::slice_axis(&t, c.axis.unwrap(), c.start.unwrap(), c.end.unwrap())
                },
                "flatten" => parity::flatten(&t),
                _ => unreachable!(),
            }
            .unwrap_or_else(|e| panic!("{op} case {ci}: {e}"));
            assert_eq!(
                out.shape(),
                c.out_shape.as_ref().unwrap(),
                "{op} c{ci}: shape {:?} != {:?}",
                out.shape(),
                c.out_shape
            );
            assert_close(
                &format!("{op} forward c{ci}"),
                out.data.as_ref(),
                &c.y,
                tol,
                tol,
            );

            let gout = tensor(&c.gout, c.out_shape.as_ref().unwrap());
            let gx: TensorND = match op
            {
                "reshape" => parity::g_reshape(&gout, &c.shape),
                "transpose" => parity::transpose2(&gout, 0, 1),
                "permute" => parity::g_permute(&gout, c.dims.as_ref().unwrap()),
                "broadcast_to" => parity::g_broadcast(&gout, &c.shape),
                "slice" => parity::g_slice(
                    &gout,
                    &c.shape,
                    c.axis.unwrap(),
                    c.start.unwrap(),
                    c.end.unwrap(),
                ),
                "flatten" => parity::g_reshape(&gout, &c.shape),
                _ => unreachable!(),
            }
            .unwrap_or_else(|e| panic!("{op} grad c{ci}: {e}"));
            assert_close(
                &format!("{op} grad c{ci}"),
                gx.data.as_ref(),
                &c.gx,
                tol,
                tol,
            );
        }
    }
}

// ------------------------------------------------------------------ //
//  Famille linear : matmul 2-D, bmm 3-D, linear+bias
// ------------------------------------------------------------------ //

#[test]
fn parity_linear_matmul() {
    let fx = load_fixture("linear", "matmul");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let a = tensor(&c.a, &c.shape);
        let b_shape = vec![c.shape[1], c.out_shape.as_ref().unwrap()[1]];
        let b = tensor(&c.b, &b_shape);
        let out = parity::matmul2(&a, &b).unwrap_or_else(|e| panic!("matmul c{ci}: {e}"));
        assert_eq!(out.shape(), c.out_shape.as_ref().unwrap());
        assert_close(
            &format!("matmul forward c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
        let gout = tensor(&c.gout, c.out_shape.as_ref().unwrap());
        let (ga, gb) = parity::d_matmul2(&gout, &a, &b).unwrap();
        assert_close(
            &format!("matmul grad-a c{ci}"),
            ga.data.as_ref(),
            &c.gx,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("matmul grad-b c{ci}"),
            gb.data.as_ref(),
            &c.gb,
            1e-4,
            1e-4,
        );
    }
}

#[test]
fn parity_linear_bmm() {
    let fx = load_fixture("linear", "bmm");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let a = tensor(&c.a, &c.shape);
        let b_shape = c
            .out_shape
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, &d)| if i == 1 { c.shape[2] } else { d })
            .collect::<Vec<_>>();
        let b = tensor(&c.b, &b_shape);
        let out = parity::bmm(&a, &b).unwrap_or_else(|e| panic!("bmm c{ci}: {e}"));
        assert_eq!(out.shape(), c.out_shape.as_ref().unwrap());
        assert_close(
            &format!("bmm forward c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
        let gout = tensor(&c.gout, c.out_shape.as_ref().unwrap());
        let (ga, gb) = parity::d_bmm(&gout, &a, &b).unwrap();
        assert_close(
            &format!("bmm grad-a c{ci}"),
            ga.data.as_ref(),
            &c.gx,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("bmm grad-b c{ci}"),
            gb.data.as_ref(),
            &c.gb,
            1e-4,
            1e-4,
        );
    }
}

#[test]
fn parity_linear_linear() {
    let fx = load_fixture("linear", "linear");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let x = tensor(&c.x, &c.shape);
        let w_shape = vec![c.out_shape.as_ref().unwrap()[1], c.shape[1]];
        let w = tensor(&c.w, &w_shape);
        let bias = tensor(&c.b, &[w_shape[0]]);
        let out =
            parity::linear(&x, &w, Some(&bias)).unwrap_or_else(|e| panic!("linear c{ci}: {e}"));
        assert_close(
            &format!("linear forward c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
        let gout = tensor(&c.gout, c.out_shape.as_ref().unwrap());
        let (gx, gw, gb) = parity::d_linear(&gout, &x, &w, true).unwrap();
        assert_close(
            &format!("linear grad-x c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("linear grad-w c{ci}"),
            gw.data.as_ref(),
            &c.gw,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("linear grad-b c{ci}"),
            gb.data.as_ref(),
            &c.gb,
            1e-4,
            1e-4,
        );
    }
}

// ------------------------------------------------------------------ //
//  Famille loss : mse_loss (mean), cross_entropy (mean) — scalaires
// ------------------------------------------------------------------ //

#[test]
fn parity_loss_mse() {
    let fx = load_fixture("loss", "mse_loss");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let pred = tensor(&c.x, &c.shape);
        let target = tensor(&c.target, &c.shape);
        let out = parity::mse_loss_mean(&pred, &target).unwrap();
        assert_eq!(out.shape(), &[] as &[usize], "mse c{ci}: must be scalar");
        assert_close(
            &format!("mse forward c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-5,
            1e-5,
        );
        let gout = c.gout.first().copied().unwrap_or(1.0);
        let gx = parity::d_mse_loss_mean(&pred, &target, gout).unwrap();
        assert_close(
            &format!("mse grad c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            1e-5,
            1e-5,
        );
    }
}

#[test]
fn parity_loss_cross_entropy() {
    let fx = load_fixture("loss", "cross_entropy");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let logits = tensor(&c.x, &c.shape);
        let out = parity::cross_entropy_mean(&logits, &c.indices).unwrap();
        assert_eq!(out.shape(), &[] as &[usize], "ce c{ci}: must be scalar");
        assert_close(
            &format!("ce forward c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-5,
            1e-5,
        );
        let gout = c.gout.first().copied().unwrap_or(1.0);
        let gx = parity::d_cross_entropy_mean(&logits, &c.indices, gout).unwrap();
        assert_close(
            &format!("ce grad c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            1e-5,
            1e-5,
        );
    }
}
