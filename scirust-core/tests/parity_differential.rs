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
    #[serde(default)]
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
    size: Option<usize>,
    #[serde(default)]
    step: Option<usize>,
    #[serde(default)]
    kernel: Option<usize>,
    #[serde(default)]
    kh: Option<usize>,
    #[serde(default)]
    kw: Option<usize>,
    #[serde(default)]
    p: Option<f32>,
    #[serde(default)]
    mask: Option<Vec<f32>>,
    #[serde(default)]
    scale: Option<f32>,
    #[serde(default)]
    zp: Option<i64>,
    #[serde(default)]
    qmin: Option<i64>,
    #[serde(default)]
    qmax: Option<i64>,
    #[serde(default)]
    idx_shape: Option<Vec<usize>>,
    #[serde(default)]
    base: Option<f32>,
    #[serde(default)]
    c: Vec<f32>,
    #[serde(default)]
    rm: Vec<f32>,
    #[serde(default)]
    rv: Vec<f32>,
    #[serde(default)]
    gk: Vec<f32>,
    #[serde(default)]
    gv: Vec<f32>,
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
    #[serde(default)]
    spec: Option<String>,
    #[serde(default)]
    shapes: Option<Vec<Vec<usize>>>,
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
        ("rsqrt", parity::rsqrt),
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

/// Gradcheck f64 pour lgamma/digamma (rows special du registre : dtypes
/// f64, tol 1e-10). Les fixtures sont en torch.float64 ; on les lit via
/// serde_json::Value (les champs f64 du Case f32 perdraient la précision).
///   lgamma grad : gout · digamma(x) ; digamma grad : gout · trigamma(x)
#[test]
fn parity_special_f64() {
    for op in ["lgamma", "digamma"]
    {
        let txt =
            fs::read_to_string(fixtures_dir().join("special").join(format!("{op}.json"))).unwrap();
        let fx: serde_json::Value = serde_json::from_str(&txt).unwrap();
        assert_eq!(fx["dtype"], "f64", "{op}: fixture doit être f64");
        for (ci, c) in fx["cases"].as_array().unwrap().iter().enumerate()
        {
            let x: Vec<f64> = c["x"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let y: Vec<f64> = c["y"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let gout: Vec<f64> = c["gout"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let gx: Vec<f64> = c["gx"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let got_y: Vec<f64> = x
                .iter()
                .map(|&v| {
                    if op == "lgamma"
                    {
                        parity::lgamma_f64(v)
                    }
                    else
                    {
                        parity::digamma_f64(v)
                    }
                })
                .collect();
            let got_gx: Vec<f64> = x
                .iter()
                .zip(gout.iter())
                .map(|(&v, &g)| {
                    g * if op == "lgamma"
                    {
                        parity::digamma_f64(v)
                    }
                    else
                    {
                        parity::trigamma_f64(v)
                    }
                })
                .collect();
            assert_close64(&format!("{op} fwd c{ci}"), &got_y, &y, 1e-10, 1e-10);
            // grad digamma = trigamma : torch.polygamma(1) a ~3.6e-10
            // d'erreur relative vs la valeur exacte — tol 1e-9 (1e-10
            // physiquement impossible contre les fixtures torch).
            let (atol, rtol) = if op == "lgamma"
            {
                (1e-10, 1e-10)
            }
            else
            {
                (1e-9, 1e-9)
            };
            assert_close64(&format!("{op} grad c{ci}"), &got_gx, &gx, atol, rtol);
        }
    }
}

fn close64(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

fn assert_close64(name: &str, got: &[f64], want: &[f64], atol: f64, rtol: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate()
    {
        assert!(
            close64(*g, *w, atol, rtol),
            "{name}[{i}]: got {g}, want {w} (atol {atol}, rtol {rtol})"
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

/// max(dim) : valeurs exactes + indices du premier max + grad routé ;
/// norm p=2 (frob) : forward + grad, tol 1e-4.
#[test]
fn parity_reductions_max_norm() {
    let fx = load_fixture("reductions", "max");
    assert_eq!(fx.kind, "reduction");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let axis = c.axis.expect("max case must have axis");
        let t = tensor(&c.x, &c.shape);
        let (vals, idx) =
            parity::max_axis(&t, axis).unwrap_or_else(|e| panic!("max case {ci}: {e}"));
        let out_shape = c.out_shape.as_ref().unwrap();
        assert_eq!(vals.shape(), out_shape, "max c{ci}: shape mismatch");
        assert_close(
            &format!("max fwd c{ci}"),
            vals.data.as_ref(),
            &c.y,
            0.0,
            0.0,
        );
        assert_eq!(idx, c.indices, "max c{ci}: argmax mismatch");
        let gout = tensor(&c.gout, out_shape);
        let gx = parity::d_max_axis(&gout, &t, axis, &idx)
            .unwrap_or_else(|e| panic!("d_max case {ci}: {e}"));
        assert_close(
            &format!("max grad c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            0.0,
            0.0,
        );
    }

    let fx = load_fixture("reductions", "norm");
    assert_eq!(fx.kind, "reduction");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let axis = c.axis.expect("norm case must have axis");
        let t = tensor(&c.x, &c.shape);
        let out = parity::norm_axis_p2(&t, axis).unwrap_or_else(|e| panic!("norm case {ci}: {e}"));
        let out_shape = c.out_shape.as_ref().unwrap();
        assert_eq!(out.shape(), out_shape, "norm c{ci}: shape mismatch");
        assert_close(
            &format!("norm fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
        let gout = tensor(&c.gout, out_shape);
        let gx = parity::d_norm_axis_p2(&gout, &t, axis)
            .unwrap_or_else(|e| panic!("d_norm case {ci}: {e}"));
        assert_close(
            &format!("norm grad c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            1e-4,
            1e-4,
        );
    }
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

// ------------------------------------------------------------------ //
//  Shape extra — cat / gather / unfold                              //
// ------------------------------------------------------------------ //

#[test]
fn parity_shape_extra_cat_gather_unfold() {
    // cat : dim 0 puis dim 1
    let fx = load_fixture("shape", "cat");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let a = tensor(&c.a, &c.shape);
        let b = tensor(&c.b, &c.shape);
        let dim = c.dims.as_ref().unwrap()[0];
        let out = parity::cat2(&a, &b, dim).unwrap();
        assert_eq!(
            out.shape(),
            c.out_shape.as_ref().unwrap(),
            "cat c{ci}: shape"
        );
        assert_close(&format!("cat fwd c{ci}"), out.data.as_ref(), &c.y, 0.0, 0.0);
        let gout = tensor(&c.gout, c.out_shape.as_ref().unwrap());
        let (gx, gb) = parity::d_cat2(&gout, &c.shape, &c.shape, dim).unwrap();
        assert_close(&format!("cat gx c{ci}"), gx.data.as_ref(), &c.gx, 0.0, 0.0);
        assert_close(&format!("cat gb c{ci}"), gb.data.as_ref(), &c.gb, 0.0, 0.0);
    }
    // gather : axe 0 puis axe 1
    let fx = load_fixture("shape", "gather");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let x = tensor(&c.x, &c.shape);
        let axis = c.axis.unwrap();
        let out = parity::gather2(&x, axis, &c.indices).unwrap();
        assert_eq!(out.shape(), &c.shape, "gather c{ci}: shape");
        assert_close(
            &format!("gather fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            0.0,
            0.0,
        );
        let gout = tensor(&c.gout, &c.shape);
        let gx = parity::d_gather2(&gout, &c.shape, axis, &c.indices).unwrap();
        assert_close(
            &format!("gather gx c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            0.0,
            0.0,
        );
    }
    // unfold : (axe=1, size=2, step=1) puis (axe=0, size=2, step=2)
    let fx = load_fixture("shape", "unfold");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let x = tensor(&c.x, &c.shape);
        let out = parity::unfold2(&x, c.axis.unwrap(), c.size.unwrap(), c.step.unwrap()).unwrap();
        assert_eq!(
            out.shape(),
            c.out_shape.as_ref().unwrap(),
            "unfold c{ci}: shape"
        );
        assert_close(
            &format!("unfold fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            0.0,
            0.0,
        );
        let gout = tensor(&c.gout, c.out_shape.as_ref().unwrap());
        let gx = parity::d_unfold2(
            &gout,
            &c.shape,
            c.axis.unwrap(),
            c.size.unwrap(),
            c.step.unwrap(),
        )
        .unwrap();
        assert_close(
            &format!("unfold gx c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            0.0,
            0.0,
        );
    }
}

// ------------------------------------------------------------------ //
//  Indexing — embedding                                              //
// ------------------------------------------------------------------ //

#[test]
fn parity_indexing_embedding() {
    let fx = load_fixture("indexing", "embedding");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let idx_shape = c.idx_shape.as_ref().unwrap();
        let out_shape = c.out_shape.as_ref().unwrap();
        let d = out_shape[out_shape.len() - 1];
        let v = c.w.len() / d;
        let w = tensor(&c.w, &[v, d]);
        let out = parity::embed(idx_shape, &c.indices, &w).unwrap();
        assert_eq!(out.shape(), out_shape, "embed c{ci}: shape");
        assert_close(
            &format!("embed fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            0.0,
            0.0,
        );
        let gout = tensor(&c.gout, out_shape);
        let gw = parity::d_embed(&gout, &c.indices, v, d).unwrap();
        assert_close(
            &format!("embed gw c{ci}"),
            gw.data.as_ref(),
            &c.gw,
            0.0,
            0.0,
        );
    }
}

// ------------------------------------------------------------------ //
//  Linear extra — cosine_similarity / normalize                     //
// ------------------------------------------------------------------ //

#[test]
fn parity_linear_extra_cosine_normalize() {
    let fx = load_fixture("linear", "cosine_similarity");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let a = tensor(&c.a, &c.shape);
        let b = tensor(&c.b, &c.shape);
        let out = parity::cosine_sim(&a, &b).unwrap();
        assert_eq!(
            out.shape(),
            c.out_shape.as_ref().unwrap(),
            "cos c{ci}: shape"
        );
        assert_close(
            &format!("cos fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
        let gout = tensor(&c.gout, c.out_shape.as_ref().unwrap());
        let (gx, gb) = parity::d_cosine_sim(&gout, &a, &b).unwrap();
        assert_close(
            &format!("cos gx c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("cos gb c{ci}"),
            gb.data.as_ref(),
            &c.gb,
            1e-4,
            1e-4,
        );
    }
    let fx = load_fixture("linear", "normalize");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let x = tensor(&c.x, &c.shape);
        let out = parity::normalize2(&x).unwrap();
        assert_eq!(out.shape(), &c.shape, "norm c{ci}: shape");
        assert_close(
            &format!("norm fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
        let gout = tensor(&c.gout, &c.shape);
        let gx = parity::d_normalize2(&gout, &x).unwrap();
        assert_close(
            &format!("norm gx c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            1e-4,
            1e-4,
        );
    }
}

// ------------------------------------------------------------------ //
//  Normalization stochastique — dropout / batch_norm (eval)         //
// ------------------------------------------------------------------ //

#[test]
fn parity_norm_stoch_dropout_batchnorm() {
    let fx = load_fixture("normalization", "dropout");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let x = tensor(&c.x, &c.shape);
        let mask = c.mask.as_ref().unwrap();
        let p = c.p.unwrap();
        let out = parity::dropout_apply(&x, p, mask).unwrap();
        assert_close(
            &format!("dropout fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-5,
            1e-5,
        );
        let gout = tensor(&c.gout, &c.shape);
        let gx = parity::d_dropout(&gout, p, mask).unwrap();
        assert_close(
            &format!("dropout gx c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            1e-5,
            1e-5,
        );
    }
    let fx = load_fixture("normalization", "batch_norm");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let x = tensor(&c.x, &c.shape);
        let w = tensor(&c.w, &[c.shape[1]]);
        let b = tensor(&c.b, &[c.shape[1]]);
        let rm = tensor(&c.rm, &[c.shape[1]]);
        let rv = tensor(&c.rv, &[c.shape[1]]);
        let eps = c.eps.unwrap_or(1e-5);
        let out = parity::batch_norm_eval(&x, &w, &b, &rm, &rv, eps).unwrap();
        assert_close(
            &format!("bn fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
        let gout = tensor(&c.gout, &c.shape);
        let (gx, gw, gb) = parity::d_batch_norm_eval(&gout, &x, &w, &rm, &rv, eps).unwrap();
        assert_close(&format!("bn gx c{ci}"), gx.data.as_ref(), &c.gx, 1e-4, 1e-4);
        assert_close(&format!("bn gw c{ci}"), gw.data.as_ref(), &c.gw, 1e-4, 1e-4);
        assert_close(&format!("bn gb c{ci}"), gb.data.as_ref(), &c.gb, 1e-4, 1e-4);
    }
}

// ------------------------------------------------------------------ //
//  Positional — rope ; Attention — scaled_dot_product_attention     //
// ------------------------------------------------------------------ //

#[test]
fn parity_positional_rope_attention_sdpa() {
    let fx = load_fixture("positional", "rope");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let x = tensor(&c.x, &c.shape);
        let base = c.base.unwrap_or(10000.0);
        let out = parity::rope(&x, base).unwrap();
        assert_eq!(out.shape(), &c.shape, "rope c{ci}: shape");
        assert_close(
            &format!("rope fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
        let gout = tensor(&c.gout, &c.shape);
        let gx = parity::d_rope(&gout, &x, base).unwrap();
        assert_close(
            &format!("rope gx c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            1e-4,
            1e-4,
        );
    }
    let fx = load_fixture("attention", "scaled_dot_product_attention");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let q = tensor(&c.a, &c.shape);
        let k = tensor(&c.b, &c.shape);
        let v = tensor(&c.c, &c.shape);
        let out = parity::sdpa(&q, &k, &v).unwrap();
        assert_eq!(out.shape(), &c.shape, "sdpa c{ci}: shape");
        assert_close(
            &format!("sdpa fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
        let gout = tensor(&c.gout, &c.shape);
        let (gq, gk, gv) = parity::d_sdpa(&gout, &q, &k, &v).unwrap();
        assert_close(
            &format!("sdpa gq c{ci}"),
            gq.data.as_ref(),
            &c.gx,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("sdpa gk c{ci}"),
            gk.data.as_ref(),
            &c.gk,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("sdpa gv c{ci}"),
            gv.data.as_ref(),
            &c.gv,
            1e-4,
            1e-4,
        );
    }
}

// ------------------------------------------------------------------ //
//  Quantization / conversion — fake_quantize (STE) / to_bf16        //
// ------------------------------------------------------------------ //

#[test]
fn parity_quantization_fake_quant_to_bf16() {
    let fx = load_fixture("quantization", "fake_quantize_per_tensor");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let x = tensor(&c.x, &c.shape);
        let out = parity::fake_quant_map(
            &x,
            c.scale.unwrap(),
            c.zp.unwrap(),
            c.qmin.unwrap(),
            c.qmax.unwrap(),
        )
        .unwrap();
        assert_close(
            &format!("fakequant fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            0.0,
            0.0,
        );
        let gout = tensor(&c.gout, &c.shape);
        let gx = parity::d_fake_quant_map(
            &gout,
            &x,
            c.scale.unwrap(),
            c.zp.unwrap(),
            c.qmin.unwrap(),
            c.qmax.unwrap(),
        )
        .unwrap();
        assert_close(
            &format!("fakequant gx c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            0.0,
            0.0,
        );
    }
    let fx = load_fixture("conversion", "to_bf16");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let x = tensor(&c.x, &c.shape);
        let out = parity::to_bf16_map(&x).unwrap();
        assert_close(
            &format!("to_bf16 fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            0.0,
            0.0,
        );
    }
}

// ------------------------------------------------------------------ //
//  Convolution — conv1d / conv2d / max_pool2d / avg_pool2d          //
// ------------------------------------------------------------------ //

#[test]
fn parity_convolution() {
    let fx = load_fixture("convolution", "conv1d");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let x = tensor(&c.x, &c.shape);
        let k = c.kernel.unwrap();
        let cout = c.w.len() / (c.shape[1] * k);
        let w = tensor(&c.w, &[cout, c.shape[1], k]);
        let b = tensor(&c.b, &[cout]);
        let out = parity::conv1d(&x, &w, &b).unwrap();
        assert_eq!(
            out.shape(),
            c.out_shape.as_ref().unwrap(),
            "conv1d c{ci}: shape"
        );
        assert_close(
            &format!("conv1d fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
        let gout = tensor(&c.gout, c.out_shape.as_ref().unwrap());
        let (gx, gw, gb) = parity::d_conv1d(&gout, &x, &w).unwrap();
        assert_close(
            &format!("conv1d gx c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("conv1d gw c{ci}"),
            gw.data.as_ref(),
            &c.gw,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("conv1d gb c{ci}"),
            gb.data.as_ref(),
            &c.gb,
            1e-4,
            1e-4,
        );
    }
    let fx = load_fixture("convolution", "conv2d");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let x = tensor(&c.x, &c.shape);
        let (kh, kw) = (c.kh.unwrap(), c.kw.unwrap());
        let cout = c.w.len() / (c.shape[1] * kh * kw);
        let w = tensor(&c.w, &[cout, c.shape[1], kh, kw]);
        let b = tensor(&c.b, &[cout]);
        let out = parity::conv2d(&x, &w, &b).unwrap();
        assert_eq!(
            out.shape(),
            c.out_shape.as_ref().unwrap(),
            "conv2d c{ci}: shape"
        );
        assert_close(
            &format!("conv2d fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
        let gout = tensor(&c.gout, c.out_shape.as_ref().unwrap());
        let (gx, gw, gb) = parity::d_conv2d(&gout, &x, &w).unwrap();
        assert_close(
            &format!("conv2d gx c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("conv2d gw c{ci}"),
            gw.data.as_ref(),
            &c.gw,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("conv2d gb c{ci}"),
            gb.data.as_ref(),
            &c.gb,
            1e-4,
            1e-4,
        );
    }
    let fx = load_fixture("convolution", "max_pool2d");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let x = tensor(&c.x, &c.shape);
        let k = c.kernel.unwrap();
        let (out, idx) = parity::max_pool2d_with_idx(&x, k, k).unwrap();
        assert_eq!(
            out.shape(),
            c.out_shape.as_ref().unwrap(),
            "maxpool c{ci}: shape"
        );
        assert_close(
            &format!("maxpool fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            0.0,
            0.0,
        );
        assert_eq!(
            idx, c.indices,
            "maxpool c{ci}: argmax tie-break must match torch (first max)"
        );
        let gout = tensor(&c.gout, c.out_shape.as_ref().unwrap());
        let gx = parity::d_max_pool2d(&gout, &x, k, k).unwrap();
        assert_close(
            &format!("maxpool gx c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            0.0,
            0.0,
        );
    }
    let fx = load_fixture("convolution", "avg_pool2d");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let x = tensor(&c.x, &c.shape);
        let k = c.kernel.unwrap();
        let out = parity::avg_pool2d(&x, k, k).unwrap();
        assert_eq!(
            out.shape(),
            c.out_shape.as_ref().unwrap(),
            "avgpool c{ci}: shape"
        );
        assert_close(
            &format!("avgpool fwd c{ci}"),
            out.data.as_ref(),
            &c.y,
            1e-5,
            1e-5,
        );
        let gout = tensor(&c.gout, c.out_shape.as_ref().unwrap());
        let gx = parity::d_avg_pool2d(&gout, &c.shape, k, k).unwrap();
        assert_close(
            &format!("avgpool gx c{ci}"),
            gx.data.as_ref(),
            &c.gx,
            1e-5,
            1e-5,
        );
    }
}

/// cholesky (row linalg) : forward only (autograd false), SPD A=M·Mᵀ+0.5I,
/// L inférieur retourné par torch.linalg.cholesky.
#[test]
fn parity_linalg_cholesky() {
    let fx = load_fixture("linalg", "cholesky");
    assert_eq!(fx.dtype, "f32");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        assert_eq!(c.kind, "cholesky", "case {ci}");
        assert_eq!(c.shape.len(), 2, "cholesky case {ci}: doit être 2-D");
        let n = c.shape[0];
        assert_eq!(c.shape[1], n, "cholesky case {ci}: carrée");
        let a = tensor(&c.a, &[n, n]);
        let l = parity::cholesky(&a).unwrap_or_else(|e| panic!("cholesky case {ci}: {e}"));
        assert_eq!(l.shape, vec![n, n]);
        assert_close(
            &format!("cholesky fwd c{ci}"),
            l.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
    }
}

/// spmv/spmm CSR (rows sparse) : f64, forward only, tol registre 1e-10.
/// Fixtures lues via serde_json::Value (cf. parity_special_f64).
#[test]
fn parity_sparse_spmv() {
    let txt = fs::read_to_string(fixtures_dir().join("sparse").join("spmv.json")).unwrap();
    let fx: serde_json::Value = serde_json::from_str(&txt).unwrap();
    assert_eq!(fx["dtype"], "f64");
    for (ci, c) in fx["cases"].as_array().unwrap().iter().enumerate()
    {
        let _n = c["n"].as_u64().unwrap() as usize;
        let rowptr: Vec<usize> = c["rowptr"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let colidx: Vec<usize> = c["colidx"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let values: Vec<f64> = c["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let x: Vec<f64> = c["x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let b: Vec<f64> = c["b"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let yv: Vec<f64> = c["yv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let ym: Vec<f64> = c["ym"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let got_v = parity::spmv_csr(&rowptr, &colidx, &values, &x)
            .unwrap_or_else(|e| panic!("spmv case {ci}: {e}"));
        assert_close64(&format!("spmv fwd c{ci}"), &got_v, &yv, 1e-10, 1e-10);
        let got_m = parity::spmm_csr(&rowptr, &colidx, &values, &b, 2)
            .unwrap_or_else(|e| panic!("spmm case {ci}: {e}"));
        assert_close64(&format!("spmm fwd c{ci}"), &got_m, &ym, 1e-10, 1e-10);
    }
}

/// solve (row sparse) : torch.linalg.solve A·x=b en f64, tol registre 1e-8.
#[test]
fn parity_sparse_solve() {
    let txt = fs::read_to_string(fixtures_dir().join("sparse").join("solve.json")).unwrap();
    let fx: serde_json::Value = serde_json::from_str(&txt).unwrap();
    assert_eq!(fx["op"], "solve");
    assert_eq!(fx["kind"], "solve");
    assert_eq!(fx["dtype"], "f64");
    for (ci, c) in fx["cases"].as_array().unwrap().iter().enumerate()
    {
        let n = c["n"].as_u64().unwrap() as usize;
        let a: Vec<f64> = c["a"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let b1: Vec<f64> = c["b1"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let b2: Vec<f64> = c["b2"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let y1: Vec<f64> = c["y1"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let y2: Vec<f64> = c["y2"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let got1 =
            parity::solve_f64(&a, &b1, n, 1).unwrap_or_else(|e| panic!("solve case {ci}: {e}"));
        assert_close64(&format!("solve 1col c{ci}"), &got1, &y1, 1e-8, 1e-8);
        let got2 = parity::solve_f64(&a, &b2, n, 2)
            .unwrap_or_else(|e| panic!("solve 2col case {ci}: {e}"));
        assert_close64(&format!("solve 2col c{ci}"), &got2, &y2, 1e-8, 1e-8);
    }
}

/// einsum (row linear) : forward only, subset 5 specs (2 opérandes,
/// transposée, 3 opérandes, diagonale, scalaire).
#[test]
fn parity_linear_einsum() {
    let fx = load_fixture("einsum", "einsum");
    assert_eq!(fx.dtype, "f32");
    for (ci, c) in fx.cases.iter().enumerate()
    {
        let shapes = c
            .shapes
            .as_ref()
            .unwrap_or_else(|| panic!("einsum case {ci}: shapes requis"));
        let operands: Vec<Vec<f32>> = [&c.x, &c.a, &c.b]
            .iter()
            .take(shapes.len())
            .map(|d| (*d).clone())
            .collect();
        let tensors: Vec<TensorND> = operands
            .iter()
            .zip(shapes.iter())
            .map(|(d, s)| tensor(d, s))
            .collect();
        let refs: Vec<&TensorND> = tensors.iter().collect();
        let spec = c
            .spec
            .as_ref()
            .unwrap_or_else(|| panic!("einsum case {ci}: spec requis"));
        let out = parity::einsum(spec, &refs)
            .unwrap_or_else(|e| panic!("einsum case {ci} ({spec}): {e}"));
        assert_close(
            &format!("einsum fwd c{ci} ({spec})"),
            out.data.as_ref(),
            &c.y,
            1e-4,
            1e-4,
        );
    }
}

//
// Regression tests discovered by the post-campaign Tensor Parity audit.
// These cases intentionally cover valid semantics that were not exercised
// by the original frozen-fixture campaign.
//

#[test]
fn regression_cat2_unequal_concat_dimension() {
    // torch.cat accepts tensors whose concatenation dimension differs,
    // provided all other dimensions match:
    // [2, 3] cat [2, 5] along dim=1 -> [2, 8].
    let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = tensor(
        &[10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        &[2, 5],
    );

    let out = parity::cat2(&a, &b, 1).expect("valid cat2 must succeed");

    assert_eq!(out.shape, vec![2, 8]);
    assert_eq!(
        out.data.as_ref(),
        &[
            1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 14.0, 4.0, 5.0, 6.0, 20.0, 21.0, 22.0, 23.0,
            24.0,
        ],
    );
}

#[test]
fn regression_avg_pool2d_backward_preserves_discarded_border_shape() {
    // Logical forward input shape is [1, 1, 5, 5].
    // k=2, stride=2 gives output [1, 1, 2, 2]; row 4 and column 4
    // are discarded by the forward operation, but backward must still
    // return a gradient with the ORIGINAL [1, 1, 5, 5] shape.
    let gout = tensor(&[1.0, 1.0, 1.0, 1.0], &[1, 1, 2, 2]);

    let gx = parity::d_avg_pool2d(&gout, &[1, 1, 5, 5], 2, 2)
        .expect("valid avg_pool2d backward must succeed");

    assert_eq!(
        gx.shape,
        vec![1, 1, 5, 5],
        "backward must preserve the original input shape, including discarded borders",
    );

    assert_eq!(
        gx.data.as_ref(),
        &[
            0.25, 0.25, 0.25, 0.25, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0,
            0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
}

#[test]
fn regression_einsum_scalar_output_has_rank_zero() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let out = parity::einsum("ij->", &[&x]).expect("valid scalar einsum must succeed");

    assert_eq!(
        out.shape,
        Vec::<usize>::new(),
        "einsum scalar output must have rank zero, not shape [1]",
    );
    assert_eq!(out.data.as_ref(), &[21.0]);
}

#[test]
fn regression_error_conv1d_kernel_larger_than_input() {
    let x = tensor(&[1.0, 2.0, 3.0], &[1, 1, 3]);
    let w = tensor(&[1.0; 5], &[1, 1, 5]);
    let b = tensor(&[0.0], &[1]);

    let err = parity::conv1d(&x, &w, &b)
        .expect_err("kernel larger than input must return a structured error");
    assert!(matches!(err, SciRustError::InvalidConfig(_)));
}

#[test]
fn regression_error_conv2d_kernel_larger_than_input() {
    let x = tensor(&[1.0; 9], &[1, 1, 3, 3]);
    let w = tensor(&[1.0; 16], &[1, 1, 4, 4]);
    let b = tensor(&[0.0], &[1]);

    let err = parity::conv2d(&x, &w, &b)
        .expect_err("kernel larger than input must return a structured error");
    assert!(matches!(err, SciRustError::InvalidConfig(_)));
}

#[test]
fn regression_error_cholesky_scalar_is_structured() {
    let scalar = tensor(&[1.0], &[]);

    let err =
        parity::cholesky(&scalar).expect_err("rank-zero input must return a structured error");
    assert!(matches!(
        err,
        SciRustError::RankMismatch { .. } | SciRustError::InvalidConfig(_)
    ));
}

#[test]
fn regression_error_spmv_empty_rowptr_is_structured() {
    let err = parity::spmv_csr(&[], &[], &[], &[])
        .expect_err("empty CSR rowptr must return a structured error");
    assert!(matches!(err, SciRustError::InvalidConfig(_)));
}

#[test]
fn regression_error_solve_wrong_rhs_length_is_structured() {
    let a = [1.0, 0.0, 0.0, 1.0];
    let b = [1.0];

    let err = parity::solve_f64(&a, &b, 2, 1)
        .expect_err("wrong RHS length must return a structured error");
    assert!(matches!(err, SciRustError::InvalidConfig(_)));
}

#[test]
fn regression_error_d_conv1d_kernel_larger_than_input() {
    let x = tensor(&[1.0, 2.0, 3.0], &[1, 1, 3]);
    let w = tensor(&[1.0; 5], &[1, 1, 5]);
    let gout = tensor(&[1.0], &[1, 1, 1]);

    let err = parity::d_conv1d(&gout, &x, &w)
        .expect_err("gradient conv1d must reject a kernel larger than the input");
    assert!(matches!(err, SciRustError::InvalidConfig(_)));
}

#[test]
fn regression_error_d_conv2d_kernel_larger_than_input() {
    let x = tensor(&[1.0; 9], &[1, 1, 3, 3]);
    let w = tensor(&[1.0; 16], &[1, 1, 4, 4]);
    let gout = tensor(&[1.0], &[1, 1, 1, 1]);

    let err = parity::d_conv2d(&gout, &x, &w)
        .expect_err("gradient conv2d must reject a kernel larger than the input");
    assert!(matches!(err, SciRustError::InvalidConfig(_)));
}

#[test]
fn regression_error_spmm_empty_rowptr_is_structured() {
    let err = parity::spmm_csr(&[], &[], &[], &[], 1)
        .expect_err("empty CSR rowptr must return a structured error");
    assert!(matches!(err, SciRustError::InvalidConfig(_)));
}

#[test]
fn regression_error_spmm_inconsistent_nnz_is_structured() {
    let rowptr = [0usize, 1];
    let values = [1.0f64];

    let err = parity::spmm_csr(&rowptr, &[], &values, &[1.0], 1)
        .expect_err("inconsistent CSR nnz arrays must return a structured error");
    assert!(matches!(err, SciRustError::InvalidConfig(_)));
}

#[test]
fn regression_error_spmm_dense_rhs_too_short_is_structured() {
    let rowptr = [0usize, 1];
    let colidx = [1usize];
    let values = [2.0f64];
    let b = [3.0f64];

    let err = parity::spmm_csr(&rowptr, &colidx, &values, &b, 1)
        .expect_err("dense RHS too short for CSR column index must return a structured error");
    assert!(matches!(err, SciRustError::InvalidConfig(_)));
}
