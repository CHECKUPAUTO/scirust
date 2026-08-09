// Direct production-parity harness for the scirust-core 2D reverse-mode stack.
//
// IMPORTANT:
// - consumes the frozen PyTorch 2.13.0 fixtures;
// - calls autodiff::reverse::{Tensor, Tape, Var} directly;
// - NEVER calls scirust_core::tensor::parity;
// - compares both forward values and gradients produced by Tape::backward.

use scirust_core::autodiff::reverse::{Tape, Tensor, Var};
use scirust_core::error::SciRustError;
use serde::Deserialize;
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
    b_shape: Vec<usize>,
    #[serde(default)]
    out_shape: Vec<usize>,
    #[serde(default)]
    scalar: Option<f32>,
    x: Vec<f32>,
    #[serde(default)]
    b: Vec<f32>,
    y: Vec<f32>,
    gout: Vec<f32>,
    gx: Vec<f32>,
    #[serde(default)]
    gb: Vec<f32>,
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURES)
}

fn load_fixture_family(family: &str, op: &str) -> Fixture {
    let path = fixtures_dir().join(family).join(format!("{op}.json"));
    let text =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read fixture {path:?}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("invalid fixture {path:?}: {e}"))
}

fn load_fixture(op: &str) -> Fixture {
    load_fixture_family("elementwise", op)
}

fn shape_2d(op: &str, case: usize, shape: &[usize]) -> (usize, usize) {
    assert_eq!(
        shape.len(),
        2,
        "{op} case {case}: 2D production harness requires rank 2, got {shape:?}"
    );
    (shape[0], shape[1])
}

fn close(a: f32, b: f32, atol: f32, rtol: f32) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: length mismatch {} vs {}",
        got.len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want).enumerate()
    {
        assert!(
            close(g, w, atol, rtol),
            "{name}[{i}]: got {g}, want {w} (atol={atol}, rtol={rtol})"
        );
    }
}

fn apply_unary<'t>(op: &str, x: Var<'t>, scalar: Option<f32>) -> Var<'t> {
    match op
    {
        "neg" => x.neg(),
        "reciprocal" => x.reciprocal(),
        "exp" => x.exp(),
        "log" => x.log(),
        "log10" => x.log10(),
        "sqrt" => x.sqrt(),
        "pow" => x.pow(scalar.expect("pow fixture missing scalar")),
        "sin" => x.sin(),
        "cos" => x.cos(),
        "tan" => x.tan(),
        "asin" => x.asin(),
        "acos" => x.acos(),
        "atan" => x.atan(),
        "sinh" => x.sinh(),
        "cosh" => x.cosh(),
        "tanh" => x.tanh(),
        "sigmoid" => x.sigmoid(),
        "relu" => x.relu(),
        _ => panic!("unsupported 2D unary production op {op}"),
    }
}

fn apply_binary<'t>(op: &str, a: Var<'t>, b: Var<'t>) -> Result<Var<'t>, SciRustError> {
    match op
    {
        "add" => a.try_add_broadcast(b),
        "sub" => a.try_sub_broadcast(b),
        "mul" => a.try_mul_broadcast(b),
        "div" => a.try_div_broadcast(b),
        "atan2" => a.try_atan2(b),
        _ => panic!("unsupported 2D binary production op {op}"),
    }
}

fn run_unary(op: &str) {
    let fx = load_fixture(op);
    assert_eq!(fx.op, op);
    assert_eq!(fx.dtype, "f32");
    assert!(
        fx.kind == "unary" || fx.kind == "unary_scalar",
        "{op}: unexpected fixture kind {}",
        fx.kind
    );

    for (ci, c) in fx.cases.iter().enumerate()
    {
        assert_eq!(c.kind, fx.kind, "{op} case {ci}: case kind drift");

        let (rows, cols) = shape_2d(op, ci, &c.shape);
        let tape = Tape::new();

        let xv = tape.input(Tensor::from_vec(c.x.clone(), rows, cols));
        let gout = tape.input(Tensor::from_vec(c.gout.clone(), rows, cols));

        let yv = apply_unary(op, xv, c.scalar);
        let y = tape.value(yv.idx());

        assert_close(
            &format!("{op} forward case {ci}"),
            &y.data,
            &c.y,
            1e-5,
            1e-5,
        );

        // PyTorch fixtures store gx for an arbitrary upstream gradient `gout`.
        // Reproduce exactly that VJP:
        //     loss = sum(y * gout)
        // then d(loss)/dx == gx.
        let loss = yv.mul(gout).sum();
        loss.backward();

        let gx = tape.grad(xv.idx());
        assert_close(
            &format!("{op} grad-x case {ci}"),
            &gx.data,
            &c.gx,
            1e-5,
            1e-5,
        );
    }
}

fn run_binary(op: &str) {
    let fx = load_fixture(op);
    assert_eq!(fx.op, op);
    assert_eq!(fx.dtype, "f32");
    assert_eq!(fx.kind, "binary");

    for (ci, c) in fx.cases.iter().enumerate()
    {
        assert_eq!(c.kind, "binary", "{op} case {ci}: case kind drift");

        let (rows, cols) = shape_2d(op, ci, &c.shape);
        let tape = Tape::new();

        let av = tape.input(Tensor::from_vec(c.x.clone(), rows, cols));
        let bv = tape.input(Tensor::from_vec(c.b.clone(), rows, cols));
        let gout = tape.input(Tensor::from_vec(c.gout.clone(), rows, cols));

        let yv = apply_binary(op, av, bv)
            .unwrap_or_else(|e| panic!("{op} case {ci}: unexpected production error: {e}"));
        let y = tape.value(yv.idx());

        assert_close(
            &format!("{op} forward case {ci}"),
            &y.data,
            &c.y,
            1e-5,
            1e-5,
        );

        let loss = yv.mul(gout).sum();
        loss.backward();

        let ga = tape.grad(av.idx());
        let gb = tape.grad(bv.idx());

        assert_close(
            &format!("{op} grad-a case {ci}"),
            &ga.data,
            &c.gx,
            1e-5,
            1e-5,
        );
        assert_close(
            &format!("{op} grad-b case {ci}"),
            &gb.data,
            &c.gb,
            1e-5,
            1e-5,
        );
    }
}

fn run_binary_broadcast(op: &str) {
    let fx = load_fixture_family("elementwise_broadcast", op);

    assert_eq!(fx.op, op);
    assert_eq!(fx.dtype, "f32");
    assert_eq!(fx.kind, "binary_broadcast");

    for (ci, c) in fx.cases.iter().enumerate()
    {
        assert_eq!(
            c.kind, "binary_broadcast",
            "{op} broadcast case {ci}: case kind drift"
        );

        let (a_rows, a_cols) = shape_2d(op, ci, &c.shape);
        let (b_rows, b_cols) = shape_2d(op, ci, &c.b_shape);
        let (out_rows, out_cols) = shape_2d(op, ci, &c.out_shape);

        assert_eq!(
            c.x.len(),
            a_rows * a_cols,
            "{op} broadcast case {ci}: A fixture length"
        );
        assert_eq!(
            c.b.len(),
            b_rows * b_cols,
            "{op} broadcast case {ci}: B fixture length"
        );
        assert_eq!(
            c.y.len(),
            out_rows * out_cols,
            "{op} broadcast case {ci}: output fixture length"
        );
        assert_eq!(
            c.gout.len(),
            out_rows * out_cols,
            "{op} broadcast case {ci}: gout fixture length"
        );
        assert_eq!(
            c.gx.len(),
            a_rows * a_cols,
            "{op} broadcast case {ci}: grad-A fixture length"
        );
        assert_eq!(
            c.gb.len(),
            b_rows * b_cols,
            "{op} broadcast case {ci}: grad-B fixture length"
        );

        let tape = Tape::new();

        let av = tape.input(Tensor::from_vec(c.x.clone(), a_rows, a_cols));
        let bv = tape.input(Tensor::from_vec(c.b.clone(), b_rows, b_cols));
        let gout = tape.input(Tensor::from_vec(c.gout.clone(), out_rows, out_cols));

        let yv = apply_binary(op, av, bv).unwrap_or_else(|e| {
            panic!("{op} broadcast case {ci}:                  unexpected production error: {e}")
        });

        let y = tape.value(yv.idx());

        assert_eq!(
            y.shape(),
            (out_rows, out_cols),
            "{op} broadcast case {ci}: output shape"
        );

        assert_close(
            &format!("{op} broadcast forward case {ci}"),
            &y.data,
            &c.y,
            1e-5,
            1e-5,
        );

        // Exact PyTorch VJP witness:
        // loss = sum(y * gout)
        let loss = yv.mul(gout).sum();
        loss.backward();

        let ga = tape.grad(av.idx());
        let gb = tape.grad(bv.idx());

        assert_eq!(
            ga.shape(),
            (a_rows, a_cols),
            "{op} broadcast case {ci}: reduced grad-A shape"
        );
        assert_eq!(
            gb.shape(),
            (b_rows, b_cols),
            "{op} broadcast case {ci}: reduced grad-B shape"
        );

        assert_close(
            &format!("{op} broadcast grad-a case {ci}"),
            &ga.data,
            &c.gx,
            1e-5,
            1e-5,
        );

        assert_close(
            &format!("{op} broadcast grad-b case {ci}"),
            &gb.data,
            &c.gb,
            1e-5,
            1e-5,
        );
    }
}

#[test]
fn production_2d_elementwise_unary_forward_and_grad() {
    // Only operators whose registry inventory explicitly contains `2d`.
    for op in [
        "neg",
        "reciprocal",
        "exp",
        "log",
        "log10",
        "sqrt",
        "pow",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "sigmoid",
        "relu",
    ]
    {
        run_unary(op);
    }
}

#[test]
fn production_2d_elementwise_binary_forward_and_grad() {
    for op in ["add", "sub", "mul", "div", "atan2"]
    {
        run_binary(op);
    }
}

#[test]
fn production_2d_elementwise_binary_broadcast_forward_and_grad() {
    for op in ["add", "sub", "mul", "div", "atan2"]
    {
        run_binary_broadcast(op);
    }
}

#[test]
fn production_2d_binary_shape_errors_are_structured() {
    for op in ["add", "sub", "mul", "div", "atan2"]
    {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let b = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));

        let err =
            apply_binary(op, a, b).expect_err("shape mismatch must return a structured error");

        assert!(
            matches!(err, SciRustError::ShapeMismatch { .. }),
            "{op}: expected ShapeMismatch, got {err:?}"
        );
    }
}

#[test]
fn production_2d_binary_cross_tape_errors_are_structured() {
    for op in ["add", "sub", "mul", "div", "atan2"]
    {
        let tape_a = Tape::new();
        let tape_b = Tape::new();

        let a = tape_a.input(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2));
        let b = tape_b.input(Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2));

        let err =
            apply_binary(op, a, b).expect_err("cross-tape misuse must return a structured error");

        assert!(
            matches!(err, SciRustError::InvalidConfig(_)),
            "{op}: expected InvalidConfig, got {err:?}"
        );
    }
}

#[test]
fn production_2d_binary_broadcast_is_symmetric_and_gradients_reduce() {
    // A=(1,3), B=(2,1) -> output=(2,3). Both operands must broadcast.
    for op in ["add", "sub", "mul", "div"]
    {
        let tape = Tape::new();
        let a = tape.input(Tensor::from_vec(vec![2.0, 4.0, 8.0], 1, 3));
        let b = tape.input(Tensor::from_vec(vec![2.0, 4.0], 2, 1));

        let y = apply_binary(op, a, b).expect("broadcastable shapes must succeed");
        assert_eq!(tape.value(y.idx()).shape(), (2, 3));

        let loss = y.sum();
        loss.backward();

        let ga = tape.grad(a.idx());
        let gb = tape.grad(b.idx());

        assert_eq!(ga.shape(), (1, 3), "{op}: grad A shape");
        assert_eq!(gb.shape(), (2, 1), "{op}: grad B shape");

        match op
        {
            "add" =>
            {
                assert_close("add ga", &ga.data, &[2.0, 2.0, 2.0], 1e-6, 1e-6);
                assert_close("add gb", &gb.data, &[3.0, 3.0], 1e-6, 1e-6);
            },
            "sub" =>
            {
                assert_close("sub ga", &ga.data, &[2.0, 2.0, 2.0], 1e-6, 1e-6);
                assert_close("sub gb", &gb.data, &[-3.0, -3.0], 1e-6, 1e-6);
            },
            "mul" =>
            {
                assert_close("mul ga", &ga.data, &[6.0, 6.0, 6.0], 1e-6, 1e-6);
                assert_close("mul gb", &gb.data, &[14.0, 14.0], 1e-6, 1e-6);
            },
            "div" =>
            {
                assert_close("div ga", &ga.data, &[0.75, 0.75, 0.75], 1e-6, 1e-6);
                assert_close("div gb", &gb.data, &[-3.5, -0.875], 1e-6, 1e-6);
            },
            _ => unreachable!(),
        }
    }

    // Reverse operand order: the former implementation rejected this case
    // because only the RHS was allowed to broadcast.
    let tape = Tape::new();
    let small = tape.input(Tensor::from_vec(vec![10.0, 20.0, 30.0], 1, 3));
    let large = tape.input(Tensor::ones(2, 3));
    let out = small
        .try_add_broadcast(large)
        .expect("left operand must also be broadcastable");
    assert_eq!(tape.value(out.idx()).shape(), (2, 3));
}

#[test]
fn production_2d_atan2_broadcast_forward_and_gradients_reduce() {
    // PyTorch-style 2-D broadcasting:
    // y=(1,3), x=(2,1) -> output=(2,3).
    //
    // atan2 derivatives:
    //   d atan2(y,x) / dy =  x / (x² + y²)
    //   d atan2(y,x) / dx = -y / (x² + y²)
    let tape = Tape::new();

    let y = tape.input(Tensor::from_vec(vec![2.0, 4.0, 8.0], 1, 3));
    let x = tape.input(Tensor::from_vec(vec![2.0, 4.0], 2, 1));

    let out = y
        .try_atan2(x)
        .expect("atan2 must accept mutually broadcastable 2-D shapes");

    let value = tape.value(out.idx());
    assert_eq!(value.shape(), (2, 3));

    let expected_forward = [
        2.0_f32.atan2(2.0),
        4.0_f32.atan2(2.0),
        8.0_f32.atan2(2.0),
        2.0_f32.atan2(4.0),
        4.0_f32.atan2(4.0),
        8.0_f32.atan2(4.0),
    ];

    assert_close(
        "atan2 broadcast forward",
        &value.data,
        &expected_forward,
        1e-6,
        1e-6,
    );

    let loss = out.sum();
    loss.backward();

    let gy = tape.grad(y.idx());
    let gx = tape.grad(x.idx());

    assert_eq!(gy.shape(), (1, 3));
    assert_eq!(gx.shape(), (2, 1));

    assert_close(
        "atan2 broadcast grad-y",
        &gy.data,
        &[0.45, 0.225, 0.07941177],
        1e-6,
        1e-6,
    );

    assert_close(
        "atan2 broadcast grad-x",
        &gx.data,
        &[-0.56764704, -0.325],
        1e-6,
        1e-6,
    );

    // Also exercise the opposite broadcast direction.
    let tape = Tape::new();
    let small_y = tape.input(Tensor::from_vec(vec![1.0, 2.0, 3.0], 1, 3));
    let large_x = tape.input(Tensor::ones(2, 3));

    let out = small_y
        .try_atan2(large_x)
        .expect("atan2 must also broadcast the left operand");

    assert_eq!(tape.value(out.idx()).shape(), (2, 3));
}
