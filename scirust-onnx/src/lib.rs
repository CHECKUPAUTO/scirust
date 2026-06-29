//! ONNX export/import for SciRust models.
//!
//! Converts `Sequential` and individual `Module` layers to an ONNX-compatible
//! JSON representation, and reads it back. The weight tensors round-trip
//! bit-for-bit, so this doubles as a **model checkpoint** format (save/load).
//!
//! The current implementation produces an ONNX-compatible JSON graph (the
//! human-readable intermediate representation). The graph has one `Gemm`
//! (weight + bias) per parameter pair with a `Relu` between layers, so it tracks
//! the model's actual layer count and every node references an initializer that
//! exists. It assumes a feed-forward `Gemm`/`Relu` topology; a protobuf encoder
//! and faithful export of arbitrary per-layer ops are future extensions. See
//! [`import_onnx_json`] / [`OnnxGraph::weights`].
//!
//! # Example
//!
//! ```ignore
//! use scirust_onnx::export_sequential_to_onnx_json;
//!
//! let json = export_sequential_to_onnx_json(&tape, &model, "my_model", (1, 784)).unwrap();
//! std::fs::write("model.onnx.json", json).unwrap();
//! ```

use scirust_core::autodiff::reverse::Tape;
use scirust_core::nn::Module;
use serde::{Deserialize, Serialize};

/// ONNX-compatible graph representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxGraph {
    pub name: String,
    pub ir_version: u32,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: u64,
    pub doc_string: String,
    pub graph: OnnxGraphDef,
}

/// Graph definition with nodes, inputs, outputs, and initializers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxGraphDef {
    pub name: String,
    pub inputs: Vec<OnnxValueInfo>,
    pub outputs: Vec<OnnxValueInfo>,
    pub nodes: Vec<OnnxNode>,
    pub initializers: Vec<OnnxTensor>,
}

/// Value info for graph inputs/outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxValueInfo {
    pub name: String,
    pub elem_type: u32, // 1 = f32
    pub shape: OnnxShape,
}

/// Tensor shape description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxShape {
    pub dims: Vec<OnnxDim>,
}

/// A single dimension (can be symbolic or fixed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxDim {
    pub dim_value: Option<u64>,
    pub dim_param: Option<String>,
}

/// A computation node in the ONNX graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxNode {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<OnnxAttribute>,
    pub name: Option<String>,
}

/// Node attribute (type/value pair).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxAttribute {
    pub name: String,
    #[serde(flatten)]
    pub value: OnnxAttrValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OnnxAttrValue {
    #[serde(rename = "INT")]
    Int { i: i64 },
    #[serde(rename = "INTS")]
    Ints { ints: Vec<i64> },
    #[serde(rename = "FLOAT")]
    Float { f: f32 },
    #[serde(rename = "STRING")]
    String { s: String },
}

/// A weight tensor stored in the graph initializer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub data_type: u32, // 1 = f32
    pub raw_data: Vec<f32>,
}

/// Export a `Sequential` model to ONNX JSON.
///
/// `input_shape` is `(batch_size, features)` where batch_size can be symbolic (-1).
pub fn export_sequential_to_onnx_json(
    tape: &Tape,
    model: &impl Module,
    model_name: &str,
    input_shape: (i64, usize),
) -> Result<String, Box<dyn std::error::Error>> {
    let param_indices = model.parameter_indices();
    let mut initializers = Vec::new();
    let mut nodes = Vec::new();

    // Collect weights as initializers
    for (i, idx) in param_indices.iter().enumerate()
    {
        let tensor = tape.value(*idx);
        initializers.push(OnnxTensor {
            name: format!("param_{}", i),
            dims: vec![tensor.rows as u64, tensor.cols as u64],
            data_type: 1, // f32
            raw_data: tensor.data.clone(),
        });
    }

    // Build a single MatMul + Add node for a Linear layer
    let in_dim = input_shape.1 as u64;
    let batch_dim = if input_shape.0 < 0
    {
        OnnxDim {
            dim_value: None,
            dim_param: Some("batch_size".into()),
        }
    }
    else
    {
        OnnxDim {
            dim_value: Some(input_shape.0 as u64),
            dim_param: None,
        }
    };

    let graph_input = OnnxValueInfo {
        name: "input".into(),
        elem_type: 1,
        shape: OnnxShape {
            dims: vec![
                batch_dim,
                OnnxDim {
                    dim_value: Some(in_dim),
                    dim_param: None,
                },
            ],
        },
    };

    // Build the computation graph from the ACTUAL parameter list: one Gemm
    // (weight + bias) per layer with a ReLU between consecutive layers, so every
    // node input references an initializer that exists. A trailing unpaired
    // weight (a bias-less final layer) becomes a plain MatMul. This replaces the
    // former hard-coded 2-layer template, which emitted dangling references to
    // param_2/param_3 for any model that did not have exactly four parameters.
    fn gemm_attrs() -> Vec<OnnxAttribute> {
        vec![
            OnnxAttribute {
                name: "alpha".into(),
                value: OnnxAttrValue::Float { f: 1.0 },
            },
            OnnxAttribute {
                name: "beta".into(),
                value: OnnxAttrValue::Float { f: 1.0 },
            },
            OnnxAttribute {
                name: "transB".into(),
                value: OnnxAttrValue::Int { i: 0 },
            },
        ]
    }

    let n_params = param_indices.len();
    let n_layers = n_params.div_ceil(2); // (weight, bias) pairs; a lone trailing weight is its own layer
    let mut prev = "input".to_string();
    for layer in 0..n_layers
    {
        let w_i = 2 * layer;
        let has_bias = w_i + 1 < n_params;
        let is_last = layer + 1 == n_layers;
        let out_name = if is_last
        {
            "output".to_string()
        }
        else
        {
            format!("hidden_{}", layer)
        };
        let mut inputs = vec![prev.clone(), format!("param_{}", w_i)];
        let (op_type, attributes) = if has_bias
        {
            inputs.push(format!("param_{}", w_i + 1));
            ("Gemm".to_string(), gemm_attrs())
        }
        else
        {
            ("MatMul".to_string(), Vec::new())
        };
        nodes.push(OnnxNode {
            op_type,
            inputs,
            outputs: vec![out_name.clone()],
            attributes,
            name: Some(format!("fc{}", layer)),
        });
        if is_last
        {
            prev = out_name;
        }
        else
        {
            let relu_out = format!("act_{}", layer);
            nodes.push(OnnxNode {
                op_type: "Relu".into(),
                inputs: vec![out_name],
                outputs: vec![relu_out.clone()],
                attributes: Vec::new(),
                name: Some(format!("relu{}", layer)),
            });
            prev = relu_out;
        }
    }
    let _ = prev; // graph output is always named "output" by construction

    // Output feature dimension = trailing dimension of the last initializer.
    let out_dim: u64 = initializers
        .last()
        .and_then(|t| t.dims.last().copied())
        .unwrap_or(0);

    let graph_output = OnnxValueInfo {
        name: "output".into(),
        elem_type: 1,
        shape: OnnxShape {
            dims: vec![
                OnnxDim {
                    dim_value: None,
                    dim_param: Some("batch_size".into()),
                },
                OnnxDim {
                    dim_value: Some(out_dim),
                    dim_param: None,
                },
            ],
        },
    };

    let onnx = OnnxGraph {
        name: model_name.into(),
        ir_version: 7,
        producer_name: "SciRust ONNX Exporter".into(),
        producer_version: env!("CARGO_PKG_VERSION").into(),
        domain: "ai.scirust".into(),
        model_version: 1,
        doc_string: "Auto-generated by SciRust ONNX exporter".to_string(),
        graph: OnnxGraphDef {
            name: model_name.into(),
            inputs: vec![graph_input],
            outputs: vec![graph_output],
            nodes,
            initializers,
        },
    };

    Ok(serde_json::to_string_pretty(&onnx)?)
}

/// Parse an ONNX-JSON graph produced by [`export_sequential_to_onnx_json`].
///
/// The model **weights** (the graph initializers) survive an export→import
/// round-trip bit-for-bit — which is what model checkpointing (save/load)
/// needs. The exported graph reflects the model's layer count (one `Gemm`/`Relu`
/// per parameter pair) and is self-consistent — every node input resolves to an
/// initializer — but it assumes a feed-forward topology, so reconstructing an
/// arbitrary computation graph from the import is future work; today the import
/// is a faithful **weight** loader.
pub fn import_onnx_json(json: &str) -> Result<OnnxGraph, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(json)?)
}

impl OnnxGraph {
    /// The weight tensors stored as graph initializers, returned as
    /// `(name, dims, row-major data)`. This is what round-trips exactly through
    /// [`export_sequential_to_onnx_json`] → [`import_onnx_json`].
    pub fn weights(&self) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        self.graph
            .initializers
            .iter()
            .map(|t| {
                (
                    t.name.clone(),
                    t.dims.iter().map(|&d| d as usize).collect(),
                    t.raw_data.clone(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_graph_serialization() {
        let graph = OnnxGraph {
            name: "test_model".into(),
            ir_version: 7,
            producer_name: "SciRust".into(),
            producer_version: "0.1.0".into(),
            domain: "test".into(),
            model_version: 1,
            doc_string: "test".into(),
            graph: OnnxGraphDef {
                name: "test".into(),
                inputs: vec![OnnxValueInfo {
                    name: "input".into(),
                    elem_type: 1,
                    shape: OnnxShape {
                        dims: vec![
                            OnnxDim {
                                dim_value: None,
                                dim_param: Some("N".into()),
                            },
                            OnnxDim {
                                dim_value: Some(784),
                                dim_param: None,
                            },
                        ],
                    },
                }],
                outputs: vec![OnnxValueInfo {
                    name: "output".into(),
                    elem_type: 1,
                    shape: OnnxShape {
                        dims: vec![
                            OnnxDim {
                                dim_value: None,
                                dim_param: Some("N".into()),
                            },
                            OnnxDim {
                                dim_value: Some(10),
                                dim_param: None,
                            },
                        ],
                    },
                }],
                nodes: vec![OnnxNode {
                    op_type: "Relu".into(),
                    inputs: vec!["input".into()],
                    outputs: vec!["output".into()],
                    attributes: vec![],
                    name: Some("relu_0".into()),
                }],
                initializers: vec![],
            },
        };

        let json = serde_json::to_string_pretty(&graph).unwrap();
        assert!(json.contains("test_model"));
        assert!(json.contains("Relu"));

        let _loaded: OnnxGraph = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn weights_roundtrip_exactly() {
        // Tricky f32 values (negatives, tiny, large, smallest positive) must
        // survive export→import bit-for-bit.
        let data = vec![0.0, -1.5, 2.5, 1e-8, -1e8, f32::MIN_POSITIVE, 42.0];
        let graph = OnnxGraph {
            name: "m".into(),
            ir_version: 7,
            producer_name: "t".into(),
            producer_version: "0".into(),
            domain: "d".into(),
            model_version: 1,
            doc_string: String::new(),
            graph: OnnxGraphDef {
                name: "m".into(),
                inputs: vec![],
                outputs: vec![],
                nodes: vec![],
                initializers: vec![OnnxTensor {
                    name: "w".into(),
                    dims: vec![1, 7],
                    data_type: 1,
                    raw_data: data.clone(),
                }],
            },
        };
        let json = serde_json::to_string(&graph).unwrap();
        let back = import_onnx_json(&json).unwrap();
        let w = back.weights();
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].0, "w");
        assert_eq!(w[0].1, vec![1, 7]);
        assert_eq!(w[0].2, data); // bit-exact
    }

    #[test]
    fn exported_model_weights_are_importable() {
        use scirust_core::autodiff::reverse::{Tape, Tensor};
        use scirust_core::nn::init::{KaimingNormal, Zeros};
        use scirust_core::nn::rng::PcgEngine;
        use scirust_core::nn::{Linear, Module};

        let tape = Tape::new();
        let mut rng = PcgEngine::new(7);
        let mut lin = Linear::new(4, 3, &KaimingNormal, &Zeros, &mut rng);
        let x = tape.input(Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], 1, 4));
        let _ = lin.forward(&tape, x);
        let params = lin.parameter_indices();

        let json = export_sequential_to_onnx_json(&tape, &lin, "lin", (-1, 4)).unwrap();
        let weights = import_onnx_json(&json).unwrap().weights();

        assert_eq!(weights.len(), params.len());
        // First param is the (KaimingNormal → non-zero) weight matrix; it must
        // round-trip exactly to the tape's value.
        let w0 = tape.value(params[0]);
        assert_eq!(weights[0].2, w0.data);
        assert!(
            weights[0].2.iter().any(|&v| v != 0.0),
            "weights should be non-zero"
        );
    }

    #[test]
    fn exported_graph_matches_param_count_with_no_dangling_refs() {
        use scirust_core::autodiff::reverse::{Tape, Tensor};
        use scirust_core::nn::init::{KaimingNormal, Zeros};
        use scirust_core::nn::rng::PcgEngine;
        use scirust_core::nn::{Linear, Module};

        let tape = Tape::new();
        let mut rng = PcgEngine::new(11);
        let mut lin = Linear::new(4, 3, &KaimingNormal, &Zeros, &mut rng);
        let x = tape.input(Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], 1, 4));
        let _ = lin.forward(&tape, x);

        let json = export_sequential_to_onnx_json(&tape, &lin, "lin", (-1, 4)).unwrap();
        let g = import_onnx_json(&json).unwrap();

        // A single Linear (2 params) → exactly one Gemm, no ReLU. The old
        // hard-coded 2-layer template wrongly emitted two Gemms here.
        let gemms = g.graph.nodes.iter().filter(|n| n.op_type == "Gemm").count();
        let relus = g.graph.nodes.iter().filter(|n| n.op_type == "Relu").count();
        assert_eq!(
            gemms, 1,
            "single Linear should export one Gemm, got {gemms}"
        );
        assert_eq!(relus, 0, "single layer should have no ReLU");

        // Every param_* a node references must exist as an initializer — the old
        // template emitted dangling param_2/param_3 references for this model.
        let init_names: std::collections::HashSet<&str> = g
            .graph
            .initializers
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        for node in &g.graph.nodes
        {
            for inp in &node.inputs
            {
                if inp.starts_with("param_")
                {
                    assert!(
                        init_names.contains(inp.as_str()),
                        "node {:?} references missing initializer {inp}",
                        node.name
                    );
                }
            }
        }

        // The graph is connected: the (only) Gemm consumes "input" and produces
        // "output".
        let gemm = g.graph.nodes.iter().find(|n| n.op_type == "Gemm").unwrap();
        assert_eq!(gemm.inputs[0], "input");
        assert_eq!(gemm.outputs[0], "output");
    }
}
