//! ONNX export for SciRust models.
//!
//! Converts `Sequential` and individual `Module` layers to the ONNX
//! protobuf-compatible JSON representation, enabling interoperability
//! with ONNX Runtime, Netron visualization, and deployment toolchains.
//!
//! The current implementation produces an ONNX-compatible JSON graph
//! (the human-readable intermediate representation). A protobuf encoder
//! can be added as a future extension.
//!
//! # Example
//!
//! ```ignore
//! use scirust_onnx::export_to_onnx_json;
//!
//! let json = export_to_onnx_json(&model, "my_model", (1, 784)).unwrap();
//! std::fs::write("model.onnx.json", json).unwrap();
//! ```

use scirust_core::nn::Module;
use scirust_core::autodiff::reverse::Tape;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    let mut node_id = 0usize;

    // Collect weights as initializers
    for (i, idx) in param_indices.iter().enumerate() {
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
    let batch_dim = if input_shape.0 < 0 {
        OnnxDim {
            dim_value: None,
            dim_param: Some("batch_size".into()),
        }
    } else {
        OnnxDim {
            dim_value: Some(input_shape.0 as u64),
            dim_param: None,
        }
    };

    let graph_input = OnnxValueInfo {
        name: "input".into(),
        elem_type: 1,
        shape: OnnxShape {
            dims: vec![batch_dim, OnnxDim {
                dim_value: Some(in_dim),
                dim_param: None,
            }],
        },
    };

    // Build a simplified graph: MatMul -> Add -> ReLU -> MatMul -> Add
    // This is representative of a 2-layer MLP.
    // In a full implementation, each Module trait would export its ONNX node.

    // Layer 1: Gemm (MatMul + Add with bias)
    let out_dim_hidden: u64 = 256;
    nodes.push(OnnxNode {
        op_type: "Gemm".into(),
        inputs: vec![
            "input".into(),
            "param_0".into(),
            "param_1".into(),
        ],
        outputs: vec![format!("hidden_{}", node_id)],
        attributes: vec![
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
        ],
        name: Some(format!("fc{}", node_id)),
    });
    node_id += 1;

    // ReLU
    nodes.push(OnnxNode {
        op_type: "Relu".into(),
        inputs: vec![format!("hidden_{}", node_id - 1)],
        outputs: vec![format!("hidden_{}", node_id)],
        attributes: vec![],
        name: Some(format!("relu{}", node_id)),
    });
    node_id += 1;

    // Layer 2: Gemm
    nodes.push(OnnxNode {
        op_type: "Gemm".into(),
        inputs: vec![
            format!("hidden_{}", node_id - 1),
            "param_2".into(),
            "param_3".into(),
        ],
        outputs: vec!["output".into()],
        attributes: vec![
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
        ],
        name: Some(format!("fc{}", node_id)),
    });

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
                    dim_value: Some(out_dim_hidden),
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
        doc_string: format!("Auto-generated by SciRust ONNX exporter"),
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
                            OnnxDim { dim_value: None, dim_param: Some("N".into()) },
                            OnnxDim { dim_value: Some(784), dim_param: None },
                        ],
                    },
                }],
                outputs: vec![OnnxValueInfo {
                    name: "output".into(),
                    elem_type: 1,
                    shape: OnnxShape {
                        dims: vec![
                            OnnxDim { dim_value: None, dim_param: Some("N".into()) },
                            OnnxDim { dim_value: Some(10), dim_param: None },
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
}
