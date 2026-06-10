//! # FusionPatterns — Base de motifs de fusion canoniques
//!
//! Regroupe les motifs de fusion supportés et fournit un moteur de matching.

use crate::graph::OpKind;

/// Un motif de fusion — séquence d'opérations qui peuvent être fusionnées.
#[derive(Debug, Clone)]
pub struct FusionPattern {
    /// Nom du motif (pour le logging).
    pub name: &'static str,
    /// Séquence d'opérations du motif (dans l'ordre d'exécution).
    pub ops: &'static [OpKindMatcher],
    /// Gain estimé en termes d'économie de mémoire (en %).
    pub memory_savings: f32,
}

/// Matcher pour un opérateur dans un motif.
/// Utilise un enum pour matcher des wildcards (ex: MatMul|Linear).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpKindMatcher {
    Exact(OpKind),
    Any(&'static [OpKind]),
}

impl OpKindMatcher {
    pub fn matches(&self, kind: OpKind) -> bool {
        match self {
            OpKindMatcher::Exact(expected) => *expected == kind,
            OpKindMatcher::Any(kinds) => kinds.contains(&kind),
        }
    }
}

/// Collection de motifs de fusion disponibles.
#[derive(Debug, Clone)]
pub struct FusionPatterns {
    patterns: Vec<FusionPattern>,
}

impl FusionPatterns {
    /// Crée les motifs par défaut.
    pub fn new() -> Self {
        let mut patterns = FusionPatterns {
            patterns: Vec::new(),
        };

        // Pattern: MatMul → SiLU (MLP hidden activation)
        patterns.add(FusionPattern {
            name: "matmul_silu",
            ops: &[
                OpKindMatcher::Any(&[OpKind::MatMul, OpKind::Linear]),
                OpKindMatcher::Exact(OpKind::SiLU),
            ],
            memory_savings: 50.0, // Économise 1 write + 1 read pour le résultat intermédiaire
        });

        // Pattern: MatMul → ReLU
        patterns.add(FusionPattern {
            name: "matmul_relu",
            ops: &[
                OpKindMatcher::Any(&[OpKind::MatMul, OpKind::Linear]),
                OpKindMatcher::Exact(OpKind::ReLU),
            ],
            memory_savings: 50.0,
        });

        // Pattern: MatMul → SiLU → LayerNorm (MLP block complet)
        patterns.add(FusionPattern {
            name: "matmul_silu_layernorm",
            ops: &[
                OpKindMatcher::Any(&[OpKind::MatMul, OpKind::Linear]),
                OpKindMatcher::Exact(OpKind::SiLU),
                OpKindMatcher::Any(&[OpKind::LayerNorm, OpKind::LayerNormFused]),
            ],
            memory_savings: 66.0, // Économise 2 writes + 2 reads
        });

        // Pattern: MatMul → LayerNorm (pre-LN transformer)
        patterns.add(FusionPattern {
            name: "matmul_layernorm",
            ops: &[
                OpKindMatcher::Any(&[OpKind::MatMul, OpKind::Linear]),
                OpKindMatcher::Any(&[OpKind::LayerNorm, OpKind::LayerNormFused]),
            ],
            memory_savings: 50.0,
        });

        // Pattern: LayerNorm → Activation (post-LN transformer)
        patterns.add(FusionPattern {
            name: "layernorm_activation",
            ops: &[
                OpKindMatcher::Any(&[OpKind::LayerNorm, OpKind::LayerNormFused]),
                OpKindMatcher::Any(&[
                    OpKind::SiLU, OpKind::Gelu, OpKind::GELU_Approx,
                    OpKind::ReLU, OpKind::Sigmoid, OpKind::Tanh,
                ]),
            ],
            memory_savings: 50.0,
        });

        // Pattern: MatMul → MatMul → Add (two-layer MLP with residual)
        patterns.add(FusionPattern {
            name: "two_layer_mlp",
            ops: &[
                OpKindMatcher::Any(&[OpKind::MatMul, OpKind::Linear]),
                OpKindMatcher::Any(&[OpKind::MatMul, OpKind::Linear]),
                OpKindMatcher::Exact(OpKind::Add),
            ],
            memory_savings: 66.0,
        });

        // Pattern: MatMul → Scale
        patterns.add(FusionPattern {
            name: "matmul_scale",
            ops: &[
                OpKindMatcher::Any(&[OpKind::MatMul, OpKind::Linear]),
                OpKindMatcher::Exact(OpKind::Scale),
            ],
            memory_savings: 50.0,
        });

        // Pattern: LayerNorm → RMSNorm (normalization chain)
        patterns.add(FusionPattern {
            name: "layernorm_rmsnorm",
            ops: &[
                OpKindMatcher::Any(&[OpKind::LayerNorm, OpKind::LayerNormFused]),
                OpKindMatcher::Exact(OpKind::RMSNorm),
            ],
            memory_savings: 33.0,
        });

        // Pattern: SsmStep → SsmStep (Mamba scan)
        patterns.add(FusionPattern {
            name: "ssm_scan",
            ops: &[
                OpKindMatcher::Exact(OpKind::SsmStep),
                OpKindMatcher::Exact(OpKind::SsmStep),
            ],
            memory_savings: 0.0, // SSM scan est séquentiel — pas de fusion possible
        });

        patterns
    }

    /// Ajoute un motif personnalisé.
    pub fn add(&mut self, pattern: FusionPattern) {
        self.patterns.push(pattern);
    }

    /// Vérifie si la séquence d'opérations correspond à un motif.
    pub fn match_sequence(&self, ops: &[OpKind]) -> Option<&FusionPattern> {
        for pattern in &self.patterns {
            if pattern.ops.len() == ops.len() {
                let all_match = pattern.ops.iter().zip(ops.iter()).all(|(matcher, op)| {
                    matcher.matches(*op)
                });
                if all_match {
                    return Some(pattern);
                }
            }
        }
        None
    }

    /// Vérifie si la paire d'opérations est un motif de fusion.
    pub fn is_pattern(&self, op_a: OpKind, op_b: OpKind) -> bool {
        self.match_sequence(&[op_a, op_b]).is_some()
    }

    /// Retourne tous les motifs dont le gain en mémoire est >= `threshold`.
    pub fn high_gain_patterns(&self, threshold: f32) -> Vec<&'static str> {
        self.patterns
            .iter()
            .filter(|p| p.memory_savings >= threshold)
            .map(|p| p.name)
            .collect()
    }

    /// Retourne la liste des motifs.
    pub fn all_patterns(&self) -> &[FusionPattern] {
        &self.patterns
    }
}

impl Default for FusionPatterns {
    fn default() -> Self {
        Self::new()
    }
}
