//! # Dynamic context-admission policy
//!
//! CCOS v0.2 used a single global `paging_threshold = 0.6`. With spatial regions
//! that bar becomes **dynamic**: whether a [`ContextRegion`] is hydrated depends
//! on token pressure, task complexity and the region's own heat/cohesion. A hot,
//! causally dense region can be admitted even when the static threshold would
//! reject it; a cold region is expelled.
//!
//! The policy is a pure, deterministic function of its inputs — no clocks, no
//! randomness — so admission decisions replay identically.

use crate::context_region::ContextRegion;
use serde::{Deserialize, Serialize};

/// Inputs that shape a single admission decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPolicy {
    /// Token budget of the target context window.
    pub available_tokens: usize,
    /// Tokens already committed to the window.
    pub current_context_size: usize,
    /// Estimated task complexity in `[0, 1]` (harder tasks justify more context).
    pub task_complexity: f32,
    /// Temperature of the region under consideration (informational / logging).
    pub region_temperature: f32,
    /// The static baseline (CCOS v0.2 used 0.6).
    pub base_threshold: f32,
}

impl Default for ContextPolicy {
    fn default() -> Self {
        ContextPolicy {
            available_tokens: 8192,
            current_context_size: 0,
            task_complexity: 0.5,
            region_temperature: 0.0,
            base_threshold: 0.6,
        }
    }
}

impl ContextPolicy {
    /// Fraction of the token budget already used, in `[0, 1]`.
    pub fn used_ratio(&self) -> f32 {
        if self.available_tokens == 0 {
            1.0
        } else {
            (self.current_context_size as f32 / self.available_tokens as f32).clamp(0.0, 1.0)
        }
    }

    /// The dynamic admission threshold. The static `base_threshold` is raised by
    /// token pressure (up to +0.3 when the window is full) and lowered by task
    /// complexity (up to −0.2): a nearly-full window admits only the hottest
    /// regions, while a complex task on an empty window is more permissive.
    pub fn dynamic_threshold(&self) -> f32 {
        (self.base_threshold + 0.3 * self.used_ratio() - 0.2 * self.task_complexity)
            .clamp(0.05, 0.95)
    }

    /// Admission score of a region in `[0, 1]`: a blend of its temperature, its
    /// (squashed) causal density and the task complexity. Hot, cohesive regions
    /// score high.
    pub fn calculate_admission_score(&self, region: &ContextRegion) -> f32 {
        // Squash unbounded density into [0, 1).
        let density = (region.causal_density / (1.0 + region.causal_density)).clamp(0.0, 1.0);
        (0.55 * region.temperature + 0.30 * density + 0.15 * self.task_complexity).clamp(0.0, 1.0)
    }

    /// Whether `region` is admitted under the current policy: its admission score
    /// must reach the dynamic threshold.
    pub fn admits(&self, region: &ContextRegion) -> bool {
        self.calculate_admission_score(region) >= self.dynamic_threshold()
    }

    /// Estimated token cost of hydrating a region (≈128 tokens/member, matching
    /// the scheduler's heuristic).
    pub fn estimate_tokens(region: &ContextRegion) -> usize {
        region.member_count() * 128
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_region::ContextRegion;

    fn region(temp: f32, density: f32, members: usize) -> ContextRegion {
        let mut r = ContextRegion::new("r", "c");
        r.temperature = temp;
        r.causal_density = density;
        r.members = (0..members).map(|i| format!("n{i}")).collect();
        r
    }

    #[test]
    fn hot_dense_region_is_admitted() {
        let policy = ContextPolicy::default();
        let hot = region(0.9, 2.0, 4);
        assert!(policy.admits(&hot), "a hot, dense region must be admitted");
    }

    #[test]
    fn cold_region_is_rejected() {
        let policy = ContextPolicy::default();
        let cold = region(0.05, 0.0, 3);
        assert!(
            !policy.admits(&cold),
            "a cold, sparse region must be expelled"
        );
    }

    #[test]
    fn token_pressure_raises_the_bar() {
        let empty = ContextPolicy {
            current_context_size: 0,
            ..ContextPolicy::default()
        };
        let nearly_full = ContextPolicy {
            current_context_size: 8000,
            ..ContextPolicy::default()
        };
        assert!(
            nearly_full.dynamic_threshold() > empty.dynamic_threshold(),
            "a fuller window must demand a higher admission score"
        );
    }

    #[test]
    fn complex_task_lowers_the_bar() {
        let simple = ContextPolicy {
            task_complexity: 0.0,
            ..ContextPolicy::default()
        };
        let complex = ContextPolicy {
            task_complexity: 1.0,
            ..ContextPolicy::default()
        };
        assert!(complex.dynamic_threshold() < simple.dynamic_threshold());
    }

    #[test]
    fn admission_is_deterministic() {
        let policy = ContextPolicy::default();
        let r = region(0.7, 1.5, 5);
        assert_eq!(
            policy.calculate_admission_score(&r),
            policy.calculate_admission_score(&r)
        );
    }

    #[test]
    fn token_estimate_scales_with_members() {
        assert_eq!(ContextPolicy::estimate_tokens(&region(0.5, 1.0, 4)), 512);
    }
}
