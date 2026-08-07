//! Deterministic Phase 9 adaptive-planning harness.

use scirust_core::nn::adaptive_latent_kv::{
    AdaptiveKvPolicyConfig, AdaptiveQualityProfile, select_adaptive_plan,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const KEY_RANK: [u16; 8] = [3_000, 5_500, 7_000, 8_300, 9_100, 9_550, 9_800, 10_000];
    const VALUE_RANK: [u16; 8] = [2_800, 5_200, 6_900, 8_100, 9_000, 9_500, 9_750, 10_000];
    const KEY_RESIDUAL: [u16; 3] = [0, 450, 850];
    const VALUE_RESIDUAL: [u16; 3] = [0, 500, 900];
    let profile = AdaptiveQualityProfile {
        key_rank_quality_bps: &KEY_RANK,
        value_rank_quality_bps: &VALUE_RANK,
        key_residual_gain_bps: &KEY_RESIDUAL,
        value_residual_gain_bps: &VALUE_RESIDUAL,
    };

    println!(
        "budget_bytes,total_bytes,worst_quality_bps,key_rank,value_rank,key_slots,value_slots,key_format,value_format,key_residual_format,value_residual_format,fingerprint"
    );
    for budget_bytes in [1_000_usize, 1_250, 1_500, 1_750, 2_000, 2_500]
    {
        let plan = select_adaptive_plan(
            AdaptiveKvPolicyConfig {
                capacity_tokens: 64,
                dimension: 8,
                minimum_rank: 2,
                maximum_rank: 6,
                maximum_residual_slots: 2,
                budget_bytes,
            },
            profile,
        )?;
        println!(
            "{budget_bytes},{},{},{},{},{},{},{},{},{},{},{}",
            plan.persistent_bytes,
            plan.worst_quality_bps,
            plan.key.rank,
            plan.value.rank,
            plan.key.residual_slots,
            plan.value.residual_slots,
            plan.key.coefficient_format.label(),
            plan.value.coefficient_format.label(),
            plan.key.residual_format.label(),
            plan.value.residual_format.label(),
            plan.fingerprint,
        );
    }
    Ok(())
}
