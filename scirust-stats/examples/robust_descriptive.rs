//! Deterministic mechanism benchmark for robust descriptive statistics.
//!
//! It builds one fixed sample of a normal signal (deterministic `SplitMix64` →
//! standard-normal inverse-CDF), then replays it under increasing **symmetric,
//! front-loaded** contamination, replacing a growing fraction of the points with a
//! single gross additive outlier. For each contamination level it prints, on one
//! line, the classical mean and median alongside the robust estimators
//! (`α`-trimmed mean, `α`-winsorized mean, median-of-means) and each estimator's
//! absolute error against the known true location.
//!
//! The output is deterministic: run it twice and the bytes are identical
//! (`cmp` / SHA-256). Formatting uses `{:.17e}` so every `f64` is printed to full
//! precision.
//!
//! # What this does and does not claim
//!
//! This is an *illustration of breakdown behaviour*, not a robustness proof. The
//! mean has a 0 % breakdown point — a single outlier moves it without bound. The
//! median, MAD-style scale, and the block estimator resist a *minority* of
//! outliers; a symmetric `α`-trimmed or `α`-winsorized mean tolerates only up to a
//! fraction `α` of contamination *per tail*. None of these estimators recovers the
//! signal once a numerical majority of the sample is adversarial. There is
//! deliberately **no claim of majority-corruption robustness** here.

use scirust_stats::{Distribution, Normal, SplitMix64, describe};
use scirust_stats::{
    MadConsistency, MedianOfMeansConfig, MedianOfMeansPartition, RobustStatsError,
    interquartile_range, median_absolute_deviation, median_of_means, trimmed_mean, weighted_median,
    winsorized_mean,
};

/// Number of samples in the fixed base signal (odd, for an unambiguous median).
const SAMPLE_COUNT: usize = 101;
/// The true location of the clean signal.
const TRUE_LOCATION: f64 = 100.0;
/// The true scale of the clean signal.
const TRUE_SCALE: f64 = 1.0;
/// A single gross additive outlier value substituted into contaminated positions.
const OUTLIER_VALUE: f64 = 1.0e3;
/// Per-tail trim fraction for the trimmed and winsorized means.
const TRIM_FRACTION: f64 = 0.25;
/// Number of blocks for the median-of-means estimator.
const MOM_BLOCKS: usize = 11;
/// Seed for the base-signal generator.
const SIGNAL_SEED: u64 = 0x5EED_0721;
/// Seed for the median-of-means partition.
const MOM_SEED: u64 = 0x0BAD_C0DE;
/// Contamination fractions replayed, in increasing order.
const CONTAMINATION_LEVELS: [f64; 7] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45];

/// Build the fixed clean sample: `TRUE_LOCATION + TRUE_SCALE · Φ⁻¹(u)` with `u`
/// drawn from a seeded `SplitMix64` and nudged into the open unit interval so the
/// inverse CDF stays finite.
fn clean_sample() -> Vec<f64> {
    let standard = Normal::standard();
    let mut rng = SplitMix64::new(SIGNAL_SEED);
    (0..SAMPLE_COUNT)
        .map(|_| {
            let u = 1.0e-6 + rng.next_f64() * (1.0 - 2.0e-6);
            TRUE_LOCATION + TRUE_SCALE * standard.quantile(u)
        })
        .collect()
}

/// Return a copy of `base` with its first `count` entries replaced by the outlier.
fn contaminate(base: &[f64], count: usize) -> Vec<f64> {
    let mut data = base.to_vec();
    for value in data.iter_mut().take(count)
    {
        *value = OUTLIER_VALUE;
    }
    data
}

fn main() -> Result<(), RobustStatsError> {
    let base = clean_sample();
    let mom_config = MedianOfMeansConfig {
        block_count: MOM_BLOCKS,
        seed: MOM_SEED,
        partition: MedianOfMeansPartition::SeededPermutation,
    };

    // Header. Lines starting with '#' are metadata; the rest is machine-readable
    // CSV so a separate parser can validate the scientific content.
    println!("# robust_descriptive deterministic mechanism benchmark");
    println!(
        "# sample_count={SAMPLE_COUNT} true_location={TRUE_LOCATION} true_scale={TRUE_SCALE} \
outlier={OUTLIER_VALUE} trim_fraction={TRIM_FRACTION} mom_blocks={MOM_BLOCKS}"
    );
    println!(
        "# columns: fraction,outliers,mean,median,trimmed_mean,winsorized_mean,median_of_means,\
mad_normal,iqr,weighted_median,err_mean,err_median,err_trimmed,err_winsorized,err_mom"
    );

    for &fraction in &CONTAMINATION_LEVELS
    {
        let outliers = (SAMPLE_COUNT as f64 * fraction).floor() as usize;
        let data = contaminate(&base, outliers);
        let uniform_weights = vec![1.0_f64; data.len()];

        let mean = describe::mean(&data);
        let median = describe::median(&data);
        let trimmed = trimmed_mean(&data, TRIM_FRACTION)?;
        let winsorized = winsorized_mean(&data, TRIM_FRACTION)?;
        let mom = median_of_means(&data, mom_config)?;
        let mad = median_absolute_deviation(&data, MadConsistency::Normal)?;
        let iqr = interquartile_range(&data)?;
        let wmed = weighted_median(&data, &uniform_weights)?;

        println!(
            "{fraction:.17e},{outliers},{mean:.17e},{median:.17e},{trimmed:.17e},\
{winsorized:.17e},{mom:.17e},{mad:.17e},{iqr:.17e},{wmed:.17e},\
{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            (mean - TRUE_LOCATION).abs(),
            (median - TRUE_LOCATION).abs(),
            (trimmed - TRUE_LOCATION).abs(),
            (winsorized - TRUE_LOCATION).abs(),
            (mom - TRUE_LOCATION).abs(),
        );
    }

    Ok(())
}
