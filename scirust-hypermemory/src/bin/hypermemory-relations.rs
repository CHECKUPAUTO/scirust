//! Deterministic relational structure-discrimination report ("F1 for relations").
//!
//! Compares the sedenion parenthesized product against real-vector baselines
//! (Sum, Hadamard, position-weighted) at the same 16 components, on order
//! sensitivity, grouping sensitivity, and noisy structure retrieval. All numbers
//! are deterministic (pure `f32` algebra, fixed-seed LCG).
//!
//! Reproduce:
//! ```text
//! cargo +nightly-2026-07-02 run --release --bin hypermemory-relations
//! ```

use scirust_hypermemory::{Encoding, grouping_sensitivity, order_sensitivity, structure_retrieval};

const SENS_SAMPLES: usize = 50_000;
const ATOM_SETS: usize = 200;
const TRIALS_PER_SET: usize = 200;
const NOISE_LEVELS: [f32; 4] = [0.0, 0.1, 0.25, 0.5];
/// Fixed seed for the retrieval experiment (reproducibility).
const RETRIEVAL_SEED: u64 = 0x51D2_C0DE;

fn main() {
    let commit = std::env::var("HYPERMEMORY_GIT_COMMIT").unwrap_or_else(|_| "<unset>".into());

    println!("scirust-hypermemory — relational structure-discrimination report");
    println!("================================================================");
    println!("commit               : {commit}");
    println!("sensitivity samples  : {SENS_SAMPLES}");
    println!("retrieval atom sets  : {ATOM_SETS}  trials/set: {TRIALS_PER_SET}");
    println!();

    // ---- order / grouping sensitivity --------------------------------------
    println!("Order & grouping sensitivity (mean relative code distance)");
    println!(
        "  {:<20} {:>18} {:>18}",
        "encoding", "order (swap a,b)", "grouping (L vs R)"
    );
    for enc in Encoding::ALL
    {
        let ord = order_sensitivity(enc, 0x0117, SENS_SAMPLES);
        let grp = grouping_sensitivity(enc, 0x6420, SENS_SAMPLES);
        println!("  {:<20} {:>18.6} {:>18.6}", enc.label(), ord, grp);
    }
    println!();

    // ---- noisy structure retrieval -----------------------------------------
    println!("Noisy structure retrieval — nearest-neighbour accuracy");
    println!("  (codebook = 12 structures over 3 fixed atoms; chance = 1/12 ≈ 0.0833)");
    print!("  {:<20}", "encoding");
    for noise in NOISE_LEVELS
    {
        print!("  noise={noise:<5}");
    }
    println!();
    for enc in Encoding::ALL
    {
        print!("  {:<20}", enc.label());
        for noise in NOISE_LEVELS
        {
            let r = structure_retrieval(enc, RETRIEVAL_SEED, ATOM_SETS, TRIALS_PER_SET, noise);
            print!("  {:>10.4}", r.accuracy());
        }
        println!();
    }
    println!();

    // ---- verdict -----------------------------------------------------------
    let sed = structure_retrieval(
        Encoding::Sedenion,
        RETRIEVAL_SEED,
        ATOM_SETS,
        TRIALS_PER_SET,
        0.1,
    );
    let sum = structure_retrieval(
        Encoding::Real(scirust_hypermemory::RealBinding::Sum),
        RETRIEVAL_SEED,
        ATOM_SETS,
        TRIALS_PER_SET,
        0.1,
    );
    let pos = structure_retrieval(
        Encoding::Real(scirust_hypermemory::RealBinding::PositionWeighted),
        RETRIEVAL_SEED,
        ATOM_SETS,
        TRIALS_PER_SET,
        0.1,
    );
    let hrr = structure_retrieval(
        Encoding::Hrr,
        RETRIEVAL_SEED,
        ATOM_SETS,
        TRIALS_PER_SET,
        0.1,
    );
    let sed_hi = structure_retrieval(
        Encoding::Sedenion,
        RETRIEVAL_SEED,
        ATOM_SETS,
        TRIALS_PER_SET,
        0.5,
    );
    let hrr_hi = structure_retrieval(
        Encoding::Hrr,
        RETRIEVAL_SEED,
        ATOM_SETS,
        TRIALS_PER_SET,
        0.5,
    );

    println!("Verdict (this harness):");
    println!(
        "  At noise 0.1: Sedenion {:.4}, HRR {:.4}, PositionWeighted {:.4}, Sum {:.4} (chance {:.4}).",
        sed.accuracy(),
        hrr.accuracy(),
        pos.accuracy(),
        sum.accuracy(),
        sed.chance()
    );
    println!(
        "  vs the naive real baselines (Sum/Hadamard/PosWeighted) the sedenion product is a clear"
    );
    println!(
        "  structure-discrimination win — they are blind to order and/or grouping by construction."
    );
    println!(
        "  BUT vs HRR — a purpose-built structural encoding (circular convolution + role vectors) —"
    );
    println!(
        "  the sedenion does NOT win: HRR matches it at low noise and is more robust at high noise"
    );
    println!(
        "  (noise 0.5: HRR {:.4} vs Sedenion {:.4}). So the algebra's structural capacity is real but",
        hrr_hi.accuracy(),
        sed_hi.accuracy()
    );
    println!(
        "  NOT superior to established structural methods — and it carries the zero-divisor collapse"
    );
    println!("  risk (F2) that HRR does not. The relation direction is bounded, not vindicated.");
}
