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
    println!("Verdict (this harness):");
    println!(
        "  At noise 0.1: Sedenion {:.4}, PositionWeighted {:.4}, Sum {:.4} (chance {:.4}).",
        sed.accuracy(),
        pos.accuracy(),
        sum.accuracy(),
        sed.chance()
    );
    println!(
        "  Commutative/associative real baselines are blind to order and/or grouping BY CONSTRUCTION;"
    );
    println!(
        "  the sedenion product discriminates both. This is a genuine capacity advantage over a plain"
    );
    println!(
        "  16-real encoding for STRUCTURE — but the same grouping is already captured by Phase 1's"
    );
    println!(
        "  explicit expression tree without the algebra. Usefulness on a real task remains unproven."
    );
}
