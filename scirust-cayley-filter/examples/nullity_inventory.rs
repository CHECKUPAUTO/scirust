use std::collections::BTreeMap;

use scirust_cayley_filter::{
    CayleyProjector, SEDENION_DIMENSION, Sedenion, zero_divisor_two_term_directions,
};

fn main() {
    let genuine = zero_divisor_two_term_directions(1.0e-12).unwrap();
    println!("genuine_zero_divisors={}", genuine.len());

    let mut counts = BTreeMap::new();

    for i in 0..SEDENION_DIMENSION
    {
        for j in (i + 1)..SEDENION_DIMENSION
        {
            for sign in [-1.0, 1.0]
            {
                let mut multiplier: Sedenion = [0.0; SEDENION_DIMENSION];
                multiplier[i] = 1.0;
                multiplier[j] = sign;

                let projector = CayleyProjector::new(multiplier, 1.0e-12).unwrap();
                *counts.entry(projector.rejected_dimension()).or_insert(0) += 1;
            }
        }
    }

    for (nullity, count) in counts
    {
        println!("nullity={nullity},count={count}");
    }
}
