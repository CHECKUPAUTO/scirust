//! Deterministic resonant-consensus closure.

use core::fmt;

use crate::{
    LinearMap16, SRCC_DIMENSION, SrccConfig, SrccError, Vector16, apply_linear_map, dot,
    squared_norm,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SrccClosureError {
    InvalidConfig(SrccError),
    EmptySeeds,
    EmptyTransports,
    NonFiniteSeed { index: usize },
    NonFiniteTransport { index: usize },
    ZeroSeedSpan,
    SeedDimensionExceedsMaximum,
}

impl fmt::Display for SrccClosureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidConfig(error) => error.fmt(formatter),
            Self::EmptySeeds => formatter.write_str("seed family must not be empty"),
            Self::EmptyTransports => formatter.write_str("transport family must not be empty"),
            Self::NonFiniteSeed { index } => write!(formatter, "seed {index} is non-finite"),
            Self::NonFiniteTransport { index } =>
            {
                write!(formatter, "transport {index} is non-finite")
            },
            Self::ZeroSeedSpan => formatter.write_str("seed family has zero span"),
            Self::SeedDimensionExceedsMaximum =>
            {
                formatter.write_str("seed span exceeds configured maximum dimension")
            },
        }
    }
}

impl std::error::Error for SrccClosureError {}

#[derive(Clone, Debug, PartialEq)]
pub struct SrccClosure {
    basis: Vec<Vector16>,
    rounds: usize,
    accepted_per_round: Vec<usize>,
    config: SrccConfig,
}

impl SrccClosure {
    pub fn build(
        seeds: &[Vector16],
        transports: &[LinearMap16],
        config: SrccConfig,
    ) -> Result<Self, SrccClosureError> {
        let config = config.validate().map_err(SrccClosureError::InvalidConfig)?;

        validate_inputs(seeds, transports)?;

        let absolute_floor = config.energy_floor.sqrt();
        let mut basis = Vec::new();

        for seed in seeds
        {
            push_orthonormal(&mut basis, *seed, absolute_floor);
        }

        if basis.is_empty()
        {
            return Err(SrccClosureError::ZeroSeedSpan);
        }

        if basis.len() > config.maximum_dimension
        {
            return Err(SrccClosureError::SeedDimensionExceedsMaximum);
        }

        let mut accepted_per_round = Vec::new();

        for _ in 0..config.maximum_rounds
        {
            if basis.len() >= config.maximum_dimension
            {
                break;
            }

            let proposals = generate_proposals(&basis, transports, config);

            let clusters = cluster_proposals(proposals, config.resonance_threshold);

            let mut expanded = basis.clone();

            for cluster in clusters
            {
                if distinct_support(&cluster, transports.len()) < config.minimum_support
                {
                    continue;
                }

                let representative = cluster_representative(&cluster);

                push_orthonormal(&mut expanded, representative, absolute_floor);

                if expanded.len() >= config.maximum_dimension
                {
                    break;
                }
            }

            let accepted = expanded.len() - basis.len();

            if accepted == 0
            {
                break;
            }

            basis = expanded;
            accepted_per_round.push(accepted);
        }

        Ok(Self {
            basis,
            rounds: accepted_per_round.len(),
            accepted_per_round,
            config,
        })
    }

    #[must_use]
    pub fn basis(&self) -> &[Vector16] {
        &self.basis
    }

    #[must_use]
    pub fn dimension(&self) -> usize {
        self.basis.len()
    }

    #[must_use]
    pub const fn rounds(&self) -> usize {
        self.rounds
    }

    #[must_use]
    pub fn accepted_per_round(&self) -> &[usize] {
        &self.accepted_per_round
    }

    #[must_use]
    pub const fn config(&self) -> SrccConfig {
        self.config
    }
}

#[derive(Clone, Debug)]
struct Proposal {
    direction: Vector16,
    transport_index: usize,
}

type ResonanceCluster = Vec<Proposal>;

fn validate_inputs(seeds: &[Vector16], transports: &[LinearMap16]) -> Result<(), SrccClosureError> {
    if seeds.is_empty()
    {
        return Err(SrccClosureError::EmptySeeds);
    }

    if transports.is_empty()
    {
        return Err(SrccClosureError::EmptyTransports);
    }

    for (index, seed) in seeds.iter().enumerate()
    {
        if seed.iter().any(|value| !value.is_finite())
        {
            return Err(SrccClosureError::NonFiniteSeed { index });
        }
    }

    for (index, transport) in transports.iter().enumerate()
    {
        if transport.iter().flatten().any(|value| !value.is_finite())
        {
            return Err(SrccClosureError::NonFiniteTransport { index });
        }
    }

    Ok(())
}

fn generate_proposals(
    basis: &[Vector16],
    transports: &[LinearMap16],
    config: SrccConfig,
) -> Vec<Proposal> {
    let mut proposals = Vec::new();

    for (transport_index, transport) in transports.iter().enumerate()
    {
        for direction in basis
        {
            let transported = apply_linear_map(transport, direction);

            let transported_norm = squared_norm(&transported).sqrt();

            if transported_norm <= config.energy_floor.sqrt()
            {
                continue;
            }

            let residual = residual_outside_span(&transported, basis);

            let residual_norm = squared_norm(&residual).sqrt();

            let novelty = residual_norm / transported_norm.max(config.energy_floor);

            if novelty <= config.novelty_threshold
            {
                continue;
            }

            proposals.push(Proposal {
                direction: normalize(residual),
                transport_index,
            });
        }
    }

    proposals
}

fn cluster_proposals(proposals: Vec<Proposal>, threshold: f64) -> Vec<ResonanceCluster> {
    let mut clusters: Vec<ResonanceCluster> = Vec::new();

    for proposal in proposals
    {
        let mut selected = None;

        for (index, cluster) in clusters.iter().enumerate()
        {
            let alignment = dot(&cluster[0].direction, &proposal.direction).abs();

            if alignment >= threshold
            {
                selected = Some(index);
                break;
            }
        }

        match selected
        {
            Some(index) => clusters[index].push(proposal),
            None => clusters.push(vec![proposal]),
        }
    }

    clusters
}

fn distinct_support(cluster: &[Proposal], transport_count: usize) -> usize {
    let mut seen = vec![false; transport_count];

    for proposal in cluster
    {
        seen[proposal.transport_index] = true;
    }

    seen.into_iter().filter(|value| *value).count()
}

fn cluster_representative(cluster: &[Proposal]) -> Vector16 {
    let anchor = cluster[0].direction;
    let mut sum = [0.0; SRCC_DIMENSION];

    for proposal in cluster
    {
        let sign = if dot(&anchor, &proposal.direction) < 0.0
        {
            -1.0
        }
        else
        {
            1.0
        };

        for (sum_value, direction_value) in sum.iter_mut().zip(proposal.direction.iter())
        {
            *sum_value += sign * direction_value;
        }
    }

    normalize(sum)
}

fn residual_outside_span(vector: &Vector16, basis: &[Vector16]) -> Vector16 {
    let mut residual = *vector;

    for _ in 0..2
    {
        for direction in basis
        {
            let coefficient = dot(direction, &residual);

            for coordinate in 0..SRCC_DIMENSION
            {
                residual[coordinate] -= coefficient * direction[coordinate];
            }
        }
    }

    residual
}

fn push_orthonormal(basis: &mut Vec<Vector16>, vector: Vector16, absolute_floor: f64) -> bool {
    let residual = residual_outside_span(&vector, basis);
    let norm = squared_norm(&residual).sqrt();

    if !norm.is_finite() || norm <= absolute_floor
    {
        return false;
    }

    basis.push(residual.map(|value| value / norm));
    true
}

fn normalize(vector: Vector16) -> Vector16 {
    let norm = squared_norm(&vector).sqrt();

    if norm == 0.0
    {
        return vector;
    }

    vector.map(|value| value / norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_vector;

    fn transport(source: usize, target: usize, coefficient: f64) -> LinearMap16 {
        let mut map = [[0.0; SRCC_DIMENSION]; SRCC_DIMENSION];
        map[target][source] = coefficient;
        map
    }

    #[test]
    fn consensus_adds_shared_direction() {
        let seeds = [basis_vector(1).unwrap()];
        let transports = [
            transport(1, 2, 1.0),
            transport(1, 2, -2.0),
            transport(1, 3, 1.0),
        ];

        let closure = SrccClosure::build(&seeds, &transports, SrccConfig::default()).unwrap();

        assert_eq!(closure.dimension(), 2);
        assert_eq!(closure.rounds(), 1);
        assert_eq!(closure.accepted_per_round(), &[1]);

        assert!(dot(&closure.basis()[1], &basis_vector(2).unwrap(),).abs() > 1.0 - 1.0e-12);
    }

    #[test]
    fn consensus_closure_grows_across_rounds() {
        let seeds = [basis_vector(1).unwrap()];

        let mut first = [[0.0; SRCC_DIMENSION]; SRCC_DIMENSION];
        first[2][1] = 1.0;
        first[3][2] = 1.0;

        let mut second = [[0.0; SRCC_DIMENSION]; SRCC_DIMENSION];
        second[2][1] = -2.0;
        second[3][2] = 3.0;

        let closure = SrccClosure::build(&seeds, &[first, second], SrccConfig::default()).unwrap();

        assert_eq!(closure.dimension(), 3);
        assert_eq!(closure.rounds(), 2);
        assert_eq!(closure.accepted_per_round(), &[1, 1]);

        assert!(dot(&closure.basis()[1], &basis_vector(2).unwrap(),).abs() > 1.0 - 1.0e-12);

        assert!(dot(&closure.basis()[2], &basis_vector(3).unwrap(),).abs() > 1.0 - 1.0e-12);
    }

    #[test]
    fn unsupported_direction_is_rejected() {
        let seeds = [basis_vector(1).unwrap()];
        let transports = [transport(1, 3, 1.0)];

        let closure = SrccClosure::build(&seeds, &transports, SrccConfig::default()).unwrap();

        assert_eq!(closure.dimension(), 1);
        assert_eq!(closure.rounds(), 0);
    }

    #[test]
    fn closure_is_deterministic() {
        let seeds = [basis_vector(1).unwrap()];
        let transports = [transport(1, 2, 1.0), transport(1, 2, -1.0)];

        let first = SrccClosure::build(&seeds, &transports, SrccConfig::default()).unwrap();

        let second = SrccClosure::build(&seeds, &transports, SrccConfig::default()).unwrap();

        assert_eq!(first, second);
    }
}
