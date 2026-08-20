//! Operational performance-analysis laws for measured service systems.
//!
//! These helpers implement deterministic relationships among directly
//! measurable quantities such as throughput, utilization, response time,
//! visit ratio, and mean service time. They are intentionally independent of
//! a stochastic arrival/service distribution. The API is suitable for
//! validating measurements, constructing analytical baselines, and comparing
//! simulations against operational identities.
//!
//! The core relationships follow the operational-analysis formulation of
//! Denning & Buzen (ACM Computing Surveys, 1978). The functions do not claim
//! that an arbitrary workload is stationary or Markovian; callers remain
//! responsible for the assumptions of any prediction made from measured
//! quantities.

use crate::engine::SimError;

fn check_finite(name: &str, value: f64) -> Result<(), SimError> {
    if value.is_finite()
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be finite"
        )))
    }
}

fn check_non_negative(name: &str, value: f64) -> Result<(), SimError> {
    check_finite(name, value)?;
    if value >= 0.0
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be non-negative"
        )))
    }
}

fn check_positive(name: &str, value: f64) -> Result<(), SimError> {
    check_finite(name, value)?;
    if value > 0.0
    {
        Ok(())
    }
    else
    {
        Err(SimError::BadInput(format!(
            "{name} = {value} must be positive"
        )))
    }
}

/// Apply the utilization law `U = X·S` for a single service center.
///
/// `throughput` is completions per unit time and `mean_service_time` is mean
/// busy service time per completion. The returned utilization is constrained
/// to `[0, 1]`; a larger product indicates measurements or a model that are
/// inconsistent with a single unit-capacity service center.
pub fn utilization_law(throughput: f64, mean_service_time: f64) -> Result<f64, SimError> {
    check_non_negative("throughput", throughput)?;
    check_non_negative("mean_service_time", mean_service_time)?;
    let utilization = throughput * mean_service_time;
    if !utilization.is_finite()
    {
        return Err(SimError::BadInput(
            "throughput * mean_service_time overflowed".to_string(),
        ));
    }
    if utilization > 1.0
    {
        return Err(SimError::BadInput(format!(
            "utilization = {utilization} exceeds unit service capacity"
        )));
    }
    Ok(utilization)
}

/// Apply Little's law `N = X·R` to obtain mean population from throughput and
/// mean response/sojourn time.
pub fn little_mean_population(throughput: f64, mean_response_time: f64) -> Result<f64, SimError> {
    check_non_negative("throughput", throughput)?;
    check_non_negative("mean_response_time", mean_response_time)?;
    let population = throughput * mean_response_time;
    if population.is_finite()
    {
        Ok(population)
    }
    else
    {
        Err(SimError::BadInput(
            "throughput * mean_response_time overflowed".to_string(),
        ))
    }
}

/// Rearrange Little's law to obtain mean response/sojourn time `R = N/X`.
///
/// Throughput must be strictly positive.
pub fn little_mean_response_time(mean_population: f64, throughput: f64) -> Result<f64, SimError> {
    check_non_negative("mean_population", mean_population)?;
    check_positive("throughput", throughput)?;
    Ok(mean_population / throughput)
}

/// Apply the forced-flow law `X_i = V_i·X_0`.
///
/// `system_throughput` is the completion rate at the reference/system level;
/// `visit_ratio` is the mean number of visits to the service center per system
/// completion.
pub fn forced_flow(system_throughput: f64, visit_ratio: f64) -> Result<f64, SimError> {
    check_non_negative("system_throughput", system_throughput)?;
    check_non_negative("visit_ratio", visit_ratio)?;
    let throughput = system_throughput * visit_ratio;
    if throughput.is_finite()
    {
        Ok(throughput)
    }
    else
    {
        Err(SimError::BadInput(
            "system_throughput * visit_ratio overflowed".to_string(),
        ))
    }
}

/// Apply the interactive response-time relation `R = M/X - Z`.
///
/// `active_population` is the mean number of users/jobs in the closed
/// think-wait cycle, `system_throughput` is completed cycles per unit time,
/// and `think_time` is mean time outside the measured service subsystem.
pub fn interactive_response_time(
    active_population: f64,
    system_throughput: f64,
    think_time: f64,
) -> Result<f64, SimError> {
    check_non_negative("active_population", active_population)?;
    check_positive("system_throughput", system_throughput)?;
    check_non_negative("think_time", think_time)?;
    let response = active_population / system_throughput - think_time;
    if response < 0.0
    {
        return Err(SimError::BadInput(format!(
            "interactive response time = {response} is negative"
        )));
    }
    Ok(response)
}

/// Mean service demand of one service center: `D = V·S`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ServiceDemand {
    visit_ratio: f64,
    mean_service_time: f64,
}

impl ServiceDemand {
    /// Construct a service demand from a non-negative visit ratio and mean
    /// service time.
    pub fn new(visit_ratio: f64, mean_service_time: f64) -> Result<Self, SimError> {
        check_non_negative("visit_ratio", visit_ratio)?;
        check_non_negative("mean_service_time", mean_service_time)?;
        Ok(Self {
            visit_ratio,
            mean_service_time,
        })
    }

    /// Mean number of visits per system completion.
    pub fn visit_ratio(self) -> f64 {
        self.visit_ratio
    }

    /// Mean service time per visit.
    pub fn mean_service_time(self) -> f64 {
        self.mean_service_time
    }

    /// Service demand `V·S` per system completion.
    pub fn demand(self) -> f64 {
        self.visit_ratio * self.mean_service_time
    }
}

/// Deterministic bottleneck summary for a collection of service demands.
#[derive(Debug, Clone, PartialEq)]
pub struct BottleneckAnalysis {
    /// Index of the first service center attaining maximum demand.
    pub bottleneck_index: usize,
    /// Maximum service demand among the centers.
    pub bottleneck_demand: f64,
    /// Sum of all service demands, a lower bound on response time when waiting
    /// is absent from the service centers represented by the demands.
    pub minimum_service_time: f64,
    /// Saturation throughput bound `1 / max_i(D_i)` for unit-capacity centers.
    pub saturation_throughput: f64,
}

/// Analyze service demands to identify the bottleneck and operational
/// saturation-throughput bound.
///
/// At least one strictly positive demand is required. Ties are deterministic:
/// the smallest index is returned as `bottleneck_index`.
pub fn analyze_bottleneck(demands: &[ServiceDemand]) -> Result<BottleneckAnalysis, SimError> {
    if demands.is_empty()
    {
        return Err(SimError::BadInput(
            "at least one service demand is required".to_string(),
        ));
    }

    let mut bottleneck_index = 0usize;
    let mut bottleneck_demand = demands[0].demand();
    let mut total = bottleneck_demand;
    for (index, demand) in demands.iter().copied().enumerate().skip(1)
    {
        let value = demand.demand();
        total += value;
        if value > bottleneck_demand
        {
            bottleneck_index = index;
            bottleneck_demand = value;
        }
    }

    if bottleneck_demand <= 0.0
    {
        return Err(SimError::BadInput(
            "at least one service demand must be positive".to_string(),
        ));
    }
    if !total.is_finite()
    {
        return Err(SimError::BadInput(
            "sum of service demands overflowed".to_string(),
        ));
    }

    Ok(BottleneckAnalysis {
        bottleneck_index,
        bottleneck_demand,
        minimum_service_time: total,
        saturation_throughput: 1.0 / bottleneck_demand,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utilization_and_little_laws_match_exact_examples() {
        assert_eq!(utilization_law(2.0, 0.25).unwrap(), 0.5);
        assert_eq!(little_mean_population(2.0, 3.0).unwrap(), 6.0);
        assert_eq!(little_mean_response_time(6.0, 2.0).unwrap(), 3.0);
        assert_eq!(forced_flow(2.0, 4.0).unwrap(), 8.0);
        assert_eq!(interactive_response_time(20.0, 1.0, 18.0).unwrap(), 2.0);
    }

    #[test]
    fn denning_buzen_bottleneck_example_is_reproduced() {
        let demands = [
            ServiceDemand::new(20.0, 0.05).unwrap(),
            ServiceDemand::new(11.0, 0.08).unwrap(),
            ServiceDemand::new(8.0, 0.04).unwrap(),
        ];
        let analysis = analyze_bottleneck(&demands).unwrap();
        assert_eq!(analysis.bottleneck_index, 0);
        assert!((analysis.bottleneck_demand - 1.0).abs() < 1e-15);
        assert!((analysis.minimum_service_time - 2.2).abs() < 1e-15);
        assert!((analysis.saturation_throughput - 1.0).abs() < 1e-15);
    }

    #[test]
    fn invalid_operational_inputs_are_rejected() {
        assert!(utilization_law(2.0, 0.75).is_err());
        assert!(utilization_law(f64::NAN, 0.1).is_err());
        assert!(little_mean_response_time(1.0, 0.0).is_err());
        assert!(interactive_response_time(1.0, 2.0, 1.0).is_err());
        assert!(analyze_bottleneck(&[]).is_err());
        let zero = [ServiceDemand::new(0.0, 1.0).unwrap()];
        assert!(analyze_bottleneck(&zero).is_err());
    }
}
