//! Shared internal validation.

use crate::FractionalError;

pub(crate) fn validate_step(step: f64) -> Result<(), FractionalError> {
    if !step.is_finite() || step <= 0.0
    {
        return Err(FractionalError::InvalidStep(step));
    }

    Ok(())
}

pub(crate) fn validate_samples(samples: &[f64]) -> Result<(), FractionalError> {
    if samples.is_empty()
    {
        return Err(FractionalError::EmptySamples);
    }

    if let Some((index, _)) = samples
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(FractionalError::NonFiniteSample(index));
    }

    Ok(())
}

pub(crate) fn validate_sample_times(sample_times: &[f64]) -> Result<(), FractionalError> {
    if sample_times.is_empty()
    {
        return Err(FractionalError::EmptySamples);
    }

    if let Some((index, _)) = sample_times
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(FractionalError::NonFiniteSampleTime(index));
    }

    for index in 1..sample_times.len()
    {
        if sample_times[index] <= sample_times[index - 1]
        {
            return Err(FractionalError::NonMonotonicSampleTimes { index });
        }
    }

    Ok(())
}
