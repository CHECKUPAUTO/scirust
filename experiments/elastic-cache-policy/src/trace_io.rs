use crate::model::TraceRow;
use std::fs;
use std::path::Path;

pub const TRACE_HEADER: &str = "trajectory_id,step,layer_id,similarity,similarity_delta,head_variance,cache_age,attention_mass,layer_fraction,refresh_cost,stale_loss";

pub fn read_trace_csv(path: impl AsRef<Path>) -> Result<Vec<TraceRow>, String> {
    let text = fs::read_to_string(path.as_ref())
        .map_err(|error| format!("cannot read {}: {error}", path.as_ref().display()))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or_else(|| "trace is empty".to_string())?;
    if header.trim() != TRACE_HEADER
    {
        return Err(format!(
            "unexpected CSV header; expected `{TRACE_HEADER}`, got `{}`",
            header.trim()
        ));
    }

    let mut rows = Vec::new();
    for (offset, line) in lines.enumerate()
    {
        let line_number = offset + 2;
        if line.trim().is_empty()
        {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() != 11
        {
            return Err(format!(
                "line {line_number}: expected 11 columns, got {}",
                fields.len()
            ));
        }
        let parse_u64 = |index: usize, name: &str| {
            fields[index]
                .parse::<u64>()
                .map_err(|error| format!("line {line_number}: invalid {name}: {error}"))
        };
        let parse_u32 = |index: usize, name: &str| {
            fields[index]
                .parse::<u32>()
                .map_err(|error| format!("line {line_number}: invalid {name}: {error}"))
        };
        let parse_f64 = |index: usize, name: &str| {
            fields[index]
                .parse::<f64>()
                .map_err(|error| format!("line {line_number}: invalid {name}: {error}"))
        };
        let row = TraceRow {
            trajectory_id: parse_u64(0, "trajectory_id")?,
            step: parse_u32(1, "step")?,
            layer_id: parse_u32(2, "layer_id")?,
            similarity: parse_f64(3, "similarity")?,
            similarity_delta: parse_f64(4, "similarity_delta")?,
            head_variance: parse_f64(5, "head_variance")?,
            cache_age: parse_f64(6, "cache_age")?,
            attention_mass: parse_f64(7, "attention_mass")?,
            layer_fraction: parse_f64(8, "layer_fraction")?,
            refresh_cost: parse_f64(9, "refresh_cost")?,
            stale_loss: parse_f64(10, "stale_loss")?,
        };
        row.validate()
            .map_err(|error| format!("line {line_number}: {error}"))?;
        rows.push(row);
    }
    if rows.is_empty()
    {
        return Err("trace contains no data rows".into());
    }
    Ok(rows)
}

pub fn write_trace_csv(path: impl AsRef<Path>, rows: &[TraceRow]) -> Result<(), String> {
    let mut output = String::with_capacity(rows.len() * 128);
    output.push_str(TRACE_HEADER);
    output.push('\n');
    for row in rows
    {
        output.push_str(&format!(
            "{},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}\n",
            row.trajectory_id,
            row.step,
            row.layer_id,
            row.similarity,
            row.similarity_delta,
            row.head_variance,
            row.cache_age,
            row.attention_mass,
            row.layer_fraction,
            row.refresh_cost,
            row.stale_loss
        ));
    }
    fs::write(path.as_ref(), output)
        .map_err(|error| format!("cannot write {}: {error}", path.as_ref().display()))
}
