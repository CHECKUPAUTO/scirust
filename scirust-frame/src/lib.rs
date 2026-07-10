//! `scirust-frame` — a lightweight typed columnar dataframe in pure Rust.
//!
//! A [`DataFrame`] holds ordered, named [`Column`]s of equal length. Columns are
//! strongly typed (`f64` / `i64` / `String` / `bool`). The crate offers the small
//! set of relational operations that show up again and again in data wrangling:
//! [`select`](DataFrame::select), [`filter`](DataFrame::filter),
//! [`head`](DataFrame::head), [`sort_by_f64`](DataFrame::sort_by_f64),
//! [`group_by_agg`](DataFrame::group_by_agg), [`inner_join`](DataFrame::inner_join)
//! and RFC-4180 CSV [read](DataFrame::from_csv) / [write](DataFrame::to_csv).
//!
//! Everything is deterministic: group and join outputs preserve first-seen /
//! left-then-right ordering, and sorts are stable. Fallible operations return
//! [`FrameError`] instead of panicking.
//!
//! ```
//! # fn main() -> Result<(), scirust_frame::FrameError> {
//! use scirust_frame::{Agg, Column, DataFrame};
//!
//! let df = DataFrame::new()
//!     .with_column("city", Column::Str(vec!["NYC".into(), "LA".into(), "NYC".into()]))?
//!     .with_column("temp", Column::F64(vec![30.0, 25.0, 32.0]))?;
//!
//! assert_eq!(df.n_rows(), 3);
//! assert_eq!(df.n_cols(), 2);
//!
//! // Mean temperature per city, groups in first-seen order (NYC, LA).
//! let means = df.group_by_agg("city", "temp", Agg::Mean)?;
//! assert_eq!(means.column_names(), vec!["city", "temp_mean"]);
//! match means.column("temp_mean")? {
//!     Column::F64(v) => assert_eq!(v, &vec![31.0, 25.0]),
//!     _ => unreachable!(),
//! }
//! #     Ok(())
//! # }
//! ```
#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;

/// Convenient crate-wide result type: `Result<T, FrameError>`.
pub type Result<T> = std::result::Result<T, FrameError>;

/// Errors returned by fallible [`DataFrame`] operations.
///
/// The crate never panics on user error; every recoverable failure surfaces as
/// one of these variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    /// A column with this name already exists in the frame.
    DuplicateColumn(String),
    /// No column with this name exists in the frame.
    UnknownColumn(String),
    /// A column's length did not match the frame's existing row count.
    LengthMismatch {
        /// The frame's current row count.
        expected: usize,
        /// The length of the offending column.
        found: usize,
    },
    /// A boolean mask length did not match the frame's row count.
    MaskLengthMismatch {
        /// The frame's row count.
        expected: usize,
        /// The length of the supplied mask.
        found: usize,
    },
    /// A column had a different type than the operation required.
    TypeMismatch {
        /// The type the operation required.
        expected: String,
        /// The type actually found.
        found: String,
    },
    /// A CSV record had the wrong number of fields for the schema.
    ColumnCountMismatch {
        /// The number of columns the schema declares.
        expected: usize,
        /// The number of fields actually parsed.
        found: usize,
    },
    /// A CSV document was malformed or a field failed to parse.
    CsvParse(String),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            FrameError::DuplicateColumn(name) => write!(f, "duplicate column name: {name}"),
            FrameError::UnknownColumn(name) => write!(f, "unknown column: {name}"),
            FrameError::LengthMismatch { expected, found } =>
            {
                write!(
                    f,
                    "column length mismatch: expected {expected}, found {found}"
                )
            },
            FrameError::MaskLengthMismatch { expected, found } =>
            {
                write!(
                    f,
                    "mask length mismatch: expected {expected}, found {found}"
                )
            },
            FrameError::TypeMismatch { expected, found } =>
            {
                write!(f, "type mismatch: expected {expected}, found {found}")
            },
            FrameError::ColumnCountMismatch { expected, found } =>
            {
                write!(
                    f,
                    "CSV column count mismatch: expected {expected}, found {found}"
                )
            },
            FrameError::CsvParse(msg) => write!(f, "CSV parse error: {msg}"),
        }
    }
}

impl std::error::Error for FrameError {}

/// A single strongly-typed column of values.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    /// 64-bit floating-point values.
    F64(Vec<f64>),
    /// 64-bit signed integer values.
    I64(Vec<i64>),
    /// UTF-8 string values.
    Str(Vec<String>),
    /// Boolean values.
    Bool(Vec<bool>),
}

impl Column {
    /// Number of elements in the column.
    pub fn len(&self) -> usize {
        match self
        {
            Column::F64(v) => v.len(),
            Column::I64(v) => v.len(),
            Column::Str(v) => v.len(),
            Column::Bool(v) => v.len(),
        }
    }

    /// Returns `true` if the column has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A short static name for the column's element type
    /// (`"f64"`, `"i64"`, `"str"` or `"bool"`).
    pub fn dtype(&self) -> &'static str {
        match self
        {
            Column::F64(_) => "f64",
            Column::I64(_) => "i64",
            Column::Str(_) => "str",
            Column::Bool(_) => "bool",
        }
    }

    /// Build a new column by gathering the rows at `indices` (in order).
    fn take(&self, indices: &[usize]) -> Column {
        match self
        {
            Column::F64(v) => Column::F64(indices.iter().map(|&i| v[i]).collect()),
            Column::I64(v) => Column::I64(indices.iter().map(|&i| v[i]).collect()),
            Column::Str(v) => Column::Str(indices.iter().map(|&i| v[i].clone()).collect()),
            Column::Bool(v) => Column::Bool(indices.iter().map(|&i| v[i]).collect()),
        }
    }
}

/// Aggregation functions for [`DataFrame::group_by_agg`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agg {
    /// Sum of the group's values.
    Sum,
    /// Arithmetic mean of the group's values.
    Mean,
    /// Number of values in the group.
    Count,
    /// Smallest value in the group.
    Min,
    /// Largest value in the group.
    Max,
}

impl Agg {
    /// Lower-case label used when naming the aggregate output column.
    fn label(self) -> &'static str {
        match self
        {
            Agg::Sum => "sum",
            Agg::Mean => "mean",
            Agg::Count => "count",
            Agg::Min => "min",
            Agg::Max => "max",
        }
    }
}

/// Column data types used to describe a CSV schema for [`DataFrame::from_csv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// Parse the column as `f64`.
    F64,
    /// Parse the column as `i64`.
    I64,
    /// Keep the column as raw UTF-8 strings.
    Str,
    /// Parse the column as `bool` (`true` / `false`).
    Bool,
}

/// An ordered collection of named, equal-length [`Column`]s.
#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    names: Vec<String>,
    cols: Vec<Column>,
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl DataFrame {
    /// Create an empty frame with no columns and no rows.
    pub fn new() -> Self {
        DataFrame {
            names: Vec::new(),
            cols: Vec::new(),
        }
    }

    /// Append a named column, consuming and returning the frame (builder style).
    ///
    /// # Errors
    /// Returns [`FrameError::DuplicateColumn`] if `name` already exists, or
    /// [`FrameError::LengthMismatch`] if `col`'s length differs from the frame's
    /// existing row count.
    pub fn with_column(mut self, name: &str, col: Column) -> Result<Self> {
        if self.col_index(name).is_some()
        {
            return Err(FrameError::DuplicateColumn(name.to_string()));
        }
        if let Some(first) = self.cols.first()
        {
            if first.len() != col.len()
            {
                return Err(FrameError::LengthMismatch {
                    expected: first.len(),
                    found: col.len(),
                });
            }
        }
        self.names.push(name.to_string());
        self.cols.push(col);
        Ok(self)
    }

    /// Number of rows (the shared length of all columns; `0` when empty).
    pub fn n_rows(&self) -> usize {
        self.cols.first().map_or(0, Column::len)
    }

    /// Number of columns.
    pub fn n_cols(&self) -> usize {
        self.cols.len()
    }

    /// The column names, in order.
    pub fn column_names(&self) -> Vec<&str> {
        self.names.iter().map(String::as_str).collect()
    }

    /// Borrow a column by name.
    ///
    /// # Errors
    /// Returns [`FrameError::UnknownColumn`] if no such column exists.
    pub fn column(&self, name: &str) -> Result<&Column> {
        self.col_index(name)
            .map(|i| &self.cols[i])
            .ok_or_else(|| FrameError::UnknownColumn(name.to_string()))
    }

    /// Project a subset of columns (in the requested order) into a new frame.
    ///
    /// # Errors
    /// Returns [`FrameError::UnknownColumn`] if any requested name is missing.
    pub fn select(&self, names: &[&str]) -> Result<DataFrame> {
        let mut out_names = Vec::with_capacity(names.len());
        let mut out_cols = Vec::with_capacity(names.len());
        for &name in names
        {
            let col = self.column(name)?;
            out_names.push(name.to_string());
            out_cols.push(col.clone());
        }
        Ok(DataFrame {
            names: out_names,
            cols: out_cols,
        })
    }

    /// Keep the rows where `mask` is `true`.
    ///
    /// # Errors
    /// Returns [`FrameError::MaskLengthMismatch`] if `mask.len() != n_rows()`.
    pub fn filter(&self, mask: &[bool]) -> Result<DataFrame> {
        if mask.len() != self.n_rows()
        {
            return Err(FrameError::MaskLengthMismatch {
                expected: self.n_rows(),
                found: mask.len(),
            });
        }
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| keep.then_some(i))
            .collect();
        Ok(self.take_rows(&indices))
    }

    /// Take the first `k` rows (or all rows if `k` exceeds `n_rows()`).
    pub fn head(&self, k: usize) -> DataFrame {
        let k = k.min(self.n_rows());
        let indices: Vec<usize> = (0..k).collect();
        self.take_rows(&indices)
    }

    /// Stable-sort every column by the values of an `f64` key column.
    ///
    /// Ordering uses [`f64::total_cmp`]; `NaN` keys always sort to the end,
    /// regardless of `ascending`.
    ///
    /// # Errors
    /// Returns [`FrameError::UnknownColumn`] if `name` is missing, or
    /// [`FrameError::TypeMismatch`] if the column is not `F64`.
    pub fn sort_by_f64(&self, name: &str, ascending: bool) -> Result<DataFrame> {
        let col = self.column(name)?;
        let keys = match col
        {
            Column::F64(v) => v,
            other =>
            {
                return Err(FrameError::TypeMismatch {
                    expected: "f64".to_string(),
                    found: other.dtype().to_string(),
                });
            },
        };
        let mut indices: Vec<usize> = (0..self.n_rows()).collect();
        indices.sort_by(|&a, &b| {
            let (x, y) = (keys[a], keys[b]);
            match (x.is_nan(), y.is_nan())
            {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) =>
                {
                    if ascending
                    {
                        x.total_cmp(&y)
                    }
                    else
                    {
                        y.total_cmp(&x)
                    }
                },
            }
        });
        Ok(self.take_rows(&indices))
    }

    /// Group by a key column and reduce an `f64` value column with `agg`.
    ///
    /// The key column may be [`Column::Str`] or [`Column::I64`]; the value column
    /// must be [`Column::F64`]. The result is a two-column frame: the distinct keys
    /// in first-seen order, followed by the aggregate (named `"<value>_<agg>"`,
    /// always `F64`).
    ///
    /// # Errors
    /// Returns [`FrameError::UnknownColumn`] for a missing column, or
    /// [`FrameError::TypeMismatch`] if the key or value column has the wrong type.
    pub fn group_by_agg(&self, key: &str, value: &str, agg: Agg) -> Result<DataFrame> {
        let key_col = self.column(key)?;
        let value_col = self.column(value)?;
        let vals = match value_col
        {
            Column::F64(v) => v,
            other =>
            {
                return Err(FrameError::TypeMismatch {
                    expected: "f64".to_string(),
                    found: other.dtype().to_string(),
                });
            },
        };
        let out_value = format!("{value}_{}", agg.label());
        match key_col
        {
            Column::Str(keys) =>
            {
                let (order, out) = group_reduce(keys, vals, agg);
                DataFrame::new()
                    .with_column(key, Column::Str(order))?
                    .with_column(&out_value, Column::F64(out))
            },
            Column::I64(keys) =>
            {
                let (order, out) = group_reduce(keys, vals, agg);
                DataFrame::new()
                    .with_column(key, Column::I64(order))?
                    .with_column(&out_value, Column::F64(out))
            },
            other => Err(FrameError::TypeMismatch {
                expected: "str or i64".to_string(),
                found: other.dtype().to_string(),
            }),
        }
    }

    /// Inner-join with `other` on a shared key column.
    ///
    /// The key column must be present in both frames and be [`Column::Str`] or
    /// [`Column::I64`] on both sides. Output columns are the left frame's columns
    /// followed by the right frame's columns except the join key; a right column
    /// whose name collides with a left column is renamed with a `_right` suffix.
    /// Rows follow left order, then right order within each matched key (so a key
    /// with multiplicity produces the cartesian product of the two groups).
    ///
    /// # Errors
    /// Returns [`FrameError::UnknownColumn`] if `on` is absent from either frame,
    /// or [`FrameError::TypeMismatch`] if the key columns are not matching
    /// `Str`/`I64` types.
    pub fn inner_join(&self, other: &DataFrame, on: &str) -> Result<DataFrame> {
        let left_key = self.column(on)?;
        let right_key = other.column(on)?;
        let (left_idx, right_idx) = match (left_key, right_key)
        {
            (Column::Str(lk), Column::Str(rk)) => join_pairs(lk, rk),
            (Column::I64(lk), Column::I64(rk)) => join_pairs(lk, rk),
            (l, r) =>
            {
                return Err(FrameError::TypeMismatch {
                    expected: format!("matching str/i64 key, left is {}", l.dtype()),
                    found: r.dtype().to_string(),
                });
            },
        };

        let mut names = Vec::with_capacity(self.n_cols() + other.n_cols());
        let mut cols = Vec::with_capacity(self.n_cols() + other.n_cols());
        for (name, col) in self.names.iter().zip(&self.cols)
        {
            names.push(name.clone());
            cols.push(col.take(&left_idx));
        }
        for (name, col) in other.names.iter().zip(&other.cols)
        {
            if name.as_str() == on
            {
                continue;
            }
            let out_name = if self.col_index(name).is_some()
            {
                format!("{name}_right")
            }
            else
            {
                name.clone()
            };
            names.push(out_name);
            cols.push(col.take(&right_idx));
        }
        Ok(DataFrame { names, cols })
    }

    /// Serialize the frame to an RFC-4180 CSV string.
    ///
    /// The first line is the header of column names; one line follows per row.
    /// Fields containing a comma, double quote, `\r` or `\n` are wrapped in double
    /// quotes with internal quotes doubled. Booleans render as `true`/`false`.
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        let header: Vec<String> = self.names.iter().map(|n| csv_quote(n)).collect();
        out.push_str(&header.join(","));
        for row in 0..self.n_rows()
        {
            out.push('\n');
            let fields: Vec<String> = self
                .cols
                .iter()
                .map(|c| csv_quote(&cell_to_string(c, row)))
                .collect();
            out.push_str(&fields.join(","));
        }
        out
    }

    /// Parse a CSV string into a frame using an explicit column schema.
    ///
    /// The first record is treated as a header and skipped; the number of columns
    /// is taken from `schema`, and each remaining record supplies one row. Each
    /// field is parsed according to its [`DType`].
    ///
    /// # Errors
    /// Returns [`FrameError::CsvParse`] for malformed CSV or an unparseable field,
    /// [`FrameError::ColumnCountMismatch`] if a record's field count differs from
    /// the schema, or [`FrameError::DuplicateColumn`] if `schema` repeats a name.
    pub fn from_csv(text: &str, schema: &[(&str, DType)]) -> Result<DataFrame> {
        let records = parse_csv_records(text)?;
        let ncol = schema.len();
        let header = records
            .first()
            .ok_or_else(|| FrameError::CsvParse("empty CSV input".to_string()))?;
        if header.len() != ncol
        {
            return Err(FrameError::ColumnCountMismatch {
                expected: ncol,
                found: header.len(),
            });
        }
        let data = &records[1..];
        for rec in data
        {
            if rec.len() != ncol
            {
                return Err(FrameError::ColumnCountMismatch {
                    expected: ncol,
                    found: rec.len(),
                });
            }
        }

        let mut df = DataFrame::new();
        for (j, (name, dtype)) in schema.iter().enumerate()
        {
            let col = match dtype
            {
                DType::F64 =>
                {
                    let mut v = Vec::with_capacity(data.len());
                    for rec in data
                    {
                        let field = rec[j].trim();
                        let parsed = field.parse::<f64>().map_err(|_| {
                            FrameError::CsvParse(format!("cannot parse '{field}' as f64"))
                        })?;
                        v.push(parsed);
                    }
                    Column::F64(v)
                },
                DType::I64 =>
                {
                    let mut v = Vec::with_capacity(data.len());
                    for rec in data
                    {
                        let field = rec[j].trim();
                        let parsed = field.parse::<i64>().map_err(|_| {
                            FrameError::CsvParse(format!("cannot parse '{field}' as i64"))
                        })?;
                        v.push(parsed);
                    }
                    Column::I64(v)
                },
                DType::Bool =>
                {
                    let mut v = Vec::with_capacity(data.len());
                    for rec in data
                    {
                        let field = rec[j].trim();
                        let parsed = match field
                        {
                            "true" => true,
                            "false" => false,
                            other =>
                            {
                                return Err(FrameError::CsvParse(format!(
                                    "cannot parse '{other}' as bool"
                                )));
                            },
                        };
                        v.push(parsed);
                    }
                    Column::Bool(v)
                },
                DType::Str => Column::Str(data.iter().map(|rec| rec[j].clone()).collect()),
            };
            df = df.with_column(name, col)?;
        }
        Ok(df)
    }

    /// Index of a column by name, if present.
    fn col_index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n.as_str() == name)
    }

    /// Build a new frame from the rows at `indices` (applied to every column).
    fn take_rows(&self, indices: &[usize]) -> DataFrame {
        DataFrame {
            names: self.names.clone(),
            cols: self.cols.iter().map(|c| c.take(indices)).collect(),
        }
    }
}

/// Group `keys`/`vals` in first-seen key order, reducing each group with `agg`.
fn group_reduce<K: Eq + Hash + Clone>(keys: &[K], vals: &[f64], agg: Agg) -> (Vec<K>, Vec<f64>) {
    let mut order: Vec<K> = Vec::new();
    let mut index: HashMap<K, usize> = HashMap::new();
    let mut groups: Vec<Vec<f64>> = Vec::new();
    for (i, k) in keys.iter().enumerate()
    {
        match index.get(k)
        {
            Some(&g) => groups[g].push(vals[i]),
            None =>
            {
                let g = order.len();
                index.insert(k.clone(), g);
                order.push(k.clone());
                groups.push(vec![vals[i]]);
            },
        }
    }
    let out: Vec<f64> = groups.iter().map(|g| aggregate(g, agg)).collect();
    (order, out)
}

/// Reduce a non-empty slice of values with the given aggregation.
fn aggregate(vals: &[f64], agg: Agg) -> f64 {
    match agg
    {
        Agg::Sum => vals.iter().sum(),
        Agg::Mean =>
        {
            if vals.is_empty()
            {
                0.0
            }
            else
            {
                vals.iter().sum::<f64>() / vals.len() as f64
            }
        },
        Agg::Count => vals.len() as f64,
        Agg::Min => vals.iter().copied().fold(f64::INFINITY, f64::min),
        Agg::Max => vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

/// Compute matched `(left_index, right_index)` pairs for an inner join.
///
/// Left order is the outer loop; within a matched key, right rows follow their
/// original order.
fn join_pairs<K: Eq + Hash + Clone>(lk: &[K], rk: &[K]) -> (Vec<usize>, Vec<usize>) {
    let mut map: HashMap<K, Vec<usize>> = HashMap::new();
    for (j, k) in rk.iter().enumerate()
    {
        map.entry(k.clone()).or_default().push(j);
    }
    let mut left_idx = Vec::new();
    let mut right_idx = Vec::new();
    for (i, k) in lk.iter().enumerate()
    {
        if let Some(rows) = map.get(k)
        {
            for &j in rows
            {
                left_idx.push(i);
                right_idx.push(j);
            }
        }
    }
    (left_idx, right_idx)
}

/// Render one cell of a column as a plain (unquoted) string.
fn cell_to_string(col: &Column, row: usize) -> String {
    match col
    {
        Column::F64(v) => v[row].to_string(),
        Column::I64(v) => v[row].to_string(),
        Column::Str(v) => v[row].clone(),
        Column::Bool(v) => if v[row] { "true" } else { "false" }.to_string(),
    }
}

/// Quote a CSV field if it contains a comma, quote, or newline.
fn csv_quote(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r')
    {
        format!("\"{}\"", field.replace('"', "\"\""))
    }
    else
    {
        field.to_string()
    }
}

/// Parse an RFC-4180 CSV document into records of string fields.
///
/// Supports quoted fields with embedded commas, doubled `""` quotes, and
/// `\n` / `\r\n` line endings. Returns [`FrameError::CsvParse`] on an
/// unterminated quoted field.
fn parse_csv_records(text: &str) -> Result<Vec<Vec<String>>> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut record: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut started = false;
    let mut i = 0;

    while i < len
    {
        let c = chars[i];
        if in_quotes
        {
            if c == '"'
            {
                if i + 1 < len && chars[i + 1] == '"'
                {
                    field.push('"');
                    i += 2;
                }
                else
                {
                    in_quotes = false;
                    i += 1;
                }
            }
            else
            {
                field.push(c);
                i += 1;
            }
        }
        else
        {
            match c
            {
                '"' =>
                {
                    in_quotes = true;
                    started = true;
                    i += 1;
                },
                ',' =>
                {
                    record.push(std::mem::take(&mut field));
                    started = true;
                    i += 1;
                },
                '\n' =>
                {
                    record.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut record));
                    started = false;
                    i += 1;
                },
                '\r' =>
                {
                    record.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut record));
                    started = false;
                    i += if i + 1 < len && chars[i + 1] == '\n'
                    {
                        2
                    }
                    else
                    {
                        1
                    };
                },
                _ =>
                {
                    field.push(c);
                    started = true;
                    i += 1;
                },
            }
        }
    }

    if in_quotes
    {
        return Err(FrameError::CsvParse(
            "unterminated quoted field".to_string(),
        ));
    }
    if started || !field.is_empty() || !record.is_empty()
    {
        record.push(field);
        records.push(record);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> DataFrame {
        DataFrame::new()
            .with_column("a", Column::I64(vec![1, 2, 3]))
            .unwrap()
            .with_column("b", Column::F64(vec![10.0, 20.0, 30.0]))
            .unwrap()
            .with_column("c", Column::Str(vec!["x".into(), "y".into(), "z".into()]))
            .unwrap()
    }

    fn f64_col(df: &DataFrame, name: &str) -> Vec<f64> {
        match df.column(name).unwrap()
        {
            Column::F64(v) => v.clone(),
            _ => panic!("expected f64 column"),
        }
    }

    fn i64_col(df: &DataFrame, name: &str) -> Vec<i64> {
        match df.column(name).unwrap()
        {
            Column::I64(v) => v.clone(),
            _ => panic!("expected i64 column"),
        }
    }

    fn str_col(df: &DataFrame, name: &str) -> Vec<String> {
        match df.column(name).unwrap()
        {
            Column::Str(v) => v.clone(),
            _ => panic!("expected str column"),
        }
    }

    #[test]
    fn column_len_and_dtype() {
        let c = Column::F64(vec![1.0, 2.0]);
        assert_eq!(c.len(), 2);
        assert!(!c.is_empty());
        assert_eq!(c.dtype(), "f64");
        assert_eq!(Column::Str(vec![]).dtype(), "str");
        assert!(Column::I64(vec![]).is_empty());
    }

    #[test]
    fn build_shape() {
        let df = sample();
        assert_eq!(df.n_rows(), 3);
        assert_eq!(df.n_cols(), 3);
        assert_eq!(df.column_names(), vec!["a", "b", "c"]);
        assert_eq!(DataFrame::new().n_rows(), 0);
        assert_eq!(DataFrame::default().n_cols(), 0);
    }

    #[test]
    fn select_reorders_and_projects() {
        let df = sample();
        let s = df.select(&["c", "a"]).unwrap();
        assert_eq!(s.column_names(), vec!["c", "a"]);
        assert_eq!(str_col(&s, "c"), vec!["x", "y", "z"]);
        assert_eq!(i64_col(&s, "a"), vec![1, 2, 3]);
    }

    #[test]
    fn head_truncates() {
        let df = sample();
        let h = df.head(2);
        assert_eq!(h.n_rows(), 2);
        assert_eq!(i64_col(&h, "a"), vec![1, 2]);
        // k beyond n_rows keeps everything.
        assert_eq!(df.head(100).n_rows(), 3);
    }

    #[test]
    fn filter_keeps_masked_rows() {
        let df = sample();
        let f = df.filter(&[true, false, true]).unwrap();
        assert_eq!(f.n_rows(), 2);
        assert_eq!(i64_col(&f, "a"), vec![1, 3]);
        assert_eq!(f64_col(&f, "b"), vec![10.0, 30.0]);
        assert_eq!(str_col(&f, "c"), vec!["x", "z"]);
    }

    #[test]
    fn sort_by_f64_ascending_descending_and_nan_last() {
        let df = DataFrame::new()
            .with_column("f", Column::F64(vec![3.0, 1.0, f64::NAN, 2.0]))
            .unwrap()
            .with_column(
                "tag",
                Column::Str(vec!["c".into(), "a".into(), "x".into(), "b".into()]),
            )
            .unwrap();

        let asc = df.sort_by_f64("f", true).unwrap();
        assert_eq!(str_col(&asc, "tag"), vec!["a", "b", "c", "x"]);
        let fa = f64_col(&asc, "f");
        assert_eq!(&fa[..3], &[1.0, 2.0, 3.0]);
        assert!(fa[3].is_nan());

        let desc = df.sort_by_f64("f", false).unwrap();
        assert_eq!(str_col(&desc, "tag"), vec!["c", "b", "a", "x"]);
        let fd = f64_col(&desc, "f");
        assert_eq!(&fd[..3], &[3.0, 2.0, 1.0]);
        assert!(fd[3].is_nan());
    }

    #[test]
    fn group_by_agg_all_functions() {
        // Groups first-seen: "a" -> [1,2,3], "b" -> [10,20].
        let df = DataFrame::new()
            .with_column(
                "k",
                Column::Str(vec![
                    "a".into(),
                    "b".into(),
                    "a".into(),
                    "a".into(),
                    "b".into(),
                ]),
            )
            .unwrap()
            .with_column("v", Column::F64(vec![1.0, 10.0, 2.0, 3.0, 20.0]))
            .unwrap();

        let sum = df.group_by_agg("k", "v", Agg::Sum).unwrap();
        assert_eq!(sum.column_names(), vec!["k", "v_sum"]);
        assert_eq!(str_col(&sum, "k"), vec!["a", "b"]);
        assert_eq!(f64_col(&sum, "v_sum"), vec![6.0, 30.0]);

        assert_eq!(
            f64_col(&df.group_by_agg("k", "v", Agg::Mean).unwrap(), "v_mean"),
            vec![2.0, 15.0]
        );
        assert_eq!(
            f64_col(&df.group_by_agg("k", "v", Agg::Count).unwrap(), "v_count"),
            vec![3.0, 2.0]
        );
        assert_eq!(
            f64_col(&df.group_by_agg("k", "v", Agg::Min).unwrap(), "v_min"),
            vec![1.0, 10.0]
        );
        assert_eq!(
            f64_col(&df.group_by_agg("k", "v", Agg::Max).unwrap(), "v_max"),
            vec![3.0, 20.0]
        );
    }

    #[test]
    fn group_by_agg_i64_keys() {
        let df = DataFrame::new()
            .with_column("k", Column::I64(vec![7, 7, 9]))
            .unwrap()
            .with_column("v", Column::F64(vec![1.0, 2.0, 5.0]))
            .unwrap();
        let g = df.group_by_agg("k", "v", Agg::Sum).unwrap();
        assert_eq!(i64_col(&g, "k"), vec![7, 9]);
        assert_eq!(f64_col(&g, "v_sum"), vec![3.0, 5.0]);
    }

    #[test]
    fn inner_join_cartesian_within_group_and_rename() {
        // Right key 2 has multiplicity 2 -> cartesian; left "name" collides.
        let left = DataFrame::new()
            .with_column("id", Column::I64(vec![1, 2]))
            .unwrap()
            .with_column("name", Column::Str(vec!["A".into(), "B".into()]))
            .unwrap();
        let right = DataFrame::new()
            .with_column("id", Column::I64(vec![2, 2, 4]))
            .unwrap()
            .with_column(
                "name",
                Column::Str(vec!["R2a".into(), "R2b".into(), "R4".into()]),
            )
            .unwrap()
            .with_column("score", Column::F64(vec![100.0, 101.0, 200.0]))
            .unwrap();

        let j = left.inner_join(&right, "id").unwrap();
        assert_eq!(j.column_names(), vec!["id", "name", "name_right", "score"]);
        assert_eq!(j.n_rows(), 2);
        assert_eq!(i64_col(&j, "id"), vec![2, 2]);
        assert_eq!(str_col(&j, "name"), vec!["B", "B"]);
        assert_eq!(str_col(&j, "name_right"), vec!["R2a", "R2b"]);
        assert_eq!(f64_col(&j, "score"), vec![100.0, 101.0]);
    }

    #[test]
    fn inner_join_str_keys_no_match_rows_dropped() {
        let left = DataFrame::new()
            .with_column("k", Column::Str(vec!["p".into(), "q".into(), "r".into()]))
            .unwrap()
            .with_column("lv", Column::I64(vec![1, 2, 3]))
            .unwrap();
        let right = DataFrame::new()
            .with_column("k", Column::Str(vec!["q".into(), "z".into()]))
            .unwrap()
            .with_column("rv", Column::I64(vec![20, 99]))
            .unwrap();
        let j = left.inner_join(&right, "k").unwrap();
        assert_eq!(j.column_names(), vec!["k", "lv", "rv"]);
        assert_eq!(str_col(&j, "k"), vec!["q"]);
        assert_eq!(i64_col(&j, "lv"), vec![2]);
        assert_eq!(i64_col(&j, "rv"), vec![20]);
    }

    #[test]
    fn csv_round_trip_with_quoting() {
        let df = DataFrame::new()
            .with_column(
                "name",
                Column::Str(vec!["hi, there".into(), "plain".into(), "q\"x".into()]),
            )
            .unwrap()
            .with_column("n", Column::I64(vec![1, 2, 3]))
            .unwrap()
            .with_column("f", Column::F64(vec![1.5, 2.0, -3.25]))
            .unwrap()
            .with_column("b", Column::Bool(vec![true, false, true]))
            .unwrap();

        let csv = df.to_csv();
        // The comma-containing field must be quoted.
        assert!(csv.contains("\"hi, there\""));
        // Internal quotes doubled.
        assert!(csv.contains("\"q\"\"x\""));

        let schema = [
            ("name", DType::Str),
            ("n", DType::I64),
            ("f", DType::F64),
            ("b", DType::Bool),
        ];
        let back = DataFrame::from_csv(&csv, &schema).unwrap();
        assert_eq!(back, df);
    }

    #[test]
    fn csv_errors() {
        // Data row has too many fields.
        let e = DataFrame::from_csv("a,b\n1,2,3", &[("a", DType::I64), ("b", DType::I64)]);
        assert!(matches!(e, Err(FrameError::ColumnCountMismatch { .. })));

        // Header field count mismatch.
        let e = DataFrame::from_csv("a,b,c\n1,2", &[("a", DType::I64), ("b", DType::I64)]);
        assert!(matches!(e, Err(FrameError::ColumnCountMismatch { .. })));

        // Unparseable number.
        let e = DataFrame::from_csv("x\nabc", &[("x", DType::F64)]);
        assert!(matches!(e, Err(FrameError::CsvParse(_))));

        // Unterminated quote.
        let e = DataFrame::from_csv("x\n\"oops", &[("x", DType::Str)]);
        assert!(matches!(e, Err(FrameError::CsvParse(_))));

        // Empty input.
        let e = DataFrame::from_csv("", &[("x", DType::Str)]);
        assert!(matches!(e, Err(FrameError::CsvParse(_))));
    }

    #[test]
    fn error_paths() {
        // Duplicate column.
        let e = DataFrame::new()
            .with_column("a", Column::I64(vec![1]))
            .unwrap()
            .with_column("a", Column::I64(vec![2]));
        assert!(matches!(e, Err(FrameError::DuplicateColumn(_))));

        // Length mismatch.
        let e = DataFrame::new()
            .with_column("a", Column::I64(vec![1, 2]))
            .unwrap()
            .with_column("b", Column::I64(vec![1, 2, 3]));
        assert!(matches!(e, Err(FrameError::LengthMismatch { .. })));

        let df = sample();
        // Unknown column.
        assert!(matches!(
            df.column("nope"),
            Err(FrameError::UnknownColumn(_))
        ));
        assert!(matches!(
            df.select(&["nope"]),
            Err(FrameError::UnknownColumn(_))
        ));

        // Mask length mismatch.
        assert!(matches!(
            df.filter(&[true, false]),
            Err(FrameError::MaskLengthMismatch { .. })
        ));

        // Sort on non-f64.
        assert!(matches!(
            df.sort_by_f64("a", true),
            Err(FrameError::TypeMismatch { .. })
        ));

        // group_by with non-f64 value column.
        assert!(matches!(
            df.group_by_agg("c", "a", Agg::Sum),
            Err(FrameError::TypeMismatch { .. })
        ));

        // Join on absent key.
        let other = DataFrame::new()
            .with_column("z", Column::I64(vec![1, 2, 3]))
            .unwrap();
        assert!(matches!(
            df.inner_join(&other, "missing"),
            Err(FrameError::UnknownColumn(_))
        ));
    }

    #[test]
    fn error_display_is_nonempty() {
        let e = FrameError::UnknownColumn("foo".to_string());
        assert!(e.to_string().contains("foo"));
        let _: &dyn std::error::Error = &e;
    }
}
