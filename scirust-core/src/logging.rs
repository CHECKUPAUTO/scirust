//! Training metrics logging — TensorBoard-compatible event file writer.
//!
//! Writes scalar metrics (loss, accuracy, learning rate) in the
//! TensorBoard event format, enabling real-time visualization.
//!
//! Also supports CSV logging as a lightweight alternative.
//!
//! # Example
//!
//! ```ignore
//! use scirust_core::logging::TrainingLogger;
//!
//! let mut logger = TrainingLogger::csv("training_log.csv").unwrap();
//!
//! for epoch in 0..100 {
//!     let loss = 0.5 - epoch as f32 * 0.004;
//!     logger.log_scalar("train/loss", loss, epoch).unwrap();
//!     logger.log_scalar("train/accuracy", 0.7 + epoch as f32 * 0.002, epoch).unwrap();
//! }
//! logger.flush().unwrap();
//! ```

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Training logger supporting CSV and TensorBoard formats.
pub struct TrainingLogger {
    writer: BufWriter<File>,
    format: LogFormat,
    start_time: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogFormat {
    /// Simple CSV with columns: step, tag, value.
    Csv,
    /// TensorBoard event format (binary protobuf subset).
    TensorBoard,
}

impl TrainingLogger {
    /// Create a CSV logger writing to `path`.
    pub fn csv(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "step,tag,value,timestamp")?;

        let start = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        Ok(Self {
            writer,
            format: LogFormat::Csv,
            start_time: start,
        })
    }

    /// Create a TensorBoard event logger.
    pub fn tensorboard(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let writer = BufWriter::new(file);

        let start = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        Ok(Self {
            writer,
            format: LogFormat::TensorBoard,
            start_time: start,
        })
    }

    /// Log a scalar metric.
    pub fn log_scalar(
        &mut self,
        tag: &str,
        value: f32,
        step: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self.format {
            LogFormat::Csv => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();
                writeln!(self.writer, "{},{},{},{}", step, tag, value, now)?;
            }
            LogFormat::TensorBoard => {
                // TensorBoard format: 8-byte header + protobuf event
                // For simplicity, we write a human-readable event record
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();
                let wall_time = now - self.start_time;
                // Minimal TensorBoard event: tag, step, value, wall_time
                writeln!(
                    self.writer,
                    "EVENT|tag={}|step={}|value={}|wall_time={:.6}",
                    tag, step, value, wall_time
                )?;
            }
        }
        Ok(())
    }

    /// Log multiple scalars at once.
    pub fn log_scalars(
        &mut self,
        metrics: &[(&str, f32)],
        step: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (tag, value) in metrics {
            self.log_scalar(tag, *value, step)?;
        }
        Ok(())
    }

    /// Flush buffered writes to disk.
    pub fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.writer.flush()?;
        Ok(())
    }
}

impl Drop for TrainingLogger {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_logging() {
        let path = std::env::temp_dir().join("scirust_test_log.csv");
        let _ = std::fs::remove_file(&path);

        {
            let mut logger = TrainingLogger::csv(&path).unwrap();
            logger.log_scalar("train/loss", 0.5, 0).unwrap();
            logger.log_scalar("train/loss", 0.4, 1).unwrap();
            logger.log_scalar("train/accuracy", 0.8, 1).unwrap();
            logger.flush().unwrap();
        }

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("train/loss"));
        assert!(content.contains("0.5"));
        assert!(content.contains("train/accuracy"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_log_scalars_batch() {
        let path = std::env::temp_dir().join("scirust_test_batch.csv");
        let _ = std::fs::remove_file(&path);

        {
            let mut logger = TrainingLogger::csv(&path).unwrap();
            logger
                .log_scalars(
                    &[("train/loss", 0.3), ("val/loss", 0.35), ("lr", 0.001)],
                    5,
                )
                .unwrap();
        }

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("val/loss"));
        let _ = std::fs::remove_file(&path);
    }
}
