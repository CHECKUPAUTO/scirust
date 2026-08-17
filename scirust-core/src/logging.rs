//! Training metrics logging — TensorBoard-compatible event file writer.
//!
//! > ⚠️ **Experimental / no consumers**: this module is not used by any
//! > crate in the workspace. The API may change or be removed; open an
//! > issue if you depend on it.
//!
//! Writes scalar metrics (loss, accuracy, learning rate) in the
//! TensorBoard event format, enabling real-time visualization.
//!
//! Also supports CSV logging as a lightweight alternative.
//!
//! # Example
//!
//! ```no_run
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

const TFRECORD_CRC_MASK_DELTA: u32 = 0xa282_ead8;
const CRC32C_POLY: u32 = 0x82f6_3b78;
const TENSORBOARD_FILE_VERSION: &str = "brain.Event:2";

/// Training logger supporting CSV and TensorBoard formats.
pub struct TrainingLogger {
    writer: BufWriter<File>,
    format: LogFormat,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogFormat {
    /// Simple CSV with columns: step, tag, value.
    Csv,
    /// TensorBoard event format: TFRecord framing containing protobuf `Event` messages.
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

        Ok(Self {
            writer,
            format: LogFormat::Csv,
        })
    }

    /// Create a TensorBoard event logger.
    ///
    /// The file uses TensorFlow's TFRecord framing and protobuf-compatible
    /// `Event`/`Summary.Value` messages. A `brain.Event:2` file-version event is
    /// emitted first, as expected by TensorBoard event readers.
    pub fn tensorboard(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);
        let event = encode_file_version_event(unix_time_seconds());
        write_tfrecord(&mut writer, &event)?;

        Ok(Self {
            writer,
            format: LogFormat::TensorBoard,
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
                writeln!(
                    self.writer,
                    "{},{},{},{}",
                    step,
                    tag,
                    value,
                    unix_time_seconds()
                )?;
            }
            LogFormat::TensorBoard => {
                let event = encode_scalar_event(unix_time_seconds(), step as u64, tag, value);
                write_tfrecord(&mut self.writer, &event)?;
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

fn unix_time_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (CRC32C_POLY & mask);
        }
    }
    !crc
}

fn masked_crc32c(bytes: &[u8]) -> u32 {
    let crc = crc32c(bytes);
    crc.rotate_right(15).wrapping_add(TFRECORD_CRC_MASK_DELTA)
}

fn write_tfrecord(writer: &mut impl Write, payload: &[u8]) -> std::io::Result<()> {
    let length = (payload.len() as u64).to_le_bytes();
    writer.write_all(&length)?;
    writer.write_all(&masked_crc32c(&length).to_le_bytes())?;
    writer.write_all(payload)?;
    writer.write_all(&masked_crc32c(payload).to_le_bytes())?;
    Ok(())
}

fn push_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn push_key(out: &mut Vec<u8>, field: u32, wire_type: u8) {
    push_varint(out, u64::from((field << 3) | u32::from(wire_type)));
}

fn push_len_delimited(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    push_key(out, field, 2);
    push_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn push_fixed64(out: &mut Vec<u8>, field: u32, value: u64) {
    push_key(out, field, 1);
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_fixed32(out: &mut Vec<u8>, field: u32, value: u32) {
    push_key(out, field, 5);
    out.extend_from_slice(&value.to_le_bytes());
}

fn encode_file_version_event(wall_time: f64) -> Vec<u8> {
    let mut event = Vec::new();
    push_fixed64(&mut event, 1, wall_time.to_bits());
    push_len_delimited(&mut event, 3, TENSORBOARD_FILE_VERSION.as_bytes());
    event
}

fn encode_scalar_event(wall_time: f64, step: u64, tag: &str, value: f32) -> Vec<u8> {
    let mut summary_value = Vec::new();
    push_len_delimited(&mut summary_value, 1, tag.as_bytes());
    push_fixed32(&mut summary_value, 2, value.to_bits());

    let mut summary = Vec::new();
    push_len_delimited(&mut summary, 1, &summary_value);

    let mut event = Vec::new();
    push_fixed64(&mut event, 1, wall_time.to_bits());
    push_key(&mut event, 2, 0);
    push_varint(&mut event, step);
    push_len_delimited(&mut event, 5, &summary);
    event
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_record(bytes: &[u8], offset: usize) -> (&[u8], usize) {
        let length_bytes: [u8; 8] = bytes[offset..offset + 8].try_into().unwrap();
        let length = u64::from_le_bytes(length_bytes) as usize;
        let length_crc = u32::from_le_bytes(bytes[offset + 8..offset + 12].try_into().unwrap());
        assert_eq!(length_crc, masked_crc32c(&length_bytes));

        let payload_start = offset + 12;
        let payload_end = payload_start + length;
        let payload = &bytes[payload_start..payload_end];
        let payload_crc = u32::from_le_bytes(bytes[payload_end..payload_end + 4].try_into().unwrap());
        assert_eq!(payload_crc, masked_crc32c(payload));
        (payload, payload_end + 4)
    }

    #[test]
    fn crc32c_matches_castagnoli_reference_vector() {
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
    }

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
                .log_scalars(&[("train/loss", 0.3), ("val/loss", 0.35), ("lr", 0.001)], 5)
                .unwrap();
        }

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("val/loss"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tensorboard_file_uses_valid_tfrecord_framing_and_events() {
        let path = std::env::temp_dir().join("scirust_test_events.tfevents");
        let _ = std::fs::remove_file(&path);

        {
            let mut logger = TrainingLogger::tensorboard(&path).unwrap();
            logger.log_scalar("train/loss", 0.25, 7).unwrap();
            logger.flush().unwrap();
        }

        let bytes = std::fs::read(&path).unwrap();
        let (version_event, next) = read_record(&bytes, 0);
        assert!(version_event
            .windows(TENSORBOARD_FILE_VERSION.len())
            .any(|window| window == TENSORBOARD_FILE_VERSION.as_bytes()));

        let (scalar_event, end) = read_record(&bytes, next);
        assert_eq!(end, bytes.len());
        assert!(scalar_event.windows(10).any(|window| window == b"train/loss"));
        assert!(scalar_event.contains(&0x2a)); // Event.summary, field 5 / wire type 2.
        assert!(scalar_event
            .windows(4)
            .any(|window| window == 0.25f32.to_bits().to_le_bytes()));

        let _ = std::fs::remove_file(&path);
    }
}
