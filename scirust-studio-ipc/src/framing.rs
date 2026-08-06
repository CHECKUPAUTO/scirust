//! Reading and writing protocol messages over a byte stream.
//!
//! One message is one line of JSON terminated by `\n` (NDJSON). That framing
//! is safe because `serde_json` escapes every control character inside
//! strings, so an encoded message can never contain a raw newline — a
//! property this module tests directly rather than assuming.
//!
//! Every read is bounded. A peer that never sends a newline, or that sends a
//! gigabyte-long line, must not be able to drive the other side into
//! unbounded allocation, so [`read_message`] takes a hard byte limit and
//! fails with [`IpcError::MessageTooLarge`] the moment it is exceeded rather
//! than after the fact.

use std::io::{BufRead, Write};

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::IpcError;

/// The default maximum encoded size of a single protocol message.
///
/// A result carrying long time-courses is genuinely large — a 250 000-step
/// run with four series is a few tens of megabytes of JSON — so this is
/// generous. It exists to bound a *runaway or hostile* peer, not to
/// second-guess a legitimate result. A caller that needs a different bound
/// passes its own to [`read_message`].
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// Encode `message` as one JSON line and write it, flushing so the peer sees
/// it immediately.
///
/// Flushing per message is what makes streamed progress actually stream: a
/// buffered writer that only flushed when full would deliver a run's
/// progress events in one burst after the run had already finished.
pub fn write_message<W, M>(writer: &mut W, message: &M) -> Result<(), IpcError>
where
    W: Write,
    M: Serialize,
{
    let encoded = serde_json::to_string(message).map_err(IpcError::Encode)?;
    debug_assert!(
        !encoded.contains('\n'),
        "serde_json must escape newlines; NDJSON framing depends on it"
    );
    writer.write_all(encoded.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

/// Read one JSON line and decode it.
///
/// Returns `Ok(None)` at a clean end of stream — the peer closed its side
/// between messages, which is a normal shutdown, not an error. A partial
/// line followed by EOF *is* an error: it means the peer died mid-message.
pub fn read_message<R, M>(reader: &mut R, max_bytes: usize) -> Result<Option<M>, IpcError>
where
    R: BufRead,
    M: DeserializeOwned,
{
    let mut line = Vec::new();
    // Read at most one byte past the limit, so exceeding it is detectable
    // without ever buffering more than the limit plus that one byte.
    let mut limited = std::io::Read::take(&mut *reader, max_bytes as u64 + 1);
    let read = limited.read_until(b'\n', &mut line)?;

    if read == 0
    {
        return Ok(None);
    }
    if line.len() > max_bytes
    {
        return Err(IpcError::MessageTooLarge { limit: max_bytes });
    }
    if !line.ends_with(b"\n")
    {
        return Err(IpcError::Truncated);
    }
    line.pop();
    // Tolerate a trailing CR so a stream that crossed a Windows text-mode
    // boundary still decodes.
    if line.ends_with(b"\r")
    {
        line.pop();
    }
    if line.is_empty()
    {
        // A blank line carries no message; skip it rather than reporting a
        // parse failure for something that is merely noise.
        return read_message(reader, max_bytes);
    }

    let message = serde_json::from_slice(&line)
        .map_err(|e| IpcError::Malformed(format!("{e} (in {} bytes)", line.len())))?;
    Ok(Some(message))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Sample {
        id: u64,
        text: String,
    }

    fn sample() -> Sample {
        Sample {
            id: 7,
            text: "hello".to_string(),
        }
    }

    #[test]
    fn round_trips_a_message() {
        let mut buf = Vec::new();
        write_message(&mut buf, &sample()).unwrap();
        let mut cursor = buf.as_slice();
        let decoded: Option<Sample> = read_message(&mut cursor, DEFAULT_MAX_MESSAGE_BYTES).unwrap();
        assert_eq!(decoded, Some(sample()));
    }

    #[test]
    fn round_trips_several_messages_in_order() {
        let mut buf = Vec::new();
        for id in 0..5
        {
            write_message(
                &mut buf,
                &Sample {
                    id,
                    text: format!("m{id}"),
                },
            )
            .unwrap();
        }
        let mut cursor = buf.as_slice();
        for id in 0..5
        {
            let decoded: Sample = read_message(&mut cursor, DEFAULT_MAX_MESSAGE_BYTES)
                .unwrap()
                .unwrap();
            assert_eq!(decoded.id, id);
        }
        let end: Option<Sample> = read_message(&mut cursor, DEFAULT_MAX_MESSAGE_BYTES).unwrap();
        assert_eq!(end, None, "clean EOF must be None, not an error");
    }

    /// The framing's core assumption, tested rather than asserted in a
    /// comment: a payload full of newlines still occupies exactly one line.
    #[test]
    fn embedded_newlines_survive_because_json_escapes_them() {
        let nasty = Sample {
            id: 1,
            text: "line one\nline two\r\n{\"fake\":\"message\"}\n".to_string(),
        };
        let mut buf = Vec::new();
        write_message(&mut buf, &nasty).unwrap();
        assert_eq!(
            buf.iter().filter(|b| **b == b'\n').count(),
            1,
            "the encoded message must occupy exactly one line"
        );

        let mut cursor = buf.as_slice();
        let decoded: Sample = read_message(&mut cursor, DEFAULT_MAX_MESSAGE_BYTES)
            .unwrap()
            .unwrap();
        assert_eq!(decoded, nasty);
    }

    /// The protocol carries computed floating-point results, so encoding
    /// and decoding must be bit-exact — "visually the same number" is not
    /// good enough when the whole point of the surrounding system is
    /// reproducibility. Pins `serde_json`'s `float_roundtrip` feature.
    #[test]
    fn floating_point_payloads_round_trip_bit_for_bit() {
        #[derive(Debug, Serialize, Deserialize)]
        struct Floats {
            values: Vec<f64>,
        }

        let values = vec![
            0.998969729210804,
            0.9988752100232527,
            0.1 + 0.2,
            std::f64::consts::PI,
            f64::MIN_POSITIVE,
            f64::MAX,
            -0.0,
            1.0e-308,
        ];
        let mut buf = Vec::new();
        write_message(
            &mut buf,
            &Floats {
                values: values.clone(),
            },
        )
        .unwrap();

        let mut cursor = buf.as_slice();
        let decoded: Floats = read_message(&mut cursor, DEFAULT_MAX_MESSAGE_BYTES)
            .unwrap()
            .unwrap();

        for (a, b) in values.iter().zip(decoded.values.iter())
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{a:?} decoded as {b:?} — float parsing is not exact"
            );
        }
    }

    #[test]
    fn empty_input_is_a_clean_end_of_stream() {
        let mut cursor: &[u8] = b"";
        let decoded: Option<Sample> = read_message(&mut cursor, DEFAULT_MAX_MESSAGE_BYTES).unwrap();
        assert_eq!(decoded, None);
    }

    #[test]
    fn a_truncated_final_line_is_an_error_not_an_end_of_stream() {
        let mut cursor: &[u8] = b"{\"id\":1,\"text\":\"no newline\"}";
        let err = read_message::<_, Sample>(&mut cursor, DEFAULT_MAX_MESSAGE_BYTES).unwrap_err();
        assert!(matches!(err, IpcError::Truncated), "{err:?}");
    }

    #[test]
    fn malformed_json_is_reported_as_malformed() {
        let mut cursor: &[u8] = b"{not json}\n";
        let err = read_message::<_, Sample>(&mut cursor, DEFAULT_MAX_MESSAGE_BYTES).unwrap_err();
        assert!(matches!(err, IpcError::Malformed(_)), "{err:?}");
    }

    #[test]
    fn a_message_over_the_limit_is_rejected_before_it_is_decoded() {
        let big = Sample {
            id: 1,
            text: "x".repeat(4096),
        };
        let mut buf = Vec::new();
        write_message(&mut buf, &big).unwrap();

        let mut cursor = buf.as_slice();
        let err = read_message::<_, Sample>(&mut cursor, 64).unwrap_err();
        assert!(
            matches!(err, IpcError::MessageTooLarge { limit: 64 }),
            "{err:?}"
        );
    }

    #[test]
    fn a_line_exactly_at_the_limit_is_accepted() {
        let mut buf = Vec::new();
        write_message(&mut buf, &sample()).unwrap();
        // buf includes the trailing newline; the limit applies to the line
        // including it, which is what read_until produces.
        let limit = buf.len();
        let mut cursor = buf.as_slice();
        let decoded: Sample = read_message(&mut cursor, limit).unwrap().unwrap();
        assert_eq!(decoded, sample());
    }

    #[test]
    fn blank_lines_between_messages_are_skipped() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"\n\n");
        write_message(&mut buf, &sample()).unwrap();
        let mut cursor = buf.as_slice();
        let decoded: Sample = read_message(&mut cursor, DEFAULT_MAX_MESSAGE_BYTES)
            .unwrap()
            .unwrap();
        assert_eq!(decoded, sample());
    }

    #[test]
    fn a_carriage_return_before_the_newline_is_tolerated() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"{\"id\":3,\"text\":\"crlf\"}\r\n");
        let mut cursor = buf.as_slice();
        let decoded: Sample = read_message(&mut cursor, DEFAULT_MAX_MESSAGE_BYTES)
            .unwrap()
            .unwrap();
        assert_eq!(decoded.id, 3);
    }
}
