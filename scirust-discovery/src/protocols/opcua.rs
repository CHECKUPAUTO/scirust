//! Sonde OPC-UA : échange minimal « Hello »/« Acknowledge » du protocole de
//! connexion UACP (OPC UA Part 6 §7.1) — la toute première chose qu'échange
//! un client OPC-UA avant même l'ouverture d'un canal sécurisé ou d'une
//! session. Plus léger qu'un `FindServers` complet, et suffisant pour
//! confirmer sans ambiguïté la présence d'un point de terminaison OPC-UA :
//! un serveur qui répond par un « Acknowledge » bien formé est un serveur
//! OPC-UA, point final.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

pub const DEFAULT_PORT: u16 = 4840;

/// Construit le message « Hello » UACP pour l'URL de point de terminaison
/// donnée (ex. `"opc.tcp://192.168.1.10:4840"`).
pub fn build_hello(endpoint_url: &str) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_le_bytes()); // protocolVersion
    body.extend_from_slice(&65536u32.to_le_bytes()); // receiveBufferSize
    body.extend_from_slice(&65536u32.to_le_bytes()); // sendBufferSize
    body.extend_from_slice(&0u32.to_le_bytes()); // maxMessageSize (0 = pas de limite annoncée)
    body.extend_from_slice(&0u32.to_le_bytes()); // maxChunkCount
    let url_bytes = endpoint_url.as_bytes();
    body.extend_from_slice(&(url_bytes.len() as i32).to_le_bytes());
    body.extend_from_slice(url_bytes);

    let mut msg = Vec::with_capacity(8 + body.len());
    msg.extend_from_slice(b"HELF"); // MessageType "HEL" + IsFinal 'F'
    let total_len = 8 + body.len() as u32;
    msg.extend_from_slice(&total_len.to_le_bytes());
    msg.extend_from_slice(&body);
    msg
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Acknowledge {
    pub protocol_version: u32,
    pub receive_buffer_size: u32,
    pub send_buffer_size: u32,
    pub max_message_size: u32,
    pub max_chunk_count: u32,
}

/// Analyse un message « Acknowledge » UACP. `None` si l'en-tête ne
/// correspond pas — endpoint qui n'est pas de l'OPC-UA, ou trame tronquée.
pub fn parse_acknowledge(buf: &[u8]) -> Option<Acknowledge> {
    if buf.len() < 8 || buf[0..3] != b"ACK"[..]
    {
        return None;
    }
    let total_len = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;
    if total_len < 28 || buf.len() < total_len
    {
        return None;
    }
    let body = &buf[8..total_len];
    Some(Acknowledge {
        protocol_version: u32::from_le_bytes(body[0..4].try_into().ok()?),
        receive_buffer_size: u32::from_le_bytes(body[4..8].try_into().ok()?),
        send_buffer_size: u32::from_le_bytes(body[8..12].try_into().ok()?),
        max_message_size: u32::from_le_bytes(body[12..16].try_into().ok()?),
        max_chunk_count: u32::from_le_bytes(body[16..20].try_into().ok()?),
    })
}

/// Envoie un « Hello » et attend l'« Acknowledge » — la seule sonde active
/// de ce module, sur une connexion TCP déjà autorisée par l'appelant
/// (`crate::scope`).
pub fn probe(
    addr: SocketAddr,
    endpoint_url: &str,
    timeout: Duration,
) -> Result<Acknowledge, String> {
    let mut stream = TcpStream::connect_timeout(&addr, timeout).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    stream
        .write_all(&build_hello(endpoint_url))
        .map_err(|e| e.to_string())?;
    let mut buf = [0u8; 512];
    let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
    parse_acknowledge(&buf[..n])
        .ok_or_else(|| "not an OPC-UA endpoint (no valid Acknowledge)".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn hello_message_has_correct_uacp_framing() {
        let hello = build_hello("opc.tcp://x:4840");
        assert_eq!(&hello[0..4], b"HELF");
        let total_len = u32::from_le_bytes(hello[4..8].try_into().unwrap()) as usize;
        assert_eq!(total_len, hello.len());
        // URL length-prefixed at the tail of the body.
        let url_len = i32::from_le_bytes(hello[28..32].try_into().unwrap());
        assert_eq!(url_len as usize, "opc.tcp://x:4840".len());
        assert_eq!(&hello[32..], b"opc.tcp://x:4840");
    }

    #[test]
    fn parse_acknowledge_roundtrip() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"ACKF");
        buf.extend_from_slice(&28u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&8192u32.to_le_bytes());
        buf.extend_from_slice(&8192u32.to_le_bytes());
        buf.extend_from_slice(&65536u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        let ack = parse_acknowledge(&buf).unwrap();
        assert_eq!(ack.receive_buffer_size, 8192);
        assert_eq!(ack.max_message_size, 65536);
    }

    #[test]
    fn parse_acknowledge_rejects_wrong_magic() {
        let buf = [b'E', b'R', b'R', b'F', 0, 0, 0, 0];
        assert!(parse_acknowledge(&buf).is_none());
    }

    #[test]
    fn parse_acknowledge_rejects_truncated_frame() {
        assert!(parse_acknowledge(b"ACKF").is_none());
    }

    #[test]
    fn probe_against_local_listener_returns_parsed_acknowledge() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut request = [0u8; 512];
            let n = socket.read(&mut request).unwrap();
            assert_eq!(&request[0..4], b"HELF");

            let mut ack = Vec::new();
            ack.extend_from_slice(b"ACKF");
            ack.extend_from_slice(&28u32.to_le_bytes());
            ack.extend_from_slice(&0u32.to_le_bytes());
            ack.extend_from_slice(&8192u32.to_le_bytes());
            ack.extend_from_slice(&8192u32.to_le_bytes());
            ack.extend_from_slice(&65536u32.to_le_bytes());
            ack.extend_from_slice(&0u32.to_le_bytes());
            socket.write_all(&ack).unwrap();
            n
        });
        let result = probe(addr, "opc.tcp://test", Duration::from_secs(2)).unwrap();
        assert_eq!(result.receive_buffer_size, 8192);
        handle.join().unwrap();
    }

    #[test]
    fn probe_rejects_non_opcua_server() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut buf = [0u8; 64];
            let _ = socket.read(&mut buf);
            socket
                .write_all(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                .unwrap();
        });
        let result = probe(addr, "opc.tcp://test", Duration::from_secs(2));
        assert!(result.is_err());
        handle.join().unwrap();
    }
}
