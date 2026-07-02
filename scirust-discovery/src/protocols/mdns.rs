//! Sonde de découverte de services façon mDNS/DNS-SD (RFC 6762/6763) : une
//! requête DNS standard (RFC 1035) pour l'énumération de services
//! (`_services._dns-sd._udp.local`), envoyée en UDP — l'exact mécanisme
//! qu'utilise n'importe quel navigateur de réseau local (imprimantes,
//! partages de fichiers, et de plus en plus d'équipements industriels
//! IIoT), pas un balayage de ports.
//!
//! `probe` prend une adresse cible explicite plutôt que de coder en dur
//! l'adresse multicast standard (`224.0.0.251:5353`) : en production
//! l'appelant y pointera, mais cela permet aussi de tester la logique de
//! bout en bout sur une boucle locale unicast, sans dépendre du support
//! multicast de l'environnement d'exécution.

use std::net::{SocketAddr, UdpSocket};
use std::time::Duration;

pub const MULTICAST_ADDR: &str = "224.0.0.251:5353";
const QTYPE_PTR: u16 = 12;
const QCLASS_IN: u16 = 1;

/// Construit une requête DNS standard pour le nom donné (ex.
/// `"_services._dns-sd._udp.local"`), type PTR, classe IN.
pub fn build_query(name: &str) -> Vec<u8> {
    let mut msg = Vec::new();
    msg.extend_from_slice(&0u16.to_be_bytes()); // ID (0 par convention mDNS)
    msg.extend_from_slice(&0u16.to_be_bytes()); // flags: requête standard
    msg.extend_from_slice(&1u16.to_be_bytes()); // QDCOUNT
    msg.extend_from_slice(&0u16.to_be_bytes()); // ANCOUNT
    msg.extend_from_slice(&0u16.to_be_bytes()); // NSCOUNT
    msg.extend_from_slice(&0u16.to_be_bytes()); // ARCOUNT
    for label in name.trim_end_matches('.').split('.')
    {
        let bytes = label.as_bytes();
        msg.push(bytes.len() as u8);
        msg.extend_from_slice(bytes);
    }
    msg.push(0); // étiquette racine
    msg.extend_from_slice(&QTYPE_PTR.to_be_bytes());
    msg.extend_from_slice(&QCLASS_IN.to_be_bytes());
    msg
}

/// Décode un nom DNS à partir de `start`, en suivant les pointeurs de
/// compression (RFC 1035 §4.1.4). Renvoie le nom et l'offset juste après sa
/// représentation *dans le flux d'origine* (avant tout saut de pointeur).
fn read_name(buf: &[u8], start: usize) -> Result<(String, usize), String> {
    let mut labels = Vec::new();
    let mut pos = start;
    let mut end_pos = None;
    let mut hops = 0;
    loop
    {
        if pos >= buf.len()
        {
            return Err("name runs past end of buffer".to_string());
        }
        let len = buf[pos];
        if len == 0
        {
            if end_pos.is_none()
            {
                end_pos = Some(pos + 1);
            }
            break;
        }
        else if len & 0xC0 == 0xC0
        {
            if pos + 1 >= buf.len()
            {
                return Err("truncated compression pointer".to_string());
            }
            if end_pos.is_none()
            {
                end_pos = Some(pos + 2);
            }
            let ptr = (((len as usize) & 0x3F) << 8) | buf[pos + 1] as usize;
            hops += 1;
            if hops > 32
            {
                return Err("compression pointer loop".to_string());
            }
            pos = ptr;
        }
        else
        {
            let label_len = len as usize;
            if pos + 1 + label_len > buf.len()
            {
                return Err("label runs past end of buffer".to_string());
            }
            labels.push(String::from_utf8_lossy(&buf[pos + 1..pos + 1 + label_len]).to_string());
            pos += 1 + label_len;
        }
    }
    Ok((
        labels.join("."),
        end_pos.expect("loop always sets end_pos before breaking or errors out"),
    ))
}

/// Analyse une réponse DNS et renvoie les noms de service annoncés (les
/// cibles des enregistrements PTR de la section réponse).
pub fn parse_service_names(buf: &[u8]) -> Result<Vec<String>, String> {
    if buf.len() < 12
    {
        return Err("frame too short for DNS header".to_string());
    }
    let qdcount = u16::from_be_bytes([buf[4], buf[5]]) as usize;
    let ancount = u16::from_be_bytes([buf[6], buf[7]]) as usize;

    let mut offset = 12;
    for _ in 0..qdcount
    {
        let (_, next) = read_name(buf, offset)?;
        offset = next + 4; // + QTYPE(2) + QCLASS(2)
    }

    let mut names = Vec::new();
    for _ in 0..ancount
    {
        let (_, next) = read_name(buf, offset)?;
        offset = next;
        if offset + 10 > buf.len()
        {
            return Err("truncated resource record header".to_string());
        }
        let rtype = u16::from_be_bytes([buf[offset], buf[offset + 1]]);
        let rdlength = u16::from_be_bytes([buf[offset + 8], buf[offset + 9]]) as usize;
        offset += 10;
        if offset + rdlength > buf.len()
        {
            return Err("truncated RDATA".to_string());
        }
        if rtype == QTYPE_PTR
        {
            let (rdname, _) = read_name(buf, offset)?;
            names.push(rdname);
        }
        offset += rdlength;
    }
    Ok(names)
}

/// Envoie une requête d'énumération de services vers `target` (en
/// production, `MULTICAST_ADDR` ; en test, une adresse de boucle locale) et
/// renvoie les noms de service reçus dans la première réponse.
pub fn probe(target: SocketAddr, name: &str, timeout: Duration) -> Result<Vec<String>, String> {
    let socket = UdpSocket::bind("0.0.0.0:0").map_err(|e| e.to_string())?;
    socket
        .set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    socket
        .send_to(&build_query(name), target)
        .map_err(|e| e.to_string())?;
    let mut buf = [0u8; 2048];
    let n = socket.recv(&mut buf).map_err(|e| e.to_string())?;
    parse_service_names(&buf[..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn query_encodes_labels_and_qtype() {
        let q = build_query("_services._dns-sd._udp.local");
        assert_eq!(&q[4..6], &1u16.to_be_bytes()); // QDCOUNT
        assert_eq!(q[12], b"_services".len() as u8);
        assert_eq!(&q[13..22], b"_services");
        let tail = &q[q.len() - 4..];
        assert_eq!(&tail[0..2], &QTYPE_PTR.to_be_bytes());
        assert_eq!(&tail[2..4], &QCLASS_IN.to_be_bytes());
    }

    fn build_ptr_response(query: &[u8], service_names: &[&str]) -> Vec<u8> {
        let mut resp = query.to_vec();
        resp[6] = 0;
        resp[7] = service_names.len() as u8; // ANCOUNT low byte
        for name in service_names
        {
            for label in name.split('.')
            {
                resp.push(label.len() as u8);
                resp.extend_from_slice(label.as_bytes());
            }
            resp.push(0);
            resp.extend_from_slice(&QTYPE_PTR.to_be_bytes()); // TYPE
            resp.extend_from_slice(&QCLASS_IN.to_be_bytes()); // CLASS
            resp.extend_from_slice(&120u32.to_be_bytes()); // TTL
            let rdata: Vec<u8> = {
                let mut r = Vec::new();
                r.push(b"myservice".len() as u8);
                r.extend_from_slice(b"myservice");
                for label in name.split('.')
                {
                    r.push(label.len() as u8);
                    r.extend_from_slice(label.as_bytes());
                }
                r.push(0);
                r
            };
            resp.extend_from_slice(&(rdata.len() as u16).to_be_bytes());
            resp.extend_from_slice(&rdata);
        }
        resp
    }

    #[test]
    fn parse_service_names_extracts_ptr_targets() {
        let query = build_query("_services._dns-sd._udp.local");
        let response = build_ptr_response(&query, &["_opcua-tcp._tcp.local"]);
        let names = parse_service_names(&response).unwrap();
        assert_eq!(names, vec!["myservice._opcua-tcp._tcp.local"]);
    }

    #[test]
    fn parse_service_names_rejects_truncated_header() {
        assert!(parse_service_names(&[0u8; 4]).is_err());
    }

    #[test]
    fn probe_over_loopback_udp_returns_service_names() {
        let responder = UdpSocket::bind("127.0.0.1:0").unwrap();
        let responder_addr = responder.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let mut buf = [0u8; 512];
            let (n, from) = responder.recv_from(&mut buf).unwrap();
            let response = build_ptr_response(&buf[..n], &["_opcua-tcp._tcp.local"]);
            responder.send_to(&response, from).unwrap();
        });
        let names = probe(
            responder_addr,
            "_services._dns-sd._udp.local",
            Duration::from_secs(2),
        )
        .unwrap();
        assert_eq!(names.len(), 1);
        assert!(names[0].contains("_opcua-tcp"));
        handle.join().unwrap();
    }
}
