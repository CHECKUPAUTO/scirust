//! Sonde Modbus TCP : requête « Read Device Identification » (code fonction
//! 0x2B, sous-type MEI 0x0E — Modbus Application Protocol V1.1b3 §6.21).
//! Une requête de lecture standard, en lecture seule, prévue par le
//! protocole précisément pour l'auto-description d'un appareil — pas un
//! sondage générique de registres.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

pub const DEFAULT_PORT: u16 = 502;

const FUNCTION_READ_DEVICE_ID: u8 = 0x2B;
const FUNCTION_READ_DEVICE_ID_ERROR: u8 = FUNCTION_READ_DEVICE_ID | 0x80;
const MEI_TYPE_READ_DEVICE_ID: u8 = 0x0E;
const READ_DEVICE_ID_BASIC: u8 = 0x01;

/// Construit l'ADU Modbus TCP pour une requête « Read Device Identification »
/// (niveau « basic » : objets 0x00 VendorName, 0x01 ProductCode,
/// 0x02 MajorMinorRevision).
pub fn build_read_device_id_request(transaction_id: u16, unit_id: u8) -> Vec<u8> {
    let pdu = [
        FUNCTION_READ_DEVICE_ID,
        MEI_TYPE_READ_DEVICE_ID,
        READ_DEVICE_ID_BASIC,
        0x00,
    ];
    let mut adu = Vec::with_capacity(7 + pdu.len());
    adu.extend_from_slice(&transaction_id.to_be_bytes());
    adu.extend_from_slice(&0u16.to_be_bytes()); // protocol id (toujours 0 pour Modbus)
    adu.extend_from_slice(&((pdu.len() + 1) as u16).to_be_bytes()); // longueur = unitId + PDU
    adu.push(unit_id);
    adu.extend_from_slice(&pdu);
    adu
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceIdentification {
    pub vendor_name: Option<String>,
    pub product_code: Option<String>,
    pub major_minor_revision: Option<String>,
    pub raw_objects: Vec<(u8, Vec<u8>)>,
}

/// Analyse une réponse « Read Device Identification ». Distingue une
/// exception Modbus (bit 0x80 posé sur le code fonction) d'une trame
/// simplement mal formée.
pub fn parse_read_device_id_response(buf: &[u8]) -> Result<DeviceIdentification, String> {
    if buf.len() < 8
    {
        return Err("frame too short for MBAP header + function code".to_string());
    }
    let pdu = &buf[7..];
    if pdu[0] == FUNCTION_READ_DEVICE_ID_ERROR
    {
        let code = pdu.get(1).copied().unwrap_or(0);
        return Err(format!(
            "Modbus exception 0x{code:02x} on Read Device Identification"
        ));
    }
    if pdu[0] != FUNCTION_READ_DEVICE_ID || pdu.get(1) != Some(&MEI_TYPE_READ_DEVICE_ID)
    {
        return Err("response is not a Read Device Identification reply".to_string());
    }
    if pdu.len() < 7
    {
        return Err("Read Device Identification response too short".to_string());
    }
    let number_of_objects = pdu[6] as usize;
    let mut idx = 7;
    let mut raw_objects = Vec::with_capacity(number_of_objects);
    for _ in 0..number_of_objects
    {
        if idx + 2 > pdu.len()
        {
            return Err("truncated object list".to_string());
        }
        let object_id = pdu[idx];
        let object_len = pdu[idx + 1] as usize;
        idx += 2;
        if idx + object_len > pdu.len()
        {
            return Err("truncated object value".to_string());
        }
        raw_objects.push((object_id, pdu[idx..idx + object_len].to_vec()));
        idx += object_len;
    }
    let text_of = |id: u8| {
        raw_objects
            .iter()
            .find(|(oid, _)| *oid == id)
            .map(|(_, v)| String::from_utf8_lossy(v).to_string())
    };
    Ok(DeviceIdentification {
        vendor_name: text_of(0x00),
        product_code: text_of(0x01),
        major_minor_revision: text_of(0x02),
        raw_objects,
    })
}

/// Envoie une requête « Read Device Identification » et analyse la réponse.
pub fn probe(
    addr: SocketAddr,
    unit_id: u8,
    timeout: Duration,
) -> Result<DeviceIdentification, String> {
    let mut stream = TcpStream::connect_timeout(&addr, timeout).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    stream
        .write_all(&build_read_device_id_request(1, unit_id))
        .map_err(|e| e.to_string())?;
    let mut buf = [0u8; 512];
    let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
    parse_read_device_id_response(&buf[..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn request_has_correct_mbap_and_pdu() {
        let req = build_read_device_id_request(7, 3);
        assert_eq!(&req[0..2], &7u16.to_be_bytes()); // transaction id
        assert_eq!(&req[2..4], &0u16.to_be_bytes()); // protocol id
        assert_eq!(&req[4..6], &5u16.to_be_bytes()); // length = unitId(1) + pdu(4)
        assert_eq!(req[6], 3); // unit id
        assert_eq!(
            &req[7..],
            &[FUNCTION_READ_DEVICE_ID, MEI_TYPE_READ_DEVICE_ID, 0x01, 0x00]
        );
    }

    fn sample_response() -> Vec<u8> {
        let mut resp = Vec::new();
        resp.extend_from_slice(&1u16.to_be_bytes()); // transaction id
        resp.extend_from_slice(&0u16.to_be_bytes()); // protocol id
        let vendor = b"Acme Corp";
        let product = b"PLC-9000";
        let pdu_len = 2 + 5 + (2 + vendor.len()) + (2 + product.len());
        resp.extend_from_slice(&((pdu_len + 1) as u16).to_be_bytes());
        resp.push(0xFF); // unit id
        resp.push(FUNCTION_READ_DEVICE_ID);
        resp.push(MEI_TYPE_READ_DEVICE_ID);
        resp.push(READ_DEVICE_ID_BASIC); // echoed read device id code
        resp.push(0x01); // conformity level
        resp.push(0x00); // more follows
        resp.push(0x00); // next object id
        resp.push(2); // number of objects
        resp.push(0x00);
        resp.push(vendor.len() as u8);
        resp.extend_from_slice(vendor);
        resp.push(0x01);
        resp.push(product.len() as u8);
        resp.extend_from_slice(product);
        resp
    }

    #[test]
    fn parse_response_extracts_vendor_and_product() {
        let id = parse_read_device_id_response(&sample_response()).unwrap();
        assert_eq!(id.vendor_name.as_deref(), Some("Acme Corp"));
        assert_eq!(id.product_code.as_deref(), Some("PLC-9000"));
        assert_eq!(id.major_minor_revision, None);
        assert_eq!(id.raw_objects.len(), 2);
    }

    #[test]
    fn parse_response_detects_modbus_exception() {
        let mut resp = vec![0u8, 1, 0, 0, 0, 3, 0xFF];
        resp.push(FUNCTION_READ_DEVICE_ID_ERROR);
        resp.push(0x01); // illegal function
        let result = parse_read_device_id_response(&resp);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("0x01"));
    }

    #[test]
    fn parse_response_rejects_wrong_function_code() {
        let resp = vec![0u8, 1, 0, 0, 0, 3, 0xFF, 0x03, 0x02, 0x01];
        assert!(parse_read_device_id_response(&resp).is_err());
    }

    #[test]
    fn probe_against_local_listener_returns_device_id() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut request = [0u8; 512];
            let n = socket.read(&mut request).unwrap();
            assert_eq!(request[7], FUNCTION_READ_DEVICE_ID);
            socket.write_all(&sample_response()).unwrap();
            n
        });
        let result = probe(addr, 0xFF, Duration::from_secs(2)).unwrap();
        assert_eq!(result.vendor_name.as_deref(), Some("Acme Corp"));
        handle.join().unwrap();
    }
}
