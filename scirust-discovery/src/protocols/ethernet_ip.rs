//! Sonde EtherNet/IP (CIP) : requête `ListIdentity` de l'encapsulation
//! EtherNet/IP (ODVA, *The CIP Networks Library, Volume 2: EtherNet/IP
//! Adaptation of CIP*) — le service que tout outil de découverte
//! EtherNet/IP légitime (RSLinx, l'utilitaire ODVA, etc.) utilise pour
//! énumérer les automates/E/S présents sur un segment.
//!
//! Port UDP par défaut 44818.
//!
//! ## Niveau de confiance (honnêteté documentaire)
//! L'en-tête d'encapsulation (24 octets, `build_request`/les 24 premiers
//! octets analysés par `parse_response`) est stable et documenté sans
//! ambiguïté dans la spécification ODVA — confiance haute. La disposition
//! interne de l'objet Identity retourné (`IdentityInfo`) suit la structure
//! couramment documentée (adresse socket embarquée, Vendor ID, Device
//! Type, Product Code, révision, état, numéro de série, nom produit en
//! chaîne courte) mais n'a pas été vérifiée face à un appareil réel dans
//! cet environnement — à valider contre un automate EtherNet/IP réel ou le
//! texte de la spécification ODVA avant un usage en production.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

pub const DEFAULT_PORT: u16 = 44818;

const COMMAND_LIST_IDENTITY: u16 = 0x0063;
const CPF_TYPE_IDENTITY: u16 = 0x0C;

/// Construit la requête `ListIdentity` : en-tête d'encapsulation de 24
/// octets, commande `ListIdentity` (0x0063), aucune donnée.
pub fn build_request() -> Vec<u8> {
    let mut req = Vec::with_capacity(24);
    req.extend_from_slice(&COMMAND_LIST_IDENTITY.to_le_bytes());
    req.extend_from_slice(&0u16.to_le_bytes()); // Length (pas de données)
    req.extend_from_slice(&0u32.to_le_bytes()); // Session Handle
    req.extend_from_slice(&0u32.to_le_bytes()); // Status
    req.extend_from_slice(&[0u8; 8]); // Sender Context (opaque, échoé)
    req.extend_from_slice(&0u32.to_le_bytes()); // Options
    req
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentityInfo {
    pub vendor_id: u16,
    pub device_type: u16,
    pub product_code: u16,
    pub revision_major: u8,
    pub revision_minor: u8,
    pub serial_number: u32,
    pub product_name: String,
}

/// Analyse une réponse `ListIdentity`. Vérifie l'en-tête d'encapsulation
/// (haute confiance) puis décode l'item Identity du CPF (voir la note de
/// confiance en tête de module).
pub fn parse_response(buf: &[u8]) -> Result<IdentityInfo, String> {
    if buf.len() < 24
    {
        return Err("frame too short for the 24-byte encapsulation header".to_string());
    }
    let command = u16::from_le_bytes([buf[0], buf[1]]);
    if command != COMMAND_LIST_IDENTITY
    {
        return Err(format!("unexpected encapsulation command 0x{command:04x}"));
    }
    let status = u32::from_le_bytes(
        buf[8..12]
            .try_into()
            .map_err(|_| "slice conversion failed")?,
    );
    if status != 0
    {
        return Err(format!(
            "encapsulation status indicates an error: 0x{status:08x}"
        ));
    }
    let data = &buf[24..];

    if data.len() < 2
    {
        return Err("missing CPF item count".to_string());
    }
    let item_count = u16::from_le_bytes([data[0], data[1]]);
    if item_count == 0
    {
        return Err("ListIdentity response has zero CPF items".to_string());
    }
    if data.len() < 6
    {
        return Err("truncated CPF item header".to_string());
    }
    let item_type = u16::from_le_bytes([data[2], data[3]]);
    if item_type != CPF_TYPE_IDENTITY
    {
        return Err(format!(
            "unexpected CPF item type 0x{item_type:04x} (expected Identity 0x0C)"
        ));
    }
    let item_len = u16::from_le_bytes([data[4], data[5]]) as usize;
    let item = &data[6..];
    if item.len() < item_len
    {
        return Err("truncated Identity item data".to_string());
    }
    let item = &item[..item_len];

    // Identity item layout: EncapProtocolVersion(2) + sockaddr_in(16) +
    // VendorID(2) + DeviceType(2) + ProductCode(2) + Revision(2:
    // major,minor) + Status(2) + SerialNumber(4) + ProductName (SHORT_STRING:
    // 1-byte length + bytes) + State(1).
    const FIXED_LEN: usize = 2 + 16 + 2 + 2 + 2 + 2 + 2 + 4;
    if item.len() < FIXED_LEN + 1
    {
        return Err("Identity item shorter than the fixed-length fields".to_string());
    }
    let vendor_id = u16::from_le_bytes([item[18], item[19]]);
    let device_type = u16::from_le_bytes([item[20], item[21]]);
    let product_code = u16::from_le_bytes([item[22], item[23]]);
    let revision_major = item[24];
    let revision_minor = item[25];
    let serial_number = u32::from_le_bytes(
        item[28..32]
            .try_into()
            .map_err(|_| "slice conversion failed")?,
    );
    let name_len = item[FIXED_LEN] as usize;
    if item.len() < FIXED_LEN + 1 + name_len
    {
        return Err("truncated product name in Identity item".to_string());
    }
    let product_name =
        String::from_utf8_lossy(&item[FIXED_LEN + 1..FIXED_LEN + 1 + name_len]).to_string();

    Ok(IdentityInfo {
        vendor_id,
        device_type,
        product_code,
        revision_major,
        revision_minor,
        serial_number,
        product_name,
    })
}

/// Envoie `ListIdentity` sur une connexion TCP déjà autorisée par
/// l'appelant et renvoie l'identité décodée.
pub fn probe(addr: SocketAddr, timeout: Duration) -> Result<IdentityInfo, String> {
    let mut stream = TcpStream::connect_timeout(&addr, timeout).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    stream
        .write_all(&build_request())
        .map_err(|e| e.to_string())?;
    let mut buf = [0u8; 512];
    let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
    parse_response(&buf[..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn request_has_correct_encapsulation_header() {
        let req = build_request();
        assert_eq!(req.len(), 24);
        assert_eq!(&req[0..2], &COMMAND_LIST_IDENTITY.to_le_bytes());
        assert_eq!(&req[2..4], &0u16.to_le_bytes()); // length = 0
    }

    fn build_sample_response(vendor_id: u16, product_name: &str) -> Vec<u8> {
        let mut identity = Vec::new();
        identity.extend_from_slice(&1u16.to_le_bytes()); // encap protocol version
        identity.extend_from_slice(&[0u8; 16]); // sockaddr_in (non décodé)
        identity.extend_from_slice(&vendor_id.to_le_bytes());
        identity.extend_from_slice(&5u16.to_le_bytes()); // device type
        identity.extend_from_slice(&42u16.to_le_bytes()); // product code
        identity.push(3); // revision major
        identity.push(1); // revision minor
        identity.extend_from_slice(&0u16.to_le_bytes()); // status
        identity.extend_from_slice(&0xDEADBEEFu32.to_le_bytes()); // serial number
        identity.push(product_name.len() as u8);
        identity.extend_from_slice(product_name.as_bytes());
        identity.push(0xFF); // state

        let mut cpf = Vec::new();
        cpf.extend_from_slice(&1u16.to_le_bytes()); // item count
        cpf.extend_from_slice(&CPF_TYPE_IDENTITY.to_le_bytes());
        cpf.extend_from_slice(&(identity.len() as u16).to_le_bytes());
        cpf.extend_from_slice(&identity);

        let mut resp = Vec::new();
        resp.extend_from_slice(&COMMAND_LIST_IDENTITY.to_le_bytes());
        resp.extend_from_slice(&(cpf.len() as u16).to_le_bytes());
        resp.extend_from_slice(&0u32.to_le_bytes()); // session handle
        resp.extend_from_slice(&0u32.to_le_bytes()); // status
        resp.extend_from_slice(&[0u8; 8]); // sender context
        resp.extend_from_slice(&0u32.to_le_bytes()); // options
        resp.extend_from_slice(&cpf);
        resp
    }

    #[test]
    fn parse_response_extracts_identity_fields() {
        let resp = build_sample_response(0x1234, "Acme PLC-5000");
        let id = parse_response(&resp).unwrap();
        assert_eq!(id.vendor_id, 0x1234);
        assert_eq!(id.product_code, 42);
        assert_eq!(id.revision_major, 3);
        assert_eq!(id.revision_minor, 1);
        assert_eq!(id.serial_number, 0xDEADBEEF);
        assert_eq!(id.product_name, "Acme PLC-5000");
    }

    #[test]
    fn parse_response_rejects_wrong_command() {
        let mut resp = build_sample_response(1, "x");
        resp[0] = 0x00;
        resp[1] = 0x00;
        assert!(parse_response(&resp).is_err());
    }

    #[test]
    fn parse_response_rejects_error_status() {
        let mut resp = build_sample_response(1, "x");
        resp[11] = 0x01; // status non nul
        assert!(parse_response(&resp).is_err());
    }

    #[test]
    fn parse_response_rejects_truncated_frame() {
        assert!(parse_response(&[0u8; 10]).is_err());
    }

    #[test]
    fn probe_against_local_listener_returns_identity() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut request = [0u8; 24];
            let n = socket.read(&mut request).unwrap();
            assert_eq!(n, 24);
            socket
                .write_all(&build_sample_response(0x0099, "Loopback Drive"))
                .unwrap();
        });
        let id = probe(addr, Duration::from_secs(2)).unwrap();
        assert_eq!(id.vendor_id, 0x0099);
        assert_eq!(id.product_name, "Loopback Drive");
        handle.join().unwrap();
    }
}
