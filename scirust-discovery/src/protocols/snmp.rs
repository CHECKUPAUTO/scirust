//! Sonde SNMPv1 : requête `GET` de `sysDescr.0` (OID `1.3.6.1.2.1.1.1.0`,
//! RFC 1213 MIB-II) — l'exact mécanisme par lequel tout outil de
//! supervision réseau légitime identifie un appareil, pas un balayage de
//! ports. Encodage/décodage BER (ITU-T X.690) minimal, limité aux quelques
//! types utilisés par un GET SNMPv1 (RFC 1157) : `SEQUENCE`, `INTEGER`,
//! `OCTET STRING`, `OBJECT IDENTIFIER`, `NULL`, et les PDU `[0]`/`[2]`
//! (GetRequest/GetResponse).
//!
//! Port UDP par défaut 161. La communauté (mot de passe en clair, hérité du
//! protocole) par défaut est `"public"`, la valeur en lecture seule quasi
//! universelle des équipements qui exposent SNMP du tout.

use std::net::{SocketAddr, UdpSocket};
use std::time::Duration;

pub const DEFAULT_PORT: u16 = 161;
pub const SYSDESCR_OID: &[u32] = &[1, 3, 6, 1, 2, 1, 1, 1, 0];

const TAG_INTEGER: u8 = 0x02;
const TAG_OCTET_STRING: u8 = 0x04;
const TAG_NULL: u8 = 0x05;
const TAG_OBJECT_IDENTIFIER: u8 = 0x06;
const TAG_SEQUENCE: u8 = 0x30;
const TAG_GET_REQUEST_PDU: u8 = 0xA0;
const TAG_GET_RESPONSE_PDU: u8 = 0xA2;

// ─── Encodage BER minimal ───────────────────────────────────────────────

fn encode_length(len: usize) -> Vec<u8> {
    if len < 0x80
    {
        vec![len as u8]
    }
    else
    {
        let mut bytes = Vec::new();
        let mut n = len;
        while n > 0
        {
            bytes.push((n & 0xFF) as u8);
            n >>= 8;
        }
        bytes.reverse();
        let mut out = vec![0x80 | bytes.len() as u8];
        out.extend(bytes);
        out
    }
}

fn encode_tlv(tag: u8, content: &[u8]) -> Vec<u8> {
    let mut out = vec![tag];
    out.extend(encode_length(content.len()));
    out.extend_from_slice(content);
    out
}

fn encode_integer(value: i32) -> Vec<u8> {
    if value == 0
    {
        return vec![0];
    }
    let mut bytes = value.to_be_bytes().to_vec();
    while bytes.len() > 1
        && ((bytes[0] == 0x00 && bytes[1] & 0x80 == 0)
            || (bytes[0] == 0xFF && bytes[1] & 0x80 != 0))
    {
        bytes.remove(0);
    }
    bytes
}

fn encode_base128(mut value: u32) -> Vec<u8> {
    let mut bytes = vec![(value & 0x7F) as u8];
    value >>= 7;
    while value > 0
    {
        bytes.push(((value & 0x7F) as u8) | 0x80);
        value >>= 7;
    }
    bytes.reverse();
    bytes
}

fn encode_oid(arcs: &[u32]) -> Vec<u8> {
    let mut out = vec![(arcs[0] * 40 + arcs[1]) as u8];
    for &arc in &arcs[2..]
    {
        out.extend(encode_base128(arc));
    }
    out
}

/// Construit une requête `GetRequest` SNMPv1 pour `oid`, sous la communauté
/// `community`, avec l'identifiant de requête `request_id` (échoué tel
/// quel par un répondeur conforme, mais non vérifié par [`parse_get_response`]).
pub fn build_get_request(community: &str, oid: &[u32], request_id: i32) -> Vec<u8> {
    let varbind = encode_tlv(
        TAG_SEQUENCE,
        &[
            encode_tlv(TAG_OBJECT_IDENTIFIER, &encode_oid(oid)),
            encode_tlv(TAG_NULL, &[]),
        ]
        .concat(),
    );
    let varbind_list = encode_tlv(TAG_SEQUENCE, &varbind);

    let mut pdu_content = Vec::new();
    pdu_content.extend(encode_tlv(TAG_INTEGER, &encode_integer(request_id)));
    pdu_content.extend(encode_tlv(TAG_INTEGER, &encode_integer(0))); // error-status
    pdu_content.extend(encode_tlv(TAG_INTEGER, &encode_integer(0))); // error-index
    pdu_content.extend(varbind_list);
    let pdu = encode_tlv(TAG_GET_REQUEST_PDU, &pdu_content);

    let mut message_content = Vec::new();
    message_content.extend(encode_tlv(TAG_INTEGER, &encode_integer(0))); // version: SNMPv1
    message_content.extend(encode_tlv(TAG_OCTET_STRING, community.as_bytes()));
    message_content.extend(pdu);
    encode_tlv(TAG_SEQUENCE, &message_content)
}

// ─── Décodage BER minimal ───────────────────────────────────────────────

fn read_length(buf: &[u8], pos: usize) -> Result<(usize, usize), String> {
    if pos >= buf.len()
    {
        return Err("truncated BER length".to_string());
    }
    let first = buf[pos];
    if first & 0x80 == 0
    {
        Ok((first as usize, pos + 1))
    }
    else
    {
        let n = (first & 0x7F) as usize;
        if pos + 1 + n > buf.len()
        {
            return Err("truncated long-form BER length".to_string());
        }
        let mut len = 0usize;
        for &b in &buf[pos + 1..pos + 1 + n]
        {
            len = (len << 8) | b as usize;
        }
        Ok((len, pos + 1 + n))
    }
}

/// Renvoie `(tag, contenu, offset juste après cette valeur)`.
fn read_tlv(buf: &[u8], pos: usize) -> Result<(u8, &[u8], usize), String> {
    if pos >= buf.len()
    {
        return Err("truncated BER tag".to_string());
    }
    let tag = buf[pos];
    let (len, content_start) = read_length(buf, pos + 1)?;
    if content_start + len > buf.len()
    {
        return Err("truncated BER content".to_string());
    }
    Ok((
        tag,
        &buf[content_start..content_start + len],
        content_start + len,
    ))
}

/// Analyse un `GetResponse` SNMPv1 et renvoie la valeur du premier
/// (unique) varbind sous forme de texte, en échouant explicitement sur un
/// `error-status` non nul ou une valeur d'un type inattendu — jamais un
/// résultat fabriqué.
pub fn parse_get_response(buf: &[u8]) -> Result<String, String> {
    let (tag, message, _) = read_tlv(buf, 0)?;
    if tag != TAG_SEQUENCE
    {
        return Err("not a SNMP message (expected top-level SEQUENCE)".to_string());
    }

    let (vtag, _, pos) = read_tlv(message, 0)?;
    if vtag != TAG_INTEGER
    {
        return Err("missing SNMP version INTEGER".to_string());
    }
    let (ctag, _, pos) = read_tlv(message, pos)?;
    if ctag != TAG_OCTET_STRING
    {
        return Err("missing community OCTET STRING".to_string());
    }
    let (ptag, pdu, _) = read_tlv(message, pos)?;
    if ptag != TAG_GET_RESPONSE_PDU
    {
        return Err(format!("expected GetResponse-PDU (0xA2), got 0x{ptag:02x}"));
    }

    let (_, _, pos) = read_tlv(pdu, 0)?; // request-id (non vérifié)
    let (estag, es_content, pos) = read_tlv(pdu, pos)?;
    if estag != TAG_INTEGER
    {
        return Err("missing error-status INTEGER".to_string());
    }
    let error_status = es_content
        .iter()
        .fold(0i64, |acc, &b| (acc << 8) | b as i64);
    if error_status != 0
    {
        return Err(format!("SNMP GetResponse error-status={error_status}"));
    }
    let (_, _, pos) = read_tlv(pdu, pos)?; // error-index (non vérifié)

    let (vblt, varbind_list, _) = read_tlv(pdu, pos)?;
    if vblt != TAG_SEQUENCE
    {
        return Err("missing VarBindList SEQUENCE".to_string());
    }
    let (vbt, varbind, _) = read_tlv(varbind_list, 0)?;
    if vbt != TAG_SEQUENCE
    {
        return Err("missing VarBind SEQUENCE".to_string());
    }
    let (_, _, value_pos) = read_tlv(varbind, 0)?; // OID (non vérifié)
    let (value_tag, value, _) = read_tlv(varbind, value_pos)?;
    match value_tag
    {
        TAG_OCTET_STRING => Ok(String::from_utf8_lossy(value).to_string()),
        other => Err(format!(
            "unexpected value type 0x{other:02x} (expected OCTET STRING)"
        )),
    }
}

/// Envoie un `GET sysDescr.0` et renvoie la description texte de l'appareil.
pub fn probe(target: SocketAddr, community: &str, timeout: Duration) -> Result<String, String> {
    let socket = UdpSocket::bind("0.0.0.0:0").map_err(|e| e.to_string())?;
    socket
        .set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    socket
        .send_to(&build_get_request(community, SYSDESCR_OID, 1), target)
        .map_err(|e| e.to_string())?;
    let mut buf = [0u8; 2048];
    let n = socket.recv(&mut buf).map_err(|e| e.to_string())?;
    parse_get_response(&buf[..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::UdpSocket as StdUdpSocket;
    use std::thread;

    #[test]
    fn get_request_matches_hand_encoded_bytes() {
        // GET public / sysDescr.0, request-id=1 — vérifié octet par octet
        // par calcul manuel de l'encodage BER (voir commentaire de conception).
        let req = build_get_request("public", SYSDESCR_OID, 1);
        let expected = vec![
            0x30, 0x26, // SEQUENCE, len 38
            0x02, 0x01, 0x00, // version = 0 (SNMPv1)
            0x04, 0x06, b'p', b'u', b'b', b'l', b'i', b'c', // community "public"
            0xA0, 0x19, // GetRequest-PDU, len 25
            0x02, 0x01, 0x01, // request-id = 1
            0x02, 0x01, 0x00, // error-status = 0
            0x02, 0x01, 0x00, // error-index = 0
            0x30, 0x0E, // VarBindList, len 14
            0x30, 0x0C, // VarBind, len 12
            0x06, 0x08, 0x2B, 0x06, 0x01, 0x02, 0x01, 0x01, 0x01, 0x00, // OID sysDescr.0
            0x05, 0x00, // NULL
        ];
        assert_eq!(req, expected);
    }

    fn build_get_response(description: &str) -> Vec<u8> {
        let varbind = encode_tlv(
            TAG_SEQUENCE,
            &[
                encode_tlv(TAG_OBJECT_IDENTIFIER, &encode_oid(SYSDESCR_OID)),
                encode_tlv(TAG_OCTET_STRING, description.as_bytes()),
            ]
            .concat(),
        );
        let varbind_list = encode_tlv(TAG_SEQUENCE, &varbind);
        let mut pdu_content = Vec::new();
        pdu_content.extend(encode_tlv(TAG_INTEGER, &encode_integer(1)));
        pdu_content.extend(encode_tlv(TAG_INTEGER, &encode_integer(0)));
        pdu_content.extend(encode_tlv(TAG_INTEGER, &encode_integer(0)));
        pdu_content.extend(varbind_list);
        let pdu = encode_tlv(TAG_GET_RESPONSE_PDU, &pdu_content);
        let mut message_content = Vec::new();
        message_content.extend(encode_tlv(TAG_INTEGER, &encode_integer(0)));
        message_content.extend(encode_tlv(TAG_OCTET_STRING, b"public"));
        message_content.extend(pdu);
        encode_tlv(TAG_SEQUENCE, &message_content)
    }

    #[test]
    fn parse_get_response_extracts_description() {
        let resp = build_get_response("Acme Industrial Switch v2.1");
        let desc = parse_get_response(&resp).unwrap();
        assert_eq!(desc, "Acme Industrial Switch v2.1");
    }

    #[test]
    fn parse_get_response_handles_long_description_requiring_long_form_length() {
        // > 127 octets force l'encodage de longueur BER en forme longue.
        let long_desc = "x".repeat(200);
        let resp = build_get_response(&long_desc);
        assert_eq!(parse_get_response(&resp).unwrap(), long_desc);
    }

    #[test]
    fn parse_get_response_rejects_error_status() {
        let mut resp = build_get_response("irrelevant");
        // error-status suit immédiatement request-id (`02 01 01`, request-id=1
        // dans build_get_response) : on corrompt les 3 octets d'après plutôt
        // que de chercher `02 01 00`, qui apparaît d'abord dans le champ
        // version (lui aussi `02 01 00`) plus tôt dans le message.
        let idx = resp
            .windows(3)
            .position(|w| w == [0x02, 0x01, 0x01])
            .unwrap();
        resp[idx + 3 + 2] = 2; // error-status = noSuchName (2)
        assert!(parse_get_response(&resp).is_err());
    }

    #[test]
    fn parse_get_response_rejects_non_snmp_buffer() {
        assert!(parse_get_response(&[0x04, 0x02, 0x00, 0x00]).is_err());
    }

    #[test]
    fn probe_over_loopback_udp_returns_description() {
        let responder = StdUdpSocket::bind("127.0.0.1:0").unwrap();
        let responder_addr = responder.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let mut buf = [0u8; 512];
            let (_n, from) = responder.recv_from(&mut buf).unwrap();
            responder
                .send_to(&build_get_response("Loopback Test Device"), from)
                .unwrap();
        });
        let desc = probe(responder_addr, "public", Duration::from_secs(2)).unwrap();
        assert_eq!(desc, "Loopback Test Device");
        handle.join().unwrap();
    }
}
