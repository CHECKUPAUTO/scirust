//! Sonde BACnet/IP : diffusion `Who-Is` globale (ANSI/ASHRAE 135 Annexe J —
//! BACnet Virtual Link Layer over UDP/IP — et clause 16.10, service
//! non confirmé `Who-Is`), et décodage de la réponse `I-Am` (clause 16.9).
//! `Who-Is` sans bornes d'instance est l'exact mécanisme par lequel tout
//! outil de supervision BACnet légitime découvre les appareils présents sur
//! un segment — pas un balayage générique.
//!
//! Port par défaut UDP 47808 (0xBAC0).

use std::net::{SocketAddr, UdpSocket};
use std::time::Duration;

pub const DEFAULT_PORT: u16 = 47808;

const BVLC_TYPE_BACNET_IP: u8 = 0x81;
const BVLC_FUNCTION_ORIGINAL_UNICAST_NPDU: u8 = 0x0A;
const BVLC_FUNCTION_ORIGINAL_BROADCAST_NPDU: u8 = 0x0B;
const PDU_TYPE_UNCONFIRMED_REQUEST: u8 = 0x10;
const SERVICE_I_AM: u8 = 0x00;

/// Construit la trame `Who-Is` globale (aucune borne d'instance — « qui que
/// vous soyez, répondez ») : BVLC (diffusion) + NPDU (destination réseau
/// diffusée, compte de sauts) + APDU (`Unconfirmed-Request`, choix de
/// service 8 = Who-Is).
pub fn build_who_is() -> Vec<u8> {
    let mut npdu_apdu = Vec::new();
    npdu_apdu.push(0x01); // NPDU version
    npdu_apdu.push(0x20); // NPDU control : destination réseau spécifiée
    npdu_apdu.extend_from_slice(&0xFFFFu16.to_be_bytes()); // DNET = diffusion globale
    npdu_apdu.push(0x00); // DLEN = 0 (pas d'adresse, diffusion sur ce réseau)
    npdu_apdu.push(0xFF); // Hop count
    npdu_apdu.push(PDU_TYPE_UNCONFIRMED_REQUEST);
    npdu_apdu.push(0x08); // choix de service : Who-Is

    let mut frame = Vec::with_capacity(4 + npdu_apdu.len());
    frame.push(BVLC_TYPE_BACNET_IP);
    frame.push(BVLC_FUNCTION_ORIGINAL_BROADCAST_NPDU);
    frame.extend_from_slice(&((4 + npdu_apdu.len()) as u16).to_be_bytes());
    frame.extend_from_slice(&npdu_apdu);
    frame
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IAm {
    /// Type d'objet BACnet annoncé (8 = Device).
    pub object_type: u16,
    /// Numéro d'instance de l'appareil.
    pub instance: u32,
}

/// Analyse une réponse `I-Am`. Ne décode que le premier paramètre
/// (l'identifiant d'objet de l'appareil) — les paramètres suivants
/// (longueur APDU max, segmentation, vendor ID) ne sont pas décodés (voir
/// `README.md`, limitation documentée plutôt que cachée).
pub fn parse_i_am(buf: &[u8]) -> Result<IAm, String> {
    if buf.len() < 4 || buf[0] != BVLC_TYPE_BACNET_IP
    {
        return Err("not a BACnet/IP frame (bad BVLC type)".to_string());
    }
    if buf[1] != BVLC_FUNCTION_ORIGINAL_UNICAST_NPDU
        && buf[1] != BVLC_FUNCTION_ORIGINAL_BROADCAST_NPDU
    {
        return Err(format!("unexpected BVLC function 0x{:02x}", buf[1]));
    }
    let bvlc_len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
    if bvlc_len > buf.len()
    {
        return Err("BVLC length exceeds frame size".to_string());
    }
    let mut pos = 4;
    if pos + 2 > buf.len()
    {
        return Err("frame too short for NPDU header".to_string());
    }
    let npdu_control = buf[pos + 1];
    pos += 2;
    // Saute DNET/DLEN/DADR/hop-count si l'en-tête NPDU annonce une
    // destination (bit 0x20), comme émis par notre propre `build_who_is`
    // et par la plupart des piles BACnet.
    if npdu_control & 0x20 != 0
    {
        if pos + 3 > buf.len()
        {
            return Err("truncated NPDU destination fields".to_string());
        }
        let dlen = buf[pos + 2] as usize;
        pos += 3 + dlen + 1; // DNET(2)+DLEN(1)+DADR(dlen)+hop count(1)
    }
    if pos + 2 > buf.len()
    {
        return Err("truncated APDU header".to_string());
    }
    if buf[pos] != PDU_TYPE_UNCONFIRMED_REQUEST
    {
        return Err("not an Unconfirmed-Request APDU".to_string());
    }
    if buf[pos + 1] != SERVICE_I_AM
    {
        return Err(format!(
            "unconfirmed service 0x{:02x} is not I-Am (0x00)",
            buf[pos + 1]
        ));
    }
    pos += 2;

    // Premier paramètre : Object Identifier, tag applicatif contexte 12,
    // longueur 4 -> octet de tag 0xC4, puis 4 octets (10 bits type objet,
    // 22 bits instance), voir ASHRAE 135 clause 20.2.14.
    if pos + 5 > buf.len() || buf[pos] != 0xC4
    {
        return Err("missing or malformed Device Object Identifier parameter".to_string());
    }
    let word = u32::from_be_bytes(
        buf[pos + 1..pos + 5]
            .try_into()
            .map_err(|_| "slice conversion failed")?,
    );
    Ok(IAm {
        object_type: (word >> 22) as u16,
        instance: word & 0x3F_FFFF,
    })
}

/// Diffuse un `Who-Is` vers `target` et renvoie le premier `I-Am` reçu.
pub fn probe(target: SocketAddr, timeout: Duration) -> Result<IAm, String> {
    let socket = UdpSocket::bind("0.0.0.0:0").map_err(|e| e.to_string())?;
    socket
        .set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    socket.set_broadcast(true).map_err(|e| e.to_string())?;
    socket
        .send_to(&build_who_is(), target)
        .map_err(|e| e.to_string())?;
    let mut buf = [0u8; 512];
    let n = socket.recv(&mut buf).map_err(|e| e.to_string())?;
    parse_i_am(&buf[..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::UdpSocket as StdUdpSocket;
    use std::thread;

    #[test]
    fn who_is_matches_the_canonical_global_broadcast_frame() {
        // La trame Who-Is globale canonique, telle qu'observée dans toute
        // capture Wireshark d'une pile BACnet/IP standard.
        let frame = build_who_is();
        assert_eq!(
            frame,
            vec![
                0x81, 0x0B, 0x00, 0x0C, 0x01, 0x20, 0xFF, 0xFF, 0x00, 0xFF, 0x10, 0x08
            ]
        );
    }

    fn build_i_am(device_instance: u32) -> Vec<u8> {
        let word = (8u32 << 22) | (device_instance & 0x3F_FFFF); // type=8 (Device)
        let mut apdu = vec![PDU_TYPE_UNCONFIRMED_REQUEST, SERVICE_I_AM, 0xC4];
        apdu.extend_from_slice(&word.to_be_bytes());
        // Paramètres restants non décodés par nous mais présents dans une
        // vraie trame : max-apdu (application unsigned), segmentation
        // (enumerated), vendor-id (application unsigned). On les inclut
        // pour vérifier qu'on s'arrête bien après le premier paramètre.
        apdu.extend_from_slice(&[0x22, 0x04, 0x00, 0x91, 0x00, 0x21, 0x18]);

        let mut npdu = vec![0x01, 0x00]; // version, control (pas de destination)
        npdu.extend_from_slice(&apdu);

        let mut frame = vec![BVLC_TYPE_BACNET_IP, BVLC_FUNCTION_ORIGINAL_BROADCAST_NPDU];
        frame.extend_from_slice(&((4 + npdu.len()) as u16).to_be_bytes());
        frame.extend_from_slice(&npdu);
        frame
    }

    #[test]
    fn parse_i_am_extracts_device_instance() {
        let frame = build_i_am(12345);
        let i_am = parse_i_am(&frame).unwrap();
        assert_eq!(i_am.object_type, 8);
        assert_eq!(i_am.instance, 12345);
    }

    #[test]
    fn parse_i_am_rejects_non_bacnet_frame() {
        assert!(parse_i_am(&[0x00, 0x00, 0x00, 0x00]).is_err());
    }

    #[test]
    fn parse_i_am_rejects_wrong_service() {
        // Who-Is (service 8) au lieu de I-Am (service 0).
        let frame = build_who_is();
        assert!(parse_i_am(&frame).is_err());
    }

    #[test]
    fn probe_over_loopback_udp_returns_i_am() {
        let responder = StdUdpSocket::bind("127.0.0.1:0").unwrap();
        let responder_addr = responder.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let mut buf = [0u8; 512];
            let (n, from) = responder.recv_from(&mut buf).unwrap();
            assert_eq!(&buf[..n], build_who_is().as_slice());
            responder.send_to(&build_i_am(99), from).unwrap();
        });
        let i_am = probe(responder_addr, Duration::from_secs(2)).unwrap();
        assert_eq!(i_am.instance, 99);
        handle.join().unwrap();
    }
}
