use super::{ParsedPayload, PayloadParser};
use crate::flow::Protocol;

/// Parseur SSH (banner et authentication).
///
/// Extrait: version SSH, methode d'authentification.
/// Detecte: versions SSH obsoletees, tentatives de downgrade.
pub struct SshParser {
    /// Versions SSH acceptables (min)
    pub min_version: (u32, u32),
}

impl SshParser {
    pub fn new() -> Self {
        Self {
            min_version: (2, 0),
        }
    }
}

impl Default for SshParser {
    fn default() -> Self {
        Self::new()
    }
}

impl PayloadParser for SshParser {
    fn protocol(&self) -> Protocol {
        Protocol::Ssh
    }

    fn parse(&self, payload: &[u8], _src_port: u16, _dst_port: u16) -> Option<ParsedPayload> {
        // SSH banner commence par "SSH-"
        if !payload.starts_with(b"SSH-")
        {
            return None;
        }

        let mut parsed = ParsedPayload::new(Protocol::Ssh);

        // Extraire la version
        if let Ok(line) = std::str::from_utf8(&payload[..payload.len().min(256)])
        {
            let banner = line.lines().next().unwrap_or("");
            parsed.command = banner.to_string();
            parsed.uri = banner.to_string();

            // Parser la version: SSH-2.0-OpenSSH_8.9
            if let Some(version_part) = banner.strip_prefix("SSH-")
            {
                let parts: Vec<&str> = version_part.split('-').collect();
                if !parts.is_empty()
                {
                    let version_nums: Vec<&str> = parts[0].split('.').collect();
                    if version_nums.len() >= 2
                    {
                        if let (Ok(major), Ok(_minor)) = (
                            version_nums[0].parse::<u32>(),
                            version_nums[1].parse::<u32>(),
                        )
                        {
                            // Detecte SSH 1.x (deprecation)
                            if major < self.min_version.0
                            {
                                parsed.anomalies.push(format!("ssh_version_{}", major));
                            }
                        }
                    }
                }
            }

            // Detecte les keywords de downgrade
            let lower = banner.to_lowercase();
            if lower.contains("1.99")
            {
                parsed.anomalies.push("ssh_compatibility_mode".to_string());
            }
        }

        parsed.payload_size = payload.len();
        Some(parsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ssh_banner() {
        let payload = b"SSH-2.0-OpenSSH_8.9\r\n";
        let parser = SshParser::new();
        let result = parser.parse(payload, 40000, 22).unwrap();
        assert_eq!(result.protocol, Protocol::Ssh);
        assert!(result.command.contains("SSH-2.0"));
        assert!(result.anomalies.is_empty());
    }

    #[test]
    fn test_parse_ssh_old_version() {
        let payload = b"SSH-1.5-OpenSSH_4.0\r\n";
        let parser = SshParser::new();
        let result = parser.parse(payload, 40000, 22).unwrap();
        assert!(result.anomalies.iter().any(|a| a.contains("ssh_version")));
    }

    #[test]
    fn test_parse_ssh_compatibility() {
        let payload = b"SSH-1.99-OpenSSH_4.0\r\n";
        let parser = SshParser::new();
        let result = parser.parse(payload, 40000, 22).unwrap();
        assert!(
            result
                .anomalies
                .iter()
                .any(|a| a.contains("compatibility_mode"))
        );
    }

    #[test]
    fn test_not_ssh() {
        let payload = b"GET / HTTP/1.1\r\n";
        let parser = SshParser::new();
        assert!(parser.parse(payload, 40000, 22).is_none());
    }
}
