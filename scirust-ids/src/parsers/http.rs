use super::{ParsedPayload, PayloadParser};
use crate::flow::Protocol;

/// Parseur HTTP basique (requete et reponse).
///
/// Extrait: methode, URI, status code, headers.
/// Detecte: User-Agent suspects, methodes inusitees, URI anormalement longues.
pub struct HttpParser {
    /// Taille max URI avant alerte
    pub max_uri_len: usize,
    /// Methodes HTTP considerées dangereuses
    pub dangerous_methods: Vec<String>,
}

impl HttpParser {
    pub fn new() -> Self {
        Self {
            max_uri_len: 2048,
            dangerous_methods: vec![
                "PUT".to_string(),
                "DELETE".to_string(),
                "TRACE".to_string(),
                "CONNECT".to_string(),
            ],
        }
    }
}

impl Default for HttpParser {
    fn default() -> Self {
        Self::new()
    }
}

impl PayloadParser for HttpParser {
    fn protocol(&self) -> Protocol {
        Protocol::Http
    }

    fn parse(&self, payload: &[u8], _src_port: u16, _dst_port: u16) -> Option<ParsedPayload> {
        if payload.len() < 10
        {
            return None;
        }

        // Detection methode HTTP
        let methods = [
            "GET ", "POST ", "PUT ", "DELETE ", "HEAD ", "OPTIONS ", "PATCH ", "CONNECT ", "TRACE ",
        ];
        let mut is_request = false;
        let mut method = String::new();

        for m in &methods
        {
            if payload.starts_with(m.as_bytes())
            {
                is_request = true;
                method = m.trim().to_string();
                break;
            }
        }

        // Detection reponse HTTP
        let is_response = payload.starts_with(b"HTTP/");
        let mut parsed = ParsedPayload::new(Protocol::Http);

        if is_request
        {
            parsed.command = method.clone();

            // Extraire l'URI
            if let Some(line_end) = payload.iter().position(|&b| b == b'\r' || b == b'\n')
            {
                let first_line = &payload[..line_end];
                if let Some(space_pos) = first_line.iter().position(|&b| b == b' ')
                {
                    let after_method = &first_line[space_pos + 1..];
                    if let Some(end) = after_method.iter().position(|&b| b == b' ')
                    {
                        parsed.uri = String::from_utf8_lossy(&after_method[..end]).to_string();
                    }
                }
            }

            // Anomalies sur la methode
            if self.dangerous_methods.iter().any(|dm| dm == &method)
            {
                parsed
                    .anomalies
                    .push(format!("dangerous_method:{}", method));
            }

            // Anomalie URI longue
            if parsed.uri.len() > self.max_uri_len
            {
                parsed
                    .anomalies
                    .push(format!("long_uri:{}", parsed.uri.len()));
            }

            // Injection de path traversal
            if parsed.uri.contains("../")
                || parsed.uri.contains("..%2f")
                || parsed.uri.contains("%2e%2e")
            {
                parsed.anomalies.push("path_traversal".to_string());
            }

            // Injection de commandes shell
            if parsed.uri.contains('$')
                || parsed.uri.contains('|')
                || parsed.uri.contains(';')
                || parsed.uri.contains('`')
            {
                parsed.anomalies.push("command_injection".to_string());
            }

            // Extraction headers
            parsed.headers = extract_headers(payload);
        }
        else if is_response
        {
            parsed.command = "RESPONSE".to_string();

            // Extraire le code de status
            if payload.len() >= 12
            {
                let status_bytes = &payload[9..12];
                if let Ok(status_str) = std::str::from_utf8(status_bytes)
                {
                    if let Ok(code) = status_str.trim().parse::<u16>()
                    {
                        parsed.response_code = Some(code);
                        if code >= 500
                        {
                            parsed.anomalies.push(format!("server_error:{}", code));
                        }
                    }
                }
            }
        }
        else
        {
            return None;
        }

        parsed.payload_size = payload.len();
        Some(parsed)
    }
}

/// Extraire les en-tetes HTTP d'un payload.
fn extract_headers(payload: &[u8]) -> Vec<(String, String)> {
    let mut headers = Vec::new();
    let text = String::from_utf8_lossy(payload);
    let mut lines = text.lines();

    // Sauter la premiere ligne (request line)
    lines.next();

    for line in lines
    {
        if line.trim().is_empty()
        {
            break;
        }
        if let Some(pos) = line.find(':')
        {
            let key = line[..pos].trim().to_string();
            let value = line[pos + 1..].trim().to_string();
            headers.push((key, value));
        }
    }

    headers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_http_get() {
        let payload =
            b"GET /index.html HTTP/1.1\r\nHost: example.com\r\nUser-Agent: Mozilla/5.0\r\n\r\n";
        let parser = HttpParser::new();
        let result = parser.parse(payload, 40000, 80).unwrap();
        assert_eq!(result.command, "GET");
        assert_eq!(result.uri, "/index.html");
        assert!(result.anomalies.is_empty());
    }

    #[test]
    fn test_parse_http_post() {
        let payload = b"POST /login HTTP/1.1\r\nHost: example.com\r\nContent-Length: 50\r\n\r\n";
        let parser = HttpParser::new();
        let result = parser.parse(payload, 40001, 80).unwrap();
        assert_eq!(result.command, "POST");
        assert_eq!(result.uri, "/login");
    }

    #[test]
    fn test_parse_http_dangerous_method() {
        let payload = b"TRACE /debug HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let parser = HttpParser::new();
        let result = parser.parse(payload, 40002, 80).unwrap();
        assert!(
            result
                .anomalies
                .iter()
                .any(|a| a.contains("dangerous_method"))
        );
    }

    #[test]
    fn test_parse_http_path_traversal() {
        let payload = b"GET /../../etc/passwd HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let parser = HttpParser::new();
        let result = parser.parse(payload, 40003, 80).unwrap();
        assert!(
            result
                .anomalies
                .iter()
                .any(|a| a.contains("path_traversal"))
        );
    }

    #[test]
    fn test_parse_http_response() {
        let payload = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
        let parser = HttpParser::new();
        let result = parser.parse(payload, 80, 40000).unwrap();
        assert_eq!(result.command, "RESPONSE");
        assert_eq!(result.response_code, Some(404));
    }

    #[test]
    fn test_parse_http_500() {
        let payload = b"HTTP/1.1 500 Internal Server Error\r\n\r\n";
        let parser = HttpParser::new();
        let result = parser.parse(payload, 80, 40000).unwrap();
        assert!(result.anomalies.iter().any(|a| a.contains("server_error")));
    }

    #[test]
    fn test_not_http() {
        let payload = b"SSH-2.0-OpenSSH_8.9\r\n";
        let parser = HttpParser::new();
        assert!(parser.parse(payload, 40000, 22).is_none());
    }
}
