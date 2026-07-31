//! Fail-closed egress policy for optional network clients.
//!
//! The community core performs no network I/O. Optional clients must first
//! authorize a strict HTTP(S) URL and then resolve it through [`ValidatedUrl`].
//! The default policy admits loopback only on ports 80, 443, and 11434.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
#[cfg(any(feature = "llm", feature = "neural-embed"))]
use std::time::Duration;

/// Maximum DNS answers accepted for one endpoint.
const MAX_RESOLVED_ADDRESSES: usize = 16;
const DEFAULT_LOOPBACK_PORTS: &[u16] = &[80, 443, 11434];

/// Why an egress attempt was refused. Values never contain URL query strings
/// or credentials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EgressError {
    Malformed(String),
    HostNotAllowed {
        host: String,
        allowlist: Vec<String>,
    },
}

impl std::fmt::Display for EgressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(reason) => write!(f, "egress refused: malformed URL ({reason})"),
            Self::HostNotAllowed { host, allowlist } => write!(
                f,
                "egress refused: host '{host}' is not allowed; configured hosts: {allowlist:?}"
            ),
        }
    }
}

impl std::error::Error for EgressError {}

pub const ALLOW_ENV: &str = "CCOS_EGRESS_ALLOW";

/// A strictly parsed endpoint before DNS resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizedUrl {
    scheme: &'static str,
    host: String,
    port: u16,
    implicit_loopback: bool,
}

impl AuthorizedUrl {
    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn scheme(&self) -> &'static str {
        self.scheme
    }

    /// Resolve once and validate every returned address. HTTP clients must pin
    /// these addresses to prevent a second DNS lookup from changing policy.
    pub fn resolve(&self) -> Result<ValidatedUrl, EgressError> {
        let mut addresses: Vec<SocketAddr> = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(|_| EgressError::Malformed(format!("host '{}' did not resolve", self.host)))?
            .take(MAX_RESOLVED_ADDRESSES + 1)
            .collect();
        if addresses.is_empty() || addresses.len() > MAX_RESOLVED_ADDRESSES {
            return Err(EgressError::Malformed(format!(
                "host '{}' returned no bounded DNS answer",
                self.host
            )));
        }
        addresses.sort_unstable();
        addresses.dedup();
        for address in &addresses {
            let ip = normalize_ip(address.ip());
            if is_always_prohibited(ip) || (self.implicit_loopback && !ip.is_loopback()) {
                return Err(EgressError::HostNotAllowed {
                    host: ip.to_string(),
                    allowlist: vec![],
                });
            }
        }
        Ok(ValidatedUrl {
            authorized: self.clone(),
            addresses,
        })
    }
}

/// An endpoint whose DNS answers all passed policy. Pass the stored addresses
/// to the client builder rather than resolving the hostname again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedUrl {
    authorized: AuthorizedUrl,
    addresses: Vec<SocketAddr>,
}

impl ValidatedUrl {
    pub fn host(&self) -> &str {
        self.authorized.host()
    }

    pub fn addresses(&self) -> &[SocketAddr] {
        &self.addresses
    }
}

/// Host-to-port policy. BTree collections make diagnostics deterministic.
#[derive(Debug, Clone)]
pub struct EgressAllowlist {
    allowed: BTreeMap<String, BTreeSet<u16>>,
}

impl Default for EgressAllowlist {
    fn default() -> Self {
        Self::localhost_only()
    }
}

impl EgressAllowlist {
    pub fn localhost_only() -> Self {
        Self {
            allowed: BTreeMap::new(),
        }
    }

    /// Add comma-separated `host` or `host:port` entries. Bare hosts admit
    /// only the standard HTTP(S) ports; malformed entries are ignored.
    pub fn from_env() -> Self {
        let mut policy = Self::localhost_only();
        if let Ok(value) = std::env::var(ALLOW_ENV) {
            for entry in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                policy.allow_entry(entry);
            }
        }
        policy
    }

    /// Explicit per-call consent. A URL admits exactly its effective port; a
    /// bare hostname admits ports 80 and 443. Invalid input widens nothing.
    pub fn allowing(mut self, url_or_host: &str) -> Self {
        if url_or_host.contains("://") {
            if let Ok(parsed) = parse_url(url_or_host) {
                self.allowed
                    .entry(parsed.host)
                    .or_default()
                    .insert(parsed.port);
            }
        } else {
            self.allow_entry(url_or_host);
        }
        self
    }

    fn allow_entry(&mut self, entry: &str) {
        let candidate = if entry.starts_with('[') || entry.matches(':').count() <= 1 {
            format!("http://{entry}")
        } else {
            return;
        };
        let Ok(parsed) = parse_url(&candidate) else {
            return;
        };
        let ports = self.allowed.entry(parsed.host).or_default();
        if entry_has_explicit_port(entry) {
            ports.insert(parsed.port);
        } else {
            ports.extend([80, 443]);
        }
    }

    pub fn allowed_hosts(&self) -> Vec<String> {
        self.allowed
            .iter()
            .map(|(host, ports)| {
                let ports = ports
                    .iter()
                    .map(u16::to_string)
                    .collect::<Vec<_>>()
                    .join("|");
                format!("{host}:{ports}")
            })
            .collect()
    }

    /// Parse and authorize without DNS. Call [`AuthorizedUrl::resolve`] before
    /// any connection and pin the resulting addresses in the client.
    pub fn authorize(&self, url: &str) -> Result<AuthorizedUrl, EgressError> {
        let parsed = parse_url(url)?;
        let ip = parsed.host.parse::<IpAddr>().ok().map(normalize_ip);
        let implicit_loopback = parsed.host == "localhost" || ip.is_some_and(|ip| ip.is_loopback());
        let allowed_ports = if implicit_loopback {
            if DEFAULT_LOOPBACK_PORTS.contains(&parsed.port)
                || self
                    .allowed
                    .get(&parsed.host)
                    .is_some_and(|ports| ports.contains(&parsed.port))
            {
                return Ok(AuthorizedUrl {
                    scheme: parsed.scheme,
                    host: parsed.host,
                    port: parsed.port,
                    implicit_loopback,
                });
            }
            DEFAULT_LOOPBACK_PORTS
        } else if let Some(ports) = self.allowed.get(&parsed.host) {
            if !ports.contains(&parsed.port) {
                return Err(EgressError::HostNotAllowed {
                    host: format!("{}:{}", parsed.host, parsed.port),
                    allowlist: self.allowed_hosts(),
                });
            }
            return Ok(AuthorizedUrl {
                scheme: parsed.scheme,
                host: parsed.host,
                port: parsed.port,
                implicit_loopback: false,
            });
        } else {
            return Err(EgressError::HostNotAllowed {
                host: parsed.host,
                allowlist: self.allowed_hosts(),
            });
        };
        if !allowed_ports.contains(&parsed.port) {
            return Err(EgressError::HostNotAllowed {
                host: format!("{}:{}", parsed.host, parsed.port),
                allowlist: self.allowed_hosts(),
            });
        }
        Ok(AuthorizedUrl {
            scheme: parsed.scheme,
            host: parsed.host,
            port: parsed.port,
            implicit_loopback,
        })
    }

    /// Backward-compatible authorization check. Network call sites must use
    /// [`Self::validate`] so DNS answers are checked and can be pinned.
    #[deprecated(note = "syntax-only authorization cannot secure a network call; use validate")]
    pub fn check(&self, url: &str) -> Result<(), EgressError> {
        self.authorize(url).map(|_| ())
    }

    pub fn validate(&self, url: &str) -> Result<ValidatedUrl, EgressError> {
        self.authorize(url)?.resolve()
    }

    pub fn is_allowed(&self, url: &str) -> bool {
        self.authorize(url).is_ok()
    }
}

#[derive(Debug)]
struct ParsedUrl {
    scheme: &'static str,
    host: String,
    port: u16,
}

fn parse_url(url: &str) -> Result<ParsedUrl, EgressError> {
    if url.is_empty()
        || !url.is_ascii()
        || url
            .bytes()
            .any(|b| b.is_ascii_control() || b.is_ascii_whitespace())
        || url.contains('\\')
    {
        return Err(EgressError::Malformed("non-canonical characters".into()));
    }
    let (raw_scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| EgressError::Malformed("missing scheme or authority".into()))?;
    let (scheme, default_port) = match raw_scheme {
        "http" => ("http", 80),
        "https" => ("https", 443),
        other => {
            return Err(EgressError::Malformed(format!(
                "scheme '{}' is not allowed",
                redact_component(other)
            )))
        }
    };
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    if authority.is_empty() {
        return Err(EgressError::Malformed("empty authority".into()));
    }
    if authority.contains('@') {
        return Err(EgressError::Malformed("URL user-info is forbidden".into()));
    }
    let (raw_host, port) = if let Some(bracketed) = authority.strip_prefix('[') {
        let close = bracketed
            .find(']')
            .ok_or_else(|| EgressError::Malformed("unterminated IPv6 host".into()))?;
        let host = &bracketed[..close];
        let suffix = &bracketed[close + 1..];
        let port = parse_port_suffix(suffix, default_port)?;
        (host, port)
    } else {
        if authority.matches(':').count() > 1 {
            return Err(EgressError::Malformed("IPv6 host must be bracketed".into()));
        }
        match authority.rsplit_once(':') {
            Some((host, raw_port)) => (host, parse_port(raw_port)?),
            None => (authority, default_port),
        }
    };
    let host = canonical_host(raw_host)?;
    Ok(ParsedUrl { scheme, host, port })
}

fn parse_port_suffix(suffix: &str, default_port: u16) -> Result<u16, EgressError> {
    if suffix.is_empty() {
        Ok(default_port)
    } else if let Some(port) = suffix.strip_prefix(':') {
        parse_port(port)
    } else {
        Err(EgressError::Malformed("characters after IPv6 host".into()))
    }
}

fn parse_port(raw: &str) -> Result<u16, EgressError> {
    if raw.is_empty() || raw.starts_with('+') || (raw.len() > 1 && raw.starts_with('0')) {
        return Err(EgressError::Malformed("non-canonical port".into()));
    }
    raw.parse::<u16>()
        .ok()
        .filter(|port| *port != 0)
        .ok_or_else(|| EgressError::Malformed("invalid port".into()))
}

fn canonical_host(raw: &str) -> Result<String, EgressError> {
    if raw.is_empty() || raw.ends_with('.') || !raw.is_ascii() {
        return Err(EgressError::Malformed("invalid host".into()));
    }
    let lower = raw.to_ascii_lowercase();
    if let Ok(ipv4) = lower.parse::<Ipv4Addr>() {
        if ipv4.to_string() != lower {
            return Err(EgressError::Malformed("non-canonical IPv4 address".into()));
        }
        if ipv4.is_unspecified() {
            return Err(EgressError::HostNotAllowed {
                host: ipv4.to_string(),
                allowlist: vec![],
            });
        }
        return Ok(ipv4.to_string());
    }
    if let Ok(ipv6) = lower.parse::<Ipv6Addr>() {
        if ipv6.to_string() != lower {
            return Err(EgressError::Malformed("non-canonical IPv6 address".into()));
        }
        let normalized = normalize_ip(IpAddr::V6(ipv6));
        if normalized.is_unspecified() {
            return Err(EgressError::HostNotAllowed {
                host: normalized.to_string(),
                allowlist: vec![],
            });
        }
        return Ok(normalized.to_string());
    }
    if lower.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
        return Err(EgressError::Malformed("non-canonical numeric host".into()));
    }
    if lower.len() > 253
        || lower.split('.').any(|label| {
            label.is_empty()
                || label.len() > 63
                || label.starts_with('-')
                || label.ends_with('-')
                || !label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        })
    {
        return Err(EgressError::Malformed("invalid DNS host".into()));
    }
    Ok(lower)
}

fn normalize_ip(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(ipv6) => ipv6
            .to_ipv4_mapped()
            .map(IpAddr::V4)
            .unwrap_or(IpAddr::V6(ipv6)),
        ipv4 => ipv4,
    }
}

fn is_always_prohibited(ip: IpAddr) -> bool {
    match normalize_ip(ip) {
        IpAddr::V4(ip) => {
            ip.is_unspecified() || ip.is_broadcast() || ip.is_multicast() || ip.is_link_local()
        }
        IpAddr::V6(ip) => ip.is_unspecified() || ip.is_multicast() || ip.is_unicast_link_local(),
    }
}

fn entry_has_explicit_port(entry: &str) -> bool {
    if let Some(close) = entry.rfind(']') {
        entry
            .get(close + 1..)
            .is_some_and(|suffix| suffix.starts_with(':'))
    } else {
        entry.matches(':').count() == 1
    }
}

fn redact_component(value: &str) -> String {
    value.chars().take(32).collect()
}

/// Apply the mandatory HTTP-client controls and DNS pinning. Automatic proxy
/// discovery and redirects are disabled; every redirect must be handled as a
/// fresh policy decision by a caller that explicitly implements redirects.
#[cfg(feature = "llm")]
pub fn secure_async_client(
    endpoint: &ValidatedUrl,
    connect_timeout: Duration,
    total_timeout: Duration,
) -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(connect_timeout)
        .timeout(total_timeout)
        .resolve_to_addrs(endpoint.host(), endpoint.addresses())
        .build()
}

#[cfg(feature = "neural-embed")]
pub fn secure_blocking_client(
    endpoint: &ValidatedUrl,
    connect_timeout: Duration,
    total_timeout: Duration,
) -> Result<reqwest::blocking::Client, reqwest::Error> {
    reqwest::blocking::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(connect_timeout)
        .timeout(total_timeout)
        .resolve_to_addrs(endpoint.host(), endpoint.addresses())
        .build()
}

/// Read an async response without trusting `Content-Length` and without
/// permitting an unbounded allocation.
#[cfg(feature = "llm")]
pub async fn response_bytes_limited(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, String> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(format!("HTTP response exceeds {limit} byte limit"));
    }
    let mut body =
        Vec::with_capacity(response.content_length().unwrap_or(0).min(limit as u64) as usize);
    while let Some(chunk) = response.chunk().await.map_err(|error| error.to_string())? {
        let next = body
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| "HTTP response length overflow".to_string())?;
        if next > limit {
            return Err(format!("HTTP response exceeds {limit} byte limit"));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[cfg(feature = "neural-embed")]
pub fn blocking_response_bytes_limited(
    response: reqwest::blocking::Response,
    limit: usize,
) -> Result<Vec<u8>, String> {
    use std::io::Read as _;
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(format!("HTTP response exceeds {limit} byte limit"));
    }
    let mut body = Vec::new();
    response
        .take(limit as u64 + 1)
        .read_to_end(&mut body)
        .map_err(|error| error.to_string())?;
    if body.len() > limit {
        return Err(format!("HTTP response exceeds {limit} byte limit"));
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local() -> EgressAllowlist {
        EgressAllowlist::localhost_only()
    }

    #[test]
    fn canonical_loopback_variants_are_allowed() {
        for url in [
            "http://127.0.0.1:11434/x",
            "http://127.0.0.2:11434/x",
            "http://[::1]:11434/x",
            "http://[::ffff:127.0.0.1]:11434/x",
            "http://localhost:11434/x",
        ] {
            assert!(local().authorize(url).is_ok(), "{url}");
        }
        let resolved = local().validate("http://localhost:11434/x").unwrap();
        assert!(resolved
            .addresses()
            .iter()
            .all(|address| normalize_ip(address.ip()).is_loopback()));
    }

    #[test]
    fn default_blocks_private_link_local_metadata_and_public_destinations() {
        for url in [
            "http://10.0.0.1/",
            "http://192.168.1.10/",
            "http://169.254.169.254/",
            "http://8.8.8.8/",
            "https://api.anthropic.com/",
            "http://0.0.0.0:11434/",
        ] {
            assert!(local().authorize(url).is_err(), "{url}");
        }
    }

    #[test]
    fn schemes_userinfo_ports_and_alternative_ip_forms_are_rejected() {
        for url in [
            "file:///etc/passwd",
            "ftp://localhost/x",
            "http://user@localhost:11434/",
            "http://localhost:22/",
            "http://127.1:11434/",
            "http://2130706433:11434/",
            "http://127.000.000.001:11434/",
            "http://[0:0:0:0:0:0:0:1]:11434/",
            "http://localhost:011434/",
            "http:\\//localhost/",
        ] {
            assert!(local().authorize(url).is_err(), "{url}");
        }
    }

    #[test]
    fn explicit_consent_is_host_and_port_scoped() {
        let policy = local().allowing("https://licensing.example:8443/claim");
        assert!(policy
            .authorize("https://licensing.example:8443/claim")
            .is_ok());
        assert!(policy
            .authorize("https://licensing.example:443/claim")
            .is_err());
        assert!(policy.authorize("https://other.example:8443/").is_err());
    }

    #[test]
    fn malformed_urls_do_not_leak_path_or_query_in_errors() {
        let error = local()
            .authorize("javascript://host/private?token=secret")
            .unwrap_err()
            .to_string();
        assert!(!error.contains("secret"));
        assert!(!error.contains("private"));
    }

    #[cfg(feature = "llm")]
    #[tokio::test]
    async fn secure_client_does_not_follow_redirects_and_bounds_bodies() {
        use std::io::{Read as _, Write as _};
        use std::net::TcpListener;
        use std::thread;

        let listener = match TcpListener::bind("127.0.0.1:0") {
            Ok(listener) => listener,
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
                eprintln!("skipping live redirect test: loopback bind denied by sandbox");
                return;
            }
            Err(error) => panic!("bind loopback test server: {error}"),
        };
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            const RESPONSES: [&str; 2] = [
                "HTTP/1.1 302 Found\r\nLocation: https://example.com/\r\nContent-Length: 0\r\n\r\n",
                "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n12345",
            ];
            // Serve both replies whether the client reuses one keep-alive
            // connection or opens a fresh one per request. Accepting exactly one
            // connection per reply raced the pool: `stream` was dropped at the end
            // of each iteration, closing the socket, so a client that reused it for
            // the second request could hit the close first and fail with
            // "Connection reset by peer" — which is exactly how this test flaked in
            // CI (`tests (release)` on run 30592290332) while passing 20/20 locally.
            let mut served = 0usize;
            while served < RESPONSES.len() {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                // Keep this connection open and answer on it for as long as the
                // client keeps sending, so a reused socket is never closed early.
                while served < RESPONSES.len() {
                    let mut request = [0u8; 1024];
                    match stream.read(&mut request) {
                        Ok(0) | Err(_) => break, // client closed: wait for the next one
                        Ok(_) => {}
                    }
                    if stream.write_all(RESPONSES[served].as_bytes()).is_err() {
                        break;
                    }
                    served += 1;
                }
            }
        });
        let url = format!("http://{address}/");
        let policy = local().allowing(&url);
        let endpoint = policy.validate(&url).unwrap();
        let client =
            secure_async_client(&endpoint, Duration::from_secs(2), Duration::from_secs(2)).unwrap();
        let redirect = client.get(&url).send().await.unwrap();
        assert_eq!(redirect.status(), reqwest::StatusCode::FOUND);
        assert_eq!(redirect.url().as_str(), url);

        let response = client.get(&url).send().await.unwrap();
        assert!(response_bytes_limited(response, 4).await.is_err());
        server.join().unwrap();
    }
}
