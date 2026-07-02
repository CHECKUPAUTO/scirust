//! `scirust-discovery` — découverte d'actifs OT/IT sûre, consentie et
//! auditée.
//!
//! Permet à un agent (le SLM `scirust-sciagent`, un autre agent connecté
//! via `scirust-mcp`, ou un opérateur humain via le CLI) de savoir *quel
//! matériel industriel est réellement présent* sur un réseau donné, sans
//! jamais dériver vers un scanner de ports générique — dangereux sur des
//! automates industriels à pile TCP/IP minimale (voir `README.md` pour les
//! sources : incident SQL Slammer / Davis-Besse 2003, étude Coffey et al.
//! 2018 sur les plantages de PLC sous sondage Nmap).
//!
//! Trois garanties structurelles, pas seulement documentaires :
//!
//! 1. **Aucun paquet sans autorisation vérifiée** — [`engine::DiscoveryEngine`]
//!    vérifie une [`scope::ScopeAuthorization`] signée (HMAC-SHA256 :
//!    plages CIDR, protocoles, fenêtre de validité, niveau de sécurité de
//!    zone IEC 62443) avant tout envoi ; une portée non signée, expirée, ou
//!    élargie après signature est rejetée.
//! 2. **Natif au protocole, jamais un scan générique** — chaque sonde
//!    (`protocols::opcua`, `protocols::modbus`, `protocols::mdns`) n'envoie
//!    que ce qu'un client légitime de ce protocole enverrait pour
//!    s'annoncer ou se connecter.
//! 3. **Tout est journalisé** — succès, échec, ou refus, dans un journal
//!    hash-chaîné SHA-256 ([`audit::AuditLog`]) qui rend toute
//!    falsification après coup détectable.

pub mod audit;
pub mod engine;
pub mod hmac;
pub mod protocols;
pub mod scope;

pub use engine::{DiscoveryEngine, DiscoveryOutcome, DiscoveryResult, Protocol};
pub use scope::ScopeAuthorization;
