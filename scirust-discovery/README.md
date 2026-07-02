# scirust-discovery

Découverte d'actifs OT/IT **sûre, consentie et auditée** : répond à la
question « quel matériel industriel est réellement présent sur ce réseau »
pour un agent (le SLM `scirust-sciagent`, un autre agent connecté via
`scirust-mcp`, ou un opérateur humain), sans jamais devenir un scanner de
ports générique.

## Pourquoi pas un simple scan de ports

Un scan générique (type Nmap) est **documenté comme dangereux** sur des
automates industriels :

- Coffey et al. (*Security and Communication Networks*, 2018) documentent un
  scan Nmap ayant mis un PLC largement déployé en défaillance, nécessitant
  un cycle d'alimentation complet — non reproductible par le fabricant.
- L'incident SQL Slammer à la centrale nucléaire de Davis-Besse (janvier
  2003) : un trafic UDP de type « scan » (pas une attaque ciblée) ayant
  traversé une liaison corporate→SCADA non pare-feu a désactivé l'affichage
  des paramètres de sûreté pendant près de cinq heures.
- **NIST SP 800-82** (Guide to OT Security) codifie la doctrine qui en
  résulte : préférer la supervision passive ; ne réserver le sondage actif
  qu'à une fenêtre de maintenance explicitement autorisée par l'exploitant.

Ce crate adopte donc une approche **native au protocole** : chaque sonde
n'envoie que ce qu'un client légitime de ce protocole enverrait pour
s'annoncer ou établir une connexion — jamais un paquet arbitraire sur un
port arbitraire.

## Le modèle zones et conduits (ISA/IEC 62443)

`ScopeAuthorization` (`src/scope.rs`) encode l'autorisation comme une
**donnée vérifiable**, pas une convention :

- une liste blanche de plages **CIDR** IPv4,
- une liste blanche de **protocoles** (`opcua`, `modbus`, `mdns`),
- une **fenêtre de validité temporelle** (`valid_from_unix`/`valid_until_unix`),
- une étiquette de **zone** et son **niveau de sécurité IEC 62443** (SL0–SL4)
  — toute zone SL3+ est refusée par défaut, un dépassement doit être
  explicite (`allow_high_security_zone: true`),
- une **signature HMAC-SHA256** (clé pré-partagée entre l'opérateur qui
  autorise et l'agent qui exécute — ce n'est pas une PKI complète, voir
  `src/hmac.rs`) : une portée non signée, expirée, ou élargie après
  signature est rejetée avant tout envoi de paquet.

`DiscoveryEngine::probe_one` (`src/engine.rs`) est le seul point d'entrée :
il appelle `ScopeAuthorization::authorize` avant toute I/O réseau, et
journalise la tentative — dans la portée ou refusée — dans un journal
hash-chaîné SHA-256 (`src/audit.rs`), sur le même principe que
`scirust-func-safety::audit`.

## Protocoles supportés

| Protocole | Mécanisme | Référence |
|---|---|---|
| **OPC-UA** | Handshake UACP `Hello`/`Acknowledge` — la première chose qu'échange tout client OPC-UA, avant même l'ouverture d'un canal sécurisé | OPC UA Part 6 §7.1 |
| **Modbus TCP** | `Read Device Identification` (code fonction 0x2B, MEI 0x0E) — lecture seule, prévue par le protocole pour l'auto-description d'un appareil | Modbus Application Protocol V1.1b3 §6.21 |
| **mDNS/DNS-SD** | Requête DNS standard d'énumération de services (`_services._dns-sd._udp.local`) | RFC 1035, RFC 6762/6763 |

D'autres protocoles natifs à faible risque identifiés par la recherche
(BACnet `Who-Is`/`I-Am`, EtherNet/IP `List Identity`, SNMP `sysDescr`) sont
documentés comme prochaine extension naturelle — même schéma :
construction/analyse pures et testées, sonde active isolée, autorisation
vérifiée en amont.

## Utilisation

```rust
use scirust_discovery::{DiscoveryEngine, Protocol, ScopeAuthorization};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

let key = b"shared-secret-negotiated-out-of-band";
let scope = ScopeAuthorization {
    operator: "alice@example.com".to_string(),
    zone: "line3-plc-zone".to_string(),
    zone_security_level: 1,
    allowed_cidrs: vec!["192.168.1.0/24".to_string()],
    allowed_protocols: vec!["opcua".to_string(), "modbus".to_string()],
    valid_from_unix: 0,
    valid_until_unix: u64::MAX,
    allow_high_security_zone: false,
    signature_hex: String::new(),
}
.sign(key);

let mut engine = DiscoveryEngine::new(scope, key.to_vec(), Duration::from_secs(2));
let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
let result = engine.probe_one("192.168.1.10".parse().unwrap(), Protocol::OpcUa, now);
println!("{result:?}");
assert!(engine.audit_log().verify_chain());
```

## Limites honnêtes (documentées, pas cachées)

- IPv4 uniquement pour l'instant (`CIDR`/`ScopeAuthorization`) — IPv6 est un
  ajout mécanique mais non fait.
- `hmac.rs` implémente une signature à clé pré-partagée, pas une PKI
  (pas de rotation ni de révocation de clé) — suffisant pour un usage à un
  seul opérateur/une seule équipe, à durcir avant un déploiement
  multi-tenant.
- Les sondes actives fournies (OPC-UA, Modbus) sont volontairement les deux
  probes les plus légers identifiés par la recherche (voir
  `docs/DOMAIN_ROADMAP.md`) ; la découverte purement **passive** (écoute de
  trafic déjà présent, sans rien émettre) reste à implémenter — elle
  nécessite une source de trames (capture réseau) que ce crate ne fournit
  pas encore.
