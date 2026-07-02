//! Sondes natives au protocole — chacune n'envoie que ce qu'un client
//! légitime de ce protocole enverrait pour établir une connexion ou
//! s'annoncer, jamais un paquet générique de scan de port. Chaque module
//! sépare la construction/l'analyse des trames (pur, testé sans réseau) de
//! l'I/O socket (`probe`, testée sur boucle locale — voir les tests de
//! chaque module).

pub mod mdns;
pub mod modbus;
pub mod opcua;
