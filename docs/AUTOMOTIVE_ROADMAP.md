# Extension automobile / Industrie 4.0 pour SciRust

## Contexte
SciRust dispose d'un socle d'inférence déterministe, de détection d'événements sur flux temporels, et de déploiement edge/embarqué (`no_std`, virgule fixe Q16.16, certification Kani). Cependant, l'intégration dans une chaîne de production automobile réelle est bloquée par l'absence de connecteurs industriels, d'extracteurs de features métier, et de modules applicatifs prêts à l'emploi.

---

## Axe 1 — Connecteurs protocoles industriels (`scirust-bridge`)

```
Implémenter un crate `scirust-opcua` :
  - Client OPC-UA (lecture de nodes temps réel, variables, événements)
  - Support du modèle de données OPC-UA (NodeId, BrowsePath)
  - Abonnement aux changements de valeur (subscription) → alimentation d'un EventStream
  - Sortie : flux normalisé compatible avec scirust-events-core::EventStream

Implémenter un crate `scirust-mqtt` :
  - Client MQTT v3.1.1 / v5 (SparkPlug B pour Industrie 4.0)
  - Publication des événements détectés (Event → payload JSON/CBOR)
  - QoS 1 minimum, keep-alive, last will testament

Optionnel avancé :
  - scirust-modbus (Modbus TCP/RTU)
  - scirust-can (CAN 2.0 / CAN FD pour bus véhicule)
```

**Contrainte :** tout en `no_std` compatible avec `scirust-edge` et `scirust-embedded`.

---

## Axe 2 — Extracteurs de features capteurs (`scirust-signal`)

```
Implémenter un crate `scirust-signal` (ou `scirust-features`) avec :

1. Domaine fréquentiel :
   - FFT (radix-2, fenêtres Hanning/Hamming/Blackman)
   - Densité spectrale de puissance (PSD)
   - Cepstre / cepstrum pour diagnostic engrenages

2. Domaine temporel :
   - RMS, crête, facteur de crête, kurtosis, skewness
   - Enveloppe (transformée de Hilbert)
   - Zero-crossing rate, autocorrélation

3. Spécifique automobile :
   - Analyse d'ordre (order tracking) pour machines tournantes
   - Détection de défauts de roulement (BPFO, BPFI, BSF)
   - Indicateurs normalisés ISO 10816 / ISO 13374

Sortie : feature vectors compatibles avec un EventClassifier ou un SpikeDetector enrichi.
```

---

## Axe 3 — Modules applicatifs (`scirust-predictive-maintenance`)

```
Implémenter un crate `scirust-pdm` (predictive maintenance) :

1. Détecteur de dégradation :
   - Calcul de Health Index à partir d'un flux de features
   - Estimation de Remaining Useful Life (RUL) par régression
   - Détection de changement de régime (CUSUM, Page-Hinkley)

2. Détecteurs spécialisés :
   - Déséquilibre moteur (1x RPM + harmoniques)
   - Défaut d'alignement (2x, 3x RPM)
   - Défaut de roulement (hautes fréquences BPFO/BPFI)
   - Cavitation pompe
   - Fuite pneumatique (analyse ultrason)

3. Classification multi-capteurs :
   - Fusion de features par Kalman
   - Modèle de diagnostic entraînable (CNN 1D / Transformer sur spectrogrammes)
   - Export SRT1 du modèle entraîné pour déploiement edge

4. Sortie standardisée :
   - Event enrichi avec : severity (INFO/WARNING/CRITICAL), remaining_life_hours,
     fault_type, component_id, maintenance_action_recommended
```

---

## Axe 4 — Certification automobile (`scirust-functional-safety`)

```
Adapter SciRust aux contraintes ISO 26262 / IEC 61508 :

1. Niveaux ASIL :
   - Configuration du niveau d'intégrité (ASIL A/B/C/D)
   - Redondance matérielle/logicielle paramétrable (dual lockstep)
   - Watchdog timer pour boucle d'inférence

2. Traçabilité exigences → code :
   - Annotation #[requirement("REQ-SAF-042")] sur fonctions critiques
   - Génération de matrice de traçabilité (requirements → tests → code)
   - Export ReqIF ou format tableur pour dossiers de certification

3. Tests de couverture :
   - MC/DC (Modified Condition/Decision Coverage) sur chemins critiques
   - Injection de fautes (fault injection) sur tenseurs et poids
   - Tests de latence pire-cas (WCET) avec `scirust-edge`

4. Mode dégradé :
   - Fallback déterministe si confiance < seuil
   - Isolation de capteur défaillant (graceful degradation)
   - Journal d'audit immuable (hash chain) de toutes les décisions
```

---

## Axe 5 — Intégration continue industrielle (`scirust-mlops`)

```
1. Pipeline d'entraînement → déploiement :
   - Entraînement sur données historiques (CSV, Parquet, base temps réel)
   - Conversion automatique → SRT1 / QSR1 int8
   - Signature du modèle (Ed25519 ou ECDSA P-256)
   - Déploiement OTA (Over-The-Air) vers flotte edge

2. Monitoring en production :
   - Dérive de données (data drift) : détection de distribution shift
   - Dérive de modèle (model drift) : divergence prédiction vs réalité
   - Alerte automatique si dérive > seuil configurable
   - Tableau de bord (export JSON ligne de commande, compatible Grafana)

3. Validation continue :
   - Benchmark de performance sur hardware cible (cycles, RAM, latence)
   - Régression de précision vs modèle de référence
   - Exécution parallèle modèle courant / nouveau modèle (shadow deployment)
   - Rollback automatique si dégradation
```

---

## Priorisation

| Priorité | Axe | Justification |
|----------|-----|---------------|
| **P0** | Axe 1 (OPC-UA + MQTT) | Sans connecteur, aucune donnée réelle n'entre dans le système |
| **P0** | Axe 2 (FFT + features) | Sans features, les détecteurs n'ont pas de signal exploitable |
| **P1** | Axe 3 (PDM) | Valeur métier directe, s'appuie sur P0 |
| **P1** | Axe 5 (MLOps) | Nécessaire pour itérer en conditions réelles |
| **P2** | Axe 4 (ISO 26262) | Prérequis réglementaire mais lourd, peut démarrer en parallèle |

---

## Métriques de succès par axe

- **Axe 1** : 1 tour de contrôle OPC-UA → EventStream en < 10 ms sur ARM Cortex-A72
- **Axe 2** : FFT 1024 points en < 100 µs sur Cortex-M7 (fixed-point Q16.16)
- **Axe 3** : F1 > 0.90 sur C-MAPSS (NASA turbofan degradation dataset)
- **Axe 4** : 100% MC/DC sur les 20 fonctions critiques d'inférence edge
- **Axe 5** : Détection de drift en < 1% de faux positifs sur dataset synthétique

---

## Statut d'implémentation (Juin 2026)

### Axes implémentés

| Axe | Crate(s) | Statut | Tests | Description |
|-----|----------|--------|-------|-------------|
| **P0 - Axe 1** | `scirust-opcua`, `scirust-mqtt` | ✅ Complété | 6 + 9 | Traits `OpcuaClient` / `MqttPublisher` + simulateurs. Feature flags pour backends réels via `opcua` / `rumqttc` |
| **P0 - Axe 2** | `scirust-signal` | ✅ Complété | 24 | FFT radix-2, 5 fenêtres, features temps/fréquence, BPFO/BPFI/BSF, order tracking |
| **P1 - Axe 3** | `scirust-pdm` | ✅ Complété | 24 | Health Index (ISO 13374), RUL linéaire+exponentiel, CUSUM, Page-Hinkley, 4 détecteurs |
| **P1 - Axe 5** | `scirust-mlops` | ✅ Complété | 19 | Data drift (PSI), model drift, shadow deployment (Promote/Keep/Inconclusive), OTA signé |
| **P2 - Axe 4** | `scirust-func-safety` | ✅ Complété | 33 | ASIL A-D, traçabilité exigences, fault injection (6 types), mode dégradé (4 niveaux), audit log hash-chainé |

### Crates supplémentaires d'intégration

| Crate | Description | Tests |
|-------|-------------|-------|
| `scirust-integration` | Lib unificatrice : `Backend`, `BackendFactory`, `PipelineConfig`, `Pipeline`, templates de code | 32 |
| `scirust-industrial` | CLI 7 commandes : discover, test-opcua, test-mqtt, gen-config, scaffold, run, doctor | — |
| `examples/industrial_monitor` | Démo de bout en bout : OPC-UA → Signal → Events → Health → RUL → MQTT → Safety → MLOps | — |

### Total

- **115 nouveaux tests** pour les crates industriels (1047 dans tout le workspace, 0 échec)
- **Documentation** : 8 langues (FR/EN/ES/DE/ZH/JA/KO/AR) mises à jour pour Documentation.md et le rapport technique
- **Prochaines étapes** : Intégration des crates `opcua` et `rumqttc` pour backends réels, dataset C-MAPSS, benchmark de performance edge

### Commandes de test rapide

```bash
# Tester le pipeline complet en mode simulé
cargo run -p industrial-monitor

# Découvrir les capteurs disponibles
cargo run -p scirust-industrial -- discover --simulated

# Générer et tester une config automotive
cargo run -p scirust-industrial -- gen-config --template automotive --stations 3 --output /tmp/cfg.json
cargo run -p scirust-industrial -- doctor --config /tmp/cfg.json
cargo run -p scirust-industrial -- run --config /tmp/cfg.json --cycles 50

# Scaffolder un projet
cargo run -p scirust-industrial -- scaffold --name my-monitor --template automotive --output /tmp
```
