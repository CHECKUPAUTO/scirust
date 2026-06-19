# SciRust — Verticaux industriels : plan d'implémentation

Complément technique de `INDUSTRIAL_ROADMAP.md` (qui couvre l'adoption /
go-to-market). Ici : les **domaines industriels** à approfondir (présents)
et à ouvrir (absents), chacun monté sur l'ADN du projet — **des garanties**,
pas seulement de la précision.

## Non-négociables (= Definition of Done par item)

1. **Rust pur, zéro FFI** ; **déterminisme bit-exact** (PRNG germé, ordre fixe).
2. **Aucune affirmation sans test** : oracle honnête ou test de propriété — pas
   de stub. Le test mesure la garantie revendiquée.
3. Le différenciateur est **toujours une garantie** : déterminisme, couverture
   conforme (sans hypothèse de distribution), borne certifiée (IBP/CROWN),
   inférence vérifiable (Freivalds), audit hash-chaîné, sûreté ASIL/SIL.
4. **CLI/démo** quand pertinent ; **docs** quand une commande CLI est ajoutée.
5. **8 gates verts** (fmt, clippy `-D warnings --all-targets`, build, test, simd,
   aarch64, doc, deny) + `--features wgpu` si concerné. **Commit + push** par item.

---

## Phase 1 — Approfondir la PdM (gains rapides, relier l'existant)

- **I1 · RUL à intervalles conformes** — `scirust-pdm` (`rul` × `conformal_guard`).
  Durée de vie restante avec **intervalle à couverture garantie** `[t_bas, t_haut]`
  sans hypothèse de distribution. Oracle : couverture empirique ≥ 1−α sur
  trajectoires de dégradation simulées germées.

- **I2 · Sévérité vibratoire ISO 10816/20816** — `scirust-pdm`.
  Zones normalisées A/B/C/D par classe de machine → verdict de conformité.
  Oracle : seuils de la norme vérifiés par cas de table.

- **I3 · MCSA — signature du courant moteur** — `scirust-signal`/`pdm`.
  Barres rotoriques / excentricité / défauts stator par bandes latérales
  `(1±2ks)·f` autour du fondamental. Oracle : signaux synthétiques à défaut
  connu → sidebands au bon offset.

## Phase 2 — Infrastructure d'estimation partagée

- **I4 · Kalman/EKF/UKF déterministes à bornes certifiées** —
  nouveau `scirust-estimation`. Estimation d'état bit-exacte + **filtrage
  ensembliste (set-membership)** à enveloppe d'erreur prouvée. Débloque I7.
  Oracle : convergence vs système linéaire connu ; l'ensemble certifié contient
  toujours l'état vrai.

## Phase 3 — Sûreté & sécurité OT (monter sur les garanties)

- **I5 · Moniteur « Simplex » certifié** — `scirust-func-safety` ×
  `scirust-core::nn::ibp` (CROWN). Contrôleur simple vérifié en repli, activé
  dès que la sortie NN quitte l'enveloppe sûre prouvée. Oracle : sur une boîte
  L∞, le moniteur ne laisse jamais passer une sortie hors enveloppe.

- **I6 · IDS pour protocoles OT/ICS** — `scirust-ids` × `opcua`/`mqtt`.
  Anomalies Modbus/DNP3/OPC-UA avec **taux de fausse-alarme garanti** (conformal).
  Oracle : FAR empirique ≤ α sur trafic normal ; injection détectée.

## Phase 4 — Nouveaux verticaux (nouveaux crates, même ADN)

- **I7 · BMS — gestion batterie** — nouveau `scirust-bms` (utilise I4).
  SoC/SoH par EKF, alerte précoce d'emballement thermique, **bornes SoH
  conformes**. Oracle : SoC suivi sur modèle de cellule simulé ; couverture SoH.

- **I8 · Réseaux électriques / smart grid** — nouveau `scirust-grid`.
  Fréquence/RoCoF, phaseurs synchronisés, îlotage, THD/harmoniques. Oracle :
  signaux réseau synthétiques à fréquence/THD connus.

- **I9 · SHM — surveillance structurelle** — nouveau `scirust-shm`.
  Analyse modale (fréquences propres, amortissement), dommage par dérive
  fréquentielle, fatigue (loi de Paris) + RUL conforme. Oracle : masse-ressort
  connu → fréquences propres exactes.

- **I10 · Médical ECG/PPG (IEC 62304)** — nouveau `scirust-biomed`.
  Arythmie avec **ensembles de prédiction conformes** + piste d'audit. Oracle :
  R-pics sur ECG synthétique ; couverture des ensembles conformes.

## Phase 5 — Preuve de certification

- **I11 · Démonstrateur DO-178C / ferroviaire SIL** — `scirust-func-safety`
  + `scirust-runtime`. Déterminisme bit-exact + inférence vérifiable +
  attestation hash-chaînée + ASIL/SIL en un **evidence pack reproductible**.
  Oracle : rejeu bit-identique + chaîne vérifiée + contre-exemple sur falsification.

---

## Ordre d'exécution

I1→I3 (PdM) → I4 (estimation) → I5,I6 (sûreté/sécurité) → I7→I10 (verticaux)
→ I11 (certification). Chaque item livré complet (code + oracle + gates +
commit/push) avant le suivant. Statut suivi dans `CHANGELOG.md`.
