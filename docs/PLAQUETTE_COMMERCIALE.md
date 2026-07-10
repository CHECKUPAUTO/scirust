# SciRust — Catalogue & offre de licences

> Plaquette commerciale · à destination des clients et intégrateurs
> Le cœur est libre pour l'usage non commercial ; l'usage professionnel se
> déverrouille **par module**, via une licence signée. Contact :
> zekrititarek@gmail.com

---

## 1. En une phrase

**SciRust est une plateforme de calcul scientifique et d'IA, écrite
intégralement en Rust, déterministe au bit près, sans aucune dépendance
C/C++/CUDA — du cloud x86 au calculateur embarqué ARM.** Vous n'achetez pas un
monolithe : vous **déverrouillez les modules dont vous avez besoin**.

Là où Python/PyTorch est conçu pour la recherche — non déterministe, lourd,
difficile à auditer — SciRust est conçu pour les environnements où l'on ne peut
pas se tromper : automobile, énergie, santé, eau, sécurité OT, industrie 4.0.

---

## 2. Trois garanties que personne ne réunit dans un seul socle

| Garantie | Ce que ça change pour vous |
|---|---|
| **Déterminisme bit-à-bit** | Le même calcul donne exactement le même résultat, octet pour octet, quel que soit le nombre de cœurs ou la machine. Indispensable pour la traçabilité, le rejeu et l'audit. |
| **Pureté & portabilité** | 100 % Rust, **zéro FFI**, zéro runtime GPU propriétaire. Surface d'audit minimale, pas de verrou fournisseur, embarquable sur cible contrainte. |
| **Certifiabilité** | Bornes de sortie *prouvées* sur les réseaux, couverture statistique *garantie* (conformal), journal d'audit chaîné par hachage. Briques visant ISO 26262 / IEC 61508 / ISO 21434 / DO-178C. |

---

## 3. Le socle, en chiffres réels

Mesuré sur le dépôt à la date d'édition (`cargo test --workspace`, exécution
complète) :

- **97 crates** dans l'espace de travail, dont **plus de 80 modules métier** ;
- **2 404 tests automatiques — 0 échec** ;
- chaque module est adossé à des **tests-oracles** (valeurs attendues dérivées à
  la main, pas des tests tautologiques) — politique « zéro stub, zéro test
  factice » ;
- validé du serveur x86 jusqu'à l'embarqué ARM (`no_std`, sans allocation, pour
  cibles Cortex-M / Jetson).

---

## 4. Le catalogue

Les modules ci-dessous sont les **unités de licence**. Chacun couvre un domaine
(un ou plusieurs crates). La commande `license-tool modules` en donne la liste
vivante.

### 4.1 Cœur IA & calcul scientifique

| Module | Couverture |
|---|---|
| `core` | Tenseurs, autodiff inverse, zoo de couches NN, optimiseurs |
| `tensor-network` | Décomposition Tensor-Train, MPS, DMRG — compression de modèles |
| `reasoning` | Raisonnement symbolique, régression symbolique, synthèse de programmes |
| `automl` | Génération de pipeline, sélection de modèle, optimisation d'hyperparamètres |
| `evolution` | GA, CMA-ES, OpenES, NSGA-II ; recherche d'architecture neuronale (NAS) |
| `reinforcement-learning` | Apprentissage par renforcement pour la découverte d'algorithmes |

### 4.2 Perception & données

| Module | Couverture |
|---|---|
| `vision` | CNN, détection d'objets, classification, segmentation |
| `audio` | MFCC, chroma, détection d'attaque, suivi de hauteur |
| `nlp` | NER, modélisation de sujets, extraction de relations, classification de texte |
| `graph` | Isomorphisme de sous-graphes, découverte de motifs, détection de communautés |
| `signal` | FFT, fenêtrage, descripteurs temps/fréquence |
| `events` | Détection d'événements et d'anomalies en flux |
| `retrieval` ⭐ | **Récupération sémantique pure (dense)** — index exact déterministe, recherche approchée des plus proches voisins (LSH), ré-ordonnancement par interaction tardive (ColBERT/MaxSim), métriques de pertinence (Recall@k, nDCG@k) et **boucle d'amélioration continue**. Une **alternative auditable au RAG**. *Add-on premium.* |

### 4.3 Verticales industrielles

| Module | Couverture | Référentiel |
|---|---|---|
| `estimation` | Kalman/EKF déterministe, estimation ensembliste (intervalles) | — |
| `navigation` | Fusion GNSS/INS faiblement couplée, multilatération TDOA | — |
| `water` | Localisation acoustique de fuites, analyse de coup de bélier | — |
| `control` | PID, LQR, QP borné, MPC à contraintes d'entrée certifiées | — |
| `battery` | SoC par EKF, alerte précoce d'emballement thermique, bornes conformes de SoH | — |
| `grid` | Fréquence/RoCoF, synchrophaseurs, THD, détection d'îlotage | — |
| `structural-health` | Analyse modale, détection d'endommagement, fatigue (loi de Paris) | — |
| `hvac` | Détection de défauts de CTA, désagrégation de charge (NILM) | — |
| `robotics` | Trajectoires à jerk limité, cinématique planaire, monitoring vitesse/séparation | ISO/TS 15066 |
| `metrology` | Propagation d'incertitude (GUM), variance d'Allan, planification de recalibration | GUM |
| `predictive-maintenance` | Indice de santé, estimation de RUL, détection de défaut machine | — |
| `spc` | Cartes Shewhart/EWMA/CUSUM, règles Western Electric, T² de Hotelling | — |
| `industrial` | Kit d'intégration, auto-config, scaffolding de pipelines de monitoring | — |

### 4.4 Sûreté, fiabilité & sécurité

| Module | Couverture | Référentiel |
|---|---|---|
| `functional-safety` | Preuves de conformité pour l'IA automobile | ISO 26262 / IEC 61508 |
| `reliability` | PFD/PFH pour architectures MooN, mapping SIL, disponibilité de Markov | IEC 61508 |
| `ot-security` | Détection d'intrusion OT/ICS (DSP + changement statistique + ML) | — |
| `mlops` | Détection de dérive, déploiement fantôme, distribution OTA de modèles | — |
| `biomed` | Détection de pics R d'ECG, classification de rythme, ensembles conformes | — |

### 4.5 Edge, embarqué & finance

| Module | Couverture |
|---|---|
| `edge` | Inférence int8 déterministe `no_std`, sans allocation (classe Cortex-M) |
| `trading` | Pipeline crypto auditable : prédictions certifiées, narration LLM, décisions scellées par preuve |

---

## 5. Le modèle de licence — déverrouillage par module

L'usage **non commercial** (étude, recherche, projets personnels, organismes
publics) est **gratuit** sous PolyForm Noncommercial 1.0.0. L'usage
**professionnel** requiert une licence commerciale, qui **déverrouille les
modules choisis**.

Ce déverrouillage n'est pas un engagement sur l'honneur : il est appliqué par un
vrai mécanisme cryptographique, le crate `scirust-license` :

- chaque licence est un fichier signé qui liste précisément les modules ouverts,
  le titulaire et la fenêtre de validité ;
- la signature est **hash-based** (signatures de Lamport sur arbre de Merkle,
  SHA-256 uniquement) — pur Rust, sans courbe elliptique, **post-quantique**,
  déterministe ;
- l'éditeur détient une graine secrète ; le binaire n'embarque que la **racine
  publique** (32 octets). **Falsifier un droit non acheté reviendrait à inverser
  SHA-256.** Toute modification de la liste de modules invalide la signature.

```text
# côté éditeur (hors-ligne, détient la graine secrète)
license-tool issue --licensee "Acme" --id L-2026-001 \
                   --modules navigation,control,functional-safety \
                   --expires 1893456000 > acme.license.json

# côté client (n'embarque que la racine publique)
license-tool inspect acme.license.json     # VALID — navigation, control, functional-safety
license-tool check   acme.license.json --module water   # DENIED (non souscrit)
```

À l'exécution, chaque module appelle `Entitlements::require(...)` : une feature
non couverte est refusée proprement, une licence expirée ou trafiquée est
rejetée.

---

## 6. Offres groupées (suggestions)

| Offre | Modules inclus | Cible |
|---|---|---|
| **Découverte** | `core` + 1 module au choix | Évaluation, PoC |
| **Perception** | `core`, `vision`, `audio`, `signal`, `events`, `retrieval` | Vision/son industriels + recherche sémantique |
| **Industrie 4.0** | `core`, `estimation`, `control`, `predictive-maintenance`, `spc`, `industrial`, `retrieval` | Monitoring & maintenance + base de connaissance |
| **Énergie** | `core`, `battery`, `grid`, `metrology`, `mlops` | Réseaux & stockage |
| **Sûreté critique** | `core`, `functional-safety`, `reliability`, `control`, `estimation` | Automobile / SIL |
| **Sécurité OT** | `core`, `ot-security`, `events`, `signal` | Cybersécurité ICS |
| **Catalogue complet** | tous les modules | Grands comptes, intégrateurs |

> **Add-on Récupération sémantique (`retrieval`) — le « RAG-killer ».** Brique à
> forte valeur ajoutée : récupération sémantique *pure*, déterministe et
> auditable, là où le RAG greffe un générateur LLM stochastique. Inclus dans les
> offres **Perception** et **Industrie 4.0**, ou en supplément de toute offre.
> **Tarif de référence : 1 USD par machine et par mois** (par calculateur
> déployé). La fenêtre mensuelle est portée par la date d'expiration de la
> licence signée (`expires_at`) ; la machine est l'unité de décompte commercial.

Au-delà des licences logicielles : **kits de certification** (dossiers de preuves
prêts à l'audit), **support & SLA**, **intégration** (OPC-UA/MQTT, mise en
service usine) et **licence edge par calculateur déployé**.

---

## 7. Pourquoi maintenant

L'IA entre dans les systèmes critiques, mais la pile dominante (Python/CUDA) n'a
pas été pensée pour la sûreté : non déterministe, immense surface d'audit,
verrou CUDA, aucune chaîne de preuve native. Faire certifier un système à base
d'IA coûte aujourd'hui des années et des millions. **SciRust occupe exactement
ce vide** — et vous n'en payez que les modules dont vous avez besoin.

---

*Le cœur de SciRust est libre pour l'usage non commercial. Pour une licence
commerciale, un devis par modules ou un kit de certification :*
**zekrititarek@gmail.com**
