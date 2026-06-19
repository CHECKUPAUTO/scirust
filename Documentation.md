# Documentation SciRust 🦀

Bienvenue dans la documentation de **SciRust**, un framework d'apprentissage profond (Deep Learning) et de calcul scientifique écrit entièrement en **Rust pur**.

## 1. Qu'est-ce que SciRust ?

SciRust est une plateforme de recherche et de développement pour l'intelligence artificielle. Contrairement à beaucoup d'autres outils (comme PyTorch ou TensorFlow) qui s'appuient sur des bibliothèques complexes en C++ ou Python, SciRust est construit de A à Z en Rust.

**Pourquoi est-ce important ?**
- **Transparence totale** : Vous pouvez lire chaque ligne de code du calcul, de la couche réseau au noyau mathématique.
- **Sécurité et Fiabilité** : Profite des garanties de mémoire et de sécurité de Rust.
- **Indépendance** : Aucune dépendance externe complexe (FFI) n'est requise.

## 2. Philosophie et Avantages Clés

SciRust n'essaie pas de remplacer les géants de l'industrie, mais propose une approche différente axée sur la **confiance** et la **reproductibilité**.

### Déterminisme Bit-à-Bit
Dans de nombreux frameworks, lancer deux fois le même calcul peut donner des résultats légèrement différents (à cause du parallélisme). SciRust garantit un **déterminisme bit-à-bit** : le résultat sera strictement identique, peu importe le nombre de processeurs utilisés. C'est crucial pour l'auditabilité.

### Auditabilité
Comme tout est en Rust, il est facile de vérifier que le code fait exactement ce qu'il dit. Il n'y a pas de "boîte noire" logicielle.

### Oracles de Validation
Chaque fonction mathématique dans SciRust est validée par rapport à un "oracle" (une référence de confiance). On ne suppose pas que le résultat est correct, on le mesure.

## 3. Domaines d'Application

SciRust est particulièrement utile dans les domaines où la précision, la sécurité et la petite taille du logiciel sont critiques :

- **Systèmes Embarqués (Edge AI)** : Grâce à sa faible empreinte et sa capacité de quantification (réduction de la taille des modèles), il tourne parfaitement sur de petits appareils.
- **Secteurs Régulés (Aérospatial, Médical, Finance)** : Là où chaque décision de l'IA doit être reproductible et explicable pour des raisons de sécurité ou de conformité.
- **Recherche Scientifique** : Pour découvrir des lois mathématiques à partir de données grâce à la régression symbolique.
- **Audit de Sécurité** : Pour les entreprises qui ont besoin de certifier l'intégralité de leur chaîne de calcul.

## 4. Ce qu'il est possible de réaliser

SciRust couvre un large éventail de techniques modernes :

- **Apprentissage Profond (Deep Learning)** : Construction de réseaux de neurones (MLP, CNN, Transformers) avec différenciation automatique (autograd).
- **Renforcement par Apprentissage (RL)** : Support complet pour le Q-Learning tabulaire, DQN et PPO avec clipping.
- **Computer Vision Avancée** : Architectures ResNet-18/34 et Vision Transformer (ViT) avec pooling global.
- **Modèles Génératifs (VAE)** : Auto-encodeurs variationnels avec trick de reparamétrage pour la génération latente.
- **Transformers et MoE** : Couches Mixture of Experts avec routage Top-k pour l'extensibilité des modèles.
- **Graphes (GNN)** : Réseaux de neurones convolutifs sur graphes (GCN) pour données structurées.
- **Speech AI et Audio** : Encodeurs audio et fonction de perte CTC pour la reconnaissance de la parole.
- **Adaptation PEFT (LoRA)** : Low-Rank Adaptation pour un ajustement efficace des modèles pré-entraînés.
- **Calcul Scientifique Avancé** : Solveur FEM (Méthode des Éléments Finis) 1D pour les équations physiques.
- **Régression Symbolique** : Découvrir des formules mathématiques (ex: `f(x) = sin(x) + x^2`) à partir d'observations.
- **Optimisation Évolutionnaire** : Utiliser des algorithmes inspirés de la nature (comme NSGA-II) pour résoudre des problèmes complexes.
- **Quantification int8** : Diviser par 4 la taille des modèles pour les faire tenir sur de petits processeurs sans perdre en précision.
- **Accélération GPU** : Utiliser la puissance des cartes graphiques via WebGPU (wgpu) ou NVIDIA Tensor Cores (cuBLAS).
- **Physics-Informed Neural Networks (PINN)** : Intégration de lois physiques (équations différentielles) directement dans la fonction de perte pour modéliser des phénomènes complexes.
- **Contrats d'Invariants Formels** : Garanties mathématiques (absence de NaN/Inf, bornes de valeurs) pour les applications critiques (médical, aérospatial).
- **Tenseurs CSR et Noyaux SpMM** : Optimisation de la mémoire et des calculs pour les modèles creux sur cibles embarquées.
- **Exécution en Enclave Sécurisée (TEE)** : Runtime durci compatible #![no_std] pour exécution isolée (TrustZone/SGX) sans allocateur OS.

## 5. Guide des Commandes

SciRust s'utilise principalement via le terminal avec `cargo`, l'outil standard de Rust.

### Installation
Ajoutez ceci à votre fichier `Cargo.toml` :
```toml
[dependencies]
scirust-core = { path = "..." }
```

### Compiler et Tester
- **Vérifier le projet** : `cargo check --workspace`
- **Lancer tous les tests** (plus de 550 tests valident le framework) : `cargo test --workspace`
- **Compiler en mode optimisé** (recommandé pour l'IA) : `cargo build --release`
- **Activer le support GPU** : Ajoutez `--features wgpu` à vos commandes.

### Exemples d'Exécution
- **Entraînement MNIST (chiffres manuscrits)** :
  ```bash
  cargo run --example mnist_classifier --release
  ```
- **Démo de compression Transformer** :
  ```bash
  cargo run -p transformer_compress --release
  ```
- **Benchmark de multiplication de matrices** :
  ```bash
  cargo run -p scirust-core --example bench_matmul --release
  ```

## 6. Exemple de Code (Entraînement rapide)

Voici comment créer et entraîner un modèle très simple en quelques lignes :

```rust
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{Sequential, Linear, ReLU, PcgEngine};

fn main() {
    let mut rng = PcgEngine::new(42);

    // Création d'un modèle simple
    let mut model = Sequential::new()
        .add(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
        .add(ReLU::new())
        .add(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));

    // Entraînement sur une boucle
    for epoch in 0..100 {
        let tape = Tape::new();
        // ... (chargement données et calcul du gradient)
        println!("Époque {}: calcul en cours...", epoch);
    }
}
```

## 7. scirust-tensor — Algèbre Tensorielle et Optimisation de Graphe

Le module `scirust-tensor` introduit une couche d'abstraction de haut niveau pour manipuler des tenseurs complexes tout en garantissant des performances maximales via la compilation de graphe.

### Pourquoi utiliser scirust-tensor ?
- **Einsum** : Écrivez des opérations complexes (Multi-Head Attention, Contrractions) en une seule ligne de code lisible.
- **Fusion d'Opérateurs** : Réduisez les accès mémoire en fusionnant les activations et les biais directement dans les kernels de calcul.
- **Déterminisme Garanti** : Comme tout SciRust, chaque calcul est reproductible bit-à-bit.

### Exemple : Multi-Head Attention
```rust
use scirust_tensor_einsum::einsum;

// Signature Einstein pour l'attention : Batch, Heads, SeqLen, Dim
// (b, h, i, d) , (b, h, j, d) -> (b, h, i, j)
let attention_scores = einsum("bhid,bhjd->bhij", &[&queries, &keys]).unwrap();
```

### Installation
Ajoutez ceci à votre `Cargo.toml` :
```toml
[dependencies]
scirust-tensor-core = { path = "scirust-tensor-core" }
scirust-tensor-einsum = { path = "scirust-tensor-einsum" }
```

## 8. Détection d'Événements (scirust-events)

Le module `scirust-events` permet d'analyser des flux de données (séries temporelles, logs, signaux) pour détecter et classifier des événements de manière déterministe. Il est conçu pour les applications critiques où la reproductibilité est essentielle.

**Exemple d'utilisation :**

```rust
use scirust_events_core::{EventStream, EventDetector};
use scirust_events_models::SpikeDetector;
use scirust_events_runtime::EventRuntime;

fn main() {
    let data = vec![0.1, 0.2, 1.5, 0.3, 0.1]; // Signal avec un spike
    let mut stream = EventStream::new(data, 3, 1);
    let detector = SpikeDetector::new(1.0, 0.5);
    let runtime = EventRuntime::new(Box::new(detector));

    let events = runtime.process_all(&mut stream);
    for e in events {
        println!("Événement détecté à t={}: {}", e.timestamp, e.confidence);
    }
}
```

- **Cas d'usage** : Surveillance industrielle, détection d'anomalies réseau, analyse de spikes neuronaux.
- **Garantie** : Déterminisme bit-à-bit sur le score d'anomalie.

---

### 8. Event Detection (scirust-events) [EN]
The `scirust-events` module provides tools to analyze data streams (time series, logs, signals) to detect and classify events deterministically. It is built for mission-critical applications where reproducibility is mandatory.

### 8. Detección de Eventos (scirust-events) [ES]
El módulo `scirust-events` permite analizar flujos de datos para detectar y clasificar eventos de forma determinista. Diseñado para aplicaciones críticas donde la reproductibilidad es fundamental.

### 8. Ereigniserkennung (scirust-events) [DE]
Das Modul `scirust-events` ermöglicht die Analyse von Datenströmen zur deterministischen Erkennung und Klassifizierung von Ereignissen. Entwickelt für kritische Anwendungen.

### 8. 事件检测 (scirust-events) [ZH-CN]
`scirust-events` 模块提供了一套用于确定性地检测和分类数据流（时间序列、日志、信号）中事件的工具。专为对可重现性有严格要求的关键任务应用而设计。

## 9. Neuro-Symbolique Avancé (scirust-neuro-symbolic)

La crate `scirust-neuro-symbolic` introduit des capacités de raisonnement hybride au sein de SciRust, combinant la puissance de l'apprentissage profond avec la rigueur de la logique symbolique.

### Modules inclus :
- **Symbolic Regression** : Recherche de lois mathématiques guidée par des réseaux de neurones.
- **Logic Programming** : Moteur Datalog pour le raisonnement sur des faits et des règles.
- **SAT/SMT Solvers** : Interfaces pour la résolution de problèmes de satisfaisabilité formelle.
- **Knowledge Graphs** : Représentation et raisonnement sur des graphes de connaissances.
- **Differentiable Reasoning** : Couches de logique floue différentiables (T-norms) intégrées aux tenseurs.

**Exemple de raisonnement flou :**
```rust
use scirust_neuro_symbolic::neural::DifferentiableLogicLayer;
use scirust_core::autodiff::reverse::Tensor;

let layer = DifferentiableLogicLayer::new("LogicLayer");
let a = Tensor::from_vec(vec![0.8], 1, 1);
let b = Tensor::from_vec(vec![0.7], 1, 1);
let result = layer.fuzzy_and(&a, &b); // 0.56
```



## 10. Surveillance Industrielle et Automobile (v0.14-dev)

SciRust inclut désormais un ensemble de crates pour la **surveillance de lignes de production industrielles**, notamment dans le domaine automobile.

### 10.1 Traitement du Signal (`scirust-signal`)

Traitement du signal pur-Rust pour l'analyse vibratoire et le diagnostic de machines :

- **FFT radix-2** (Cooley-Tukey, forward + inverse)
- **Fenêtres** : Hanning, Hamming, Blackman, Blackman-Harris, Flat-top
- **Features temporelles** : RMS, facteur de crête, kurtosis, skewness, taux de passage par zéro, autocorrélation, énergie, entropie
- **Features spectrales** : PSD, centroïde spectral, étalement, entropie spectrale, rolloff, puissance de bande, flatness
- **Diagnostic de roulements** : BPFO, BPFI, BSF, FTF avec détection de défauts dans le spectre d'enveloppe
- **Analyse d'ordre** : order tracking, rééchantillonnage angulaire, spectre d'ordre pour machines à vitesse variable

```rust
use scirust_signal::{fft_real, hanning, rms, kurtosis, crest_factor};

let signal: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin()).collect();
let window = hanning(1024);
let r = rms(&signal);
let k = kurtosis(&signal);
let cf = crest_factor(&signal);
```

### 10.2 Connecteur OPC-UA (`scirust-opcua`)

Connecte les PLC/SCADA industriels au pipeline SciRust :

- **Trait `OpcuaClient`** : abstraction pour lecture de variables, abonnement, browse
- **`SimulatedOpcuaClient`** : 8 capteurs simulés (vibration 3 axes, température moteur/liquide, pression hydraulique, courant moteur, débit liquide)
- **Bridge** : conversion des valeurs OPC-UA → `EventStream` SciRust
- Prêt pour l'intégration d'un vrai stack OPC-UA (crate `opcua`) via feature flag

### 10.3 Publication MQTT (`scirust-mqtt`)

Publie les événements détectés vers des brokers MQTT pour l'Industrie 4.0 :

- **Trait `MqttPublisher`** : abstraction de publication
- **Format SparkPlug B** : payloads compatibles Industrie 4.0
- **Sévérité** : Info / Warning / Critical (dérivée du score de confiance)
- **`SimulatedMqttPublisher`** : backend de test sans broker réel
- **`MonitoringStation`** : configuration de station de surveillance

### 10.4 Maintenance Prédictive (`scirust-pdm`)

Modules de maintenance prédictive pour machines industrielles :

- **Health Index** : score 0..1 combinant plusieurs indicateurs capteurs, avec lissage EMA et classification ISO 13374 (Good/Degraded/Warning/Critical/Failed)
- **RUL (Remaining Useful Life)** : estimateurs linéaire et exponentiel avec intervalles de confiance 95%
- **Détection de changement** : CUSUM (ISO 7870) et Page-Hinkley pour détection de régime
- **Détecteurs spécialisés** :
  - `ImbalanceDetector` : déséquilibre rotor (pic 1x RPM dominant)
  - `MisalignmentDetector` : désalignement (pics 2x/3x RPM)
  - `BearingFaultDetector` : défauts de roulement (BPFO/BPFI/BSF/FTF)
  - `CavitationDetector` : cavitation de pompe (haute kurtosis + bande HF)

### 10.5 MLOps Industriel (`scirust-mlops`)

Opérations ML pour le déploiement industriel continu :

- **Détection de dérive** : Data drift via Population Stability Index (PSI), Model drift via MAE relative
- **Shadow deployment** : exécution parallèle modèle production / modèle candidat, recommandation Promote/Keep/Inconclusive
- **OTA signé** : distribution de modèles Over-The-Air avec signature cryptographique et vérification d'intégrité

### 10.6 Sûreté de Fonctionnement (`scirust-func-safety`)

Conformité ISO 26262 / IEC 61508 pour l'IA automobile :

- **ASIL A-D** : niveaux d'intégrité, configuration automatique (lockstep, watchdog, latence max, redondance)
- **Traçabilité exigences** : matrice exigences → code → tests, export JSON, rapport de certification
- **Fault injection** : 6 types de fautes (bit-flip, stuck-at, noise, zero-out, scale-shift, overflow), tests par lots
- **Mode dégradé** : 4 niveaux (Full → Reduced → Safety → Emergency), hystérésis, safe state
- **Audit log hash-chainé** : journal immuable des décisions de sécurité, vérification d'intégrité de chaîne

### 10.7 Kit d'Intégration (`scirust-integration`)

Librairie unificatrice pour simplifier l'intégration industrielle :

- **`Backend`** : abstraction unifiée OPC-UA + MQTT avec feature flags (`real-opcua`, `real-mqtt`)
- **`BackendFactory`** : création automatique, fallback simulé → réel
- **`PipelineConfig`** : configuration JSON complète (backend, stations, capteurs, Health Index, RUL, drift)
- **`Pipeline`** : pipeline complet Backend → Signal → Events → Health → RUL → MQTT → Audit
- **Templates** : génération de projets (`minimal`, `automotive`, `bearing`, `pdm`)

### 10.8 CLI Industriel (`scirust-industrial`)

Outil en ligne de commande pour faciliter l'intégration :

```bash
# Découvrir les capteurs disponibles sur le PLC
scirust-industrial discover --simulated

# Tester la connexion OPC-UA
scirust-industrial test-opcua --simulated --samples 5

# Tester la connexion MQTT
scirust-industrial test-mqtt --simulated

# Générer un fichier de configuration
scirust-industrial gen-config --output config.json --template automotive --stations 3

# Générer un projet complet de surveillance
scirust-industrial scaffold --name line3-monitor --template automotive

# Lancer le pipeline de surveillance
scirust-industrial run --config config.json --cycles 100 --report report.json

# Diagnostiquer les problèmes d'intégration
scirust-industrial doctor --config config.json
```

### 10.9 Détection d'Intrusions Réseau (`scirust-ids`)

Système de détection d'intrusions (IDS) complet intégré au pipeline SciRust :

- **Capture réseau** — trait `NetworkCapture` + `SimulatedCapture`, conversion `RawPacket` → `Flow`
- **Parseurs protocoles** — HTTP (path traversal, command injection, méthodes dangereuses), DNS (domain length, tunneling TXT, NXDOMAIN), SSH (version downgrade)
- **Détecteurs d'attaques** :
  - `PortScanDetector` — scan vertical, horizontal, complet
  - `DdosDetector` — SYN flood, RST flood, volumétrique, applicatif
  - `BruteForceDetector` — password, dictionary, credential stuffing
  - `DnsTunnelDetector` — tunneling DNS (exfiltration)
  - `BeaconDetector` — beaconing C2 (régularité temporelle)
- **Apprentissage ML** — autoencodeur de reconstruction pour détection d'anomalies non supervisée, calibration de seuil automatique
- **Corrélation d'alertes** — multi-attack, escalation (scan → brute force), attaques coordonnées
- **Export SIEM** — JSON, NDJSON, CEF (ArcSight), Syslog (RFC 5424), LEEF (QRadar)
- **Moteur intégré** — orchestre capture → parsing → détection → corrélation → ML → SIEM

```rust
use scirust_ids::*;

let mut engine = IdsEngine::with_defaults();
let mut window = FlowWindow::new(0.0, 60.0);
// ... remplir window avec des flux réseau ...
let report = engine.analyze(&window, 1000.0);

// Corréler et exporter
let mut correlator = AlertCorrelator::with_defaults();
let correlations = correlator.add_results(&report.results, report.timestamp);

let mut siem = SiemExporter::with_defaults();
siem.push_results(&report.results, report.timestamp, "my-ids");
let json = siem.flush().unwrap();
```

```bash
cargo test -p scirust-ids   # 66 tests, 0 failures
cargo run -p ids_demo       # Démo 4 scénarios d'attaque
```

### 10.10 Exemple d'Intégration Complète (`industrial-monitor`)

L'exemple `industrial_monitor` démontre la chaîne complète :

```
OPC-UA (PLC) → Signal Processing → Event Detection → Health Index
→ RUL Estimation → CUSUM → MQTT Publishing → Audit Log → Functional Safety → MLOps Drift
```

```bash
cargo run -p industrial-monitor
```

---

## 11. Nouvelles Fonctionnalités (v0.14-dev)

### Checkpointing & Reprise d'Entraînement
Sauvegarde et restauration complète de l'état d'entraînement (poids + optimiseur + époque). Format JSON interopérable.

### DataLoader avec Batch/Shuffle/Prefetch
Itérateur de mini-batch avec mélange Fisher-Yates déterministe, seeds reproductibles, et option drop_last.

### ONNX Export
Export des modèles au format ONNX (JSON intermédiaire) pour interopérabilité avec ONNX Runtime, Netron, et les chaînes de déploiement.

### Automatic Mixed Precision (AMP)
Entraînement en précision mixte FP16/BF16 avec loss scaling dynamique et détection d'overflow.

### Differential Privacy (DP-SGD)
Protection de la vie privée via clipping des gradients + bruit gaussien calibré. Moments accountant pour le suivi du budget (ε, δ).

### Model Pruning
Élagage par magnitude, structuré (colonnes/lignes), et Lottery Ticket Hypothesis avec rewinding.

### Distributed Training
Primitives all-reduce en ring topology pour l'entraînement data-parallèle multi-processus.

### TensorBoard Logging
Enregistrement de métriques au format CSV et TensorBoard pour visualisation en temps réel.

### Neural Architecture Search (NAS)
Recherche évolutionnaire d'architectures sur l'espace (couches, dimensions, activations) avec optimisation multi-objectif.

### Optimiseurs Avancés
RMSprop, AdamW (weight decay découplé), et LAMB (Layer-wise Adaptive Moments) désormais accessibles.

### Opérations Fusionnées
Kernels matmul+SiLU, matmul+GELU, matmul+LayerNorm en un seul passage mémoire pour réduire la pression sur la bande passante.

## 12. Conclusion

SciRust est le framework de choix pour ceux qui privilégient la **compréhension** et la **rigueur** sur la vitesse brute ou la facilité de Python. C'est un outil puissant pour bâtir une IA de confiance, de la recherche à l'embarqué.

---
*Pour plus de détails techniques, consultez le rapport complet dans `paper/SciRust-technical-report.md`.*

## 13. Recherche → Fonctions (extensions de la tape N-D)

La tape autograd N-D porte désormais une pile d'apprentissage profond complète,
chaque brique adossée à un papier de recherche et à un test (gradient check ou
oracle). Voir [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md) (14/20 livrés).

- **LM décodeur causal** entraînable de bout en bout (embeddings token + position,
  attention causale, cross-entropy fusionnée) ; sur-apprend une séquence exactement.
- **Couches LLaMA** : RMSNorm, SwiGLU, bloc LLaMA, RoPE, attention groupée /
  multi-requête (GQA/MQA).
- **Optimiseurs déterministes** : Adam, AdamW, Lion, Muon (Newton–Schulz), Schedule-Free, AdEMAMix et SOAP (Adam dans la base propre de Shampoo).
- **IA certifiable** : Interval Bound Propagation **et CROWN** (bornes plus
  serrées par relaxation linéaire) — bornes de sortie *prouvées*
  et certificat de robustesse.
- **Réductions reproductibles** indépendantes de l'ordre (bit-identiques quel que
  soit le nombre de threads).
- **Décodage spéculatif exact** ; **FlashAttention** (softmax en ligne) ;
  **DeltaNet** (attention linéaire à règle delta) ;
  **Mamba** (état-espace sélectif / scan sélectif) ;
  **RetNet** (rétention / attention linéaire) ;
  **GLA** (attention linéaire à porte) ;
  **HGRN** (RNN linéaire à porte) ;
  **Neural ODE** (backprop à travers un solveur RK4) ; un réseau de neurones informé par la physique (PINN) qui résout un problème aux limites avec le résidu de l'EDP dans la fonction de perte.
- **Compression** : élagage Wanda (activation-aware), SmoothQuant, GPTQ (quantification int8 des poids par feedback d'erreur d'ordre 2), AWQ (quantification int8 des poids basée sur une recherche et consciente des activations).

Nouvelles commandes CLI :
- `scirust certify [--seed N] [--eps E]` — bornes prouvées d'un MLP ReLU (IBP **et** CROWN, les bornes plus serrées par relaxation linéaire, côte à côte).
- `scirust lm [...] [--opt adam|adamw|lion|schedule-free|ademamix|soap|lookahead|lamb|adan|adafactor|shampoo|prodigy]` — entraîne le LM décodeur N-D.
- `scirust deltanet [--seed N] [--steps S]` — entraîne une couche DeltaNet (attention linéaire à règle delta) à une seule tête pour ajuster une séquence ; affiche la réduction de la MSE.
- `scirust mamba [--seed N] [--steps S]` — entraîne une couche Mamba à état-espace sélectif (scan S6) pour ajuster une séquence ; affiche la réduction de la MSE.
- `scirust retnet [--seed N] [--steps S]` — entraîne une couche de rétention RetNet (attention linéaire, forme récurrente ≡ forme parallèle) pour ajuster une séquence ; affiche la réduction de la MSE.
- `scirust gla [--seed N] [--steps S]` — entraîne une couche d'attention linéaire à porte GLA (porte d'oubli dépendante des données) pour ajuster une séquence ; affiche la réduction de la MSE.
- `scirust hgrn [--seed N] [--steps S]` — entraîne un mélangeur de tokens HGRN à RNN linéaire à porte (porte d'oubli bornée inférieurement) pour ajuster une séquence ; affiche la réduction de la MSE.
- `scirust rwkv [--seed N] [--steps S]` — entraîne une couche de mélange temporel RWKV (WKV ; décroissance temporelle par canal + bonus) pour ajuster une séquence ; affiche la réduction de la MSE.
- `scirust conformal [--seed N] [--alpha A]` — intervalles conformes à couverture garantie (sans hypothèse de distribution).
- `scirust calibrate [--seed N]` — mise à l'échelle de température ; ajuste T pour réduire l'erreur de calibration attendue (ECE) sans modifier l'exactitude.
- `scirust pinn [--seed N] [--steps S]` — réseau informé par la physique ; résout le BVP `u''=−u` (résidu de l'EDP dans la loss), vérifié vs `sin x`.
- `scirust gptq [--seed N] [--samples S] [--damp D]` — quantification int8 des poids GPTQ ; affiche la réduction d'erreur de calibration par rapport au round-to-nearest.
- `scirust awq [--seed N] [--samples S] [--grid G]` — quantification int8 des poids AWQ consciente des activations ; affiche l'exposant de mise à l'échelle sélectionné et la réduction d'erreur de calibration par rapport au round-to-nearest.
- **scirust bitnet [--seed N]** — quantification ternaire {-1,0,+1} des poids BitNet b1.58 (~1,58 bit/poids) ; vérifie la multiplication matricielle sans multiplication.

## 14. CLI Industriel — Référence Complète

Le CLI `scirust-industrial` facilite l'intégration de SciRust avec les systèmes industriels réels.

### Installation

```bash
cargo install --path scirust-industrial   # fournit le binaire `scirust-industrial`
# ou en place : cargo run -p scirust-industrial -- <commande>
```

### Commandes

| Commande | Description | Options |
|----------|-------------|---------|
| `discover` | Liste les capteurs disponibles sur le serveur OPC-UA | `--endpoint`, `--filter`, `--simulated` |
| `test-opcua` | Teste la connexion OPC-UA et lit des valeurs | `--endpoint`, `--simulated`, `--samples N` |
| `test-mqtt` | Teste la connexion MQTT et publie un message | `--host`, `--port`, `--simulated`, `--topic` |
| `gen-config` | Génère un fichier de configuration de pipeline | `--output`, `--template`, `--stations N`, `--line-id` |
| `scaffold` | Génère un projet de surveillance complet | `--name`, `--output`, `--template` |
| `run` | Lance un pipeline de surveillance depuis un fichier de config | `--config`, `--cycles N`, `--report` |
| `doctor` | Diagnostique les problèmes d'intégration | `--config` |

### Templates disponibles pour `gen-config` et `scaffold`

| Template | Description |
|----------|-------------|
| `minimal` | 1 station, backend simulé, détection de spikes |
| `automotive` | Ligne automobile multi-stations avec diagnostic roulement, RUL, MQTT, audit |
| `bearing` | Détection de défauts de roulement (FFT enveloppe, BPFO/BPFI/BSF) |
| `pdm` | Maintenance prédictive (Health Index, RUL, CUSUM) |

### Flux d'intégration recommandé

```bash
# 1. Scaffold un projet
scirust-industrial scaffold --name line3-monitor --template automotive

# 2. Vérifier que tout fonctionne
cd line3-monitor
scirust-industrial doctor --config config.json

# 3. Personnaliser la configuration
# Éditer config.json : endpoint OPC-UA, broker MQTT, capteurs, seuils

# 4. Passer en mode réel (optionnel)
# Éditer Cargo.toml : décommenter les features real-opcua / real-mqtt
# Éditer config.json : backend_type "opcua"

# 5. Lancer la surveillance
scirust-industrial run --config config.json --cycles 1000
```

### Passage du mode simulé au mode réel

Le mode simulé fonctionne sans aucun matériel. Pour passer en production :

1. **OPC-UA réel** : Ajouter `features = ["real-opcua"]` à `scirust-integration` dans `Cargo.toml`, ajouter la dépendance `opcua = "0.13"`, et changer `backend_type` en `"opcua"` dans `config.json`.
2. **MQTT réel** : Ajouter `features = ["real-mqtt"]`, ajouter `rumqttc = "0.24"`, et configurer `host`/`port` du broker.

Le `BackendFactory` gère automatiquement le fallback : si le backend réel échoue, il bascule vers le mode simulé.

## 15. Détection de Motifs (Pattern Detection)

SciRust propose huit crates spécialisées dans la détection de motifs et de régularités à travers différents types de données.

### 15.1 Vision par Ordinateur (`scirust-vision`)

Détection de motifs visuels dans les images et vidéos :
- **CNN** — réseaux convolutifs pour classification, segmentation et détection d'objets
- **HOG** (Histogram of Oriented Gradients) — descripteur de forme pour la détection de piétons et objets
- **LBP** (Local Binary Patterns) — descripteur de texture invariant à l'illumination
- **Haar** — cascades de classifieurs pour la détection rapide de visages
- **Canny** — détection de contours avec seuillage par hystérésis
- **Otsu** — binarisation automatique par seuillage adaptatif
- **NMS** (Non-Maximum Suppression) — suppression des détections redondantes
- **Template Matching** — correspondance de gabarits par corrélation normalisée

### 15.2 Reconnaissance Audio (`scirust-audio`)

Extraction et reconnaissance de motifs dans les signaux audio :
- **MFCC** (Mel-Frequency Cepstral Coefficients) — coefficients cepstraux pour la reconnaissance vocale
- **Chroma** — vecteurs chromatiques pour l'analyse harmonique et la détection d'accords
- **YIN** — suivi de hauteur tonale (pitch tracking) robuste par autocorrélation
- **Onset Detection** — détection des débuts de notes via flux spectral
- **Spectral Features** — centroïde, étalement, rolloff, flatness et entropie spectrale

### 15.3 Motifs de Graphes (`scirust-graph`)

Analyse structurelle des graphes pour la découverte de motifs topologiques :
- **Isomorphisme de sous-graphes** — recherche de motifs exacts dans un graphe hôte (VF2)
- **Découverte de motifs** (motif discovery) — énumération et comptage de sous-graphes fréquents
- **Détection de communautés** — Louvain, Leiden, label propagation, modularité
- **Centralité d'intermédiarité** (betweenness centrality) — identification des nœuds-ponts

### 15.4 Motifs Séquentiels (`scirust-sequential`)

Modélisation et recherche de motifs dans les séquences temporelles et textuelles :
- **HMM** (Modèles de Markov Cachés) — modélisation d'états latents dans les séquences
- **CRF** (Champs Aléatoires Conditionnels) — étiquetage de séquences avec contraintes de transition
- **Viterbi** — décodage du chemin d'états le plus probable
- **Baum-Welch** — estimation des paramètres HMM par EM
- **DTW** (Dynamic Time Warping) — alignement temporel de séquences à vitesses variables
- **KMP** (Knuth-Morris-Pratt) — recherche exacte de motifs dans un texte
- **Boyer-Moore** — recherche rapide de motifs par sauts de caractères

### 15.5 Analyse Multivariée (`scirust-multivariate`)

Extraction de motifs dans les données multidimensionnelles :
- **PCA** (Analyse en Composantes Principales) — réduction dimensionnelle par variance maximale
- **ICA** (Analyse en Composantes Indépendantes) — séparation de sources par indépendance statistique
- **K-Means++** — partitionnement avec initialisation optimisée des centroïdes
- **Mahalanobis** — distance tenant compte de la covariance pour la détection d'anomalies
- **MDS** (Multidimensional Scaling) — projection en basse dimension préservant les distances
- **CCA** (Analyse de Corrélation Canonique) — recherche de relations entre deux ensembles de variables

### 15.6 Détection Non Supervisée (`scirust-unsupervised`)

Détection de motifs anormaux sans étiquettes :
- **Autoencodeur** — reconstruction et score d'anomalie par erreur résiduelle
- **Forêt d'isolement** (Isolation Forest) — partitionnement aléatoire pour isoler les anomalies
- **DBSCAN** — clustering par densité avec identification automatique du bruit
- **LOF** (Local Outlier Factor) — facteur local d'aberration basé sur la densité du voisinage
- **GMM** (Modèles de Mélange Gaussien) — détection par probabilité sous le modèle
- **One-Class SVM** — frontière de décision autour des données normales

### 15.7 Motifs Saisonniers (`scirust-seasonal`)

Détection de cycles et de tendances dans les séries temporelles :
- **STL** — décomposition saisonnière et de tendance par LOESS
- **ACF/PACF** — fonctions d'autocorrélation et autocorrélation partielle pour l'identification de périodicités
- **Mann-Kendall** — test non paramétrique de tendance monotone
- **CUSUM saisonnier** — détection de ruptures dans les séries avec composante saisonnière

### 15.8 NLP Avancé (`scirust-nlp-advanced`)

Traitement avancé du langage naturel pour la détection de motifs textuels :
- **NER** (Reconnaissance d'Entités Nommées) — extraction de personnes, lieux, organisations
- **LDA** (Allocation de Dirichlet Latente) — modélisation thématique de corpus
- **TextRank** — extraction de phrases clés et résumé automatique par graphe
- **MinHash** — estimation rapide de similarité par hachage sensible à la localité (LSH)
- **NaiveBayes** — classification textuelle probabiliste avec lissage de Laplace

## 16. Création d'Algorithmes (Algorithm Creation)

SciRust offre six crates dédiées à la génération, la synthèse et l'optimisation automatique d'algorithmes.

### 16.1 AutoML (`scirust-automl`)

Automatisation du pipeline d'apprentissage automatique :
- **Optimisation bayésienne** — recherche efficace d'hyperparamètres par modélisation par processus gaussien
- **Sélection de modèles** — comparaison automatique et classement de candidats
- **Ensembles** — stacking, bagging, boosting et voting de modèles
- **Feature engineering** — génération, transformation et sélection automatique de caractéristiques

### 16.2 Synthèse de Programmes (`scirust-synthesis`)

Génération automatique de code par synthèse inductive :
- **30+ opérateurs** — bibliothèque d'opérations mathématiques et logiques composables
- **Sketch-based** — synthèse guidée par un squelette partiel de programme
- **Stratégies de recherche** — bottom-up (synthèse ascendante), top-down (descendante), programmation génétique (GP), beam search

### 16.3 Génération d'Algorithmes (`scirust-algogen`)

Création automatique d'algorithmes classiques par composition :
- **Tri** — 10 stratégies (quick, merge, heap, bubble, insertion, selection, shell, counting, radix, bucket)
- **Recherche** — 8 stratégies (binaire, linéaire, exponentielle, Fibonacci, saut, interpolation, ternaire, A*)
- **Graphes** — Dijkstra, Bellman-Ford, Floyd-Warshall, Kruskal, Prim, Tarjan
- **Programmation dynamique** — knapsack, LCS, LIS, édition de distance
- **Diviser pour Régner** — Strassen, Karatsuba, exponentiation rapide
- **Analyse de complexité** — estimation automatique en temps et espace (notation O)

### 16.4 Transformation de Code (`scirust-codetrans`)

Manipulation et optimisation de code source par AST :
- **AST** — représentation en arbre syntaxique abstrait pour Rust, Python et C
- **20 règles d'optimisation** — constant folding, inlining, dead code elimination, copy propagation, strength reduction
- **Refactoring** — renommage, extraction de fonction, simplification conditionnelle
- **Transpilation** — conversion automatique Rust → Python et Rust → C

### 16.5 RL pour la Découverte d'Algorithmes (`scirust-rl-algo`)

Recherche d'algorithmes par renforcement :
- **REINFORCE** — gradient de politique pour l'exploration d'espaces de programmes
- **Actor-Critic** — apprentissage de politique avec fonction de valeur
- **Q-Learning** — recherche d'algorithmes par valeur d'état-action
- **MCTS** (Monte Carlo Tree Search) — exploration arborescente pour la synthèse constructive
- **Méta-apprentissage** — apprentissage de stratégies de synthèse transférables entre domaines

### 16.6 Échafaudage de Code (`scirust-scaffold`)

Génération de code structuré par templates et DSL :
- **DSL** — langage dédié à la description d'architectures algorithmiques
- **Génération multi-langage** — sortie en Rust, Python, C et pseudocode
- **16 templates** — squelettes de projets pré-configurés pour cas d'usage courants
- **Analyse de code** — inspection, métriques et validation du code généré
