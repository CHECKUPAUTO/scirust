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
  **Neural ODE** (backprop à travers un solveur RK4) ; un réseau de neurones informé par la physique (PINN) qui résout un problème aux limites avec le résidu de l'EDP dans la fonction de perte.
- **Compression** : élagage Wanda (activation-aware), SmoothQuant, GPTQ (quantification int8 des poids par feedback d'erreur d'ordre 2), AWQ (quantification int8 des poids basée sur une recherche et consciente des activations).

Nouvelles commandes CLI :
- `scirust certify [--seed N] [--eps E]` — bornes prouvées d'un MLP ReLU (IBP **et** CROWN, les bornes plus serrées par relaxation linéaire, côte à côte).
- `scirust lm [...] [--opt adam|adamw|lion|schedule-free|ademamix|soap|lookahead]` — entraîne le LM décodeur N-D.
- `scirust deltanet [--seed N] [--steps S]` — entraîne une couche DeltaNet (attention linéaire à règle delta) à une seule tête pour ajuster une séquence ; affiche la réduction de la MSE.
- `scirust mamba [--seed N] [--steps S]` — entraîne une couche Mamba à état-espace sélectif (scan S6) pour ajuster une séquence ; affiche la réduction de la MSE.
- `scirust conformal [--seed N] [--alpha A]` — intervalles conformes à couverture garantie (sans hypothèse de distribution).
- `scirust calibrate [--seed N]` — mise à l'échelle de température ; ajuste T pour réduire l'erreur de calibration attendue (ECE) sans modifier l'exactitude.
- `scirust pinn [--seed N] [--steps S]` — réseau informé par la physique ; résout le BVP `u''=−u` (résidu de l'EDP dans la loss), vérifié vs `sin x`.
- `scirust gptq [--seed N] [--samples S] [--damp D]` — quantification int8 des poids GPTQ ; affiche la réduction d'erreur de calibration par rapport au round-to-nearest.
- `scirust awq [--seed N] [--samples S] [--grid G]` — quantification int8 des poids AWQ consciente des activations ; affiche l'exposant de mise à l'échelle sélectionné et la réduction d'erreur de calibration par rapport au round-to-nearest.
