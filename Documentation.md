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
- **Lancer tous les tests** (plus de 250 tests valident le framework) : `cargo test --workspace`
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
        .push(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
        .push(ReLU)
        .push(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));

    // Entraînement sur une boucle
    for epoch in 0..100 {
        let tape = Tape::new();
        // ... (chargement données et calcul du gradient)
        println!("Époque {}: calcul en cours...", epoch);
    }
}
```

## 7. Conclusion

SciRust est le framework de choix pour ceux qui privilégient la **compréhension** et la **rigueur** sur la vitesse brute ou la facilité de Python. C'est un outil puissant pour bâtir une IA de confiance, de la recherche à l'embarqué.

---
*Pour plus de détails techniques, consultez le rapport complet dans `paper/SciRust-technical-report.md`.*
