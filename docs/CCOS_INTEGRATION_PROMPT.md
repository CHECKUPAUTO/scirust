# Prompt — Intégration ccos × récupération sémantique pure (challenger RAG)

> À coller dans une session Claude Code qui a **les deux repos en portée** :
> `ccos` **et** `scirust`. Le but : brancher la plateforme de récupération
> sémantique *pure* de SciRust (`scirust-retrieval`) sur les embeddings que ccos
> possède déjà, puis mesurer face au RAG existant. Pur Rust, zéro FFI,
> déterministe.

---

```text
MISSION
Intègre la plateforme de récupération sémantique *pure* de SciRust (crate
`scirust-retrieval`) dans le projet ccos, pour challenger le RAG existant de ccos
sur la pertinence — de façon déterministe, pure-Rust, zéro FFI. Tu ne réécris pas
d'embeddings : tu implémentes le trait `Encoder` de scirust-retrieval AU-DESSUS de
la source d'embeddings que ccos possède déjà, puis tu mesures.

PHILOSOPHIE (non négociable)
- 100 % Rust, zéro FFI, pas de runtime GPU propriétaire.
- Déterminisme bit-à-bit : RNG seedé, accumulation f32 à ordre fixe.
- Tests-oracles honnêtes (valeurs dérivées à la main), jamais ajustés pour coller
  à une sortie buggée. Zéro stub, zéro TODO.
- `cargo clippy --workspace --all-targets -- -D warnings` propre, `cargo fmt` propre,
  MSRV 1.89.
- Branche descriptive, PR en draft. Ne jamais pousser ailleurs sans permission.

PRÉ-REQUIS
1. Confirme que les crates `scirust-retrieval` et `scirust-license` sont accessibles
   (même workspace, ou chemin relatif). Ajoute-les en dépendances `path` du crate
   ccos qui fait la récupération :
     scirust-retrieval = { path = "../scirust/scirust-retrieval" }
     scirust-license   = { path = "../scirust/scirust-license" }   # si gating voulu
   (Adapte le chemin réel ; ne duplique PAS le code de retrieval dans ccos.)
2. Repère dans ccos : (a) la source d'embeddings actuelle (le composant qui
   transforme un texte en vecteur dense), sa dimension D, et si elle est
   déterministe ; (b) le retriever RAG actuel et son jeu d'évaluation (requêtes +
   documents pertinents connus). Tu t'en serviras comme oracle de comparaison.

TÂCHE 1 — Adapter l'encodeur de ccos (le pont)
Crée un type, p.ex. `CcosEncoder`, qui enveloppe la source d'embeddings de ccos et
implémente `scirust_retrieval::Encoder` :
    impl Encoder for CcosEncoder {
        fn embedding_dim(&self) -> usize { /* D de ccos */ }
        fn encode(&mut self, text: &str) -> Vec<f32> { /* embedding ccos du texte */ }
        // encode_batch a un défaut ; surcharge-le si ccos sait batcher efficacement.
    }
Test-oracle : le même texte encode au même vecteur (déterminisme) ; la dimension
retournée == D.

TÂCHE 2 — Câbler la récupération sémantique pure
- Construis un `SemanticRetriever::new(CcosEncoder::…)`, indexe le corpus de ccos
  (`index_text(id, texte)`), interroge (`retrieve(query, k) -> Vec<Scored>`).
- Ajoute aussi un `HybridRetriever::new(encoder, rrf_k)` (dense + BM25 fusionnés
  par RRF) pour la voie hybride.
- Vérifie l'invariant de base : une requête identique à un document le retrouve en
  rang 1 avec un cosinus ~1.0.

TÂCHE 3 — CHALLENGER RAG (le cœur)
Sur le jeu d'évaluation de ccos, calcule côte-à-côte, avec
`scirust_retrieval::metrics`, pour le RAG actuel de ccos ET pour la récupération
pure (dense + hybride) :
    recall_at_k, precision_at_k, mean_reciprocal_rank, average_precision, ndcg_at_k
Produis un tableau comparatif (k = 1, 5, 10) et un court verdict chiffré : où la
récupération pure gagne/perd, et le gain de déterminisme/auditabilité (même requête
→ même résultat, bit pour bit ; aucune étape générative qui hallucine).
N'invente pas les chiffres : exécute la mesure et rapporte la sortie réelle.

TÂCHE 4 — (option) Amélioration continue + premium
- `ImprovementLoop::new(D, dim_out, seed, cfg)` : enregistre les paires
  (requête, doc pertinent) confirmées par ccos, `train_cycle()`, et montre la
  courbe Recall@k qui monte cycle après cycle.
- Gating premium : protège l'entrée commerciale derrière
  `RetrievalAccess::unlock(&entitlements)` (module `Module::Retrieval`). Pour le
  modèle 1 USD/machine/mois, émets une licence node-lockée
  (`License::new(..).with_node_lock(machine_id)`) et vérifie-la avec
  `verify_license_on_node(.., machine_id)`. En dev, utilise `demo_vendor()` /
  `demo_root()`.

LIVRABLES
- Branche descriptive (p.ex. `ccos-pure-retrieval`), PR en draft.
- Le pont `CcosEncoder` + le câblage retriever, sous tests.
- Le benchmark RAG-vs-pur avec les métriques RÉELLES dans la description de PR.
- Tout vert : tests, clippy -D warnings, fmt.

API DE RÉFÉRENCE (scirust-retrieval, déjà sur master)
- trait Encoder { fn embedding_dim(&self)->usize; fn encode(&mut self,&str)->Vec<f32>;
  fn encode_batch(&mut self,&[String])->Vec<Vec<f32>> }
- SemanticRetriever::new(E) ; .index_text(u64,&str)->Result<(),RetrievalError> ;
  .retrieve(&str,usize)->Vec<Scored> ; .len()/.is_empty()
- HybridRetriever::new(E, rrf_k:f32) ; .index_text ; .retrieve
- ImprovementLoop::new(dim_in,dim_out,seed,cfg:ContrastiveConfig) ; .record ;
  .train_cycle()->Vec<f32> ; .evaluate_recall_at_k(eval,corpus,k)
- RetrievalAccess::unlock(&Entitlements)->Result<Self,LicenseError> ;
  .semantic_retriever(E) / .hybrid_retriever(E,f32) / .improvement_loop(..)
- metrics::{recall_at_k, precision_at_k, reciprocal_rank, mean_reciprocal_rank,
  average_precision, ndcg_at_k}
- Scored { id:u64, score:f32 }
- scirust_license::{Module::Retrieval, verify_license, verify_license_on_node,
  License, node_fingerprint, demo_vendor, demo_root}

NOTE : Encoder::encode prend &mut self (autorise un cache interne d'embeddings).
Si la source ccos est immuable, ignore simplement la mutabilité.
```
