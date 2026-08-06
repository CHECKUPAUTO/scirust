# SciRust → CCOS — Renforcer le moat causal, sécuritaire et temporel face au RAG

> **Statut : proposition de conception (design pass).** Ce document ne modifie
> aucun code ; il propose des nouveautés priorisées, chacune reliée à un
> algorithme SciRust *mature*, à un point d'intégration CCOS *réel*, et à une
> analyse de déterminisme honnête. Il est le produit d'une cartographie
> exhaustive des deux dépôts (`CCOS` + `scirust`) et d'une revue adversariale de
> chaque idée (faisabilité, préservation de `replay == live`, maturité SciRust,
> différenciation *structurelle* vs RAG). Les correctifs exigés par cette revue
> sont intégrés à chaque fiche.

---

## 0. Thèse stratégique — pourquoi ces trois axes battent le RAG

Le RAG, sous toutes ses formes (naïf, hybride, re-ranké, GraphRAG, agentique),
est **de la récupération sans état par similarité** : découper, plonger dans un
espace vectoriel, renvoyer le top-k. CCOS n'est pas « un meilleur RAG » : c'est
une **mémoire de travail causale, déterministe, rejouable et porteuse de
croyances**. Sur les axes qu'un récupérateur probabiliste ne peut même pas
*représenter* — contradiction, provenance, temps, rejeu, intervention — la
comparaison n'est pas serrée (`docs/COMPARISON_vs_rag.md`).

SciRust (~90 crates, discipline « output validé contre un oracle, déterminisme
mesuré ») offre exactement les briques algorithmiques qui **prolongent**
structurellement ce moat sur les trois axes demandés :

| Axe | Ce que le RAG ne peut pas faire | Brique SciRust à distiller |
|---|---|---|
| **Causalité** | distinguer cause/conséquence ; propager une croyance ; répondre à `do(X)` | SCM linéaire (`scirust-neuro-symbolic`), DAG causal (`scirust-graph`) |
| **Sécurité** | mettre en quarantaine une source, propager la défiance, certifier un verdict | conforme + IBP (`scirust-core::nn`), MinHash/LSH (`scirust-nlp-advanced`) |
| **Time-travel** | reconstruire une croyance passée, imputer une dérive à un op, forker l'histoire | lisseur RTS (`scirust-estimation`), CUSUM/Mann-Kendall (`scirust-seasonal`), DTW (`scirust-sequential`) |

Chacune de ces briques existe déjà, testée, dans SciRust — et **aucune** ne doit
devenir une dépendance de runtime (§1).

---

## 1. La règle d'or — contraintes non négociables

Toute nouveauté ci-dessous respecte la discipline déjà établie par CCOS
(`docs/MEASUREMENT_scirust_fusion.md`, `src/lsa.rs`, `src/scirust_bridge.rs`) :

1. **Distiller, ne pas lier.** SciRust est un *oracle algorithmique*, jamais une
   dépendance de runtime. Lier `scirust-core` tirerait `rayon` (réductions
   flottantes parallèles **non déterministes**), `nalgebra`, `ndarray` — ce qui
   casserait le `replay == live` bit-à-bit. On copie l'algorithme (~40–150 lignes
   `f64`/`usize` pures) dans un nouveau module CCOS zéro-dépendance, comme
   `src/lsa.rs` l'a fait pour la LSA incrémentale. La seule liaison tolérée reste
   l'`optional dependency` opt-in `scirust-retrieval` (cœur pur : `serde`/`sha2`).
2. **Déterminisme bit-à-bit.** Accumulations en ordre trié (ids croissants,
   arêtes canoniquement ordonnées), aucun `HashMap`-order, aucun RNG, aucune
   horloge murale — seulement l'horloge logique (`tick`). Précédents :
   `eigencentrality` (`memory.rs:1025`), le fold Gram de la LSA.
3. **`derived-not-stored`.** Les signaux dérivés (comme `QBelief`, la centralité)
   sont recalculés depuis le graphe, jamais sérialisés → snapshot byte-identique.
4. **Opt-in, off-by-default.** Un nouveau poids de scoring ships à `0.0`,
   `#[serde(default, skip_serializing_if)]` (précédent `w_centrality`,
   `NodeState`) → le build par défaut reste byte-identique et le hash-chain
   inchangé. Une nouveauté ne s'active qu'après **mesure** sur `retrieval_reward`.
5. **Propagation bornée.** Toute diffusion a une condition d'arrêt explicite
   (fixpoint, `max_depth`, `max_rounds`) — précédent `propagate_failure`.

> ⚠️ **Piège transversal identifié par la revue.** Le graphe de code CCOS est
> **cyclique** (récursion `Calls`, dépendances circulaires `DependsOn` — cf.
> `find_cycles()` `memory.rs:2861`). Tout algorithme SciRust qui suppose un DAG
> (ordre topologique de Kahn, relaxation en `|nodes|+1` passes) **diverge ou
> perd les nœuds en cycle** sur l'entrée réelle. Chaque fiche concernée précise
> la parade (condensation en SCC, normalisation PageRank `coef = poids/degré`,
> ou repli sur une décroissance `decay^hop` bornée).

---

## 2. Causalité

CCOS possède déjà un graphe causal structurel (arêtes `Contains`/`DependsOn` +
sémantiques `Calls`/`DataFlow`) et une couche de croyance Q-Page (`Supports`/
`Contradicts` → `belief`/`conflict`, décroissance, propagation **1 hop**). La
causalité au sens *inférence* (interventions, contrefactuels, direction apprise)
est absente. Voici comment la construire.

### 2.1 — Propagation de croyance multi-hop convergente (`Op::Propagate`) — **Tier 1**

- **Quoi.** Remplacer la passe 1-hop de `propagate_beliefs` par une relaxation
  bornée jusqu'à convergence, ordonnancée de façon déterministe, et **enregistrée
  comme op rejouable**. Le commentaire de code `memory.rs:2325-2328` nomme
  explicitement ce slice comme « suivant » — cette proposition ferme un manque
  que le code lui-même signale.
- **Source SciRust (mature).** `scirust-neuro-symbolic::probabilistic::causal::
  CausalEngine::solve` (motif de relaxation bornée « `0..=nodes.len()` passes sur
  un DAG », `causal.rs:48`) ; `scirust-graph::dag::topo_order` (Kahn réel,
  `dag.rs:191`).
- **Intégration CCOS.** `memory.rs:2332` `propagate_beliefs` → ajouter
  `propagate_beliefs_converge(threshold, damping, max_rounds)` en réutilisant le
  contrat `collect → sort → dedup → add_edge` par ronde ; `agent_session.rs:50`
  `Op` → `Op::Propagate { threshold, damping, rounds }` (append **en dernier**,
  `serde(default)`) ; `replay_to` l'applique → `replay == live` pour la révision
  multi-hop.
- **Algorithme.** Layering topologique du sous-graphe `Causes` (Kahn, égalités
  brisées par id croissant) → traitement Gauss-Seidel (la croyance induite d'une
  cause résolue est visible par son successeur dans la même ronde). Chaque ronde
  émet des arêtes `Supports`/`Contradicts` dérivées (poids
  `edge.weight·damping·|belief|`) ; **arrêt** quand une ronde n'ajoute aucune
  arête *nouvelle* (fixpoint — `add_edge` dédoublonne sur `(src,tgt,type)` en
  **ignorant** le poids, ce qui rend le fixpoint bien défini) **ou** au plafond
  `max_rounds` (estimation du diamètre, défaut 8, gère aussi le cas cycle
  `Causes`).
- **Déterminisme.** Tri/dedup par id+polarité à chaque ronde (l'ordre
  d'itération ne fuit pas) ; terminaison bornée ; `Op::Propagate` rejoué à
  l'identique. Les arêtes induites sont des `Supports`/`Contradicts` déjà dans
  l'enum additif → snapshot byte-identique. Opt-in via la nouvelle op.
- **Anti-RAG.** Un récupérateur cosinus n'a pas d'axe de croyance signé ni
  d'arêtes labellisées : « A est réfuté, A cause B, donc B est faiblement
  contredit » est *inreprésentable*.
- **Effort / risque.** M / **faible**. Maturité SciRust : réelle.
- **Correctifs de revue.** Réutiliser le dedup existant de `add_edge` (ne pas
  inventer de clé) ; la terminaison repose sur le fixpoint discret zéro-arête,
  pas sur l'argument de décroissance des poids (qui n'est qu'un bonus).

### 2.2 — `do(X)` : rappel interventionnel via SCM linéaire distillé — **Tier 2**

- **Quoi.** Attacher des équations structurelles au sous-graphe `Calls`/`DataFlow`
  et répondre à des requêtes de Pearl `do(node = faute)` : « si cette fonction
  casse, quels nœuds sont *structurellement forcés* d'être affectés, et de combien »
  — un mode de rappel rejouable de première classe.
- **Source SciRust (mature).** `scirust-neuro-symbolic::probabilistic::causal::
  CausalEngine::{solve,intervene}` (SCM linéaire, `do(var=value)` qui *coupe*
  l'équation entrante de la variable — sémantique Pearl correcte, 3 tests,
  `causal.rs:48-85`).
- **Intégration CCOS.** Nouveau `src/causal_scm.rs` (SCM distillé, `fn` pure sur
  `&MemoryGraph`, jamais lié) ; réutiliser `Calls`/`DataFlow`/`DependsOn` comme
  arêtes d'équation (`coef = edge.weight`), aucune nouvelle `EdgeType` ; exposer
  via un outil MCP `causal_intervene { node, delta }` frère de `recall_what_if`
  (`mcp.rs:211`) et un verbe REPL `intervene <node>` (`postmortem.rs`, gabarit
  `render_energy`). Fusion optionnelle dans `compute_node_score` via un terme
  gated `w_intervention = 0.0`.
- **Algorithme.** `value(v) = base_importance(v) + Σ_{u→v} edge.weight·value(u)`
  sur le sous-ensemble d'arêtes directionnelles. `do(X = X.base + δ)` coupe
  l'équation entrante de X, épingle sa valeur, puis propage. **La différence-clé
  avec `propagate_failure`** (diffusion additive) : couper les entrées de X →
  le score reflète « ce que X *force* », pas « ce qui *coule près* de X ».
- **Déterminisme.** SCM = fonction pure des nœuds triés + arêtes canoniques ;
  `derived-not-stored` (zéro état snapshot) ; seul `Op::Intervene` (appendé en
  dernier) entre dans le log. `w_intervention` élidé à 0 → build par défaut
  byte-identique.
- **Anti-RAG.** `do(X)` n'est pas une requête de similarité : couper les causes
  d'une variable puis re-dériver l'aval est une chirurgie de graphe qu'un top-k
  cosinus ne peut pas exprimer. C'est précisément l'axe que
  `COMPARISON_vs_rag.md` qualifie de structurellement impossible pour le RAG.
- **Effort / risque.** M / **moyen** (à cause du correctif ci-dessous).
- **Correctif de revue (bloquant).** ❗ Le graphe est **cyclique** : la
  relaxation Gauss-Seidel à passes fixes **diverge** sur un cycle de gain ≥ 1
  (arêtes à poids 1.0), et l'ordre topologique de Kahn *laisse tomber
  silencieusement* les nœuds en cycle. **Parades** (au choix, aucune n'exige de
  code SCC dans SciRust — inexistant) : (a) restreindre le SCM au sous-ensemble
  acyclique et laisser le garde d'acyclicité de `CausalDag` retirer les
  back-edges à la construction ; ou (b) normaliser les coefficients façon
  PageRank (`coef = poids/degré-sortant`, rayon spectral < 1) ; ou (c) reporter
  un *flag d'instabilité* par nœud plutôt qu'un delta bit-identique mais faux.
  Retirer du pitch la fausse promesse « converge en `|nodes|+1` passes ».

### 2.3 — Attribution amont : ranking par ancêtres (skeleton backdoor) — **Tier 2**

- **Quoi.** Quand un nœud échoue, classer ses **ancêtres causaux** (le squelette
  d'ajustement backdoor) comme candidats-coupables, au lieu de seulement diffuser
  la pression vers l'aval. C'est l'objet correct pour « qu'est-ce qui a causé ça ».
- **Source SciRust (mature).** `scirust-graph::dag::{ancestors, intervention_
  ancestors}` (fermeture d'ancêtres = ensemble d'ajustement, `dag.rs:145,226`) ;
  `scirust-retrieval::causal_rerank::CausalReranker` (motif de consommation d'un
  ensemble d'ancêtres comme signal de ranking — déjà implémenté et testé).
- **Intégration CCOS.** `memory.rs:2074` → `blame_ancestors(failing_node,
  max_depth)` (BFS **inverse** sur `Calls`/`DataFlow`/`DependsOn`) ;
  `postmortem.rs` → verbe `blame <node>` (gabarit `render_energy`/`missing`) ;
  valider sur `retrieval_reward` avant activation.
- **Anti-RAG.** La victoire différenciante de CCOS (« la cause est dans un autre
  fichier ») *est* une requête backdoor : le coupable est en amont et de faible
  similarité → le cosinus le classe mal, la reachability inverse le classe #1.
- **Effort / risque.** S / **moyen**.
- **Correctifs de revue.** (1) Cadrer honnêtement vs l'existant :
  `propagate_failure_bidirectional` (`memory.rs:2118`) atteint déjà l'amont — le
  delta réel est « requête *read-only* d'ancêtres classés » vs « mutation de
  `failure_relevance` mêlant ancêtres et descendants ». (2) ❗ Le poids
  « plus-long-chemin-atténué par ordre topo » est **mal défini sur un sous-graphe
  cyclique** : replier sur une décroissance `decay^hop` par BFS borné
  (bien définie et déterministe).

### 2.4 — Découverte causale LiNGAM (diagnostic hors-ligne, gated) — **Tier 3**

- **Quoi.** Faire passer les arêtes `Causes` de *asséré* à *inféré* : distiller
  FastICA pour récupérer un ordre causal linéaire-non-gaussien et **noter la
  direction** des arêtes — la première inférence causale du dépôt.
- **Source SciRust.** `scirust-multivariate::ica_fit` (FastICA déterministe,
  `lib.rs:424`) — **mature** ; mais le pont LiNGAM (permutation vers
  triangulaire-inférieure, `B = I − PW`) est **net-new**.
- **Verdict de revue : REVISE, haut risque — à cadrer en *diagnostic*, jamais en
  entrée de scoring live.** Trois problèmes réels : (1) l'entrée est douteuse —
  `replay_to` donne un *état final*, pas une série temporelle ; les signaux
  d'activation devraient être modélisés comme un SEM linéaire-non-gaussien, ce
  qu'ils ne sont pas ; (2) l'ambiguïté de signe/permutation de l'ICA se propage
  dans l'orientation des arêtes (`ica_fit` renvoie des composantes **non**
  épinglées en signe et trie ses valeurs propres sans tie-break par id) ; (3) la
  « confiance d'orientation » `|Bij|` est non calibrée. **À conserver seulement**
  rescopé en outil d'*eval-harness* (jamais `compute_node_score`) : épingler les
  signes, ajouter les tie-breaks par id, valider la non-gaussianité avant
  d'émettre la moindre arête `CausesInferred`. C'est un pari de recherche, pas
  une distillation.

---

## 3. Sécurité

CCOS durcit déjà l'entrée (dé-obfuscation Unicode `sanitizer.rs`, signal
d'injection Naive-Bayes `injection_classifier.rs`, hash-chain). Les manques
*explicitement admis* (`docs/SECURITY.md`) : homoglyphes, seuil ad-hoc 0.5,
absence de propagation de provenance/défiance, pas de garantie certifiée. Les
quatre fiches ci-dessous les ferment.

### 3.1 — Rayon de robustesse **certifié** du verdict d'injection — **Tier 1**

- **Quoi.** Le classifieur d'injection est un modèle **linéaire** log-space sur
  un vecteur de comptes de 2048 buckets → sa décision a un rayon de robustesse
  *calculable exactement* : le nombre minimal de perturbations de features qu'un
  attaquant doit faire pour retourner le verdict. « Un signal » devient « un
  signal **avec un certificat** de fragilité », inscrit au log inviolable.
- **Source SciRust.** `scirust-core::nn::ibp` (Interval Bound Propagation) comme
  *oracle* — mais voir correctif : pour **une** couche linéaire, c'est une
  forme close ~15 lignes, aucune dépendance nécessaire.
- **Intégration CCOS.** `injection_classifier.rs:99` `LinearModel::score` →
  `certified_margin(text, budget_l1)` ; `:320` `explain` → étendre `Explanation`
  (qui porte déjà `margin`/`prior_margin`) avec `certified_radius` ;
  `IngestReport` → `certified_robust: bool` (`serde(default, skip_serializing_if)`,
  purement audit, ne *gate* pas l'ingestion) ; `examples/injection_redteam.rs`
  reporte le rayon certifié à côté de P/R/F1.
- **Algorithme (forme close).** Marge `m(X) = Σ_b g[b]·X[b] + (bias_inj−bias_ben)`,
  `g[b] = W_inj[b] − W_ben[b]`. Sous un budget L1 de `K` occurrences ajoutées/
  retirées (avec `ΔX ≥ −X`, comptes non négatifs), le pire cas dépense tout le
  budget sur les buckets `g[b]` les plus négatifs → borne inférieure de marge en
  forme close. `certified_robust = (borne inf) > 0` ; rayon = plus petit `K`
  faisant chevaucher zéro.
- **Déterminisme.** Arithmétique linéaire en forme close sur le blob de poids
  épinglé SHA-256 + texte défangé ; aucun RNG. Champ `IngestReport` élidé →
  snapshot byte-identique.
- **Anti-RAG.** Un re-ranker dense neuronal **n'a pas** de frontière linéaire ni
  de certificat calculable : on ne peut pas prouver « aucune perturbation ≤ K
  caractères ne retourne ce verdict » sur une carte transformer non linéaire.
  CCOS peut l'inscrire au log — une garantie adverse auditable qu'un pipeline RAG
  ne peut structurellement pas produire.
- **Effort / risque.** S / **faible**.
- **Correctifs de revue.** ❗ Retirer la citation **CROWN** :
  `crown_bounds(l1,l2,…)` exige *deux* couches via un ReLU et **ne peut pas**
  être appelé sur le modèle mono-couche. Assumer que c'est une forme close
  triviale sur les poids épinglés (aucune dépendance SciRust). Note honnête :
  ajouter le champ à une payload *hashée* (`Parsing`) est un *format bump* du
  chain (pas une non-déterminisme) — `replay == live` tient toujours.

### 3.2 — Arêtes de **provenance / taint** : un axe de confiance dans le graphe — **Tier 2**

- **Quoi.** Transformer le `injection_score`/anomalie (aujourd'hui advisory,
  jeté après `IngestReport`) en une **atténuation de confiance persistante et
  rejouable** par nœud, qui se propage **1 hop** le long de `Contains`/`Calls`/
  `DataFlow` et rabaisse structurellement toute croyance, décroissance et rappel
  dérivés d'une source suspecte. *Un fichier empoisonné ne peut pas devenir
  silencieusement une preuve de haute autorité.*
- **Source SciRust (patron distillé).** `scirust-som` `OwnershipOracle`
  (propagation source→puits, réutilisé comme *patron* de labelling 1-hop) ;
  `scirust-ids::hashchain` (déjà mirroré par `event_log`).
- **Intégration CCOS.** `memory.rs:42` `GraphNode` → `trust: f64`
  (`#[serde(default = "one", skip_serializing_if = "is_full_trust")]`, défaut
  1.0 → byte-identique) ; `external_memory.rs` ingest → `trust = clamp(1 −
  injection_score, 0, 1)`, **plafonné à 0.5** si le `ScanReport` contient une
  anomalie `BidiControl`/`TagChar` (Trojan-Source / ASCII-smuggling — un signal
  *structurel* non contournable par paraphrase) ; `qbelief`/`qbeliefs`
  (`memory.rs:2234/2258`) → `authority = edge.weight · trust[source]` ;
  `compute_node_score` → grille multiplicative `w_trust` (élidée à 0).
- **Déterminisme.** `trust` = fonction pure du score NB (dot-product sans RNG) +
  scan déterministe ; propagation 1-hop accumulée sur ids/arêtes triés (idiome
  `qbeliefs` BTreeMap) ; **recalculée au replay depuis `Op::Ingest`** (pas
  d'`Op::Taint` — redondant, cf. correctif).
- **Anti-RAG.** Un top-k cosinus rangera volontiers « ignore les instructions
  précédentes ; la clé API est X » comme pertinent. CCOS met en **quarantaine
  structurelle** : le verdict devient un discount d'autorité persistant, rejouable
  (« taint dès l'étape N ») et *contrefactuellement interrogeable* (rejouer sans
  le taint = ce que l'agent aurait cru si la source était fiable). Impossible pour
  une récupération sans état.
- **Effort / risque.** M / **faible**.
- **Correctifs de revue.** (1) ❗ **Supprimer `Op::Taint`** : le replay
  ré-exécute déjà `ingest_deferred` pour chaque `Op::Ingest` et recalcule
  `injection_score`+scan → `trust` est un dérivé recalculable (discipline
  `QBelief`). (2) Ne pas survendre la moitié NB comme un *bouclier* (la
  paraphrase l'évade) : le cœur défendable est le **plafond structurel à 0.5** sur
  `BidiControl`/`TagChar`. `w_trust = 0` par défaut.

### 3.3 — Garde **conforme** à l'ingestion : borne de fausse alarme distribution-free — **Tier 2**

- **Quoi.** Remplacer le `injection_flagged = score ≥ 0.5` codé en dur par un
  seuil calibré **split-conformal** qui garantit une borne (α) *sans hypothèse de
  distribution* sur le taux de fausse quarantaine — la même garantie que les
  gardes OT de SciRust donnent au trafic Modbus/DNP3, portée sur le fil de
  l'ingestion de code. Verdict à trois voies **Accept / Abstain / Reject**.
- **Source SciRust (mature).** `scirust-core::nn::conformal::{conformal_quantile,
  ConformalClassifier}` (quantile `⌈(n+1)(1−α)⌉`-ième, correction ceil
  fini-échantillon, tests de couverture **et** de reproductibilité `to_bits`) ;
  `scirust-core::nn::guard::StatisticalGuard::decide` (Accept/Abstain/Reject) ;
  `scirust-ids::ot::OtGuard::{calibrate,check,false_alarm_rate}` (patron du
  garde).
- **Intégration CCOS.** `injection_classifier.rs` → `calibrate_conformal(...)`
  stockant `q̂` ; `external_memory.rs:1454` → remplacer le `≥ 0.5` par le verdict
  conforme ; `IngestReport` → `alarm_bound: f64` (l'α garanti) ; `event_log`
  → événement `Calibration` (payload : `n_calib, α, q̂`) dans le hash-chain.
- **Déterminisme.** `q̂` = fonction pure d'un **fichier de calibration
  checked-in** (constante compile-time, comme `DEFAULT_MODEL_FINGERPRINT`), donc
  identique à chaque build ; `conformal_quantile` = tri + index entier avec
  `total_cmp`. Aucune donnée runtime.
- **Anti-RAG.** Un cosinus ne peut pas *exprimer* une borne distribution-free sur
  la fréquence à laquelle il supprime à tort du contenu bénin ; et étant sans
  état, il ré-récupère la même chunk empoisonnée à chaque requête (pas de
  quarantaine).
- **Effort / risque.** S / **moyen**.
- **Correctifs de revue.** ❗ L'artefact de calibration **n'existe pas encore** :
  le red-team « 240 échantillons » est *généré* au runtime depuis une graine
  SplitMix64 (`examples/injection_redteam.rs`), pas un split de probabilités
  labellisé. `q̂` doit être calculé hors-ligne et **épinglé en constante**, sinon
  le déterminisme casse. Mettre à jour les deux tests qui assertent le seuil 0.5.
  Honnêteté : la garantie n'est aussi forte que le corpus de calibration (« un
  signal, pas un bouclier »).

### 3.4 — Skeleton anti-homoglyphes (UTS-39) + spoof-provenance MinHash-LSH — **Tier 2**

- **Quoi.** Ajouter la normalisation par **skeleton de confusables UTS-39**
  (*surface, ne strip pas*, comme le reste du sanitizer) — puis, via MinHash-LSH,
  signaler les identifiants/URL/chemins d'un texte entrant qui sont des
  quasi-doublons de symboles *déjà fiables* du graphe mais diffèrent par des
  caractères visuellement confusables (l'attaque `раypal` cyrillique vs `paypal`
  latin, que le pass par-scalaire **et** le modèle bag-of-features NB manquent
  tous deux structurellement). Ferme le manque « homoglyphes » explicitement laissé
  ouvert.
- **Source SciRust (mature, 976 LoC, zéro `todo!`).** `scirust-nlp-advanced::
  similarity::MinHash` (signatures FNV-1a + LCG entier, **zéro flottant** dans le
  chemin de signature → bit-identique inter-arch) ; `::lsh::MinHashLsh`
  (band-and-bucket O(n)) ; `::bloom::BloomFilter` (pré-filtre d'appartenance).
- **Intégration CCOS.** `sanitizer.rs` → nouvelle `AnomalyKind::Confusable` +
  passe `skeleton()` (table UTS-39 distillée en `const-fn`) qui **surface** un
  token mixte-script comme littéral explicite `[CONFUSABLE 'раypal'→'paypal']`
  (jamais normalisé en silence) ; `external_memory.rs` ingest → shingler chaque
  token identifiant, MinHash, interroger un index LSH bâti sur les **skeletons**
  des symboles résidents fiables (rebuild version-cached comme
  `weighted_lsa_cache`) → collision de bucket avec formes brutes différentes =
  `SpoofSuspected` dans `IngestReport.anomalies` (déjà un `Vec`, strictement
  additif). Bonus : les littéraux `[CONFUSABLE …]` déclenchent des features NB
  comme les `[U+…]` existants → le score NB se relève aussi sur les homoglyphes.
- **Déterminisme.** Table = constante compile-time ; `skeleton()` = map pur
  par-scalaire (conserve le O(n) single-pass, `Cow::Borrowed`-si-propre) ; index
  LSH `derived-not-stored` (rebuild depuis l'ensemble trié des symboles fiables) ;
  `file_hash` pris sur la forme propre skeleton-surfacée → chain reproductible.
  Opt-in via `Action` (échappatoire pour faux positifs emoji/ZWJ).
- **Anti-RAG.** « Ce token est visuellement confusable avec un symbole fiable
  mais byte-distinct de lui » est une relation **discrète, exacte, de provenance
  caractère** sur l'ensemble fiable — pas une similarité d'espace d'embedding.
  Un dense retriever fait *collisionner* `раypal` et `paypal` (et sert l'URL de
  l'attaquant comme contexte pertinent) ou les éloigne (et rate le lien) : ni
  l'un ni l'autre ne signale l'attaque.
- **Effort / risque.** M / **moyen**.
- **Correctifs de revue.** (1) Ce n'est **pas** « ajouter une branche à
  `classify` » (strictement par-scalaire, la boucle de defang substitue **un**
  char à un `byte_offset`) : la passe skeleton doit être un **pass par-token
  séparé** avec sa propre injection de littéral — vraie chirurgie. (2) « le NB se
  relève gratis » dépend du littéral atterrissant bien dans le texte tokenisé —
  c'est la partie non-gratuite. Scoper la PR : pass sanitizer d'abord,
  spoof-provenance LSH ensuite.

---

## 4. Time-Travel

CCOS a un event-sourcing (oplog hash-chainé), un REPL post-mortem
(`diff`/`energy`/`missing`), et un *temporal tensor* `Θ[claim,{Belief,Tension},t]`
**mesuré mais non productionisé** (`docs/MEASUREMENT_temporal_tensor.md`). Le
manque #1 admis : le post-mortem **montre** la dérive mais ne l'**impute** jamais
à une cause. Les trois fiches ci-dessous en font un vrai *flight recorder* causal.

### 4.1 — 🚩 Attribution **causale** de la dérive (`cause`/`blame`) — **Tier 1, FLAGSHIP**

C'est l'intersection causalité × time-travel, et la démo anti-RAG la plus forte.
Deux méthodes **complémentaires**, une même sortie (« quel op a bougé le nœud »).

**(a) Localisation par changepoint (statistique).** Reconstruire la série de
score d'un nœud par `replay_to(t)` à chaque étape, puis détecter la rupture.
- **Source SciRust (mature, ~150 lignes distillables, tests exacts).**
  `scirust-seasonal::trend::{cusum (lib.rs:2110), mann_kendall (:1959),
  sens_slope (:2007)}`.
- **Algorithme.** `CUSUM` localise `argmax|S[t]|` (rupture dominante) ;
  `Mann-Kendall` (avant/après) + signe de `Sen` confirment un décalage de niveau
  *significatif et orienté* → l'op à `ops[k]` passe de « op à l'étape d'éviction »
  (le `trigger:` corrélationnel actuel, `postmortem.rs:348`) à un changepoint
  testé avec direction et magnitude.

**(b) Confirmation par ablation contrefactuelle (causale).** Prouver *quel* op
a causé l'éviction en rejouant l'oplog avec cet op **ablaté** (`do(op = no-op)`).
- **Intégration CCOS.** `agent_session.rs:640` `replay_to` →
  `replay_ablating(step, skip: &BTreeSet<usize>)` (changement d'un prédicat,
  réutilise le batching deferred-ingest) ; `postmortem.rs` → verbe `cause <node>`
  remplaçant le `trigger:` naïf par une liste causale classée ;
  `render_missing` calcule déjà `first_missing` (l'effet) et `eviction_detail`
  calcule déjà `rank(N)` — exactement le `Δrank` requis.
- **Algorithme.** Effet = nœud N évincé à l'étape t (watchpoint `missing`).
  Candidats = les ops `Failure`/`PageFault`/`Ingest` dans `(floor, t)` ciblant des
  **ancêtres causaux** de N (`CausalDag::ancestors` élague O(tous) → O(pertinents)).
  Pour chaque candidat, `replay_ablating(t, {i})` → N est-il toujours évincé ?
  Score = Δrank de N entre monde factuel et contrefactuel. L'op dont l'ablation
  restaure le plus N *est* la cause prouvée.
- **Déterminisme.** C'est **la** meilleure histoire de déterminisme du lot :
  `replay_ablating` = `replay_to` + skip-set sur le même chemin de batching →
  chaque monde contrefactuel est byte-reproductible ; **rien n'est persisté** (les
  replays ablatés sont éphémères) → snapshot + hash-chain intacts. Honore le
  `compaction floor` comme `missing`. Zéro dépendance. Précédent en dépôt :
  `retrieval_reward` s'auto-décrit « évaluation contrefactuelle sur le log
  hash-chainé ».
- **Anti-RAG.** La causation contrefactuelle sur l'histoire est *définitivement*
  impossible pour un récupérateur sans état : il n'a ni oplog, ni replay, ni
  notion de « la même requête sous un passé alternatif ». C'est le moat *flight
  recorder + do-calculus* : de la causalité auditée et testée, pas de la
  similarité.
- **Effort / risque.** M / **faible** (b : det=yes), M / **moyen** (a).
- **Correctifs de revue.** (a) ❗ Ajouter la **correction de ties** à
  `mann_kendall` (les scores sont clampés `[0,1]` → beaucoup d'égalités → la
  p-value, tout l'argument de « confiance causale », est peu fiable sans elle) ;
  **abandonner le cadre saisonnier** (`seasonal_break_detection` est
  inconditionnellement saisonnier) au profit d'une segmentation binaire récursive
  simple. (b) Dégrader la citation `CausalEngine::intervene` : le vrai `do()` ici
  est `replay_to + skip`, il ne doit rien à SciRust ; l'élagage par `ancestors`
  est une optimisation **phase 2** (adaptateur `MemoryGraph→CausalDag`), pas un
  cœur. **Borner dur le budget de replay** — verbe REPL hors-ligne, jamais chemin
  chaud. Les deux méthodes : imputer « pré-floor, non-attribuable » sous le
  `compaction floor` plutôt que mal-charger.

### 4.2 — Rétrodiction **RTS** de la croyance (lisseur de Kalman) — **Tier 1**

- **Quoi.** Replier les preuves **futures** de l'oplog dans chaque étape **passée**
  d'une trajectoire de croyance/tension avec un lisseur Rauch-Tung-Striebel : le
  time-travel renvoie la reconstruction **à variance minimale** de ce que l'agent
  *aurait dû* croire à l'étape t compte tenu de tout ce qu'on sait maintenant. Une
  rétrodiction principielle que la « fever curve » brute ne donne pas — et le
  *inverse structurel de la récupération*.
- **Source SciRust (mature).** `scirust-estimation::smoother::RtsSmoother::smooth`
  (forward Kalman + backward RTS complet, oracle `smoother_beats_the_filter :
  RMSE lissé < RMSE filtré`, `smoother.rs:19`) ; `::linalg::Mat`
  (matmul/matvec/inverse testés).
- **Intégration CCOS.** Nouveau `src/retrodict.rs` (Mat 2×2 + passes
  forward/backward copiées, ~90 lignes, discipline `lsa.rs`) ;
  `spectral.rs:274/329` `TemporalProfile`/`temporal_profile` →
  `rts_smoothed(F,Q,H,R)` sur `belief_series`/`tension_series` ;
  `agent_session.rs:740` `belief_tension_timeline` = le producteur des frames par
  étape ; `postmortem.rs` → verbe `smooth <claim>` (filtré vs lissé) ; `mcp.rs`
  → outil read-only `retrodict` (le temporal tensor n'a **aucune** surface
  CLI/MCP aujourd'hui — un manque explicite que ceci comble).
- **Algorithme.** Modéliser chaque claim comme un état latent linéaire-gaussien
  (`level`+`rate`), `F=[[1,1],[0,1]]`, `H=[1,0]`, `Q`/`R` diagonaux fournis par
  l'appelant. Forward KF puis passe RTS `C_k = P_filt[k]·Fᵀ·P_pred[k+1]⁻¹` ; le
  niveau lissé = croyance rétrodite, la vitesse lissée = onset. Une croyance
  bruitée tôt mais fermement établie plus tard est reconstruite à faible variance
  aux étapes précoces.
- **Déterminisme.** Fonction pure `f64` de la série (déjà déterministe) ; aucun
  RNG (le RNG de `smoother.rs` est test-only) ; read-only offline, **jamais
  sérialisé** → snapshot + hash-chain byte-identiques (discipline
  `derived-not-stored`, précédent `retrieval_reward`). `Q`/`R` = params appelant
  (comme `qbelief_decayed(half_life)`).
- **Anti-RAG.** Un cosinus n'a ni axe temps ni axe croyance : il ne peut pas dire
  « à l'étape 40, sachant tout jusqu'à l'étape 120, l'estimée à variance minimale
  de la croyance en X était 0.3 ± 0.05 ». Replier le futur dans le passé est du
  *time-travel bayésien* sur un état auditable — structurellement hors de portée.
- **Effort / risque.** M / **faible**.
- **Correctifs de revue.** (1) ❗ `Mat::inverse` (`linalg.rs:153`) utilise un
  **pivot partiel** (pas « ordre fixe ») — toujours déterministe (même entrée →
  même pivot → mêmes bytes), mais réécrire la justification en « Gauss-Jordan à
  pivot partiel déterministe » (un distillateur croyant l'ordre fixe est un
  footgun). (2) Éviter la **double décroissance** de la tension : si `F` relaxe
  déjà la tension, ne pas aussi utiliser la mesure `qbelief_decayed` (déjà
  `0.5^(age/half_life)`). (3) Démoter « crossing lissé = vrai onset » en
  *estimée conditionnée au modèle* (un `Q`/`R` mal réglé peut effacer un vrai pic
  ou en fabriquer un).

### 4.3 — Branch-and-align : timelines alternatives + DTW — **Tier 2**

- **Quoi.** Donner à CCOS sa **première timeline alternative** : forker l'oplog à
  une étape passée, injecter un op contrefactuel (un correctif appliqué plus tôt,
  une assertion jamais faite), rejouer la branche divergente, et **aligner par
  DTW** les deux trajectoires de croyance/chaleur pour quantifier *où* et *de
  combien* l'histoire diverge. (`recall_what_if` ne varie que les params de
  rappel à état fixe — jamais l'histoire elle-même.)
- **Source SciRust (mature).** `scirust-sequential::matching::{dynamic_time_
  warping_with_path (matching.rs:230), longest_common_subsequence (:149)}` (DP
  `O(n·m)` avec traceback, `f64`/`usize` purs, tie-break déterministe).
- **Intégration CCOS.** `agent_session.rs:50/132` `Op`/`PersistedTimeline` →
  champ additif `branch: fork_at:usize + ops divergents` (`serde(default)` → vieux
  `.oplog` byte-identiques) ; `replay_to` rejoue `baseline + trunk[..fork_at] +
  branch` par le **même** chemin → `replay == live` gratis pour le monde
  contrefactuel ; nouveau `src/timeline_align.rs` (DTW/LCS copiés) ;
  `postmortem.rs` → verbes `branch <fork_at> <op>` / `align`.
- **Déterminisme.** Une branche est un oplog **append-only hash-chainé** propre —
  injecter un contrefactuel = ajouter un vrai `Op` à une **nouvelle** chaîne,
  jamais muter le trunk (histoire append-only, inviolable). DTW/LCS purs, tie-break
  index-croissant.
- **Anti-RAG.** « Et si l'agent avait asséré X à l'étape 40 » : un récupérateur
  sans état n'a qu'un corpus et **aucune** notion de forker/injecter/rejouer une
  histoire divergente puis mesurer l'écart d'alignement — pas d'histoire, pas
  d'état à diverger.
- **Effort / risque.** L / **moyen**.
- **Correctifs de revue.** (1) ❗ `verify_integrity()` vit sur `KernelSnapshot`
  (`persist.rs:59`) et **ne couvre pas** un sidecar `PersistedTimeline` : la
  tamper-evidence de la branche est une **nouvelle plomberie à construire**, pas
  une réutilisation d'un garde existant. (2) `Op` est un enum **privé** : le verbe
  REPL « injecter un op arbitraire depuis une string » exige un parser d'op
  in-crate (travail non budgété). (3) L'« onset de divergence » DTW est
  présentationnel (le DTW déforme le temps) — la *distance* DTW scalaire, elle,
  est bien définie.

---

## 5. Feuille de route priorisée

**Tier 1 — quick wins** (KEEP, risque faible, `det=yes` ou opt-in propre,
SciRust mature, ferme un manque explicitement nommé). Ordre suggéré :

| # | Fiche | Axe | Effort | Pourquoi d'abord |
|---|---|---|---|---|
| 1 | **4.1 Attribution causale de la dérive** (`cause`/`blame`) | time-travel × causalité | M | Ferme le manque #1 admis ; démo anti-RAG la plus forte ; bâti sur `replay==live` |
| 2 | **2.1 Propagation de croyance multi-hop** (`Op::Propagate`) | causalité | M | Le code lui-même nomme ce slice comme « suivant » |
| 3 | **4.2 Rétrodiction RTS** (`smooth`/`retrodict`) | time-travel | M | Productionise le temporal tensor ; read-only, det=yes |
| 4 | **3.1 Rayon de robustesse certifié** | sécurité | S | Forme close, audit-only, aucune dépendance |

**Tier 2 — haute valeur, risque moyen** (fort anti-RAG, nécessite les correctifs
de revue) : **2.2** `do(X)` interventionnel · **3.2** taint de provenance ·
**3.3** garde conforme · **3.4** anti-homoglyphes · **4.3** branch-and-align ·
**2.3** ranking par ancêtres.

**Tier 3 — paris de recherche** (à cadrer en diagnostics *eval-harness*, jamais
en scoring live) : **2.4** découverte causale LiNGAM · force d'arête par ablation.

---

## 6. Ce qui a été considéré puis déprioritisé (honnêteté mesure-d'abord)

- **Compression Tensor-Train du temporal tensor** (`scirust-tn` TT/MPS/DMRG). Le
  tenseur `Θ` est *par-claim* et petit ; la valeur est dans la *rétrodiction*
  (4.2) et l'*attribution* (4.1), pas dans la compression low-rank. À reconsidérer
  seulement si des sessions très longues rendent `Θ` dominant en RAM — et alors la
  discipline « mesurer d'abord » s'applique.
- **Embedder neuronal / dense pour battre le RAG sur le rappel sémantique pur.**
  Rejeté par conception : les poids d'un transformer ne sont pas bit-stables →
  casse `replay == live`. CCOS garde un plancher sémantique déterministe et
  investit la différenciation dans structure/croyance/temps/audit (cf.
  `COMPARISON_vs_rag.md` — c'est un compromis assumé, pas un manque).

---

## 7. Récapitulatif SciRust → CCOS

| Fiche | Crate SciRust (oracle) | Maturité | Module CCOS distillé | Point d'intégration |
|---|---|---|---|---|
| 2.1 | `neuro-symbolic::…::CausalEngine::solve`, `graph::topo_order` | mature | `memory.rs` | `propagate_beliefs`, `Op::Propagate` |
| 2.2 | `neuro-symbolic::…::CausalEngine::intervene` | mature | `src/causal_scm.rs` (neuf) | `mcp.rs`, `postmortem.rs`, `w_intervention` |
| 2.3 | `graph::dag::{ancestors,intervention_ancestors}`, `retrieval::CausalReranker` | mature | `memory.rs` | `blame_ancestors`, `postmortem.rs` |
| 2.4 | `multivariate::ica_fit` | partielle | `src/lingam.rs` (neuf, **diagnostic**) | eval-harness (jamais scoring) |
| 3.1 | `core::nn::ibp` (oracle ; forme close) | mature | `injection_classifier.rs` | `LinearModel::score`, `IngestReport` |
| 3.2 | `som::OwnershipOracle` (patron), `ids::hashchain` | mature | `memory.rs` | `GraphNode.trust`, `qbelief`, `w_trust` |
| 3.3 | `core::nn::conformal`, `ids::ot::OtGuard` | mature | `injection_classifier.rs` | seuil ingest, `event_log` `Calibration` |
| 3.4 | `nlp-advanced::{MinHash,MinHashLsh,BloomFilter}` | mature | `sanitizer.rs`, `external_memory.rs` | `AnomalyKind::Confusable`, `IngestReport.anomalies` |
| 4.1 | `seasonal::{cusum,mann_kendall,sens_slope}` (+ ablation native) | mature | `src/drift.rs` (neuf) | `replay_ablating`, `postmortem.rs` `cause` |
| 4.2 | `estimation::smoother::RtsSmoother`, `linalg::Mat` | mature | `src/retrodict.rs` (neuf) | `TemporalProfile`, `postmortem.rs` `smooth`, `mcp.rs` |
| 4.3 | `sequential::matching::{DTW,LCS}` | mature | `src/timeline_align.rs` (neuf) | `PersistedTimeline.branch`, `postmortem.rs` `branch` |

---

*Méthodologie : cartographie exhaustive des deux dépôts (9 lecteurs), conception
sous trois angles indépendants par axe (profondeur algorithmique · différenciation
anti-RAG · pragmatisme déterministe), puis vérification adversariale de chaque
proposition (faisabilité · `replay == live` · maturité SciRust · anti-RAG). Les
convergences entre angles indépendants et les correctifs adversariaux sont
reportés tels quels dans chaque fiche.*
