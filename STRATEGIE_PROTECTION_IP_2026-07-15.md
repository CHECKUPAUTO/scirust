# SciRust — Stratégie de protection anti-clonage & audit simulé

_Audit d'architecture et stratégie de verrouillage IP — 2026-07-15_
_Méthode : reconnaissance des crates réelles (`scirust-autodiff`, `scirust-simd`, `scirust-transpiler`,
`scirust-gpu`, `scirust-macros`, `scirust-simd-macros`, `scirust-license`), conception de 16 mécanismes
sur 4 piliers, red-team adverse par mécanisme, puis revue de correction numérique, revue de préjudice
utilisateur et revue de solidité juridique/forensique._

---

## 0. Résumé exécutif (la vérité inconfortable, puis ce qui marche)

Ton modèle de menace est explicite : **un concurrent qui a accès au code source**, sait faire de la
rétro-ingénierie, et veut recréer un outil fonctionnellement identique en réécrivant/masquant le code.

Contre **cet** adversaire précis, la conclusion de l'audit est nette et il faut la regarder en face :

> **Aucun canary mathématique, aucun watermark de codegen et aucune obfuscation de macro ne survit à une
> réimplémentation depuis les sources.** Toute protection « côté client » que tu compiles dans ton binaire
> est, par construction, du code source que l'adversaire édite ou n'émet tout simplement jamais. La
> quasi-totalité des 16 mécanismes conçus a été notée « effort pour retirer = trivial » par le red-team.

Ce n'est pas un échec de conception : c'est une propriété fondamentale. Un watermark ne peut pas être à la
fois **invisible au compilateur / neutre sur le résultat** (condition pour ne pas casser une lib
scientifique) **et porteur au point de survivre à une réécriture** (condition pour prouver un vol). Ces deux
propriétés sont contradictoires. Les mécanismes qui touchent aux bits de sortie (pour être détectables en
boîte noire) **cassent la reproductibilité numérique de tes utilisateurs honnêtes** pour une valeur
dissuasive quasi nulle contre le cloneur — le pire des deux mondes.

**Ce qui, en revanche, a une valeur réelle — et c'est là qu'il faut mettre l'effort :**

| Levier | Ce qu'il protège réellement | Contre qui |
|---|---|---|
| **Signatures asymétriques Lamport/Merkle** (réutiliser `scirust-license/hashsig.rs`) sur les artefacts émis | Preuve juridique *non-forgeable* de provenance | Le redistributeur **verbatim** d'artefacts que TU as produits (fuite client, repackaging) |
| **Similarité du code source lui-même** (schéma de tuilage GEMM, layout de packing, constantes, commentaires) | La vraie pièce à conviction copyright/secret d'affaires | Le copieur de source — c'est ça la preuve, pas le watermark |
| **Licensing / node-lock à refus gracieux** (le gate `CoreKernels` existe déjà) | Le revenu : empêcher l'usage non-licencié par des clients honnêtes | L'utilisateur qui n'a pas payé, PAS le cloneur |
| **Tripwires neutres** (canary d'exécution en thread-local, statics `#[used]`) | Attraper le copieur *paresseux* verbatim à coût ~nul | Celui qu'un simple `diff` condamne déjà |

**La règle d'or de cette stratégie :** ne jamais confondre trois choses distinctes —
1. **Dissuasion / tripwire** (coût ~nul, attrape les paresseux, aucune valeur juridique),
2. **Preuve juridique** (signatures asymétriques + custody + horodatage),
3. **Licensing** (protège le revenu, pas l'algorithme).

Chaque mécanisme ci-dessous est classé dans l'une de ces catégories, **honnêtement**. Tout ce qui est vendu
comme « preuve irréfutable » alors que c'est un tripwire est un piège qui se retournera contre toi au
tribunal (le red-team juridique détaille pourquoi : p-values fictives, résidus forgeables, défense
« création indépendante » servie sur un plateau par ta propre doc de tolérance).

---

## 1. Modèle de menace et principes

### 1.1 Le spectre des adversaires (à ne jamais mélanger)

1. **Redistributeur verbatim** : reprend tes binaires/artefacts tels quels. → Les signatures asymétriques
   le clouent. Facile.
2. **Lessiveur d'artefacts** : reprend ton code généré (WGSL, Rust émis) et le passe dans un formateur /
   minifieur / round-trip naga. → Survivent seulement les canaux liés à la sémantique + la signature si le
   support est préservé. Moyen.
3. **Cloneur depuis les sources** (TON adversaire déclaré) : réimplémente la logique. → **Rien** côté
   binaire ne survit. Sa seule vulnérabilité est la **similarité substantielle du code source** qu'il a
   copié, et le **licensing** s'il veut utiliser TON binaire.

Toute cette stratégie consiste à : (a) maximiser la preuve contre 1 et 2 à coût quasi nul, (b) accepter que
contre 3 la défense est **juridique et contractuelle**, pas technique, et préparer le terrain pour ce combat.

### 1.2 Garde-fous non-négociables (issus des revues sécurité)

Ces contraintes priment sur tout gain de protection. Elles sont détaillées en §7.

- **G1 — Neutralité numérique prouvée.** SciRust est une lib scientifique/DL. Aucun watermark ne doit
  modifier un résultat visible par l'utilisateur. Les mécanismes qui perturbent les ULP de poids faible
  (résidus d'ordre de réduction, « Channel B » de l'autodiff, schedule WGSL) **cassent la
  reproductibilité bit-à-bit** et sont **écartés**.
- **G2 — Zéro préjudice utilisateur.** Pas de corruption, pas de biais silencieux, pas de gate matériel qui
  refuse en plein calcul, pas d'I/O fichier caché sur le chemin de calcul, pas de fingerprint par-licencié
  dans des sorties numériques que l'utilisateur peut publier.
- **G3 — Solidité forensique.** Ne présenter comme *preuve* que les signatures asymétriques, avec custody de
  la graine + racine publique horodatée. Tout le reste est un *tripwire*, jamais une pièce à conviction.
- **G4 — Transparence.** Tout watermarking est divulgué dans l'EULA ; une cible de build « sans watermark »
  (`--no-default-features`) est fournie pour les utilisateurs sensibles à la reproductibilité.

---

## 2. Verdict par pilier (vue d'ensemble)

| Pilier | Meilleur mécanisme conçu | Effort de retrait (red-team) | Verdict |
|---|---|---|---|
| **1. Canary maths** | Canary d'exécution neutre dans `chain()` ; résidus d'ordre de réduction | trivial | **Tripwire neutre = OUI. Résidus numériques = NON (nocifs).** |
| **2. Watermark codegen** | Bannière signée Lamport/Merkle sur l'artefact émis (`emit.rs`) | faible | **OUI comme provenance/traçage de fuite. NON comme anti-clone.** |
| **3. Obfuscation macros** | CFF + prédicats opaques + MBA au moment de l'expansion | trivial | **NON.** Sépararble du calcul, class d'obfuscation la plus mature à désobfusquer, taxe de perf. Garder l'idée de *gating de build* uniquement. |
| **4. Binding environnemental** | Node-lock GPU + gate d'entitlement `CoreKernels` | trivial (anti-clone) | **OUI comme licensing honnête à refus gracieux. NON comme forensique.** |

---

## 3. Pilier 1 — Canary traps mathématiques

### 3.1 Ce qui a été conçu (ancré dans le vrai code)

`scirust-autodiff/src/lib.rs` fait passer **toute** la propagation de tangente non-linéaire du mode forward
par un seul entonnoir : `fn chain(factor, deriv) -> f64` (lignes 50-53), qui gère déjà le cas
`deriv == 0.0 -> 0.0` (garde anti-`0*inf=NaN`). C'est le point d'ancrage idéal — un seul `#[inline]` couvre
Div, `f64/Dual` Div, `powi`, `powf`, `sqrt`, `ln`.

Deux familles de canary y ont été conçues :

**(A) Canary d'EXÉCUTION, strictement neutre — À RETENIR (tripwire).**
Le canary ne touche jamais le `f64` retourné. Il lit `factor`/`deriv` et plie un digest de chemin
d'exécution, dérivé d'une graine liée à ta racine Merkle vendeur, dans un accumulateur `thread_local`.

```rust
// scirust-autodiff/src/lib.rs — dérivé hors-ligne, gravé en const (drift-guard test comme DEMO_ROOT_HEX)
const PROV_TAG: u64 = 0x_....; // = 8 premiers octets de hashsig::hash(b"SRL.canary", vendor_root, b"scirust-autodiff")

#[cfg(feature = "canary")]
thread_local! { static CANARY: core::cell::Cell<u64> = core::cell::Cell::new(0); }

#[inline(always)]
fn chain(factor: f64, deriv: f64) -> f64 {
    #[cfg(feature = "canary")]
    CANARY.with(|c| {
        let h = c.get().rotate_left(7)
            ^ PROV_TAG.wrapping_mul((deriv == 0.0) as u64 + 1)
            ^ (factor.to_bits() >> 52)
            ^ (deriv.to_bits() & 0xFFFF);
        c.set(h);
    });
    // BRANCHE NUMÉRIQUE : identique octet-pour-octet à l'original.
    if deriv == 0.0 { 0.0 } else { factor * deriv }
}
```

Un « probe » figé (fermeture + entrées fixes) produit un digest 64-bit reproductible **uniquement** depuis
cette source. Le harnais de neutralité (fichier `tests/canary_neutrality.rs`) prouve, via proptest sur tout
le domaine (inf/NaN/subnormaux/zéros signés), que `chain(f,d).to_bits() == chain_ref(f,d).to_bits()`.

**(B) Canary NUMÉRIQUE (biais ULP / ordre de réduction) — À NE PAS EXPÉDIER.**
L'idée : exploiter la non-associativité IEEE-754 comme empreinte. Soit dans l'autodiff (« Channel B » :
choisir entre factorisations équivalentes `1/x` vs `x/x²` qui arrondissent différemment), soit dans les
réductions SIMD (`PermFold` : arbre de sommation à permutation clé dans `portable::dot_f32`,
`dispatch::sdot_f32_avx2`, `gemm.rs`).

### 3.2 Verdict red-team

- **Canary A** : *trivial* à retirer. En boîte noire il **ne porte aucun signal** (par construction il ne
  touche pas la sortie). Il n'attrape que le copieur verbatim des sources — que tu condamnes déjà par un
  `diff`. Valeur marginale ~0 au-delà de la preuve de copie que le diff fournit. **Reste utile comme
  tripwire quasi-gratuit**, jamais comme preuve.
- **Canary B** : *trivial* aussi (réécrire la réduction en ordre canonique / activer FMA / fast-math /
  changer d'ISA l'efface incidemment) **ET activement nocif** (voir §3.3). Le red-team juridique le classe
  **CRITIQUE** : p-values fictives (les 64 probes ne sont pas indépendantes — un seul choix d'arbre fixe la
  classe de résidu pour toutes, l'effectif réel ≈ 1), et le résidu est dérivé de la racine **publique** donc
  **forgeable** — ça prouve au mieux « connaissance d'une donnée publique », pas l'accès à un secret.

### 3.3 Revue de correction numérique (pourquoi B est écarté)

Mesuré sur 99 999 entrées échantillonnées : la réécriture `ln` `1/x -> x/x²` diffère sur **25 %** des
entrées ; la réécriture `div` `f/g² -> (f/g)/g` sur **35 %** (~1 ULP chacune). Ce ne sont pas des cas de
bord rares. Dans des optimiseurs itérés ou des solveurs d'ODE, ces erreurs ULP par pas **s'amplifient** en
trajectoires visiblement divergentes. De plus la « borne à quelques ULP » ne tient que pour des sommes bien
conditionnées : sur du dot/GEMM à forte compensation (résultat proche de zéro), l'écart entre deux ordres de
sommation est **non borné relativement au résultat** et peut dépasser les tolérances 1e-3/1e-2 des tests
GEMM.

Détail piégeux confirmé : même le swap « Channel A » `a-b -> -(b-a)` **n'est pas** bit-exact — il transforme
`+0.0` en `-0.0` quand les deux contributions sont égales (ex. dérivée de `x/x`), et ce zéro signé se
propage (`1.0/deriv -> -inf` vs `+inf`, `copysign`, `atan2`, `signbit`). `assert_eq!` le masque
(`-0.0 == 0.0`). **Ne garder que le swap commutatif des addends de `Mul`** (bit-exact pour les valeurs
finies/zéros), et le classer « dans la tolérance », pas « bit-exact ».

### 3.4 Recommandation pilier 1

- **Expédier** : le canary d'exécution neutre (A) **derrière une feature `canary` non-défaut**, présenté
  honnêtement comme tripwire anti-copie-verbatim, avec son harnais de neutralité en CI. Coût ~nul.
- **Ne pas expédier** : tout résidu numérique (B, PermFold, résidus d'ordre). Si tu y tiens malgré tout,
  c'est **opt-in par défaut-off**, exclu de tout chemin déterministe (`deterministic_fp32_gemm`, KahanSum,
  Zq/INT8/Q15.16/Q31.32/`dequantize_int4`), avec **clé unique commune à tous les licenciés** (jamais
  par-siège) et divulgation dans la doc de reproductibilité.
- **Garder intacts comme ancres d'intégrité** : les chemins à contrat bit-exact
  (`dequantize_int4_simd_matches_scalar_bit_exact`) — ne jamais y injecter de mark ; s'en servir au
  contraire comme détecteurs d'altération.

---

## 4. Pilier 2 — Watermarking transpileur / codegen

### 4.1 Le seul mécanisme à valeur juridique réelle : bannière signée sur l'artefact émis

Point d'ancrage réel : `scirust-transpiler/src/emit.rs::emit_module` (lignes 15-25) est **l'unique** point
où `PRELUDE + join(emit_func)` est concaténé — les deux front-ends (Python, MATLAB) y passent. C'est le
chokepoint parfait pour signer.

Le mécanisme réutilise **tel quel** `scirust-license/src/hashsig.rs` (SHA-256 Lamport OTS + Merkle,
déterministe) — aucune crypto réinventée :

```
//! srl-emit:v1 root=<hex8> leaf=<u32> sig=<MerkleSig::to_hex>
```

- La bannière est un commentaire Rust (`//!`) : le lexer la jette, **l'artefact compilé est identique
  bit-à-bit** (neutralité G1 respectée). Le champ `fmt_f64` n'est **jamais** touché (aucune perturbation de
  littéral).
- La signature porte sur une **canonicalisation** de l'artefact (`C(src)` : retire la bannière, retire les
  commentaires en respectant les littéraux chaîne, collapse les espaces). Elle survit donc à
  l'indentation / au reformattage / à `rustfmt`.
- Un concurrent **ne peut pas transplanter ta bannière** sur son propre code : son digest canonique ne
  matchera pas, `hashsig::verify` renvoie `false`.
- **La graine secrète maîtresse ne quitte jamais le poste vendeur.** La signature se fait dans une **passe
  hors-ligne séparée**, pas dans le transpileur expédié. Le binaire client ne porte que la **racine publique
  32-octets** (motif `DEMO_ROOT_HEX` + drift-guard test). `emit_module` prend un `Option<&Provenance>` :
  `None` => sortie identique à aujourd'hui (tests snapshot et builds OSS inchangés).

Un outil `prov verify artefact.rs` (nouveau `src/bin/prov.rs`) recanonicalise, recalcule le digest, et
appelle `scirust_license::hashsig::verify(&EMIT_ROOT, &digest, &sig)` contre la racine **publique** épinglée.
Un expert tiers, avec la seule racine publique, **reproduit le verdict et ne peut pas forger** — c'est
exactement ce qui le rend admissible.

**Canal de redondance covert (survit au strip de commentaires)** : `PRELUDE` (`emit.rs:28-345`) est ~25
fonctions libres `np::` mutuellement indépendantes (`sum`, `dot`, `prod`, `max`, `zeros`, `linspace`…).
L'ordre des items est sémantiquement inerte en Rust → les émettre dans une permutation dérivée de la clé
(K=8 → 8! = 40320 ordres). Survit à `rustfmt` et à la minification ; tombe sous une passe AST qui re-trie.
**À présenter comme corroboration faible, jamais comme preuve autonome** (voir §4.3).

### 4.2 Verdict red-team

- Contre le **cloneur depuis les sources** : **zéro**. Il rebuild depuis le chemin `None`/OSS et n'émet
  aucun watermark. Le watermark protège l'**artefact**, pas l'**outil**.
- Contre le **lessiveur d'artefacts** : un seul round-trip AST (`syn` + `prettyplease`, ~40 lignes) tue les
  deux couches (commentaires jetés → bannière morte ; items re-triés → permutation à 0 bit). « Sémantiquement
  inerte » = « canonicalisable par l'attaquant ».
- Contre le **redistributeur verbatim** : **fort et propre**. Tant que la bannière est présente, la
  signature asymétrique est une preuve non-forgeable, reproductible par un expert neutre. C'est la vraie
  valeur, et elle sert surtout à **tracer quel client a fuité** (le `leaf` OTS est un numéro de série unique).

### 4.3 Revue juridique (les conditions sine qua non)

Le red-team juridique classe la couche signature **HIGH** *utilisable seulement si* trois choses, absentes de
tous les designs, sont établies **avant toute distribution** :

1. **Custody de la graine maîtresse** : générée dans un HSM, procédure loggée et à accès contrôlé.
2. **Ancrage horodaté de la racine publique** : publiée dans un registre immuable et horodaté
   (Certificate-Transparency, tag git signé et poussé, notarisation/ancrage) **avant** l'artefact suspect —
   sinon la défense « le vendeur a planté la signature après coup » reste sans réponse.
3. **Reproduction par un expert tiers**, pas par le vendeur.

Autres points **MEDIUM/LOW** à intégrer :
- Une signature valide prouve la **provenance des octets, pas l'acte de copie** par ce défendeur. La
  cadrer plutôt comme **retrait d'information de gestion de droits (DMCA §1202)** / violation de licence,
  dont la force réelle est la non-forgeabilité + résistance au retrait.
- Les canaux combinatoires à petit alphabet (permutation 1/40320) calculent leur FP contre un auteur
  uniforme idéal ; les vrais auteurs indépendants se regroupent sur des ordres conventionnels (alphabétique,
  graphe d'appel). **Calibrer le taux de faux positifs empiriquement** contre un corpus réel avant de citer
  la moindre probabilité.

### 4.4 Recommandation pilier 2

- **Expédier** : la **passe de signature hors-ligne** (couche 1) sur les artefacts émis, réutilisant
  `hashsig`, + l'outil `prov verify`, + la discipline custody/horodatage de §4.3. C'est ton unique dispositif
  à réelle valeur de tribunal, et il est peu coûteux et sûr (G1 respecté).
- **Garder en corroboration** : la permutation `PRELUDE` (couche covert), honnêtement scopée « ne tient que
  contre rustfmt-seul, tombe sous normalisation AST ».
- **Abandonner** : toute prétention que ceci défend contre le clone-depuis-source. Et **abandonner** le
  canal WGSL équivalent (steganographie d'identifiants/ordre dans les shaders) — naga efface CH1/CH2 avant
  même l'exécution sur backends non-passthrough, donc le programme GPU déployé ne porte aucun mark
  (détectable seulement sur la source pré-naga, qu'un réimplémenteur n'embarque jamais).

---

## 5. Pilier 3 — Obfuscation des macros procédurales

### 5.1 Ce qui a été conçu

Points d'ancrage réels : `scirust-macros/src/lib.rs` (le corps `_grad` généré, lignes 144-179) et
`scirust-simd-macros/src/lib.rs` (les 4 ré-émissions `#block` avx2/sse2/neon/scalar, lignes 62-100). Trois
couches keyed par une graine vendeur 128-bit dérivée via `hashsig::hash(b"SRL.wm", …)` :

1. **Control-flow flattening** : découper le corps en `S0..Sk`, dispatcher via une machine à états
   `loop { match __st { … } }` dont les labels sont un PRP keyed (constantes splitmix64 déjà présentes dans
   `scirust-gpu/deterministic.rs:201-204`). L'ordre d'exécution est préservé.
2. **Prédicats opaques** sur les transitions : `if OPAQUE(__st) { real } else { dead }` avec une identité
   MBA toujours-vraie mais non-repliable par LLVM (`black_box` sur un opérande).
3. **Labels MBA** : les constantes de label émises comme expressions MBA de la graine.

L'arithmétique `f64`/`Dual`/SIMD est **copiée token-pour-token** dans les bras d'états — rien de numérique
n'est touché (MBA reste dans le domaine `u64`).

### 5.2 Verdict red-team + perf (pourquoi on abandonne)

- **Séparabilité fatale** : par conception « rien de numérique n'est touché ». L'adversaire avec les sources
  ne ship jamais l'empreinte — il réimplémente les deux macros triviales (`#[autodiff]` émet des seeds
  `Dual::var/primal` + `.grad()` ; `#[simd]` émet 4 copies `target_feature` + une échelle
  `is_x86_feature_detected`), quelques heures.
- **CFF est la classe d'obfuscation la PLUS mature à désobfusquer** : le prédicat `(a|b)==(a^b)+(a&b)` est
  une identité MBA de manuel que GAMBA/msynth/SSPAM réduisent instantanément ; `black_box` devient un nœud
  identité ; angr/miasm relinéarise le `loop{match}`. Pire, le doc de conception vante le prédicat comme un
  « idiome rare et distinctif » — c'est un **signal greppable qui LOCALISE** les fonctions instrumentées pour
  l'attaquant.
- **Taxe de perf sur les utilisateurs honnêtes** : `black_box` sur le dispatch SIMD et le hot path autodiff
  **inhibe l'inlining/la vectorisation** exactement là où la lib doit être rapide. La revue préjudice classe
  ça **MEDIUM** : coût de perf permanent pour les honnêtes, watermark qui se retire gratuitement.

### 5.3 Ce qu'on garde de ce pilier

Uniquement l'**hygiène de build à deux clés** (bonne pratique, pas anti-clone) :
- Feature `obf` non-défaut sur les crates macros + `proc_macro::tracked_env` pour forcer-off en build
  reproductible.
- Split de profil émis : bras propre sous `#[cfg(debug_assertions)]`, bras (éventuellement obfusqué) sous
  `#[cfg(not(debug_assertions))]`. Le cfg-stripping a lieu **avant** l'abaissement HIR/MIR → le dev loop /
  `cargo test` reste **identique octet-pour-octet à aujourd'hui**, coût zéro en debug.

**Recommandation** : ne PAS injecter CFF/prédicats/MBA dans les hot paths numériques. Rediriger l'effort
vers la crypto de signature (§4) et le licensing (§6). L'obfuscation de macro contre un adversaire qui tient
les sources est du théâtre coûteux.

---

## 6. Pilier 4 — Binding environnemental / attestation (non-destructif)

### 6.1 Ce qui existe déjà et ce qui a été conçu

Bonne nouvelle : `scirust-license` fournit déjà tout le nécessaire — `verify_license_on_node` (lib.rs:286),
`module_gate!` (gate.rs), `node_fingerprint` **salé par identité de licence** (license.rs:181, avec un test
« la machine_id brute ne doit pas fuiter »), et le motif token zéro-sized `_sealed:()` inconstructible sans
un `Entitlements::require` réussi. Le crate est **sans horloge ni réseau** (le `now:u64` est fourni par
l'hôte) → **aucun phone-home**, vérification 100 % hors-ligne.

Deux bindings ont été conçus :
- **Node-lock GPU** : dériver un `cap_hash` de `adapter.get_info()` (`wgpu_backend.rs:675`), le passer comme
  `machine_id` à `verify_license_on_node` **avant** tout `create_shader_module`. Un shader extrait dans le
  harnais d'un concurrent n'atteint jamais ce handshake → le dispatch refuse (`Err`), jamais de corruption.
- **Mur d'entitlement `CoreKernels`** sur `sgemm_tiled`/`dgemm_tiled` (renommés `*_impl` `pub(crate)`,
  corps **inchangés**), ré-exposés en méthodes du token.

### 6.2 Verdict red-team

Comme anti-clone : **trivial** à retirer (l'adversaire supprime l'appel `verify_*`, dé-`pub` les kernels,
retire la dépendance `scirust-license`). Un token zéro-sized ne compile en **aucun** code machine — c'est une
barrière de visibilité, et l'accès aux sources dissout une barrière. Valeur forensique contre le cloneur :
**zéro** (le vrai signal de copie reste le **corps du micro-kernel** copié, orthogonal au gate).

### 6.3 Revue de préjudice utilisateur (les pièges à éviter absolument)

La revue classe les gates à **refus dur** en **HIGH** — c'est l'anti-feature classique qui punit les clients
payants :
- `adapter.get_info().name` est une **chaîne instable** (upgrade de driver, Mesa/lavapipe vs GPU réel,
  migration VM/conteneur, reprogrammation d'instance cloud, swap eGPU). Un licencié légitime peut être
  **refusé en plein workload**.

**Équivalents non-destructifs obligatoires :**
- (a) **soft-fail / warn-and-continue** sur nœud non reconnu, avec une **période de grâce hors-ligne
  généreuse**, jamais un refus de calcul ;
- (b) lier au mieux à une **classe de capacité stable et grossière** (famille vendeur / feature set), pas à
  la chaîne exacte ;
- (c) rester **hors-ligne** (aucun phone-home) ;
- (d) fournir un **chemin de ré-activation manuelle hors-ligne** documenté, pour qu'un changement d'environ-
  nement ne nécessite jamais de contacter le vendeur ;
- (e) **jamais** de gate pouvant `Err` **dans ou après** un calcul partiellement exécuté ;
- (f) fournir une licence d'éval/démo + un chemin de licence par variable d'env, pour qu'une config
  transitoire ne verrouille jamais un utilisateur honnête.

**Interdits stricts (G2) :** pas d'I/O fichier sur le chemin de calcul (le « journal de provenance append-only »
conçu écrit un fichier depuis `derivative/gradient/backward` — à retirer : échoue en env read-only/HPC/
conteneur, remplit le disque, course entre process, constitue un logging non-divulgué) ; pas d'identité
par-licencié dans des sorties numériques publiables ; pas de payload destructif/anti-tamper.

### 6.4 Recommandation pilier 4

- **Expédier** : le gate d'entitlement + node-lock **comme licensing produit honnête**, à **refus gracieux**
  (`Result`, `Err` documenté, instantanément réversible en fournissant une licence), avec **tous** les
  garde-fous §6.3. C'est une vraie valeur — contre l'utilisateur qui n'a pas payé et le partage casual — et
  le node-lock salé/préservant la vie privée est bien conçu.
- **Ne pas vendre** ça comme anti-clone ni comme forensique. Contre le cloneur, ça ne lie rien.
- **Ajouter** un variant `Module::Gpu`/`Module::Autodiff` propre (plutôt que sur-scoper `Module::Core`) pour
  séparer les entitlements.

---

## 7. Les garde-fous, en détail (revues sécurité)

### 7.1 Correction numérique (G1)
- **Exclusion dure** de tout canary d'ordre de réduction des chemins `deterministic_fp32_gemm`, `KahanSum`,
  `Zq`/`INT8`/`Q15.16`/`Q31.32`/`dequantize_int4`. Ajouter un **golden test** qui épingle les bits de sortie
  de `deterministic_fp32_gemm` contre un vecteur de référence **checké-in** (cross-build, pas seulement
  run-to-run — le test actuel `…is_bit_reproducible` ne vérifie que run-to-run et laisserait passer un ordre
  keyed fixe) + une assertion de graphe d'appel que la réduction watermarkée est **inatteignable** depuis les
  entrées déterministes.
- Restater la borne de neutralité **relativement à `sum|termes|`**, pas au résultat. Rendre le self-check
  Kahan-oracle **obligatoire pour toutes** les variantes de réduction + test sur cas mal conditionnés
  (compensation délibérée, grand K).
- Harnais `to_bits()` sur **tout le domaine** (inf/NaN/subnormaux/zéros signés) obligatoire pour tout
  mécanisme prétendu neutre. Documenter l'exception charge-utile NaN both-operands (dépend de l'ordre sur SSE
  x86).

### 7.2 Préjudice utilisateur (G2)
Voir §6.3. En résumé : refus gracieux jamais dur, aucun I/O fichier sur le chemin de calcul, aucune identité
par-licencié dans les sorties numériques, obfuscation/canary **hors** hot path (bookkeeping une fois par
process en init froid, jamais par-op), et **opt-out + divulgation EULA** pour tout.

### 7.3 Solidité forensique (G3)
- Seules les couches **signature asymétrique** sont présentables comme preuve, et seulement contre
  redistribution verbatim, **et** seulement avec custody HSM + racine horodatée + reproduction par expert
  tiers (§4.3).
- **Reclasser explicitement en tripwire/dissuasion** : tous les résidus ULP/ordre de réduction (p-values
  non-indépendantes = fictives ; résidus dérivés de racine publique = forgeables) et tous les canary
  d'exécution neutres (aucun signal en boîte noire ; n'attrapent que le copieur verbatim déjà condamné par
  diff).
- **Reconnaître le retournement de ta propre doc** : le crate documente que les réductions diffèrent
  bit-à-bit selon backend/largeur et ne sont garanties qu'en tolérance. Cette même clause qui rend le
  watermark « neutre » **détruit sa valeur probante** (défense « juste un autre ordre de sommation valide »).
- **Bit-identité d'un autodiff dual-number n'est PAS probante** : forward-mode duals et tape reverse-mode
  sont de manuel ; le déterminisme IEEE-754 rend des résultats identiques **attendus** de toute
  implémentation correcte indépendante. Aucun rapport d'expert ne doit prétendre l'inverse.
- Construire la vraie pièce à conviction autour de la **similarité substantielle du source** (schéma de
  tuilage, packing, constantes, commentaires, identifiants) — préserver et documenter la provenance des
  kernels.

---

## 8. Plan d'action priorisé

**P0 — Fondations juridiques (à faire AVANT toute distribution) :**
1. Générer la graine maîtresse Merkle en HSM, procédure loggée. Publier la racine publique dans un registre
   horodaté immuable (tag git signé + CT log). Sans ça, aucune signature n'a de valeur au tribunal.
2. Divulguer le watermarking dans l'EULA ; documenter une cible `--no-default-features` sans watermark.

**P1 — Ce qui a une valeur réelle (à implémenter) :**
3. Passe de **signature hors-ligne** sur `emit_module` (réutiliser `hashsig`) + outil `prov verify`
   (§4.1/§4.4). Domaine `b"scirust-emit:v1\0"`, `EMIT_ROOT_HEX` + drift-guard test.
4. **Licensing à refus gracieux** : finaliser `CoreKernels` / node-lock avec **tous** les garde-fous §6.3
   (soft-fail, grâce hors-ligne, classe de capacité, ré-activation manuelle). Ajouter `Module::Gpu`.
5. **Golden tests de reproductibilité** cross-build sur les chemins déterministes (§7.1) — protège tes
   utilisateurs ET sert d'ancre anti-altération.

**P2 — Tripwires quasi-gratuits (honnêtement étiquetés) :**
6. Canary d'exécution neutre dans `chain()` derrière feature `canary` non-défaut + harnais de neutralité.
7. Permutation `PRELUDE` covert comme corroboration faible.
8. Hygiène de build deux-clés (§5.3) pour isoler tout code sensible au profil release.

**À NE PAS FAIRE (anti-patterns nocifs, résumé) :**
- ❌ Perturber les bits de sortie (résidus d'ordre, Channel B, schedule WGSL) — casse la repro, nocif, se
  retire gratis.
- ❌ Clé de watermark **par-licencié** dans du numérique — deux installs licenciés donneraient des résultats
  différents (fatal pour une lib scientifique).
- ❌ Gate matériel à **refus dur** sur `adapter.get_info()` — verrouille les clients payants.
- ❌ I/O fichier / logging caché sur le chemin de calcul.
- ❌ `black_box`/CFF/MBA dans les hot paths numériques.
- ❌ Présenter un résidu forgeable dérivé de racine publique comme « preuve keyée ».

---

## 9. Annexe — Points d'ancrage réels (carte de recon)

| Crate | Fichier:symbole | Rôle / opportunité |
|---|---|---|
| autodiff | `lib.rs:50-53` `fn chain` | Entonnoir unique de la tangente forward — canary d'exécution neutre |
| autodiff | `lib.rs:18-30` `Dual::var/primal/new` | Amorçage de tangente |
| autodiff | `lib.rs:507-534` `derivative_1d/gradient_2d/3d` | Frontière de driver (probes / gate) |
| simd | `portable.rs:151/170`, `dispatch.rs:324` | Réductions de dot (ancres d'intégrité, **pas** de résidu) |
| simd | `gemm.rs` `micro_kernel_8x16`/`sgemm_tiled` | Crown-jewel — similarité source = la vraie preuve ; mur d'entitlement |
| simd | `dequantize_int4…bit_exact` | Contrat bit-exact — ancre anti-altération, **jamais** de mark |
| transpiler | `emit.rs:15-25` `emit_module` | Chokepoint unique — signature + bannière |
| transpiler | `emit.rs:28-345` `PRELUDE` | Canal covert de permutation |
| transpiler | `emit.rs:1484` `fmt_f64` | **Interdit** de toucher (repro f64) |
| gpu | `wgpu_backend.rs:675` `adapter.get_info()` | Node-lock (classe de capacité, soft-fail) |
| gpu | `deterministic.rs` `verify_bit_exact` | Contrat déterministe — golden test cross-build |
| macros | `scirust-macros/lib.rs:144-179` | Gating de build (pas d'obfuscation en hot path) |
| license | `hashsig.rs` (Lamport/Merkle SHA-256, déterministe) | **Réutiliser** pour toute signature |
| license | `lib.rs:286` `verify_license_on_node`, `gate.rs` `module_gate!` | Licensing à refus gracieux |
| license | `license.rs:181` `node_fingerprint` (salé) | Node-lock préservant la vie privée |

---

_Fin de l'audit. La ligne directrice tient en une phrase : contre un cloneur qui a tes sources, ta défense
est juridique et contractuelle, pas technique — alors investis l'effort d'ingénierie là où il compose avec
le droit (signatures asymétriques provenance + horodatage + custody, similarité de source préservée,
licensing honnête), et n'expédie jamais un « piège » qui abîme tes utilisateurs honnêtes pour un cloneur
qu'il ne ralentit pas._
