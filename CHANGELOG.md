# Changelog

Le format suit [Keep a Changelog](https://keepachangelog.com/) ;
versions sémantiques à partir de la prochaine release taguée.

## [Non publié]

### Ajouté — `scirust-sigma` : bornes structurelles σ (« couvercle de zéro ») + audit des epsilons
Nouvelle crate feuille **sans dépendance externe** (`std` seul) qui nomme et
encode l'invariant numérique jusqu'ici implicite du contrat de déterminisme :

- **σ = couvercle de zéro par régime.** Chaque voie numérique déterministe
  (`scirust-gpu/src/deterministic.rs`) a un plus petit positif représentable :
  entier exact `1`, fixe Q15.16 `2⁻¹⁶`, fixe Q31.32 `2⁻³²`, f32 *sanitized*
  `f32::MIN_POSITIVE`, f32 brut / f64 brut = plus petit sous-normal. Constantes
  nommées (`SIGMA_SANITIZED_F32`, `SIGMA_RAW_F32`, `SIGMA_RAW_F64`, `SIGMA_Q15_16`,
  `SIGMA_Q31_32`), `sigma_f32`/`sigma_f64`, `guard_denominator_f32/f64`, et
  l'**invariant central** `is_valid_guard_f32` (une garde anti-zéro sous σ est
  *morte* sur la voie sanitized : `sanitize_f32` l'écrase). Comportements de bord
  (0, négatif, NaN, régime sans σ f32) définis et testés — 12 tests unitaires
  aux valeurs bit-à-bit (`to_bits()`).
- **Test d'alignement** (`tests/sanitize_alignment.rs`) : affirme, sans coupler
  la crate à `scirust-gpu`, que le seuil de `sanitize_f32` (= `f32::MIN_POSITIVE`)
  est bit-identique à `SIGMA_SANITIZED_F32`. Casse si l'un bouge sans l'autre.
- **Binaire `epsilon-audit`** (std-only ; `sha2` déjà au lockfile scelle le
  rapport) : scanner lexical maison (hors commentaires/chaînes) qui classe les
  ~14 400 littéraux flottants `< 1.0` du workspace en A (algorithme, ne pas
  migrer) / B (garde zéro, cible σ) / C (test) / D (convergence) / U (non classé),
  et produit `docs/EPSILON_AUDIT.md` (rapport déterministe, scellé SHA-256).
- **Gate CI `--check`** : sort ≠ 0 si une garde f32 sous σ_sanitized subsiste
  hors test dans `scirust-gpu/src`. Exit 0 sur l'arbre actuel (aucune garde
  morte sur la voie sanitized ; 686 littéraux gpu/src inspectés). Sécurité :
  contrôle préventif d'une classe de défauts (garde morte → `Inf`/`NaN`
  silencieux) invisible en revue humaine, sans surface d'approvisionnement
  ajoutée, artefact scellé, binaire strictement en lecture seule.

### Ajouté — crates voisines : carte CUSUM (`scirust-spc`) et incertitude élargie GUM (`scirust-metrology`)
Deux compléments dans les crates voisines du tolérancement, chacun vérifié par
un cross-check Monte-Carlo embarqué dans ses tests :

- **`scirust-spc::cusum`** : **carte de contrôle CUSUM** tabulaire bilatérale.
  Sommes cumulées `Cᵢ⁺ = max(0, Cᵢ₋₁⁺ + (xᵢ−μ₀) − K)` / `Cᵢ⁻` avec valeur de
  référence `K = k·σ` et intervalle de décision `H = h·σ`, signal de dérive, et
  **ARL** par l'approximation de Siegmund (`b = h+1,166`), combinée en bilatéral
  `1/ARL = 1/ARL⁺ + 1/ARL⁻`. Complète EWMA comme détecteur à mémoire des petites
  dérives soutenues. *Cross-check* : taux de fausse alarme Monte-Carlo (N(0,1))
  vs l'ARL₀ ≈ 168 pour `k=0,5, h=4`.
- **`scirust-metrology::expanded`** : **incertitude élargie** GUM. Degrés de
  liberté effectifs de Welch–Satterthwaite `ν_eff = u_c⁴ / Σ(uᵢ⁴/νᵢ)`, facteur
  d'élargissement `k = t_{(1+p)/2}(ν_eff)` (quantile de Student par développement
  de Cornish–Fisher, exact quand `ν_eff→∞ ⇒ k→1,96`), incertitude élargie
  `U = k·u_c` et intervalle de couverture. Boucle le GUM après `combined_uncertainty`.
  *Cross-check* : quantiles `t` vs tables (t₀,₉₇₅(10)=2,228, …).

### Ajouté — plan & économie : ajustements ISO 286, échantillonnage double/séquentiel, perte de Taguchi (`scirust-tolerance`)
Trois modules qui bordent la cotation : la table normalisée des ajustements, les
plans d'échantillonnage à taille moyenne réduite, et le coût de non-qualité relié
à l'inertie. Chaque module est vérifié par cross-check de fuzzing contre une
**référence indépendante** :

- **`fits`** : **limites et ajustements ISO 286**. Tolérance normalisée `ITn` à
  partir du facteur `i = 0,45·∛D + 0,001·D` (µm) et des multiplicateurs de grade
  (IT5–IT18), écarts fondamentaux d'**arbre** `d, e, f, g, h` (formules ISO
  vérifiées), et **classification d'ajustement** trou/arbre en système à trou de
  base H (jeu maxi/mini, catégorie jeu / incertain / serrage). *Cross-check* :
  identité « étendue de jeu = IT_trou + IT_arbre », recomputation indépendante du
  `IT` par la formule du facteur `i`, monotonie en grade.
- **`sequential`** : **échantillonnage double et séquentiel** (SPRT de Wald).
  Plan double `(n1,c1,r1,n2,c2)` de CE binomiale `Pa(p) = P(d1≤c1) + Σ P(d1=k)·P(d2≤c2−k)`
  et nombre moyen d'échantillon `ASN = n1 + n2·P(c1<d1<r1)` ; SPRT à deux droites
  frontières `d = s·n ∓ h` (accepter / rejeter / continuer). *Cross-check* :
  CE et ASN du plan double vs Monte-Carlo direct ; garantie de la CE du SPRT aux
  deux points de conception.
- **`taguchi`** : **perte de Taguchi et coût de non-qualité**. Perte quadratique
  `L = k(y−T)²`, coefficient `k = A/Δ²` calé sur le coût à la limite, et l'identité
  `E[L] = k·(σ²+δ²) = k·I²` — la raison exacte pour laquelle le tolérancement
  inertiel minimise directement la perte de Taguchi. Variantes plus-petit/plus-grand-
  c'est-mieux et **tolérance économique** `Δ = Δ₀·√(A/A₀)`. *Cross-check* : Monte-Carlo
  de la perte quadratique vs `k·I²` ; équilibre de la tolérance économique.

Câblés dans `scirust-mcp` (`tolerance_fits`, `tolerance_sequential`,
`tolerance_taguchi`). Fuzz global : **118 476 checks / 0 erreur** sur 29 modules.

### Ajouté — atelier : échantillonnage aux attributs, interférence contrainte-résistance, étude de capabilité par sous-groupes (`scirust-tolerance`)
Trois modules qui complètent la boîte à outils qualité d'atelier : accepter un
lot sans mesurer, chiffrer la fiabilité d'un **ajustement** aléatoire, et mener
une vraie **étude de capabilité** à sous-groupes rationnels. Chaque module est
vérifié par cross-check de fuzzing contre une **référence indépendante** :

- **`attributes`** : **échantillonnage aux attributs** (ISO 2859-1 / MIL-STD-105).
  Plan simple `(n, c)` — accepter si défectueux `≤ c` — de courbe d'efficacité
  binomiale exacte `Pa(p) = Σ_{d≤c} C(n,d) pᵈ(1−p)ⁿ⁻ᵈ` (récurrence stable), avec
  **conception à deux points** (balayage de `c` croissant, plus petit `n` tenant
  le point producteur) et qualité moyenne après contrôle (AOQ). *Cross-check* :
  Monte-Carlo direct de la règle d'acceptation vs la CE binomiale ; les plans
  conçus tiennent les deux points nominaux.
- **`interference`** : **interférence contrainte-résistance** et fiabilité
  d'ajustement. Fiabilité `R = P(S > L) = Φ(β)`, `β = (μ_S−μ_L)/√(σ_S²+σ_L²)`
  (indice de fiabilité), et analyse d'**ajustement** alésage/arbre : jeu
  `C = alésage − arbre ∼ N(μ_h−μ_s, σ_h²+σ_s²)`, `P(jeu > 0)` (ajustement libre)
  vs `P(jeu < 0)` (serrage) — la probabilité qu'une paire tirée au hasard
  s'assemble comme prévu, qu'un pire-cas min/max ne donne pas. *Cross-check* :
  Monte-Carlo de `P(S>L)` vs la forme close ; identités de partition du jeu.
- **`subgroup`** : **étude de capabilité par sous-groupes rationnels** (MSA AIAG /
  ISO 22514-2). Sépare la dispersion **intra-sous-groupe** (court terme,
  `σ̂ = R̄/d₂ = s̄/c₄` via les constantes de cartes de contrôle) qui porte les
  indices de **capabilité** `Cp`/`Cpk`, de la dispersion **globale** (long terme)
  qui porte les indices de **performance** `Pp`/`Ppk` : un grand `Cp` avec un
  petit `Pp` signale un procédé stable mais qui dérive. *Cross-check* :
  recalcul indépendant du σ global ; accord des estimateurs `R̄/d₂` et `s̄/c₄` ;
  identité du `Cp`.

Câblés dans `scirust-mcp` (`tolerance_attributes_plan`, `tolerance_interference`,
`tolerance_subgroup_capability`). Fuzz global : **113 336 checks / 0 erreur** sur
26 modules.

### Ajouté — qualité de procédé : échantillonnage aux mesures, Six-Sigma, attribution des causes (`scirust-tolerance`)
Trois modules qui prolongent la couche mesure & analyse vers le **pilotage
qualité** que les suites concurrentes (Minitab, Q-DAS) offrent autour de la
cotation : accepter un lot sur mesures, chiffrer le rendement d'un procédé
multi-étapes, et remonter des données à la cause. Chaque module est vérifié par
cross-check de fuzzing contre une **référence indépendante** :

- **`variables`** : **échantillonnage aux mesures** (ISO 3951 / MIL-STD-414,
  forme `k`). Accepte le lot quand la distance normalisée `Q = (limite−x̄)/σ ≥ k` ;
  courbe d'efficacité en forme close `Pa(p) = Φ(√n_eff·(z_p−k))` avec `z_p = −Φ⁻¹(p)`,
  et **conception à deux points** `√n = (z_{1−α}+z_{1−β})/(z_aql−z_rql)`,
  `k = (z_aql·z_{1−β}+z_rql·z_{1−α})/(z_{1−α}+z_{1−β})` depuis (AQL, RQL, α, β).
  Méthodes `σ` connu et `s` inconnu (échantillon gonflé de `1+k²/2`) ; écart-type
  maximal admissible pour un lot centré `MSD = (USL−LSL)/(2k)`, pendant aux mesures
  du budget inertiel `I_max`. *Cross-check* : Monte-Carlo direct de la règle
  d'acceptation vs la CE close ; identité `MSD`.
- **`sixsigma`** : **comptabilité de rendement Six-Sigma**. DPU, DPMO, rendement
  de passage `Y = e^(−DPU)`, **rendement roulé** `RTY = ∏ Yᵢ` (la probabilité
  qu'une pièce franchisse *toutes* les étapes sans reprise, invisible sur une
  seule capabilité), rendement normalisé `RTY^(1/étapes)`, et les conversions
  rendement↔niveau sigma↔DPMO `Z = Φ⁻¹(Y)+décalage` avec le décalage `1,5σ`
  Motorola (d'où « 6σ ⇒ 3,4 DPMO »). *Cross-check* : aller-retours vs la queue
  normale indépendante ; `RTY` vs produit explicite ; `−ln Y = DPU` de Poisson.
- **`attribution`** : **attribution des causes pilotée par les données**. Ajuste
  l'assemblage mesuré aux composants co-mesurés par moindres carrés `y ≈ β₀ + Σ βⱼxⱼ`
  et décompose la variance expliquée par l'identité exacte (MCO avec constante)
  `Σⱼ βⱼ·Cov(xⱼ,y) = Var(ŷ) = R²·Var(y)` : sensibilités **empiriques** `βⱼ` (à
  confronter aux `αⱼ` de conception), parts signées `cⱼ = βⱼ·Cov(xⱼ,y)/Var(y)`
  (somment à `R²`, mesure de Pratt) et **reste inexpliqué** `1−R²` qui trahit une
  cause hors du jeu mesuré. *Cross-check* : identité `Σcⱼ = R²` ; récupération des
  coefficients générateurs ; `c = corr²` à un seul régresseur.

Câblés dans `scirust-mcp` (`tolerance_variables_plan`, `tolerance_six_sigma`,
`tolerance_attribution`). Fuzz global : **111 926 checks / 0 erreur** sur 23
modules.

### Ajouté — la couche mesure & analyse du tolérancement inertiel (`scirust-tolerance`)
Six modules qui portent la crate au niveau des produits concurrents (Minitab,
Q-DAS, 3DCS, CETOL) sur ce qu'ils font *autour* de la cotation : qualifier le
moyen de mesure, borner statistiquement, séparer les leviers, ajuster une loi,
approfondir la GD&T et chiffrer l'incertitude d'un indice. Chaque module est
vérifié par cross-check de fuzzing contre une **référence indépendante** :

- **`msa`** : **Gage R&R croisé par ANOVA** (MSA AIAG). Décomposition du modèle
  `yᵢⱼₖ = μ + Partᵢ + Opⱼ + (Part·Op)ᵢⱼ + εᵢⱼₖ` en composantes de variance
  répétabilité `EV` / reproductibilité `AV` / pièce `PV`, avec `%R&R` (variation
  d'étude), `%contribution`, `%tolérance = 6σ_GRR/(USL−LSL)`, `ndc = ⌊1,41·σ_PV/σ_GRR⌋`
  et verdict AIAG (bandes 10 %/30 %). *Cross-check* : identité de décomposition
  des sommes de carrés ; constructions à répétabilité/reproductibilité nulles.
- **`interval`** : **intervalles statistiques de tolérance** (ISO 16269-6).
  Facteur bilatéral de Howe `k = z_{(1+p)/2}·√(ν(1+1/n)/χ²_{ν,α})` et unilatéral
  de Natrella (forme fermée) ; tendent tous deux (lentement, en `1−1,645√(2/ν)`)
  vers le quantile normal. *Cross-check* : couverture Monte-Carlo du vrai contenu.
- **`sensitivity::dual_contributions`** : **GeoFactor / sensibilité duale** — pour
  chaque contributeur, magnification géométrique `|αᵢ|`, part sur la **moyenne**
  d'assemblage `αᵢδᵢ` (somment à `δ_Y`) *et* part sur la **variance** `αᵢ²σᵢ²/σ_Y²`
  (somment à 1), à la manière de 3DCS/CETOL : distingue une cote à **recentrer**
  d'une cote à **resserrer**, ce que la seule part de variance masque.
- **`distfit`** : **ajustement de loi** (ISO 22514-2). Familles Normale /
  Lognormale / Rayleigh / Weibull (régression des rangs médians), meilleur ajuste-
  ment par vraisemblance maximale, et **capabilité par percentiles**
  `Cp = (USL−LSL)/(X₀.₉₉₈₆₅−X₀.₀₀₁₃₅)`. *Cross-check* : aller-retour `cdf∘quantile` ;
  la Normale retrouve le `Cp` classique ; récupération des paramètres.
- **`position` (GD&T avancée)** : **condition virtuelle / résultante** (`VC`
  interne `MMC−t`, externe `MMC+t`), **décalage de référence** (glissement depuis
  le MMB) et **position composée** à deux étages (PLTZF/FRTZF). *Cross-check* :
  monotonie et bornes des enveloppes vs la taille réelle du détail.
- **`capability` (IC de capabilité)** : intervalle de confiance **exact** (χ²) sur
  `Cp` et **grand-échantillon** (Bissell) sur `Cpk`. *Cross-check* : couverture
  Monte-Carlo de l'IC de `Cp`.

Câblés dans `scirust-mcp` (`tolerance_gage_rr`, `tolerance_statistical_interval`,
`tolerance_dual_sensitivity`, `tolerance_distribution_fit`, `tolerance_gdt`,
`tolerance_capability_ci`). Fuzz global : **98 858 checks / 0 erreur**.

### Ajouté — transpileur : **MATLAB `range(v)`** — étendue statistique, prouvée contre Octave réel (Phase 2, incrément 49)
`range(v) = max(v) − min(v)` (étendue de l'échantillon), composée depuis les nœuds de
réduction `Max`/`Min` déjà vérifiés — aucun nouveau nœud SIR, std-only. L'inférence de
type reconnaît `range` comme une réduction (argument vecteur).

- `range(v)` : vecteur → scalaire (différence max−min).

Un cas d'oracle (`range(v)` sur un vecteur de 7 éléments). **Oracle 140/140** (200
essais chacun) ; **97 tests unitaires** (1 nouveau).
*Non-vacuité* : remplacer la soustraction par une addition (`max+min` au lieu de
`max−min`) fait diverger le cas `range` — la composition est bien portante.

### Ajouté — transpileur : **MATLAB `fftshift` / `ifftshift`** — centrage du spectre, prouvés contre Octave réel (Phase 2, incrément 48)
Compagnons de la FFT : `fftshift(v)` ramène la composante de fréquence nulle au
centre (échange des deux moitiés = `circshift` de `⌊n/2⌋`) et `ifftshift(v)` l'inverse
(`circshift` de `⌈n/2⌉`) — inverses **exacts** pour les longueurs paires **et
impaires**. Nouveaux nœuds SIR `Fftshift`/`Ifftshift` (vecteur→vecteur) et helpers de
prélude déterministes bâtis sur `np::circshift`, réutilisant la réindexation modulaire
déjà prouvée.

- `fftshift(v)`, `ifftshift(v)` : vecteur réel → vecteur (même longueur).
- S'appliquent naturellement sur un spectre de magnitude réel : `fftshift(abs(fft(x)))`.

Trois cas d'oracle (`fftshift`/`ifftshift` en longueur **impaire** pour distinguer
`⌊·⌋`/`⌈·⌉`, plus `fftshift(abs(fft(x)))` — FFT routée + abs complexe + décalage).
**Oracle 139/139** (200 essais chacun) ; **96 tests unitaires** (1 nouveau).
*Non-vacuité* : faire utiliser `⌊n/2⌋` à `ifftshift` (au lieu de `⌈n/2⌉`) fait diverger
le cas `ifftshift` en longueur impaire tandis que `fftshift` reste vert — la distinction
plancher/plafond est bien portante.

### Ajouté — transpileur : **MATLAB `fft` / `ifft`** routées vers `scirust-signal`, prouvées contre Octave réel (Phase 2, incrément 47)
Premier **routage de traitement du signal** côté MATLAB : `fft(x)` (DFT complexe
d'un vecteur réel), `ifft(c)` (DFT inverse) et `abs(fft(x))` (spectre de magnitude)
émettent le kernel FFT vérifié de `scirust-signal` plutôt que de le redériver —
réutilisant exactement la machinerie complexe (`Fft`/`Ifft`/`ComplexArray`/`ComplexAbs`)
déjà prouvée côté Python.

- `fft(x)` : vecteur réel → vecteur complexe (spectre complet à N points).
- `ifft(c)` : vecteur complexe → vecteur complexe (DFT inverse, `1/N`).
- `abs(z)` sur un spectre complexe → tableau réel des magnitudes (routé
  distinctement de l'`abs` réel élément par élément).

Le harnais de l'oracle sérialise désormais les résultats complexes d'Octave en
`(re, im)` entrelacés pour s'aligner sur la sortie Rust `ComplexArray` (un
`ifft(fft(x))` qu'Octave réduit au réel est complété par des parties imaginaires
nulles). Trois cas d'oracle prouvés contre Octave réel (compilés via cargo avec
`scirust-signal`). **Oracle 136/136** (200 essais chacun) ; **95 tests unitaires**
(1 nouveau).
*Non-vacuité* : router `fft` vers `rfft` (demi-spectre) fait diverger les trois cas
FFT (longueurs 10/16 et 5/8, crash du round-trip) — le routage `fft` est bien portant.

### Ajouté — transpileur : **MATLAB `sec` / `csc` / `cot`** — trigonométrie réciproque, prouvée contre Octave réel (Phase 2, incrément 46)
Achève la trigonométrie : les fonctions réciproques `sec = 1/cos`, `csc = 1/sin`,
`cot = 1/tan`, chacune appliquant la fonction trig de base (scalaire ou élément par
élément) puis prenant l'inverse via le nouveau helper `reciprocal` (`1.0 / e`,
scalaire ou diffusion).

- `sec(x)`, `csc(x)`, `cot(x)` : scalaire ou vecteur (élément par élément).

Quatre cas d'oracle (`sec`/`csc`/`cot` scalaires sur des plages hors pôles, plus
`sec(flip(v))` élément par élément). **Oracle 133/133** (200 essais chacun) ;
**94 tests unitaires** (1 nouveau).
*Non-vacuité* : router `sec` vers `sin` (au lieu de `cos`) fait diverger les deux cas
`sec` tandis que `csc`/`cot` restent verts — le mappage réciproque est bien portant.

### Ajouté — transpileur : **MATLAB `asind` / `acosd` / `atand`** — trigonométrie inverse en degrés, prouvée contre Octave réel (Phase 2, incrément 45)
Complète la famille trigonométrique en degrés : `asind`/`acosd`/`atand` appliquent
l'inverse `asin`/`acos`/`atan` (résultat en radians, scalaire ou élément par élément)
puis convertissent l'angle en **degrés** (`× 180/π`, via le helper `scale_by_const`
partagé avec `rad2deg`).

- `asind(x)`, `acosd(x)`, `atand(x)` : scalaire ou vecteur (élément par élément).
- Domaines : `asind`/`acosd` sur `[-1, 1]` ; `atand` sur tous les réels.

Quatre cas d'oracle (`asind`/`acosd`/`atand` scalaires, plus `atand(flip(v))` élément
par élément). **Oracle 129/129** (200 essais chacun) ; **93 tests unitaires** (1 nouveau).
*Non-vacuité* : remplacer le facteur `180/π` par `90/π` pour la trigo inverse en
degrés fait diverger les quatre cas tandis que `rad2deg` (facteur propre) reste vert —
le facteur de conversion est bien portant.

### Ajouté — transpileur : **MATLAB `sind` / `cosd` / `tand`** — trigonométrie en degrés, prouvée contre Octave réel (Phase 2, incrément 44)
Trigonométrie à argument en **degrés** : `sind`/`cosd`/`tand` convertissent l'argument
en radians (`× π/180`, scalaire ou par diffusion) puis appliquent `sin`/`cos`/`tan`
(scalaire ou élément par élément). La logique de conversion est factorisée dans un
helper `scale_by_const` partagé avec `deg2rad`/`rad2deg`.

- `sind(x)`, `cosd(x)`, `tand(x)` : scalaire ou vecteur (élément par élément).

*Frontière (honnête)* : les cas particuliers MATLAB (zéro exact / `Inf` exact aux
multiples de 90°) ne sont **pas** répliqués — la définition simple `f(x·π/180)` est
utilisée (l'oracle tire des angles aléatoires qui n'atteignent jamais ces points).

Quatre cas d'oracle (`sind`/`cosd`/`tand` scalaires, plus `cosd(flip(v))` élément par
élément). **Oracle 125/125** (200 essais chacun) ; **92 tests unitaires** (1 nouveau).
*Non-vacuité* : remplacer le facteur `π/180` par `π/90` pour la trigo en degrés fait
diverger les quatre cas `sind`/`cosd`/`tand` tandis que `deg2rad` (facteur propre)
reste vert — le facteur de conversion est bien portant.

### Ajouté — transpileur : **MATLAB `circshift(v, k)`** — décalage circulaire, prouvé contre Octave réel (Phase 2, incrément 43)
Réindexation modulaire : `circshift(v, k)` décale circulairement le vecteur de `k`
positions (`result[i] = v[(i−k) mod n]`, longueur inchangée), avec `k` arrondi à
l'entier le plus proche et réduit modulo `n` — donc **tout signe / toute amplitude**
est valide. Nouveau nœud SIR `Circshift { arr, k }` et helper de prélude déterministe
`np::circshift` (arithmétique via `rem_euclid`, sûre pour les décalages négatifs).

- `circshift(v, k)` : `v` vecteur, `k` scalaire entier (littéral ou variable) →
  vecteur décalé (même longueur).

Deux cas d'oracle (`circshift(v, 2)` positif, `circshift(v, -3)` négatif). **Oracle
121/121** (200 essais chacun) ; **91 tests unitaires** (1 nouveau).
*Non-vacuité* : inverser le sens du décalage (`i + k` au lieu de `i − k`) fait
diverger les deux cas `circshift` (positif ET négatif) — le sens de la réindexation
est bien portant.

### Ajouté — transpileur : **MATLAB `gradient(v)`** — gradient numérique à pas unitaire, prouvé contre Octave réel (Phase 2, incrément 42)
Différentiation numérique : `gradient(v)` renvoie un vecteur de **même longueur**
que l'entrée, par différences **centrées** à l'intérieur `(v[i+1] − v[i−1])/2` et
**unilatérales** aux deux extrémités (`v[1] − v[0]`, `v[n−1] − v[n−2]`), à pas
unitaire — exactement `gradient` de MATLAB/Octave. Nouveau nœud SIR `Gradient`
(vecteur→vecteur, comme `diff`) et helper de prélude déterministe `np::gradient`.

- `gradient(v)` : `v` vecteur → vecteur de dérivées numériques (même longueur).
- Cas limites : `gradient([x]) = [0]` ; `gradient([]) = []`.

Un cas d'oracle (`gradient(v)` sur un vecteur de 7 éléments). **Oracle 119/119**
(200 essais chacun) ; **90 tests unitaires** (2 nouveaux — routage + structure de la
formule centrée/unilatérale).
*Non-vacuité* : diviser la différence centrée par `3` au lieu de `2` fait diverger
le cas `gradient` (dès l'indice intérieur) tandis que `diff` reste vert — le facteur
central est bien portant.

### Ajouté — transpileur : **MATLAB `log2` / `asinh` / `acosh` / `atanh`** — log base-2 et trigonométrie hyperbolique inverse, prouvées contre Octave réel (Phase 2, incrément 41)
Achève le vocabulaire élémentaire : le **logarithme base 2** `log2` et les trois
inverses hyperboliques **arc-sinus** `asinh` / **arc-cosinus** `acosh` /
**arc-tangente** `atanh`, chacune une fonction unaire s'appliquant en scalaire **ou**
élément par élément (même mécanisme éprouvé que `sin`/`asin`), mappée 1:1 sur la
méthode `f64` correspondante.

- `log2(x)`, `asinh(x)`, `acosh(x)`, `atanh(x)` : scalaire ou vecteur (élément par
  élément).
- Domaines : `log2` sur `(0, ∞)` ; `asinh` sur tous les réels ; `acosh` sur `[1, ∞)` ;
  `atanh` sur `(-1, 1)`.

Cinq cas d'oracle (`log2`, `asinh`, `acosh`, `atanh` scalaires sur des plages dans
le domaine, plus `atanh(flip(v))` élément par élément). **Oracle 118/118** (200 essais
chacun) ; **89 tests unitaires** (1 nouveau).
*Non-vacuité* : router `atanh` vers la méthode `asinh` fait diverger les deux cas
`atanh` (scalaire ET élément par élément) tandis que `log2`/`asinh`/`acosh` restent
verts — le mappage nom→méthode est bien portant.

### Ajouté — transpileur : **MATLAB `tan` / `asin` / `acos`** — trigonométrie élémentaire et inverse, prouvées contre Octave réel (Phase 2, incrément 40)
Complète le vocabulaire trigonométrique : la **tangente** `tan` et les inverses
**arc-sinus** `asin` / **arc-cosinus** `acos`, chacune une fonction unaire qui
s'applique en scalaire **ou** élément par élément (même mécanisme éprouvé que
`sin`/`cos`/`atan`), mappée 1:1 sur la méthode `f64` correspondante.

- `tan(x)`, `asin(x)`, `acos(x)` : scalaire ou vecteur (élément par élément).
- Domaines : `asin`/`acos` sur `[-1, 1]` ; `tan` fini hors des pôles `±π/2`.

Quatre cas d'oracle (`tan`, `asin`, `acos` scalaires sur des plages dans le
domaine, plus `asin(flip(v))` élément par élément). **Oracle 113/113** (200 essais
chacun) ; **88 tests unitaires** (1 nouveau).
*Non-vacuité* : router `asin` vers la méthode `acos` fait diverger les deux cas
`asin` (scalaire ET élément par élément) tandis que `acos` et `tan` restent verts —
le mappage nom→méthode est bien portant.

### Ajouté — transpileur : **MATLAB `norm(v, p)`** — p-norme vectorielle finie générale, prouvée contre Octave réel (Phase 2, incrément 39)
`norm` accepte désormais une **seconde forme** `norm(v, p)` calculant la p-norme
vectorielle `(Σ |vᵢ|^p)^{1/p}` pour un `p` **fini** quelconque (`norm(v, 1)` = somme
des valeurs absolues, `norm(v, 2)` = norme euclidienne, etc.). Composée depuis des
nœuds déjà vérifiés : `abs` élément par élément, `.^ p` par diffusion, somme à ordre
fixe, puis puissance scalaire `^(1/p)`. La forme à un argument `norm(v)` (2-norme)
est inchangée.

- `norm(v, p)` : `v` vecteur, `p` scalaire fini → p-norme.
- L'inférence de type reconnaît le premier argument de `norm(v, p)` comme un
  vecteur (comme `polyval`), `p` restant scalaire.

*Frontière (honnête)* : les normes `p = Inf`/`-Inf` et la norme matricielle
(spectrale) sont des quantités distinctes et restent **refusées**.

Deux cas d'oracle (`norm(v, 1)` littéral, `norm(v, p)` avec `p` fuzzé dans `[1, 5]`).
**Oracle 109/109** (200 essais chacun) ; **87 tests unitaires** (1 nouveau).
*Non-vacuité* : remplacer l'exposant `1/p` par `2/p` (numérateur `1`→`2`) fait
diverger les deux cas p-norme tandis que la 2-norme `norm(v)` (chemin séparé) reste
verte — l'exposant réciproque est bien portant.

### Ajouté — transpileur : **MATLAB `logspace(a, b, n)`** — constructeur de vecteur logarithmique, prouvé contre Octave réel (Phase 2, incrément 38)
Frère de `linspace` : `logspace(a, b, n)` produit `n` points espacés
logarithmiquement, `10^a .. 10^b`. Défini exactement comme `10 .^ linspace(a, b, n)`,
il hérite donc des **extrémités exactes** de `linspace` (`y(end) = 10^b`) et du
cas `logspace(a, b, 1) = [10^b]`. Nouveau nœud SIR `Logspace { a, b, n }` (mêmes
règles de type/scan que `Linspace`) et helper de prélude déterministe `np::logspace`
bâti sur `np::linspace`.

- `logspace(a, b, n)` : `a`, `b` scalaires, `n` compte entier (littéral ou
  `length(x)`) → vecteur de `n` valeurs `10^a..10^b`.

*Frontière (honnête)* : le cas particulier MATLAB « si `b == pi`, points entre
`10^a` et `pi` » n'est **pas** répliqué — la définition simple `10.^linspace` est
utilisée (l'oracle tire des bornes aléatoires qui ne valent jamais `pi`).

Un cas d'oracle (`logspace(a, b, 6)`). **Oracle 107/107** (200 essais chacun) ;
**86 tests unitaires** (1 nouveau). *Non-vacuité* : remplacer la base `10` par `2`
dans `np::logspace` fait diverger le cas `logspace` seul (base erronée) tandis que
tous les autres restent verts — la base 10 est donc bien portante.

### Ajouté — transpileur : **MATLAB `mod` / `rem` élément par élément sur tableaux** prouvés contre Octave réel (Phase 2, incrément 37)
`mod` et `rem` deviennent **vectoriels**. Le nouveau helper `lower_modrem`
assemble `a − b·floor(a/b)` (resp. `a − b·fix(a/b)`) en déléguant chaque étape
arithmétique à `ew_or_broadcast`, de sorte que la même logique couvre scalaires,
deux vecteurs, ou une diffusion scalaire↔vecteur (avec `floor`/`fix` appliqué
élément par élément quand le quotient est un tableau).

- `mod(a, b)` / `rem(a, b)` acceptent désormais des **vecteurs** (élément par
  élément) et des mélanges scalaire↔vecteur (diffusion), en plus des scalaires.

Deux cas d'oracle (`mod(cumsum(v), 3)`, `rem(cumsum(v), 3)` en diffusion).
**Oracle 106/106** (200 essais chacun) ; **85 tests unitaires** (1 nouveau).
*Non-vacuité* : faire utiliser `floor` à `rem` (au lieu de `fix`) fait diverger
les cas `rem` à dividende négatif (scalaire ET élément par élément) tandis que
`mod` reste vert — la distinction `floor`/`fix` est donc bien portante.

### Ajouté — transpileur : **MATLAB `deg2rad` / `rad2deg` + `sign` élément par élément** prouvés contre Octave réel (Phase 2, incrément 36)
Conversions d'angle et `sign` vectoriel :

- **`deg2rad(x)`** / **`rad2deg(x)`** — conversion degré↔radian (multiplication par
  `π/180` resp. `180/π`), scalaire ou par diffusion sur un vecteur (réutilise
  `ScalarBin`/`ScalarBroadcast`, aucune nouvelle primitive).
- **`sign(v)`** — forme **élément par élément** de `sign` (nouveau nœud
  `ArraySign`, `-1/0/+1` par élément) ; `sign(x)` scalaire reste le nœud `Sign`.
  Dispatch non ambigu sur le type de l'argument (unaire).

Trois cas d'oracle : `deg2rad(x)` (scalaire), `rad2deg(cumsum(v))` (diffusion),
`sign(cumsum(v))` (élément par élément). **Oracle 104/104** (200 essais chacun) ;
**84 tests unitaires** (1 nouveau ; un ancien test mis à jour car `sign` accepte
désormais un vecteur). *Non-vacuité* : inverser les branches `>`/`<` de
`ArraySign` fait diverger le cas (|Δ|=2, ROUGE).

### Ajouté — transpileur : **MATLAB `atan2` / `hypot` / `max` / `min` élément par élément sur tableaux** prouvés contre Octave réel (Phase 2, incrément 35)
Les fonctions math à deux arguments deviennent **vectorielles** : elles se
dispatchent désormais sur le type des opérandes (scalaire∘scalaire → scalaire,
tableau∘tableau → élément par élément, mélange scalaire↔tableau → diffusion),
réutilisant les nœuds `EwBinFn`/`BroadcastFn` (créés pour `.^`). Le nouveau
helper `lower_math2` centralise cette logique pour `atan2`/`hypot`/`max`/`min`.

- `atan2(y, x)`, `hypot(a, b)`, `max(a, b)`, `min(a, b)` acceptent maintenant des
  **vecteurs** (élément par élément) et des mélanges scalaire↔vecteur (diffusion),
  en plus des scalaires ; l'ordre des opérandes est préservé pour `atan2`.

Trois cas d'oracle : `atan2(cumsum(v), flip(v))` (élément par élément),
`hypot(cumsum(v), 2)` (diffusion), `max(…)−min(…)` sur tableaux — les opérandes
étant construits par des builtins renvoyant des tableaux, l'inférence coule
naturellement (sans code mort). **Oracle 101/101** (200 essais chacun) ; **83
tests unitaires** (1 nouveau ; un ancien test mis à jour car `hypot` accepte
désormais un vecteur). *Non-vacuité* : inverser l'ordre des opérandes élément par
élément fait diverger `atan2` (ROUGE) tandis que `max`/`min`, commutatifs,
restent verts.

### Ajouté — transpileur : **MATLAB `expm1` / `log1p`** prouvés contre Octave réel (Phase 2, incrément 34)
Deux fonctions math **précises près de zéro**, mappées sur les méthodes IEEE
exactes `f64::exp_m1` / `f64::ln_1p` et intégrées au motif `MATH_FNS`
(scalaire *et* élément par élément) :

- **`expm1(x)`** = `exp(x) − 1` sans perte de précision pour `x` proche de 0.
- **`log1p(x)`** = `ln(1 + x)` sans perte de précision pour `x` proche de 0.

Deux cas d'oracle (`expm1` scalaire ; `log1p` élément par élément). **Oracle
98/98** (200 essais chacun) ; **82 tests unitaires** (1 nouveau). *Non-vacuité* :
mapper `expm1` sur `exp` au lieu de `exp_m1` décale le résultat de 1 et passe
l'oracle au ROUGE.

### Ajouté — transpileur : **MATLAB `conv` + `polyval`** prouvés contre Octave réel (Phase 2, incrément 33)
Deux classiques du traitement du signal / numérique, câblés via des helpers
déterministes du prélude :

- **`conv(a, b)`** — convolution linéaire complète (longueur `len(a)+len(b)−1`,
  ordre d'accumulation fixe donc bit-reproductible) ; les **deux** opérandes sont
  inférés vecteurs.
- **`polyval(p, x)`** — évaluation de Horner du polynôme de coefficients `p`
  (degré décroissant) au point scalaire `x` ; `p` est inféré vecteur, `x` reste
  scalaire.

Deux cas d'oracle. **Oracle 96/96** (200 essais chacun) ; **81 tests unitaires**
(1 nouveau). *Non-vacuité* : remplacer la récurrence de Horner `acc·x + p[i]` par
`acc + x·p[i]` fait diverger `polyval` et passe l'oracle au ROUGE.

### Ajouté — transpileur : **MATLAB `kron` + `cumtrapz`** prouvés contre Octave réel (Phase 2, incrément 32)
Deux opérations vectorielles supplémentaires, câblées via des helpers
déterministes du prélude :

- **`kron(a, b)`** — produit de Kronecker de deux vecteurs (`out[i·n+j] =
  a[i]·b[j]`, longueur `len(a)·len(b)`) ; les **deux** opérandes sont inférés
  vecteurs.
- **`cumtrapz(v)`** — intégrale trapézoïdale **cumulée** à pas unité (premier
  élément `0`, même longueur).

Deux cas d'oracle. **Oracle 94/94** (200 essais chacun) ; **80 tests unitaires**
(1 nouveau). *Non-vacuité* : échanger l'ordre d'imbrication des boucles de `kron`
réordonne la sortie et passe l'oracle au ROUGE.

### Ajouté — transpileur : **MATLAB `diag` (surchargé) + `trapz`** prouvés contre Octave réel (Phase 2, incrément 31)
Deux ajouts, dont la **surcharge `diag`** de MATLAB, désambiguïsée par le **type
de l'opérande** (jamais deviné) :

- **`diag(A)`** où `A` est une matrice → **extraction** de la diagonale (vecteur ;
  nouveau nœud `DiagExtract`).
- **`diag(v)`** où `v` est un vecteur → **construction** d'une matrice diagonale
  (réutilise le nœud `Diag` existant, déjà prouvé côté Python via `np.diag`).
- **`trapz(v)`** — intégration trapézoïdale à pas unité (`Σ ½·(v[i−1]+v[i])`).

Trois cas d'oracle, exerçant les deux voies de `diag` naturellement :
`diag(A' * A)` (extraction — diagonale de la matrice de Gram) et
`diag(cumsum(v))` (construction), plus `trapz(v)`. **Oracle 92/92** (200 essais
chacun) ; **79 tests unitaires** (2 nouveaux). *Non-vacuité* : retirer le facteur
`½` de `trapz` double le résultat et passe l'oracle au ROUGE.

### Ajouté — transpileur : **MATLAB `trace(A)` + `cross(a, b)`** prouvés contre Octave réel (Phase 2, incrément 30)
Deux opérations classiques, câblées via des helpers déterministes du prélude :

- **`trace(A)`** — somme de la diagonale d'une matrice (scalaire) ; `A` inférée
  matrice depuis l'intrinsèque.
- **`cross(a, b)`** — produit vectoriel de deux vecteurs 3D ; les **deux**
  opérandes sont inférés vecteurs.

Deux cas d'oracle. **Oracle 89/89** (200 essais chacun) ; **77 tests unitaires**
(1 nouveau). *Non-vacuité* : inverser un signe dans une composante de `cross`
fait diverger le cas (|Δ|≈1) et passe l'oracle au ROUGE.

### Ajouté — transpileur : **MATLAB opérateur de transposition `A'` / `A.'`** prouvé contre Octave réel (Phase 2, incrément 29)
Ajoute l'opérateur postfixe de **transposition** — omniprésent en MATLAB — routé
vers le nœud `Transpose` de la SIR (déjà prouvé côté Python via `A.T`).

- **`A'`** (transposée conjuguée) et **`A.'`** (transposée simple) — identiques
  pour les matrices réelles. Le lexer reconnaît `'` (postfixe, jamais une chaîne
  puisque le sous-ensemble n'a pas de chaînes de caractères), le parser
  l'applique en postfixe (liant plus fort que `^`), le lowering exige une matrice.
- `A'` **prouve** que son opérande est une matrice (nouvelle preuve-matrice),
  d'où l'inférence sans indice.

Deux cas d'oracle : `B = A'` (transposition simple) et `C = A' * A` (matrice de
Gram, composant transposition et produit matriciel). **Oracle 87/87** (200 essais
chacun) ; **76 tests unitaires** (1 nouveau). *Non-vacuité* : parser `A'` comme
l'identité (sans transposer) fait perdre à `A` son type matrice et casse les deux
cas (ROUGE).

### Ajouté — transpileur : **MATLAB produit matriciel `A*b` / `A*B`** routé vers `scirust-solvers`, prouvé contre Octave réel (Phase 2, incrément 28)
Complète l'algèbre linéaire MATLAB : l'opérateur `*` (multiplication **matricielle**
en MATLAB, distincte de l'élément-par-élément `.*`) se route vers les kernels
vérifiés `matvec` / `matmul` de `scirust-solvers` quand l'opérande de gauche est
une **matrice** (inférée depuis `det`/`inv`/`eig`/`\`).

- **`A * x`** (matrice × vecteur) → `matvec`.
- **`A * B`** (matrice × matrice) → `matmul` (sortie matrice).
- Sinon, `*` reste la multiplication scalaire ou la diffusion scalaire↔tableau ;
  `A * b` avec deux tableaux reste refusé (pointant vers `.*`), et les formes
  matricielles non gérées (scalaire×matrice, `matrice/`) sont refusées clairement.

Désambiguïsation **sûre** : le routage n'a lieu que si l'opérande matrice est
déjà typé matrice par une autre opération (jamais deviné). Les cas d'oracle
l'exercent naturellement, sans code mort : **résidu `r = A*(A\b) ≈ b`** (matvec)
et **`C = A*inv(A) ≈ I`** (matmul). **Oracle 85/85** (200 essais chacun) ; **75
tests unitaires** (1 nouveau). *Non-vacuité* : transposer la matrice dans
`matvec` fait diverger à la fois le résidu MATLAB et le cas `A @ b` Python
(émetteur partagé) et passe l'oracle au ROUGE.

### Ajouté — transpileur : **MATLAB `linspace(a, b, n)`** — constructeur de vecteur, prouvé contre Octave réel (Phase 2, incrément 27)
Première **construction de tableau** du front-end MATLAB (jusqu'ici les tableaux
venaient des paramètres ou de transformations). `linspace(a, b, n)` produit `n`
points régulièrement espacés de `a` à `b` inclus, avec **extrémités exactes**
(comme MATLAB, qui force `y(end) = b`) et le cas `linspace(a, b, 1) = [b]`.

- `a`, `b` sont des scalaires ; `n` est un **entier** (littéral ou expression
  entière comme `length(x)`), abaissé via le chemin d'index entier existant.
- Câblé via un helper déterministe du prélude.

Un cas d'oracle (longueur fixe 6, `a`/`b` tirés au sort). **Oracle 83/83** (200
essais chacun) ; **74 tests unitaires** (1 nouveau). *Non-vacuité* : utiliser le
pas `(b−a)/n` au lieu de `(b−a)/(n−1)` fait diverger les points intérieurs et
passe l'oracle au ROUGE (200/200, |Δ|≈0,13).

### Ajouté — transpileur : **MATLAB `var` / `std` / `median`** — statistiques de réduction, prouvées contre Octave réel (Phase 2, incrément 26)
Trois réductions statistiques (vecteur → scalaire), alignées exactement sur la
convention d'Octave (vérifiée empiriquement avant câblage) :

- **`var(v)`** — variance **d'échantillon**, normalisée par **`N−1`** (comme la
  valeur par défaut de MATLAB, pas `N`) ; `0` pour `N < 2`.
- **`std(v)`** — écart-type d'échantillon (`√var`).
- **`median(v)`** — médiane (valeur du milieu ; moyenne des deux valeurs
  centrales pour une longueur paire).

Câblées via des helpers déterministes du prélude ; l'argument est inféré vecteur
(ajoutées à `is_reduction`). Trois cas d'oracle (var+std, médiane paire, médiane
impaire). **Oracle 82/82** (200 essais chacun) ; **73 tests unitaires** (1
nouveau). *Non-vacuité* : normaliser `var` par `N` au lieu de `N−1` fait diverger
le cas (200/200, |Δ|≈0,89) et passe l'oracle au ROUGE.

### Ajouté — transpileur : **MATLAB `cumprod` / `cummax` / `cummin` / `flip`** — fonctions vecteur → vecteur, prouvées contre Octave réel (Phase 2, incrément 25)
Quatre fonctions natives supplémentaires (un vecteur en entrée, un vecteur en
sortie), sur le même modèle que `cumsum`/`diff`/`sort` (helpers déterministes du
prélude, argument inféré vecteur) :

- **`cumprod(v)`** — produit cumulé en ordre gauche→droite fixe.
- **`cummax(v)`** / **`cummin(v)`** — maximum / minimum courant.
- **`flip(v)`** — vecteur inversé.

Quatre cas d'oracle. **Oracle 79/79** (200 essais chacun) ; **72 tests
unitaires** (le test des builtins vectoriels couvre désormais les sept fonctions).
*Non-vacuité* : remplacer `>` par `<` dans `cummax` fait diverger le cas
(200/200, |Δ|=∞) et passe l'oracle au ROUGE.

### Ajouté — transpileur : **MATLAB `cumsum` / `diff` / `sort`** — fonctions vecteur → vecteur, prouvées contre Octave réel (Phase 2, incrément 24)
Trois fonctions natives non ambiguës (un vecteur en entrée, un vecteur en
sortie), câblées via de nouveaux helpers déterministes du prélude :

- **`cumsum(v)`** — somme cumulée en ordre **gauche→droite fixe** (donc
  bit-reproductible), même longueur.
- **`diff(v)`** — différences consécutives `v[i+1] − v[i]` (longueur `n−1`).
- **`sort(v)`** — tri **croissant** (`sort` de MATLAB).

L'argument est inféré comme un vecteur à partir de l'intrinsèque. Trois cas
d'oracle. **Oracle 76/76** (200 essais chacun) ; **72 tests unitaires** (1
nouveau ; le helper `sig_of` des tests cible désormais le `pub fn` de premier
niveau pour ne pas confondre une fonction utilisateur avec un homonyme du
prélude, p. ex. `np::cumsum`). *Non-vacuité* : inverser l'ordre de soustraction
de `diff` (`v[i−1] − v[i]`) nie chaque élément et passe l'oracle au ROUGE
(200/200, |Δ|≈2,4).

### Ajouté — transpileur : **MATLAB puissance élément par élément `.^` sur tableaux** prouvée contre Octave réel (Phase 2, incrément 23)
Première opération **vectorielle** à deux arguments — l'idiome au cœur de MATLAB.
Ajoute une infrastructure réutilisable (`SirExpr::EwBinFn` pour tableau∘tableau,
`SirExpr::BroadcastFn` pour scalaire↔tableau, variante `MathFn2::Powf`) qui
resservira aux formes élément-par-élément de `max`/`min`/`atan2`/`hypot`.

- **`v .^ w`** (deux tableaux) → `np::ew2(v, w, |x, y| x.powf(y))`.
- **`v .^ 2`** (base tableau, exposant scalaire) → `np::map1(v, |x| x.powf(2))`.
- **`2 .^ v`** (base scalaire, exposant tableau) → `np::map1(v, |x| (2).powf(x))`
  — l'ordre des opérandes est préservé (`.^` n'est pas commutatif).
- **`^` sur un tableau** (puissance matricielle `mpower`) reste **refusé** avec un
  diagnostic pointant vers `.^`.

`f64::powf` reproduit `.^` d'Octave (vérifié, y compris exposant entier). Trois
cas d'oracle. **Oracle 73/73** (200 essais chacun) ; **71 tests unitaires** (2
nouveaux). *Non-vacuité* : inverser l'ordre de diffusion de `2 .^ v` (calculer
`v .^ 2`) fait diverger le cas (200/200, |Δ|≈0,83) et passe l'oracle au ROUGE.

### Ajouté — transpileur : **MATLAB `max(a,b)` / `min(a,b)` (2-arg) + `power(a,b)`** prouvés contre Octave réel (Phase 2, incrément 22)
Réutilise le nœud math binaire (`MathFn2`) pour les formes à deux arguments de
`max`/`min`, distinguées de la **réduction** à un argument par le **nombre
d'arguments** :

- **`max(a, b)`** / **`min(a, b)`** (deux scalaires) → `f64::max` / `f64::min`.
  La forme à un argument reste la réduction sur un vecteur ; l'inférence de type
  ne marque plus les opérandes de la forme à deux arguments comme des tableaux
  (garde `args.len() == 1` sur la preuve de réduction).
- **`power(a, b)`** → forme fonctionnelle de `a ^ b` (partage l'abaissement de
  `^` : un exposant entier se replie sur `powi`).

Opérandes scalaires. Deux cas d'oracle. **Oracle 70/70** (200 essais chacun) ;
**69 tests unitaires** (2 nouveaux). *Non-vacuité* : échanger `max`/`min` dans la
forme à deux arguments inverse le signe de l'étendue `max−min` et passe l'oracle
au ROUGE (200/200, |Δ|≈6,2).

### Ajouté — transpileur : **MATLAB `atan2` / `hypot`** — fonctions math à deux arguments, prouvées contre Octave réel (Phase 2, incrément 21)
Ajoute à la SIR un **nœud math scalaire binaire** réutilisable
(`SirExpr::ScalarBinFn` + `MathFn2`), émis en `(l).méthode(r)`, et câble les deux
premières fonctions à deux arguments côté MATLAB :

- **`atan2(y, x)`** — arctangente à quatre quadrants (`f64::atan2`), l'ordre des
  arguments est significatif.
- **`hypot(a, b)`** — longueur euclidienne `√(a²+b²)` sans dépassement
  (`f64::hypot`).

Opérandes scalaires dans ce sous-ensemble. Vérifiées empiriquement contre Octave
(cas des quatre quadrants inclus) avant câblage. Deux cas d'oracle. **Oracle
68/68** (200 essais chacun) ; **67 tests unitaires** (1 nouveau). *Non-vacuité* :
inverser l'ordre des arguments d'`atan2` fait diverger le cas (200/200,
|Δ|≈0,22) et passe l'oracle au ROUGE — tandis que `hypot`, symétrique, reste
vert (le test prouve donc la position, pas seulement la présence).

### Ajouté — transpileur : **MATLAB `round` / `fix` / `mod` / `rem` / `sign`** — arrondi et arithmétique modulaire, prouvés contre Octave réel (Phase 2, incrément 20)
Élargissement du vocabulaire scalaire MATLAB, chaque fonction alignée exactement
sur la sémantique d'Octave (vérifié empiriquement avant câblage) :

- **`round(x)`** — arrondi au plus proche, **départager en s'éloignant de zéro**
  (`f64::round`). C'est *volontairement* différent de l'arrondi bancaire de NumPy
  (`round half to even`), donc `round` n'est câblé que sur le chemin MATLAB.
- **`fix(x)`** — troncature **vers zéro** (`f64::trunc`).
- **`mod(a, b)`** — modulo, résultat du **signe du diviseur** ;
  abaissé en `a - b·floor(a/b)`.
- **`rem(a, b)`** — reste, résultat du **signe du dividende** ;
  abaissé en `a - b·fix(a/b)`.
- **`sign(x)`** — `-1 / 0 / +1` avec **`sign(0) = 0`** (sémantique MATLAB,
  distincte de `f64::signum`) ; nouveau nœud `SirExpr::Sign` émis en `if/else`
  lié (l'argument n'est évalué qu'une fois).

`round`/`fix` fonctionnent aussi élément par élément (comme les autres
intrinsèques math) ; `mod`/`rem`/`sign` sont scalaires dans ce sous-ensemble.
Quatre cas d'oracle. **Oracle 66/66** (200 essais chacun) ; **66 tests
unitaires** (3 nouveaux). *Non-vacuité* : inverser les branches `+1`/`-1` de
`sign` fait diverger le cas (200/200, |Δ|=2) et passe l'oracle au ROUGE.

### Ajouté — transpileur : **MATLAB `norm` / `dot` / `eig`** — normes, produit scalaire et valeurs propres, prouvés contre Octave réel (Phase 2, incrément 19)
Poursuite de la couverture MATLAB large et sûre, en réutilisant des kernels déjà
vérifiés (aucune nouvelle primitive à prouver depuis zéro). Trois intrinsèques,
tous non ambigus :

- **`norm(v)`** — norme euclidienne (2-norme) d'un **vecteur**, abaissée en
  `sqrt(sum(v .* v))` à partir des nœuds SIR existants (restreinte à un vecteur ;
  la `norm` d'une matrice est la norme spectrale, une quantité différente —
  refusée avec un diagnostic).
- **`dot(a, b)`** — produit scalaire, routé vers la réduction `np::dot` à **ordre
  fixe** (bit-reproductible). L'inférence de type marque désormais les **deux**
  opérandes comme vecteurs.
- **`eig(A)`** — valeurs propres (ordre croissant) d'une matrice **symétrique**,
  routées vers `scirust_solvers::eigen_symmetric`. `eig` d'Octave renvoie des
  valeurs propres réelles croissantes pour une entrée symétrique, donc cette
  route est prouvée sur entrées symétriques (via `SymMatrix` dans l'oracle) ;
  `A` est inférée matrice à partir de l'intrinsèque.

Trois cas d'oracle : `norm(v)`, `dot(a,b)`, `eig(A)`. **Oracle 62/62** (200
essais chacun) ; **63 tests unitaires** (3 nouveaux). *Non-vacuité* : remplacer
`v .* v` par `v + v` dans `norm` fait diverger le cas (99/200, |Δ|≈1,4) et passe
l'oracle au ROUGE.

### Ajouté — transpileur : **MATLAB algèbre linéaire — `det` / `inv` / `\` (résolution)** routés vers `scirust-solvers`, prouvés contre Octave réel (Phase 2, incrément 18)
La couverture MATLAB gagne l'algèbre linéaire du cœur numérique, réutilisant les
kernels vérifiés de `scirust-solvers` déjà branchés côté Python. (1) **`det(A)`**
et **`inv(A)`** deviennent des intrinsèques MATLAB (déterminant scalaire,
inverse matrice 2-D). (2) **`A \ b`** — l'opérateur de division à gauche, la
manière idiomatique de résoudre `Ax = b` en MATLAB — se lexe (`\`), se parse
(`MBinOp::LDiv`) et se lowerise en `LinSolve` (factorisation LU). (3) **Inférence
de paramètre matrice** : les arguments de `det`/`inv` et la gauche de `\`
prouvent qu'un paramètre est une **matrice** (`infer_param_ty` teste désormais
la preuve-matrice avant la preuve-tableau) ; `\` exige une matrice à gauche et un
vecteur à droite (diagnostic clair sinon). L'oracle Octave sérialise désormais
les sorties matricielles en **ordre ligne-major** (`r.'`) et les vecteurs en
**colonne** pour aligner la sémantique `A \ b`. Trois cas d'oracle : `det(A)`,
`inv(A)` (sortie matrice), `A \ b` (résolution). **Oracle 59/59** (200 essais
chacun) ; **60 tests unitaires** (3 nouveaux). *Non-vacuité* : sérialiser `inv`
en colonne-major fait diverger le cas non symétrique et passe l'oracle au ROUGE.

### Ajouté — transpileur : **MATLAB multi-sorties `[a,b] = f(…)` + vocabulaire MATLAB élargi** prouvés contre Octave réel (Phase 2, incrément 17)
Vers une couverture MATLAB large et sûre. (1) **Fonctions multi-sorties** :
`function [o1, o2, …] = f(x) … end` transpile vers un `pub fn … -> (T0, T1, …)`
(retour tuple), réutilisant la machinerie `RetTy::Tuple`/`ReturnTuple` du côté
Python (incrément 16). Lexer étendu (`[`/`]` avec suivi de profondeur), parser
(liste de sorties entre crochets), lowering (`ReturnTuple` des variables de
sortie). (2) **Intrinsèques MATLAB alignés sur Python** : nouvelles fonctions
math `log`/`log10`/`floor`/`ceil`/`sinh`/`cosh`/`atan` et réductions
`prod`/`mean`/`max`/`min` (les réductions comptent aussi comme preuve de tableau
pour l'inférence des paramètres). L'oracle Octave capture désormais plusieurs
sorties (`[o1,…] = f(args)`). Quatre cas d'oracle : `[s,d]=sumdiff`,
`[n,ss]=normstats`, `[lo,mu,hi]=stats3` (min/mean/max), `mathx` (log/floor/atan).
**Oracle 56/56** (200 essais chacun) ; **57 tests unitaires** (4 nouveaux).
Non-vacuité re-vérifiée : inverser l'ordre des sorties fait diverger les cas
multi-sorties et passe l'oracle au ROUGE.

### Ajouté — transpileur : **retours de tuple généraux** (`return a, b`) prouvés contre NumPy réel (Phase 2, incrément 16)
Complète l'histoire des tuples côté **production** : une fonction peut renvoyer
plusieurs valeurs (`def minmax(x): return np.min(x), np.max(x)` → `-> (f64, f64)`).
Nouveautés : `RetTy` (retour simple ou tuple, hors du treillis `Copy` `Ty`),
`SirStmt::ReturnTuple`, parsing Python de `return e0, e1, …`, et sérialisation
oracle des retours-tuple (impression champ par champ `r.0`, `r.1`, …). Les
éléments de tuple sont restreints aux **scalaires** dans ce sous-ensemble ; un
retour mixte simple/tuple est refusé, et une fonction renvoyant un tuple ne peut
pas être appelée comme valeur (diagnostic clair). Trois cas d'oracle : `addsub`
(a+b, a-b), `minmax` (min, max), `stats3` (min, mean, max). **Oracle 52/52**
(200 essais chacun) ; **53 tests unitaires**. Non-vacuité re-vérifiée : inverser
l'ordre des éléments du tuple à l'émission fait diverger les trois cas (|Δ|≈5)
et passe l'oracle au ROUGE.

### Ajouté — transpileur : **vocabulaire numérique élargi** (log/floor/ceil/sinh/…, prod/mean/max/min) prouvé contre NumPy réel (Phase 2, incrément 15)
Sept nouvelles fonctions math élémentaires (scalaire ou tableau) — `np.log`
(→ `ln`), `np.log10`, `np.floor`, `np.ceil`, `np.sinh`, `np.cosh`, `np.arctan`
(→ `atan`) — et quatre réductions — `np.prod`, `np.mean` (désucrée en
`sum(a)/len(a)`, sans nouveau nœud), `np.max`, `np.min`. Nouveaux `MathFn`
(Ln/Log10/Floor/Ceil/Sinh/Cosh/Atan) et `SirExpr::Prod`/`Max`/`Min` avec
helpers de prélude à ordre pinné (`prod` ascendant reproductible). Les
réductions comptent aussi comme preuve de tableau pour l'inférence de type des
paramètres. Cinq cas d'oracle (log+log10, floor+ceil, sinh/cosh/arctan,
max−min+mean, prod). **Oracle 49/49** (200 essais chacun) ; **48 tests
unitaires**. Non-vacuité re-vérifiée : mapper `np.log` sur `log10` fait diverger
le cas (|Δ|≈0,9) et passe l'oracle au ROUGE.

### Ajouté — transpileur : **`np.linalg.qr`** (déstructuration `Q, R = …`) prouvé contre NumPy réel (Phase 2, incrément 14)
Deuxième noyau multi-sorties, sur le même point d'extension `TupleExpr` que la
SVD. `Q, R = np.linalg.qr(A)` transpile vers la QR de Householder vérifiée
`scirust_solvers::linalg::qr_decompose` (`Q` orthogonale via `.q()`, `R`
triangulaire sup via `.r()`). Sur une matrice **carrée**, `q()` (m×m) coïncide
avec la Q réduite de numpy, donc les formes collent. Comme les signes de Q/R
dépendent de la jauge, la preuve porte sur la **reconstruction invariante**
`Q @ R ≈ A`. **Oracle 44/44** (200 essais chacun) ; **45 tests unitaires**.
Non-vacuité re-vérifiée : intervertir `q()`/`r()` (émettre `(R, Q)`) fait
diverger la reconstruction (|Δ|≈0,48) et passe l'oracle au ROUGE.

### Ajouté — transpileur : **Python élargi** (appels de fonctions utilisateur + listes) prouvés contre NumPy réel (Phase 2, incrément 13)
Le transpileur compose désormais **plusieurs fonctions** : une `def` transpilée
peut en appeler une autre définie **plus tôt** dans le module (define-before-use),
et les **listes littérales** `[a, b, c]` deviennent des `Vec<f64>`. Nouveautés :
carte de signatures `FuncSig`/`Sigs` tissée à travers le lowering (chaque
fonction voit les signatures des précédentes), `SirExpr::UserCall`
(appel direct type-checké) et `SirExpr::ArrayLit`, plus le parsing Python des
listes. **Inférence de type sans annotation à travers les appels** : un paramètre
passé à une fonction utilisateur hérite du type du paramètre correspondant du
callee — d'où `sumdbl(x)` où `x` est inféré `&[f64]` uniquement parce que `dbl`
attend un tableau. Les paramètres des fonctions appelées sont restreints à
scalaire/tableau (coercition d'argument non ambiguë à l'émission). Quatre cas
d'oracle : composition scalaire (`sumsq`→`sq`), composition tableau sans
annotation (`sumdbl`→`dbl`), chaîne à 3 niveaux (`chain`→`twice`→`inc`), et liste
littérale comme vecteur de poids (`wavg` via `np.dot`). **Oracle 43/43**
(200 essais chacun) ; **43 tests unitaires** (7 nouveaux). Non-vacuité
re-vérifiée : injecter un décalage `+ 1.0` dans l'émission d'appel fait diverger
les trois cas de composition (ROUGE) tandis que la liste littérale reste verte.

### Ajouté — transpileur : **tuples + `np.linalg.svd`** prouvés contre NumPy réel (Phase 2, incrément 12)
Premier **noyau multi-sorties** et première **déstructuration de tuple**.
`U, S, Vh = np.linalg.svd(A)` transpile vers la SVD fine vérifiée
`scirust_solvers::linalg::svd`, avec `Vh = Vᵀ` pour coller au troisième retour
de `numpy.linalg.svd`. Nouveautés : `TupleExpr` (enum des appels producteurs de
tuple, hors du treillis `Copy` `Ty`), `SirStmt::LetTuple` (bind déstructurant
`let (n0, n1, …): (T0, T1, …) = …`), `SirExpr::Diag` (`np.diag(v)` → matrice
diagonale carrée), et le parsing Python de la cible-tuple `a, b, c = …`. Sur une
matrice **carrée**, la SVD fine == SVD complète, donc les formes collent à
numpy. Deux cas d'oracle prouvent la route de deux manières complémentaires :
(a) **valeurs singulières** `S` (uniques, décroissantes) comparées directement à
numpy ; (b) **reconstruction** `U @ diag(S) @ Vh ≈ A` (invariante de jauge, donc
robuste aux ambiguïtés de signe de U/V — et exerçant réellement U et V).
**Oracle 39/39** (200 essais chacun) ; **36 tests unitaires** (5 nouveaux).
Non-vacuité re-vérifiée : retirer la transposée de `Vh` fait diverger la
reconstruction (|Δ|≈1,3) et passe l'oracle au ROUGE, tandis que le cas
« valeurs singulières » reste vert — preuve que la reconstruction exerce U et V.

### Ajouté — transpileur : **front-end MATLAB/Octave** prouvé contre Octave réel (Phase 2, incrément 11)
Deuxième langue source, sur la **même** SIR + émetteur que Python — donc même
déterminisme et mêmes noyaux `scirust-*` vérifiés. Front-end dédié
(`src/front_matlab/{lexer,parser,ast,mod}.rs`) + lowering `src/lower_matlab.rs`,
et API publique `transpile_matlab` / `transpile_matlab_to_sir`. Sémantique MATLAB
gérée : indexation **1-based** (`a(i)` → `a[i-1]`), plages `for` **inclusives**
(`1:n` → `1..n+1`), opérateurs **élémentaires** `.*`/`./`/`.^` (opérandes
inférés tableaux) vs scalaires `* / ^`, comparaisons dont `~=`, `if`/`elseif`/
`else` + `while`, et **retour par variable de sortie**. Nouveau
`SirStmt::Declare` (déclaration hoistée sans initialiseur, validée par
l'analyse d'assignation-définie de Rust) pour les locales/sorties d'abord
assignées en branche. L'oracle exécute les cas MATLAB contre **Octave réel**
(9 cas × 200 essais) en plus des cas Python contre NumPy. Non-vacuité MATLAB :
casser l'indexation 1-based (`i-1` → `i-2`) fait planter `mysum` et passe
l'oracle au ROUGE.

### Ajouté — transpileur : matrice-matrice `A @ B` + transpose `A.T` (Phase 1, incrément 10)
Complète l'algèbre linéaire dense. `A.T` (transpose) et `A @ B` (produit
matrice-matrice) → `scirust_solvers::Matrix::transpose`/`matmul`. Nouveautés :
`SirExpr::Matmul`/`Transpose` ; helper d'émission `as_matrix` qui accepte
indifféremment une matrice-paramètre plate ou une valeur `Matrix` produite,
d'où le **chaînage** (`A @ A.T`, et les opérations matricielles acceptant un
`MatrixVal`). Cas d'oracle : `A.T` et `A @ A.T` (matrice de Gram) vs numpy.
**Oracle 28/28** (200 essais chacun) ; 24 tests unitaires.

### Ajouté — transpileur : `np.linalg.inv` (retour matrice 2-D) (Phase 1, incrément 9)
Premier **retour de matrice 2-D** : `np.linalg.inv(A)` transpile vers
`scirust_solvers::Matrix::inverse` et renvoie une valeur `scirust_solvers::Matrix`
(qui porte sa forme). Nouveau `Ty::MatrixVal`, `SirExpr::Inv` ; l'oracle
sérialise un retour matriciel en aplatissant row-major (via `rows()`/`row()`),
comparé à `numpy.linalg.inv`. **Oracle 26/26** (200 essais chacun) ; 23 tests
unitaires.

### Ajouté — six extensions du tolérancement inertiel (`scirust-tolerance`)
Six nouveaux modules qui étendent la crate au-delà de la chaîne linéaire
indépendante et de la seule cotation de position, chacun vérifié par
cross-check de fuzzing contre une **référence indépendante** :

- **`montecarlo`** : simulation Monte-Carlo de tolérances. Lois composant
  (normale, uniforme, triangulaire, moments exacts), RNG déterministe graine
  (xorshift64\* + Box–Muller), et `simulate` qui pousse `n` tirages à travers
  une fonction de transfert quelconque `Y = f(X₁…Xₙ)` → moyenne, dispersion,
  **inertie à la cible**, ppm, rendement, percentiles `0,135/50/99,865 %`.
  *Cross-check* : une combinaison linéaire de normales reproduit `Σαμ`, `Σα²σ²`.
- **`correlated`** : chaînes **corrélées** et **non-linéaires**. Forme
  quadratique `I_Y² = (α∘I)ᵀ R (α∘I)` (se réduit au `√(Σα²I²)` de `chain` pour
  `R=𝕀`), linéarisation par différences finies (`gradient`), variance
  `gᵀΣg` (`correlated_variance`), et correction du second ordre de la moyenne
  `f(μ)+½ Σ Hᵢᵢσᵢ²`. *Cross-check* : gradient vs dérivée analytique ; moyenne
  du second ordre vs le moment exact d'une quadratique ; vs Monte-Carlo.
- **`geometry`** : le reste des caractéristiques **ISO 1101** — rectitude,
  planéité, circularité, cylindricité (forme, par moindres carrés), parallélisme
  / perpendicularité / angularité (orientation, zone `L·sin Δθ`), profil et
  battement — chacune avec sa lecture **inertielle** (RMS des écarts).
  *Cross-check* : orthogonalité des résidus du plan des moindres carrés ; forme
  parfaite → 0 ; orientation vs produits vectoriel/scalaire.
- **`sensitivity`** : analyse de contribution — part de chaque composant dans
  l'inertie d'assemblage `cᵢ = αᵢ²Iᵢ²/I_Y²` (et version corrélée), triée. Pointe
  les cotes à resserrer. *Cross-check* : les parts somment à 1 et égalent le
  recalcul direct.
- **`process`** : allocation à **procédés discrets** — sac-à-dos à choix
  multiple résolu **exactement** par frontière de Pareto des états
  `(poids, coût)` non dominés : choisir un procédé `{inertie, coût}` par
  composant minimisant le coût sous budget d'inertie (statistique ou pire cas).
  *Cross-check* : vs énumération exhaustive.
- **`drift`** : capabilité court terme vs long terme — variance de dérive
  uniforme `σ_lt = √(σ_st² + d²/3)`, décalage Motorola `1,5σ`
  (`Cpk↔Ppk`), et ppm long terme. *Cross-check* : `σ_lt` vs un Monte-Carlo d'une
  moyenne qui dérive plus le bruit intra-lot.
- **`scirust-mcp`** : six nouveaux outils — `tolerance_monte_carlo`,
  `tolerance_geometry`, `tolerance_sensitivity`, `tolerance_discrete_allocate`,
  `tolerance_drift`, `tolerance_correlated`.

Le harnais `fuzz_crosscheck` couvre désormais **14 modules** — **76 534
vérifications, 0 erreur** à 1500 instances.

### Ajouté — tolérancement non-normal + position GD&T (`scirust-tolerance`)
Deux modules qui étendent le tolérancement inertiel au-delà de l'hypothèse
normale et à la cotation de position :

- **`nonnormal`** (nouveau module) : tolérancement statistique **non-normal**
  à partir des quatre premiers moments (moyenne, écart-type, asymétrie `S`,
  aplatissement d'excès `K`). L'inertie `I = √(δ²+σ²)` étant *sans hypothèse
  de distribution* (c'est la RMS d'écart à la cible), c'est la **conformité**
  qui dépend de la forme. `cornish_fisher_quantile` donne le `p`-quantile par
  l'expansion de Cornish–Fisher `x_p = μ + σ·w(Φ⁻¹(p))` ; `nonnormal_ppm`
  inverse l'expansion à chaque limite pour la non-conformité en ppm ;
  `clements_capability` fournit le `Cp`/`Cpk` percentile de Clements (1989)
  sur données asymétriques. Les trois **se réduisent exactement** aux résultats
  normaux classiques quand `S = K = 0`. L'inversion `w(z)=t` d'un cubique n'est
  bien posée que sur la **branche monotone** autour de `z=0` : le solveur
  localise cette branche (marche + bornage), rabat une cible extérieure sur son
  extrémité (une limite loin en queue ⇒ contribution ≈ 0, pas de racine
  parasite) et bissecte à l'intérieur — valide pour une non-normalité modérée
  et des limites dans le corps de la distribution (régime usuel de capabilité).
- **`position`** (nouveau module) : cotation de **position GD&T / ISO GPS** et
  sa forme inertielle. `true_position = 2·√(Δx²+Δy²)` (écart diamétral),
  `mmc_bonus`/`total_position_tolerance` (bonus au maximum de matière selon
  `FeatureType::Internal`/`External`), `coord_to_position`/`position_to_coord`
  (conversion zone `±` ↔ zone diamétrale `Ø`), et l'**inertie de position**
  `√(Iₓ²+I_y²)` — puisque `E[Δx²+Δy²] = Iₓ²+I_y²`, exactement la
  `vector_inertia` des deux axes, ce qui rattache la position au cadre inertiel.
- **`scirust-mcp`** : nouveaux outils `tolerance_nonnormal_capability`
  (ppm non-normal + capabilité de Clements) et `tolerance_position`
  (position vraie + bonus MMC + inertie de position).
- **Cross-check par fuzzing** (`fuzz_crosscheck`) étendu à ces deux modules :
  réduction exacte au normal, cohérence aller-retour de l'inversion
  Cornish–Fisher sur son domaine valide, monotonie de la queue vs asymétrie,
  et identités radiales de la position — **0 erreur** sur 10 000 instances. Le
  fuzzing a révélé et corrigé une racine parasite de l'inversion pour une cible
  sous le minimum de la branche monotone (queue basse gonflée) ; d'où le
  solveur robuste marche-bornage-bissection.
- **Visualisation** (`scirust-tolerance/viz/inertia_cone.html`) : page HTML
  autonome et interactive du **cône d'inertie** — la carte d'acceptation
  `(δ, σ)` (demi-disque inertiel vs triangle Cpk), le cône 3D `z = √(δ²+σ²)`
  coupé par le plan `I_max`, la distribution du lot, et la lecture en direct de
  `I`/`Cpi`/`Cpm`/`Cp`/`Cpk`/ppm en glissant le point de lot ou les curseurs.
  Sans dépendance réseau, thème clair/sombre.

### Ajouté — transpileur : couverture de test exhaustive + script global
Objectif « tester **toutes** les fonctions codées » : l'oracle différentiel
couvre désormais **chaque** intrinsèque et opérateur supporté. Nouveaux cas —
`np.sin`/`np.cos`/`np.abs` (scalaire), `np.exp` (scalaire **et** élémentaire
sur tableau), l'opérateur `**`, et `np.ones` + `len` (tableau en sortie) —
portant l'oracle à **19/19** (200 essais chacun vs NumPy réel). Ajout du script
`scripts/test_transpiler.sh` qui lance en un point la suite complète (17 tests
unitaires + oracle) avec rapport clair et code de sortie non nul si une seule
fonction transpilée diverge de NumPy.

### Ajouté — transpileur : routage `np.linalg.det` (Phase 1, incrément 4)
Deuxième noyau routé vers `scirust-solvers` : `np.linalg.det(A)` transpile vers
`scirust_solvers::Matrix::from_row_major(...).determinant()` (déterminant par LU
prouvé). Réutilise l'infrastructure `Ty::Matrix` + oracle bi-mode (compilation
cargo). `SirExpr::Det` ajouté ; inférence de paramètre matrice étendue à l'arg 0
de `np.linalg.det`. Nouveau cas d'oracle sur des matrices 4×4 comparé à
`numpy.linalg.det`. **Oracle 14/14** (200 essais chacun) ; 17 tests unitaires.

### Ajouté — transpileur : routage vers les noyaux vérifiés (Phase 1, incrément 3)
Premier **routage vers un noyau `scirust-*` vérifié** : `np.linalg.solve(A, b)`
est transpilé vers `scirust_solvers::linalg::solve` (résolution LU prouvée) au
lieu d'être re-dérivé en Rust std. C'est le différenciateur central de la
conception — on ne ré-implémente pas la numérique, on route vers des noyaux
oracle-validés.
- SIR : `Ty::Matrix` (matrice 2-D plate row-major), `SirExpr::LinSolve`,
  fonction `required_crates(&SirModule)` qui déclare les crates `scirust-*`
  nécessaires ; inférence des paramètres matrice (arg 0 de `np.linalg.solve`).
- **Oracle bi-mode** : les cas std-only compilent toujours avec `rustc` seul ;
  les cas routés compilent en projet cargo autonome dépendant (par chemin) de
  `scirust-solvers`, avec un target partagé (l'arbre de deps se compile une
  fois). Nouveau cas : `np.linalg.solve` sur des systèmes 5×5 à diagonale
  dominante, comparé à `numpy.linalg.solve`. **Oracle 13/13** (200 essais
  chacun). 16 tests unitaires.

### Ajouté — transpileur : boucles `while` (Phase 1, incrément 2)
Le sous-ensemble Python du transpileur entrant supporte désormais les **boucles
`while`** (condition = comparaison scalaire), débloquant les algorithmes
itératifs (Newton, point fixe, bisection). Prouvé par le même oracle
différentiel contre NumPy réel avec deux cas de **méthode de Newton** — à
nombre d'itérations fixe et à condition de convergence (le nombre d'itérations
dépend des données mais reste identique côté Rust et NumPy, les opérations
flottantes étant bit-identiques). **Oracle 12/12** (200 essais chacun) ; 14
tests unitaires. `SirStmt::While` ajouté ; émetteur, parseur et inférence de
paramètres étendus.

### Ajouté — transpileur : contrôle de flux `if`/`elif`/`else` (Phase 1, incrément 1)
Extension du sous-ensemble Python avec le **contrôle de flux scalaire**, prouvée
par le même oracle différentiel contre NumPy réel :
- front-end : instructions `if`/`elif`/`else` (`elif` désucré en `if` imbriqué
  dans la branche `else`) ; opérateurs de comparaison `< <= > >= == !=` comme
  conditions booléennes (une comparaison n'est valide qu'en condition, jamais
  comme valeur — sinon refusée).
- SIR : `Ty::Bool`, `SirStmt::If`, `SirExpr::Cmp` ; inférence de paramètres et
  émetteur étendus ; les branches suivent la même règle « initialiser avant »
  que les boucles.
- oracle : 3 nouveaux cas (relu, clamp, sign) → **10/10 cas conformes**
  (200 essais chacun) ; 13 tests unitaires.

### Ajouté — synthèse de tolérances à coût minimal (`scirust-tolerance`)
Le « calcul optimal » du tolérancement inertiel : nouveau module `optimize`
qui minimise le coût total de fabrication `Σᵢ bᵢ·Iᵢ^(−rᵢ)` (modèle
coût-tolérance en puissance inverse, Chase & Greenwood) sous **plusieurs
exigences fonctionnelles simultanées** `√(Σᵢ αₖᵢ² Iᵢ²) ≤ I_max,ₖ`. En
variables `vᵢ=Iᵢ²` le coût est convexe et les contraintes linéaires, donc
programme convexe à dualité forte : le lagrangien se sépare par composant
(`Iᵢ = ((rᵢ/2)bᵢ/sᵢ)^{1/(rᵢ+2)}`, `sᵢ=Σₖ μₖ αₖᵢ²`) et le dual est
maximisé par une mise à jour multiplicative invariante d'échelle
`μₖ ← μₖ·(atteintₖ²/I_max,ₖ²)^γ` dont le point fixe est exactement le point
KKT (contrainte active ⇒ atteint=budget, contrainte lâche ⇒ μₖ→0). Pour une
exigence unique, reproduit exactement la forme close `Allocation::CostOptimal`.
Fournit `Component`, `Requirement`, `optimize`/`optimize_with`,
`OptimizeResult` (inerties, coût total, multiplicateurs/prix duaux, exigences
actives), et la **frontière de Pareto coût-qualité** `cost_quality_frontier`.
Vérifié par : égalité à la forme close mono-exigence, satisfaction des
conditions KKT à deux exigences, coût ≤ allocation naïve par-exigence, et
monotonie de la frontière. **Cross-check par fuzzing** (exemple
`fuzz_optimize`) sur 1500+ instances aléatoires contre un certificat
d'optimalité indépendant purement primal (faisabilité + « chaque composant
épinglé » : aucune inertie ne peut croître sans violer une contrainte, ce
qui est nécessaire à l'optimalité puisque le coût décroît strictement en I).
Le fuzzing a révélé qu'une exécution ayant atteint `max_iters` sur des
contraintes quasi-parallèles pouvait laisser une contrainte marginalement
dépassée (~4 ppm) ; corrigé par un **garde-fou de faisabilité** (resserrement
uniforme final `f = 1/maxₖ(atteintₖ/I_max,ₖ)`) qui **garantit** désormais que
l'allocation retournée respecte toujours chaque budget — préférable, pour du
tolérancement, à une solution légèrement infaisable. Nouvel outil MCP
`tolerance_optimize_cost`.

### Ajouté — tolérancement de forme et modal (`scirust-tolerance`)
Complément « surface + modal » de la thèse d'Adragna (*Tolérancement des
Systèmes Assemblés, une approche par le Tolérancement Inertiel et Modal*,
tel-00403876 ; arXiv:1002.0251) qui étend le tolérancement inertiel d'une
caractéristique scalaire à une surface mesurée entière :

- **`form`** (nouveau module) : `FormBatch` sur une matrice de mesures
  (parts × points, écart au nominal). L'**inertie de surface**
  `I_S = √((1/m) Σⱼ Iⱼ²)` est la moyenne quadratique des inerties de points,
  égale à la RMS de tous les écarts au nominal — vérifié par l'identité
  `I_S² = (1/(m·n)) Σᵢⱼ xᵢⱼ²`. Fournit aussi les inerties par point, le point
  le pire, et la signature de forme moyenne.
- **`modal`** (nouveau module) : décomposition modale des défauts de forme
  « à la manière des séries de Fourier ». `ModalBasis` (base DCT-II
  exactement orthonormée, base utilisateur, ou orthonormalisation de
  Gram-Schmidt d'une base FEM), `decompose`/`reconstruct`/`residual_norm`
  (Parseval `Σ λₖ² = ‖d‖²`), et `modal_inertias` dont l'identité de
  partition **`Σₖ Iₖ² = m·I_S²`** rend le tolérancement des modes (petit
  jeu de budgets physiques : mode 0 = taille, 1 = inclinaison, 2 = ovalité…)
  équivalent au tolérancement de toute la surface.
- **`spatial`** (nouveau module) : **tolérancement inertiel 3D par
  torseurs de petits déplacements** (SDT, d'après Bourdet & Clément ;
  Adragna/Samper/Pillet, arXiv:1002.0253). L'écart d'un point vaut
  `d(M) = T + R × OM`, et l'écart normal `e(M) = d(M)·n = T·n + R·(OM×n)
  = g(M)·θ` avec le vecteur d'influence `g = [n ; OM×n]`. `Torsor`,
  `Feature` (échantillon points+normales), `fit_torsor` (association aux
  moindres carrés `θ=(GᵀG)⁻¹Gᵀe` par élimination de Gauss avec pivot,
  renvoie `None` si la surface est sous-contrainte — un plan seul
  n'observe que 3 DDL), `form_residual` (défaut de forme résiduel, à
  passer à `modal`), et l'**inertie de surface** `I_S² = θ̄ᵀHθ̄ + tr(HΣ_θ)`
  avec `H=(1/m)Σ g gᵀ` — la combinaison statistique exacte du défaut de
  **localisation** (T) et d'**orientation** (R), avec sa décomposition
  location/orientation/couplage. La forme analytique est vérifiée égale à
  l'empirique (via `FormBatch`) et l'association vérifiée par
  aller-retour sur une pièce datum 3-2-1 pleine échelle. Ceci **remplace**
  l'ancienne limite « non livré » : la géométrie 3D par torseurs est
  maintenant fournie et vérifiée.
- **`scirust-mcp`** : nouveaux outils `tolerance_form_modal` (inertie de
  surface + décomposition modale) et `tolerance_3d_surface_inertia`
  (inertie de surface 3D + décomposition localisation/orientation).

### Ajouté — plateforme de trading crypto agentique (`scirust-trader` + `scirust-mcp`)
Extension majeure du MVP `scirust-trader` (marché→indicateurs→modèle→
certification→risque→LLM→preuve) en une boîte à outils de trading niveau
plateforme pro, **entièrement pilotable par un LLM agentic via MCP** et
**simulation/paper-trading d'abord** (aucune exécution d'ordre réel exposée ;
les données de marché live Binance restent en lecture derrière `--features
live`). Tout est en Rust pur, déterministe (même entrée ⇒ même sortie et mêmes
empreintes de preuve), sans nouvelle dépendance.

- **Indicateurs (`indicators.rs`)** — +12 indicateurs pro au-delà de
  RSI/MACD/ATR/Bollinger/SMA/EMA : Stochastic (%K/%D), ADX/DMI (+DI/−DI,
  lissage de Wilder correct, amorce ADX à `2·période−1`), OBV, VWAP glissant,
  Williams %R, CCI (déviation absolue moyenne), MFI, ROC, momentum, Z-score,
  Chaikin Money Flow, Supertrend (bandes ATR + logique de report/retournement),
  canaux Donchian et Keltner, extrema glissants.
- **Figures chartistes (`patterns.rs`)** — détection déterministe de doji,
  marteau/pendu, marteau inversé/étoile filante, marubozu, engulfing, piercing
  line/dark cloud, étoiles du matin/soir, trois soldats/corbeaux.
- **Carnet d'ordres (`orderbook.rs`)** — microstructure : mid, micro-price
  pondéré par la taille, spread (bps), profondeur, imbalance, **VWAP
  d'exécution en marchant le carnet**, slippage et liquidité dans X bps.
- **Ordres & moteur d'appariement (`orders.rs`)** — types Market/Limit/
  Stop/StopLimit/TakeProfit, TIF (GTC/IOC/FOK), post-only/reduce-only, frais
  maker/taker, modèle de slippage, arrondi tick/lot, et une logique de fill
  *paper* déterministe sur chandelier (sémantique de backtest standard).
- **Portefeuille (`portfolio.rs`)** — comptes multi-actifs, positions nettées
  long/short (coût moyen, PnL réalisé/latent, retournement à travers zéro),
  équité mark-to-market, exposition brute/nette, rééquilibrage vers des poids
  cibles, prix de liquidation isolé (levier).
- **Métriques (`metrics.rs`)** — Sharpe, Sortino, Calmar, CAGR, volatilité
  annualisée, max drawdown, Ulcer Index, VaR/CVaR historiques, Kelly
  (discret & continu), win-rate, profit factor, expectancy, corrélation, bêta.
- **Stratégies (`strategy.rs`)** — trait `Strategy` + archétypes : croisement
  SMA/EMA, RSI mean-reversion, MACD, breakout Bollinger/Donchian, Supertrend,
  momentum ; fabrique par nom + paramètres (pilotable en langage naturel).
- **Backtest événementiel (`backtest.rs`)** — décision à la clôture,
  exécution à l'ouverture suivante (**pas de look-ahead**), frais/slippage
  réels, journal de trades round-trip, rapport de performance complet,
  comparaison buy-and-hold.
- **Découverte d'opportunités (`scanner.rs`)** — le cœur du « trouve-moi des
  trades qui respectent ces conditions, avec un objectif de profit de X » :
  backteste chaque stratégie × symbole, lit le signal courant, filtre sur les
  contraintes (retour, drawdown, Sharpe, win-rate, profit factor, direction),
  dimensionne un plan entrée/stop/take-profit/taille basé ATR, classe, et
  **scelle chaque opportunité + le rapport avec une preuve SHA-256** vérifiable.
- **Exécution de micro-ordres (`execution.rs`)** — découpe d'un ordre parent
  en ordres enfants rapides : TWAP, VWAP (profil de volume), POV, Iceberg,
  micro-burst, et trajectoire optimale **Almgren-Chriss**
  (`x_j=X·sinh(κ(T−t_j))/sinh(κT)`, `η̃=η−½γτ`), plus simulation de qualité
  d'exécution (VWAP réalisé, slippage vs prix d'arrivée).
- **Market making (`marketmaking.rs`)** — quotes optimales **Avellaneda-
  Stoikov** : prix de réservation `r=s−q·γ·σ²·(T−t)`, spread optimal
  `γ·σ²·(T−t)+(2/γ)·ln(1+γ/κ)`, skew d'inventaire, approximation GLFT.
- **Signaux microstructure (`microstructure.rs`)** — Order-Flow Imbalance
  (Cont-Kukanov-Stoikov), imbalance de flux de trades, VPIN (toxicité de flux,
  classification bulk-volume), lambda de Kyle (impact prix).
- **Graphes SVG (`chart.rs`)** — chandeliers + overlays d'indicateurs +
  marqueurs d'entrée/sortie et courbes d'équité, en SVG autonome que le LLM
  affiche directement (« fournir des graphes »).
- **Outils MCP (`scirust-mcp/src/tools/trader.rs`)** — 26 outils exposant tout
  le pipeline à n'importe quel agent MCP : `trader_market_data`,
  `trader_indicators`, `trader_patterns`, `trader_signal`, `trader_backtest`,
  `trader_scan_opportunities`, `trader_orderbook`, `trader_size_position`,
  `trader_execution_plan`, `trader_market_making_quotes`,
  `trader_microstructure`, `trader_metrics`, `trader_chart`,
  `trader_certified_predict` (prédiction ML bornée par IBP), `trader_portfolio`
  (état du portefeuille : PnL réalisé/latent, équité mark-to-market, exposition
  brute/nette, prix de liquidation avec levier), `trader_rebalance`
  (ordres pour atteindre des poids cibles) et `trader_dashboard` (rapport HTML
  autonome : opportunités + preuves + cartes de métriques + courbe d'équité) —
  le portefeuille et le reporting se pilotent au chat.
- **Tableau de bord (`dashboard.rs`)** — génération d'une page HTML autonome
  (CSS inline, SVG embarqué, thème clair/sombre) réunissant le scan
  d'opportunités et un backtest ; « montre-moi » devient un rapport visuel
  partageable plutôt qu'un mur de JSON.
- **Robustesse anti-surapprentissage (`robustness.rs` + 2 outils MCP)** — un
  scanner qui garde la meilleure de nombreuses stratégies trouve forcément des
  flukes ; deux garde-fous : `walk_forward` (backtest sur segments séquentiels
  indépendants → **consistance out-of-sample** = fraction de fenêtres
  gagnantes, pour distinguer un edge durable d'un ajustement de courbe) et
  `monte_carlo` (ré-échantillonnage bootstrap **déterministe** du journal de
  trades → bandes de percentiles d'équité, distribution du max drawdown,
  probabilité de perte et de **ruine**). Outils MCP `trader_walkforward` et
  `trader_monte_carlo`.
- **Construction de portefeuille (`portfolio_opt.rs` + 1 outil MCP)** — passer
  d'un signal par actif à une **allocation multi-actifs** : matrice de
  covariance et de corrélation des rendements, volatilités annualisées, et
  quatre méthodes de pondération — poids égaux, **inverse-vol**,
  **inverse-variance** et **variance minimale** (inversion Gauss-Jordan
  régularisée par crête, repli sur inverse-variance si singulière),
  long-only en option. Diagnostics de risque : contributions au risque par
  actif, **ratio de diversification** et variance de portefeuille. L'outil
  MCP `trader_portfolio_construct` prend des rendements (ou des séries OHLCV
  alignées), renvoie des poids cibles + la matrice de corrélation, et se
  branche sur `trader_rebalance` pour émettre les ordres — « construis-moi un
  portefeuille » se pilote au chat.
- **Détection de régime de marché (`regime.rs` + 1 outil MCP)** — lire *l'état
  du marché* avant de choisir comment y trader. Trois lectures orthogonales
  fusionnées en une taxonomie de six régimes (haussier/baissier × calme/volatil,
  plus range et crise) : **volatilité réalisée glissante** classée par
  percentile (calme / élevé / crise, la volatilité étant auto-corrélée —
  Mandelbrot 1963), **force de tendance** = pente OLS du log-prix normalisée par
  la volatilité (un t-stat signal/bruit), et **exposant de Hurst** par analyse
  R/S (Hurst 1951, Mandelbrot & Wallis 1969 ; `H>0.5` tendanciel/momentum,
  `H<0.5` retour à la moyenne). Les labels par barre alimentent une **matrice de
  transition de Markov** empirique → durées de régime attendues et occupation
  stationnaire (long terme). L'outil MCP `trader_regime` renvoie le régime
  courant, une **posture recommandée** (famille de stratégie + levier à adapter
  aux conditions), et toute la dynamique de transition — déterministe.
- **Optimisation de paramètres anti-surapprentissage (`optimize.rs` + 1 outil
  MCP)** — répondre honnêtement à « quels paramètres utiliser ? ». Un balayage
  naïf qui garde le meilleur backtest ne fait que sur-ajuster le passé ; ce
  module reproduit la validation d'un desk systématique : (1) **découpe**
  l'historique en une portion *train* et un *holdout* jamais vu par la
  recherche ; (2) **explore** la grille sur le train uniquement, en classant les
  candidats non par leur meilleur ajustement plein-échantillon mais par leur
  **consistance walk-forward hors-échantillon** (via `robustness`) — un jeu de
  paramètres qui ne marche que sur une fenêtre chanceuse est mal classé même
  in-sample ; (3) **confirme** les finalistes sur le holdout, la dégradation de
  Sharpe train→holdout (`overfit_gap`) trahissant le surapprentissage ; (4) rend
  un **verdict** en clair (robuste / partiel / surappris). L'outil MCP
  `trader_optimize` accepte une grille explicite `{param:[valeurs]}` ou des
  grilles par défaut par stratégie, cinq objectifs de classement, et borne le
  balayage (`max_combos`, échantillonnage régulier) — déterministe.
- **Arbitrage statistique / pairs trading (`pairs.rs` + 2 outils MCP)** — trader
  la *relation* entre deux actifs plutôt que la direction de l'un : marché-neutre
  (long une jambe, short l'autre), rentable même dans un marché plat ou baissier.
  Boîte à outils quant standard : **ratio de couverture** par MCO (β tel que
  `A−βB` soit stationnaire), test de **cointégration** d'Engle-Granger (t-stat de
  Dickey-Fuller sur le coefficient AR(1) de retour à la moyenne du spread),
  **demi-vie** de retour à la moyenne (Ornstein-Uhlenbeck), exposant de Hurst du
  spread (confirmation indépendante `H<0.5`), et **z-score** du spread pour le
  signal (short le spread quand il est riche, long quand il est bon marché).
  `trader_pair_analyze` analyse une paire (cointégration + couverture + signal +
  verdict) ; `trader_pair_scan` teste toutes les paires d'un panier et classe les
  plus tradables (spread le plus stationnaire en tête) — déterministe.
- **Options / dérivés (`options.rs` + 2 outils MCP)** — une nouvelle classe
  d'instruments : une créance à effet de levier, convexe, sensible à la
  **volatilité**. Boîte à outils du desk d'options : **pricing Black-Scholes-
  Merton** de calls/puts européens (avec rendement de portage/dividende continu),
  les **Grecs** en conventions de marché (delta, gamma, véga par point de vol,
  thêta par jour, rhô par point de taux), la **volatilité implicite** par
  bissection robuste bornée (bornes de non-arbitrage vérifiées), et l'analyse
  (moneyness, valeur intrinsèque/temps, point mort, probabilité risque-neutre de
  finir dans la monnaie). Agrégation de **livre d'options** : Grecs nets d'un
  portefeuille de jambes + la quantité de spot qui **neutralise le delta**
  (couverture). `trader_option_price` price une option (+ Grecs + VI) ;
  `trader_option_book` agrège un livre et calcule la couverture delta —
  déterministe (validé : parité call-put, aller-retour de VI, valeurs de
  référence Black-Scholes).
- **CLI (`scirust trader …`)** — nouvelles sous-commandes `strategies`,
  `scan` (scan d'opportunités sur données mock, preuve vérifiée), `chart`
  (écrit un SVG de courbe d'équité) et `dashboard` (écrit un rapport HTML).
- **Connexion aux portefeuilles (`wallet.rs` + 7 outils MCP)** — plomberie
  conforme aux protocoles reconnus, **watch-only / dry-run par défaut** :
  Keccak-256 et HMAC-SHA256 en Rust pur (vérifiés contre les vecteurs
  Ethereum et RFC 4231), adresses EVM avec checksum **EIP-55** (vérifié
  contre les 4 exemples canoniques), parsing d'URI de pairing **WalletConnect
  v2** + namespaces `eip155`/CAIP-2, construction de transactions **EIP-1559**
  avec hash de signature (RLP + keccak, non signé), séparateur de domaine et
  digest **EIP-712**, signature de requêtes REST d'exchange (Binance/Coinbase,
  HMAC), et un connecteur watch-only + lecture de solde JSON-RPC (derrière
  `live`). **Sécurité** : toute action qui signe ou déplace des fonds est
  verrouillée derrière une `WalletAuthorization` signée hors-bande avec une
  clé côté serveur (`SCIRUST_WALLET_KEY`) que le LLM ne peut pas fabriquer ;
  les secrets d'exchange proviennent d'une variable d'environnement
  (`SCIRUST_EXCHANGE_SECRET`) et ne transitent jamais par la conversation.
  Outils MCP : `wallet_validate_address`, `wallet_parse_walletconnect_uri`,
  `wallet_walletconnect_namespace`, `wallet_build_evm_transaction`,
  `wallet_eip712_hash`, `wallet_sign_exchange_request`,
  `wallet_authorization_status`.
- **Durcissement de l'autorisation portefeuille (revue de sécurité, avant toute
  exécution réelle)** — le modèle d'autorisation est renforcé pour éliminer les
  contournements d'un simple plafond en valeur native, **sans activer la moindre
  signature réelle** (aucune signature ECDSA n'existe ; l'autorisation reste un
  jeton de capacité pur). `WalletAuthorization` est désormais lié au *contexte de
  la transaction* — allowlist de destinataires (`allowed_to`) et de **sélecteurs
  de calldata** (`allowed_selectors`, vide ⇒ transferts natifs uniquement, ce qui
  bloque un `transfer` ERC-20 à `value=0` qui esquivait le plafond natif),
  plafond par transaction **et budget cumulé** (`cumulative_budget_wei`), et un
  mode **lié** (`bound_tx_hash`) à usage unique qui n'autorise qu'une transaction
  au hash exact. Un `SpendLedger` applique l'usage unique et le budget cumulé
  (anti-rejeu). L'encodage canonique signé est **préfixé en longueur** (plus
  d'ambiguïté de délimiteur). Le contrôle de fenêtre de validité utilise
  l'horloge **serveur**, jamais un temps fourni par le client. Côté exchange, la
  signature REST refuse **toujours** les endpoints de retrait / transfert /
  gestion de clés et respecte une allowlist opérateur optionnelle
  (`SCIRUST_EXCHANGE_ALLOWED_PATHS`). Tout reste en simulation ; l'exécution
  réelle derrière `live` reste non implémentée et exige une revue dédiée.

### Ajouté — verticaux industriels D2-D8 de `docs/DOMAIN_ROADMAP.md`
Chaque domaine documenté dans la feuille de route de marché reçoit maintenant
une implémentation (ou, quand une pièce ne peut pas être vérifiée avec une
confiance suffisante pour du code de sécurité, une limite honnête explicite
plutôt qu'une formule devinée) :

- **`scirust-grid`** (existant, complété — D2 protection réseau) : nouveaux
  modules `state_estimation` (estimation d'état par moindres carrés pondérés
  `x̂=(HᵀWH)⁻¹HᵀWz`, détection de mauvaises données par test du χ² global et
  test du plus grand résidu normalisé, Abur & Expósito — vérifié contre un
  exemple 3-nœuds calculé indépendamment) et `distance_relay` (comparateur
  mho multi-zones, IEEE C37.113 §5.2).
- **`scirust-biomed`** (existant, complété — D3 dispositifs médicaux) :
  nouveau module `control` (`pid`, `iob`, `insulin_safety`, `barrier`) — PID
  à anti-windup conditionnel, suivi d'insuline active par décroissance
  exponentielle, supervision par seuils (suspension sur glycémie basse,
  sortie de mode automatique), et un filtre de sécurité **Control Barrier
  Function** (Ames et al., IEEE TAC 2017) résolu en forme close. Chaque
  module porte un avertissement de non-usage clinique explicite : ceci
  démontre des techniques de contrôle certifiable, pas un algorithme de
  dosage homologable.
- **`scirust-maritime`** (nouvelle crate — D5 maritime autonome) :
  `colregs` (classification de rencontre COLREG par relèvement relatif),
  `cpa_tcpa` (évaluation du risque de collision, vérifié contre un exemple
  travaillé à deux navires : TCPA≈54.5min, CPA≈3.41nm), `thrust_allocation`
  (allocation de poussée DP par pseudo-inverse pondérée, Fossen 2011,
  vérifiée contre la pseudo-inverse de Moore-Penrose numpy).
- **`scirust-fab`** (nouvelle crate — D6 semi-conducteurs) : `r2r`
  (contrôleur EWMA run-to-run, Sachs, Hu & Ingolfsson 1995, vérifié contre
  un exemple travaillé et une preuve de convergence géométrique) et `pca`
  (FDC multivarié T²/SPE, Kourti & MacGregor 1995, sur la SVD générale de
  `scirust-solvers`) — construit par-dessus `scirust-spc` (`EwmaChart`,
  `HotellingT2`) déjà présent, sans le dupliquer.
- **`scirust-agtech`** (nouvelle crate — D7 agriculture de précision) :
  pipeline de nettoyage de carte de rendement déterministe et auditable
  (`outlier_filter` : filtres global + local, Sudduth & Drummond 2007 ;
  `idw` : interpolation par pondération inverse à la distance) répondant à
  la divergence documentée entre QGIS/Agro-Map/Farm Works (Walczykova et
  al. 2018). `agpl` expose le modèle des trois paramètres de risque
  ISO 25119-2 (Sévérité/Exposition/Contrôlabilité, vérifié contre le texte
  normatif) mais **n'implémente délibérément pas** la fonction de décision
  `S×E×C→AgPL` : le graphe de risque complet (Figure 1, §6.3.7) n'apparaît
  dans aucune source ouverte vérifiable trouvée.
- **`scirust-fatigue`** (nouvelle crate — D4 fatigue structurelle) :
  `rainflow` (comptage de cycles ASTM E1049-85 §5.4.4, port de l'algorithme
  à pile vérifié valeur par valeur contre la bibliothèque de référence PyPI
  `rainflow` sur deux séquences indépendantes) et `miner` (règle de
  Palmgren-Miner de cumul de dommage, courbe S-N en loi de puissance
  générique — aucune courbe de matériau réel n'est prétendue).
- **`scirust-sis`** (complété — D8 nucléaire) : nouveau module
  `reactor_trip` (`architecture_with_bypass`, `pfd_avg_during_bypass`) —
  reconfiguration du vote MooN quand un canal est en dérivation pour
  maintenance (IEC 61513 §6.2.3.5, réduit `N` sans changer `M`), construit
  entièrement sur les primitives déjà vérifiées de `Architecture` et
  `pfd_moon`. La méthodologie de seuil ISA-67.04 reste documentée mais non
  implémentée (limite honnête, pas un oubli).
- **`scirust-tolerance`** (nouvelle crate — tolérancement inertiel) : la
  méthode de M. Pillet et du laboratoire SYMME (Adragna, Pillet, Formosa,
  Samper — arXiv:1002.0270), qui tolérance l'**inertie**
  `I = √(δ² + σ²)` (l'écart quadratique moyen à la cible, soit
  `√(E[perte de Taguchi]/k)`) plutôt que la distance à un intervalle. Cinq
  modules : `inertia` (type `Inertia`, estimation d'échantillon avec `Î²`
  estimateur non biaisé de `I²`, perte de Taguchi, budget `I_max`, cône
  d'inertie), `capability` (`Cp`/`Cpk`/`Cpm`/`Cpmk`/`Pp`/`Ppk`, l'indice
  inertiel `Cpi = I_max/I` — égal à `Cpm` au budget `Cp=1` —, non-conformité
  en ppm avec une queue `erfc` fiable jusqu'à 6σ), `chain` (analyse et
  répartition de chaînes de cotes 1D : pire cas / statistique / pondérée /
  garantie d'un `Cpk` par le coefficient `ICC = √(Cpk²+n/9)`, **vérifié
  contre le tableau 2 de arXiv:1002.0270** : `0.033`/`0.075`/`0.060`),
  `chart` (carte de pilotage inertiel avec limite `UPL(α) = I_max·√(χ²_{n;1−α}/n)`
  et recommandation recentrer / réduire la dispersion), `sampling`
  (échantillonnage d'acceptation par inertie, Pillet & Maire — loi du χ²
  **non-centré** `n·Î²/σ² ~ χ'²(n, λ=n·δ²/σ²)`, courbe d'efficacité et
  synthèse d'un plan `(n, k)` satisfaisant risques fournisseur α et
  client β), et `special` (`erf`/`erfc`/CDF normale/quantile χ² et **CDF
  du χ² non-centré**, validés contre valeurs de référence — dont des
  ancres Monte-Carlo indépendantes pour le χ² non-centré). Le module
  `inertia` couvre aussi le **mélange de lots** (`I_c² = Σ pᵢ Iᵢ²`, un
  avantage clé du tolérancement inertiel), la combinaison multi-DOF/3D
  (`vector_inertia`), la correction de l'inertie observée pour l'incertitude
  de mesure, et une répartition **à coût minimal** (`CostOptimal`, minimum
  lagrangien en forme close, vérifié par les conditions KKT). Pur Rust,
  dépendance unique `serde`. Découvert et corrigé par une passe de
  vérification adverse : saturation de `erf` à `|x|≥6` (débordement→NaN
  pour grand `x`).
- **`scirust-mcp`** : un outil par domaine ci-dessus
  (`grid_state_estimate`, `biomed_cbf_safe_dose`, `maritime_collision_risk`,
  `fab_r2r_update`, `agtech_clean_yield_map`, `fatigue_rainflow_damage`,
  `sis_reactor_trip_bypass`, `tolerance_inertial_capability`,
  `tolerance_chain_allocate`, `tolerance_acceptance_plan`) — chaque domaine
  ajouté devient immédiatement pilotable par un agent, conformément à la
  doctrine du connecteur unique de `docs/DOMAIN_ROADMAP.md`.

### Ajouté — algèbre linéaire et solveurs
- **`scirust-solvers`** : **SVD aléatoire** (Halko, Martinsson & Tropp 2011 —
  projection sur sous-espace aléatoire germé par un `SplitMix64` déterministe
  maison, avec itérations de puissance optionnelles et ré-orthonormalisation
  QR) pour approximer la SVD tronquée d'une matrice sans la décomposer en
  entier ; **accélération d'Anderson** (Walker & Ni 2011) pour les
  itérations à point fixe, réduite à des moindres carrés sans contrainte
  résolus par la QR déjà présente. Même graine ⇒ sortie bit-identique.
- **`scirust-reliability`** : nouvelle formule générale `pfd_moon(m, n, ...)`
  généralisant PFDavg à toute architecture `M`-parmi-`N` au-delà des cinq
  tabulées par IEC 61508-6 Annexe B (validée contre les cinq cas nommés et
  contre 2oo4/3oo4 par dérivation indépendante — voir la doc du module pour
  le near-miss de généralisation naïve qui a motivé cette vérification
  poussée). `scirust-sis::voting::Architecture::pfd_avg` s'y replie
  désormais au lieu de refuser les architectures non tabulées (2oo4, etc.).
- **`scirust-sis`** : nouveau mode de panne « déclenchement intempestif »
  (`fault_injection::simulate_demand_with_spurious`) — modélise un canal
  bloqué en position déclenchée, indépendamment des pannes dangereuses non
  détectées déjà modélisées.
- **`scirust-discovery`** : trois nouveaux protocoles de découverte —
  BACnet/IP (Who-Is/I-Am), SNMPv1 (GET sysDescr.0, encodeur/décodeur BER
  minimal maison), EtherNet/IP (CIP ListIdentity — en-tête d'encapsulation à
  confiance élevée, disposition interne de l'item Identity documentée comme
  moins vérifiée faute de matériel réel pour confirmer).

### Ajouté — sûreté fonctionnelle des procédés (IEC 61511/61508 — SIS)
- **`scirust-reliability`** (existant, complété) : ajout des architectures de
  vote manquantes `pfd_2oo2` (`λDU·T1`, pas de terme β — un 2oo2 n'a aucune
  redondance à vaincre pour une défaillance dangereuse) et `pfd_1oo3`
  (`(1−β)³(λT1)³/4 + β·λT1/2`), complétant la famille MooN
  1oo1/1oo2/2oo2/2oo3/1oo3. `Sil` dérive maintenant `Ord` (bande la plus
  haute = garantie la plus forte). Nouveau test de validation contre un
  exemple publié externe (Lundteigen & Rausand, NTNU, ch. 8, diapo 27/43 :
  2oo3, λDU=1e-6/h, τ=8760h, β=10% → PFDavg≈5.00e-4), en plus des dérivations
  à la main déjà présentes.
- **`scirust-sis`** (nouvelle crate) : la couche systèmes/logique par-dessus
  ces primitives — architectures de vote `M`-parmi-`N` (évaluation de votes
  en décision de déclenchement), boucle SIF complète (capteurs → automate
  logique → éléments finaux, PFDavg total = somme des sous-systèmes, pratique
  ISA-TR84.00.02 standard), injection de pannes (démontre empiriquement
  qu'un 2oo3 tolère un canal en panne mais qu'un 2oo2 non), matrices
  cause-à-effet évaluées déterministiquement, dimensionnement d'intervalle
  de test de preuve par inversion numérique de PFDavg (réutilise
  `scirust-solvers::roots::bisection`), et un journal d'audit hash-chaîné
  SHA-256 des décisions de déclenchement et des changements de matrice
  cause-à-effet — motivé directement par l'attaque Triton/Trisis (2017)
  contre des automates de sécurité Triconex Schneider. Exposé comme outils
  MCP (`sis_verify_sif_loop`, `sis_size_proof_test_interval`). Marque le
  domaine D1 de `docs/DOMAIN_ROADMAP.md` comme fait.

### Ajouté — connecteur d'agent (MCP) et découverte OT/IT sûre
- **`scirust-mcp`** (nouvelle crate) : serveur [Model Context Protocol](https://modelcontextprotocol.io)
  (JSON-RPC 2.0, transport stdio) exposant les capacités de SciRust — solveurs numériques, outils de
  développement du SLM `scirust-sciagent`, découverte OT/IT — comme des **outils MCP standard**,
  appelables par n'importe quel agent (le SLM embarqué, Claude, ChatGPT, un script) sans glue code
  spécifique par intégration. Réutilise l'implémentation existante des outils de développement
  (`scirust_sciagent::agentic::tools::Tool::builtins()`) plutôt que de la dupliquer. Chaque
  `tools/call` — succès ou échec — est journalisé dans une chaîne hash SHA-256 (`AuditLog`), sur le
  modèle de `scirust-func-safety::audit` mais avec un vrai SHA-256 plutôt qu'un hash maison. Outils
  fournis par défaut : `dev_*` (hérités du SLM), `linalg_eigen_symmetric`, `linalg_svd`,
  `linalg_gmres`, `discovery_scan`, et l'échappatoire générique `scirust_cli`.
- **`scirust-discovery`** (nouvelle crate) : découverte d'actifs OT/IT **sûre, consentie et
  auditée** — jamais un scanner de ports générique (dangereux sur des automates industriels : voir
  l'incident SQL Slammer/Davis-Besse 2003 et l'étude Coffey et al. 2018 citées dans son `README.md`).
  Sondes natives au protocole uniquement : handshake OPC-UA UACP `Hello`/`Acknowledge`, Modbus TCP
  `Read Device Identification` (0x2B/0x0E), énumération de services mDNS/DNS-SD. Aucun paquet n'est
  envoyé sans une `ScopeAuthorization` **signée HMAC-SHA256** validant la cible contre une liste
  blanche de plages CIDR, une liste blanche de protocoles, une fenêtre de validité temporelle, et un
  niveau de sécurité de zone IEC 62443 (zones SL3+ refusées par défaut). Chaque tentative — dans la
  portée ou refusée — est journalisée dans une chaîne hash SHA-256. Exposé comme outil MCP
  (`discovery_scan`) dont la clé d'autorisation vit côté serveur (`SCIRUST_DISCOVERY_KEY`), jamais
  dans les arguments d'appel — un agent ne peut pas s'auto-autoriser.
- **`docs/DOMAIN_ROADMAP.md`** (nouveau) : recherche de marché sur les secteurs régulés (sûreté
  procédés IEC 61511, protection réseau électrique IEC 61850, dispositifs médicaux IEC 62304,
  aéronautique DO-178C, maritime autonome DNV/IMO MASS, semi-conducteurs SEMI, agriculture de
  précision ISO 25119, nucléaire IEC 61513) où le déterminisme et l'auditabilité de SciRust
  apportent une valeur documentée et non déjà couverte par les crates existantes.

### Ajouté — algèbre linéaire (`scirust-solvers`)
- **Décomposition en valeurs propres symétrique** (`linalg::eigen_symmetric`) : tridiagonalisation
  de Householder + algorithme QL implicite à décalage de Wilkinson (portage du couple `tred2`/`tql2`
  d'EISPACK, domaine public). Primitive **publique et réutilisable**, contrairement à
  l'implémentation Jacobi cyclique privée et dupliquée dans `scirust-multivariate` pour la seule PCA.
- **SVD dense générale** (`linalg::svd`) : Jacobi à un côté (Hestenes 1958), pour n'importe quelle
  matrice `(m, n)` — pseudo-inverse, moindres carrés de rang déficient — complémentaire de la SVD
  tronquée basée `nalgebra` de `scirust-core::tn::ops` (pensée pour les réseaux de tenseurs).
- **GMRES(m) redémarré et BiCGSTAB** (`linalg::gmres`, `linalg::bicgstab`) : solveurs itératifs
  matrix-free pour systèmes `A·x=b` **non symétriques** (Saad & Schultz 1986 ; van der Vorst 1992),
  jusqu'ici couverts uniquement par le gradient conjugué (SPD seulement). Orthogonalisation de
  Arnoldi séquentielle (Gram-Schmidt modifié), déterministe.
- **Préconditionneur de Jacobi** (`linalg::precond::JacobiPreconditioner`), utilisable avec
  `gmres_preconditioned`/`bicgstab_preconditioned`.
- **Gradient projeté spectral** (`optimize::spg`) : optimisation sous contraintes de boîte
  (Birgin, Martínez & Raydan 2000), pas de Barzilai-Borwein + recherche linéaire d'Armijo non
  monotone — jusqu'ici seul un QP de boîte ad hoc existait dans `scirust-control`.

### Ajouté — simulation quantique par réseaux de tenseurs
- **Simulateur de circuits quantiques MPS / Tensor-Train** (`quantum::Mps`/`MpsNode`) : représente
  un état à `n` qubits par une **chaîne de tenseurs de rang 3** au lieu des `2ⁿ` amplitudes d'un
  state-vector dense ⇒ tant que l'intrication reste modérée, le coût passe de **exponentiel** à
  `O(n·χ³)` (`χ` = dimension de liaison bornant l'intrication à chaque coupe). Une porte 1-qubit
  contracte une `2×2` dans l'indice physique en place ; une porte **2-qubits** sur des qubits
  adjacents (1) contracte les deux nœuds en un tenseur `θ`, (2) **applique** la porte `4×4`,
  (3) reforme une matrice et exécute une **SVD tronquée** (la SVD **maison** `tn::ops::truncated_svd`,
  **Rust pur via nalgebra — zéro FFI**), gardant au plus `χ` valeurs singulières pour plafonner la
  dimension de liaison. Amplitudes réelles `f32` (portes réelles `H`/`X`/`Z`/`CNOT`/`CZ`/`Ry`) ;
  les amplitudes complexes (phase/`S`/`T`/`Rz`) sont un travail futur. Oracle honnête (pas de
  mock) : le MPS **reproduit exactement le state-vector dense** (simulateur de référence en clair)
  sur un circuit **aléatoire** de 5 qubits / 40 portes + Bell `(|00⟩+|11⟩)/√2` (bond 2) + GHZ
  3-qubits ; **troncation saine** (état produit → bond 1 ; cap `χ=1` ⇒ approximation de haute
  fidélité) ; norme préservée ; déterminisme bit-exact. La même machinerie contraction + SVD
  tronquée **est** la compression de poids Tensor-Train déjà présente (`tn::tt_decompose`,
  `nn::tt_linear`) — directement réutilisable pour compresser des LLM locaux (SLHAv2).
  *Note d'architecture* : refus délibéré de `openblas-src`/`cuSOLVER` (FFI C/CUDA, briseraient la
  thèse zéro-FFI + déterminisme bit-exact) et de `faer` (Rust pur mais redondant avec nalgebra) —
  la SVD maison existante suffit.

### Ajouté — synergie d'écosystème (CCOS, SLHAv2)
- **Commandes CLI de la synergie** (`scirust kvcache | guard | attest`) : exposent les primitives
  ci-dessous en ligne de commande, déterministes par `--seed`. `kvcache [--budget B]` compresse une
  séquence KV et affiche le **ratio de compression** + la **fidélité cosinus** de l'attention vs
  pleine précision (et le soft-paging borné avec `--budget`) ; `guard [--alpha A]` calibre le guard
  et affiche la **couverture empirique** (≥ 1−α) + des verdicts Accept/Abstain/Reject ; `attest`
  enregistre des inférences vérifiables dans le **journal hash-chaîné**, vérifie la chaîne, rejette
  une inférence falsifiée et démontre l'inviolabilité. Documentées dans `docs/REFERENCE.md` et dans
  les **8 langues** (`Documentation*.md`).
- **Guard à garantie statistique** (`nn::guard::StatisticalGuard`) : une porte de réponse à
  **garantie de couverture sans hypothèse de distribution**, pour alimenter le `guard` de **CCOS**
  (valider/abstenir sur la sortie d'un modèle) sans seuil ad-hoc. À partir des probabilités de
  classe d'une décision, le guard forme l'**ensemble de prédiction conforme** (#21,
  `ConformalClassifier`) et en tire un verdict : une seule classe franchit `1−q̂` ⇒ **Accept** ;
  plusieurs ⇒ **Abstain** (ambigu) ; aucune ⇒ **Reject** (hors-distribution). La calibration
  conforme garantit que la vraie classe est dans l'ensemble avec proba **≥ 1−α** sur données
  échangeables, *quelle que soit la distribution* — le guard ne laisse donc prouvablement pas
  filer la bonne réponse plus d'une fraction `α` du temps. Oracle honnête : **couverture empirique
  ≥ 1−α** sur données fraîches (3-classes, déterministe) + logique de verdict (confiant→Accept,
  partagé→Abstain, plat/OOD→Reject). Les deep ensembles (#40) donnent un signal épistémique
  complémentaire pour le flag OOD.
- **Codec KV accéléré SIMD, bit-exact** (`scirust_simd::ops::dequantize_int4_into`, câblé dans
  `nn::elastic_kv_cache`) : la déquantification INT4 (`out[i]=code[i]·échelle`) passe par le kernel
  SIMD `mul_f32` ; étant **élémentaire** (pas de réduction) et un produit IEEE-754 identique par
  lane et en scalaire, le résultat est **bit-identique entre largeurs SIMD et plateformes** — le
  chemin rapide du codec KV **sans casser le déterminisme** (les réductions cosinus/attention
  restent sur le chemin déterministe). Oracle : SIMD ≡ scalaire **bit-exact** pour toute longueur
  (y compris < une lane) et une plage d'échelles.
- **Journal d'attestation hash-chaîné** (`scirust_runtime::attest`) : le pont de l'**inférence
  vérifiable** de scirust (`vinfer` #80) vers l'`event_log` de **CCOS**. Chaque `InferenceEvent`
  fige l'engagement du modèle, le hash de l'entrée et le hash de la sortie, et se chaîne au
  précédent par un **hash SHA-256** (`entréeₙ = H(entréeₙ₋₁ ‖ seq ‖ engagement ‖ entrée ‖ sortie)`)
  — exactement la forme append-only et inviolable de CCOS, donc les inférences d'un runtime scirust
  s'ingèrent dans son journal d'audit. Recalculer la chaîne re-dérive la **même tête** (replay
  déterministe) ; toute mutation ou réordonnancement la **casse**. `attest_and_record` vérifie en
  plus, *avant* d'ajouter, que la paire `(entrée, sortie)` est une inférence **authentique** du
  modèle engagé (Freivalds sur `GF(p)`, #80) — la chaîne n'atteste donc que des inférences réelles.
  Oracle honnête : la chaîne se vérifie et se rejoue (même tête) ; falsification d'un événement /
  réordonnancement **détectés** ; une inférence authentique est attestée et chaînée tandis qu'une
  sortie **falsifiée est rejetée** (journal inchangé). Complète la pile de preuve (#3, `proof`,
  DiFR #5, `vinfer` #80).
- **KV-cache compressé élastique** (`nn::elastic_kv_cache`) : la primitive déterministe
  partagée derrière **SLHAv2** (compresser le KV-cache pour faire tourner un LLM dans le cache
  du CPU plutôt que sur un GPU hors de prix) et **CCOS** (paging à mémoire bornée), bâtie sur la
  quantification et le déterminisme de scirust. Une paire clé/valeur d'attention est compressée
  en une `KvTile` par quantification **INT4 à deux niveaux** (base symétrique + **résidu** INT4 —
  le « residual tracking » de SLHAv2), chaque niveau à **échelles adaptatives par groupe**
  (`quantize_int4_grouped` : une échelle plus fine par groupe de canaux ⇒ « adaptive scaling »
  cosine-aware de SLHAv2, dans l'esprit per-canal de KVQuant #68), ce qui porte la fidélité
  **cosinus** au-delà de 0,99 tout en réduisant l'empreinte plusieurs fois par rapport au `f32`. L'`ElasticKvCache` conserve ces
  tuiles sous un **budget** optionnel et évince la plus ancienne au dépassement (soft-paging /
  mémoire élastique — l'abstraction de paging commune avec CCOS), et sert l'attention directement
  depuis les tuiles compressées en réutilisant `contiguous_attention` (#63), si bien que le seul
  écart avec un cache pleine précision est l'erreur de compression (mesurée). Oracle honnête :
  reconstruction à **fidélité cosinus** élevée (>0,95, le niveau résidu battant strictement la
  base seule) ; **attention compressée ≈ pleine** (cosinus >0,99) ; **ratio de compression** ≥3×
  vs `f32` ; cache **borné** sous budget (la plus ancienne évincée) et **bit-exact déterministe**.
  Codec exposé (`quantize_int4`/`dequantize_int4`/`KvTile`/`cosine_similarity`) pour être consommé
  par SLHAv2/CCOS. Rejoint KVQuant (#68) et PagedAttention (#63) dans la pile KV-cache.

### Corrigé
- **SIMD `portable` — bug d'alignement (résultats faux, non déterministe)** :
  `add_f32/f64_inplace`, `dot_f32/f64` et `fma_f32` (`scirust-simd::portable`)
  découpaient **chaque opérande indépendamment** via `as_simd`/`as_simd_mut`.
  Quand deux slices avaient des alignements mémoire différents (fréquent : dépend
  de l'allocation), les boucles SIMD du cœur appariaient des lanes **décalées** →
  résultats **incorrects**, de façon **non déterministe** (d'où le test
  `test_add_f32_inplace` qui échouait ~30–50 % des lancers). Réécrites avec
  `chunks_exact`, qui apparie le bloc k de chaque slice à l'identique quel que
  soit l'alignement. Ajout d'un test de régression couvrant tous les décalages
  relatifs (add/dot/fma vs référence scalaire) ; 12/12 lancers verts. Au passage,
  un `needless_return` dans `complex.rs` (chemin `portable-simd`) corrigé.

### Ajouté — campagne « faire grandir scirust »
- **Reluplex — vérification *complète* de style SMT** (`nn::ibp::reluplex_verify`/
  `reluplex_unstable_count`, Katz et al. 2017, roadmap #4) : une recherche de **satisfiabilité**
  d'un contre-exemple par **case-splitting des phases ReLU** — mais **paresseuse**, la signature de
  Reluplex : un neurone dont l'intervalle de pré-activation reste entièrement d'un côté de 0 sur la
  boîte est **stable**, donc sa phase est **forcée** (jamais scindée) ; seuls les neurones
  **instables** sont scindés, soit `2^instables` feuilles au lieu des `2^cachés` de l'énumération
  *eager* du MILP (#31). Sur chaque feuille (un patron ReLU complet) le réseau est affine et un
  contre-exemple est cherché en minimisant chaque marge sur la région du patron (le **LP 2D exact**
  partagé avec le vérificateur MILP) ; on renvoie le **premier** contre-exemple trouvé (SAT) ou
  Robust. Distinct du branch-and-bound (#26, scinde le domaine d'entrée) et du MILP (#31, énumère
  *tous* les patrons) par le **splitting paresseux des phases ReLU**. Oracle honnête : **accord avec
  MILP** sur tout un balayage de rayons (deux méthodes exactes ⇒ mêmes décisions) ; contre-exemple
  réel (marge ≤ 0, dans la boîte) ; à petit rayon, **moins de neurones scindés** que `cachés`
  (élimination par bornes) ; déterministe. Réseau (2 entrées, 1 couche). **Clôt la pile de
  vérification** (IBP, CROWN, zonotopes, DeepPoly, randomized smoothing, Lipschitz, CROWN-IBP, BaB,
  MILP, Reluplex).
- **Inférence vérifiable — argument cryptographique compact** (`scirust_runtime::vinfer`,
  ZK-based Verifiable ML, roadmap #80) : prolonge les certificats `proof` de la ré-exécution
  bit-exacte vers une **garantie de soundness succincte**. Le modèle (une couche linéaire entière
  quantifiée sur le corps premier `GF(p)`, `p = 2³¹−1`) est **engagé** par le hachage de ses poids.
  Pour vérifier une sortie batchée `Y` revendiquée pour des entrées `X`, le vérifieur exécute la
  **vérification de Freivalds** sur `GF(p)` : tirer un `r` aléatoire et tester `W·(X·r) = Y·r`.
  Calculer `W·(X·r)` coûte `O(out·in + in·b)` contre `O(out·in·b)` pour recalculer `Y = W·X`, donc
  pour un batch c'est **succinct** (sous-linéaire dans le coût de recalcul). Un `Y` faux passe avec
  proba `≤ 1/p` par défi, donc quelques défis donnent une erreur de soundness négligeable. Le défi
  `r` est dérivé par **Fiat-Shamir** d'un hachage de `(engagement, X, Y)`, donc non-interactif et
  **lié à la sortie revendiquée** (le prouveur ne peut pas adapter `Y` à un `r` connu). Oracle
  honnête : accepte une inférence correcte (déterministe) ; **soundness** — sur 1000 falsifications
  aléatoires d'une entrée de la sortie, **toutes** rejetées ; l'engagement **lie** le modèle
  (vérifier contre l'engagement d'un autre modèle échoue) ; Fiat-Shamir **lie** la sortie (la sortie
  valide d'**autres** entrées est rejetée pour `X`). Fournit la **soundness** cryptographique (la
  sortie provient prouvablement du modèle engagé), **pas** le zero-knowledge — le vérifieur détient
  les poids ; les zk-SNARK cachant les poids restent hors périmètre. Couronne la pile de preuve
  (sommation reproductible #3, certificats `proof`, DiFR #5).
- **DiFR — vérification d'inférence malgré le non-déterminisme** (`scirust_runtime::difr::difr_verify`,
  2025, roadmap #5) : les certificats [`proof`] vérifient une inférence par **ré-exécution
  bit-exacte** — ce qui ne marche que si le vérificateur reproduit l'arithmétique du prouveur à
  l'identique. Or sur un **matériel différent** (largeurs SIMD, FMA, nombre de threads) la sommation
  flottante est **non-déterministe**, donc un contrôle bit-exact rejetterait des sorties pourtant
  honnêtes. DiFR vérifie *malgré* cela : il recompute une **référence canonique** avec
  `reproducible_dot` (produits et somme accumulés en `f64`, indépendant de l'ordre) et accepte la
  sortie revendiquée ssi elle se trouve dans une **enveloppe d'erreur flottante saine** de cette
  référence. *Tout* calcul `f32` honnête — dans *n'importe quel* ordre de sommation — est
  prouvablement dans l'enveloppe (donc accepté) ; une sortie **falsifiée** au-delà est rejetée.
  L'enveloppe est la borne d'arrondi du produit scalaire `γ·Σ|termes|` propagée à travers les
  couches (la ReLU est 1-lipschitzienne, elle la transmet sans l'amplifier) et reste **minuscule**
  (quelques ppm de l'échelle d'activation), si bien que le contrôle attrape toute falsification
  signifiante. Oracle honnête : accepte un calcul `f32` dans un **ordre de sommation différent** ;
  enveloppe **saine** (1000 ordres aléatoires, tous acceptés) et **fine** (< 0,001 de l'échelle) ;
  **rejette** une falsification (au-delà de l'enveloppe, ici de quoi changer la classe prédite) ;
  déterministe. Prolonge la sommation reproductible (#3) et l'outillage de preuve d'inférence.
- **MILP — vérification *exacte*** (`nn::ibp::milp_min_margin`/`milp_verify_robustness`, Tjeng
  et al. 2019, roadmap #31) : la vérification exacte d'un réseau ReLU par la formulation MILP.
  L'observation clé : les **patrons d'activation** des ReLU sont précisément les variables
  **binaires** du MILP, et sur le domaine d'un patron fixé le réseau est **affine**. Pour un petit
  réseau (2 entrées, 1 couche cachée) on **énumère** les patrons et on résout chaque LP
  **exactement** — la marge `logitₜ − logitⱼ` y est affine, minimisée sur la boîte intersectée
  avec les demi-espaces d'activation du patron par **énumération des sommets** du polygone 2D (pas
  de simplexe fragile : robuste et exact). Le minimum global sur tous les patrons et toutes les
  classes concurrentes est donc **exact** ; `> 0` ⇒ robuste, sinon l'argmin est un **contre-exemple
  exact**. Oracle honnête : le minimum énumérée **égale la force brute** (il minore toute valeur
  d'une grille fine et la grille s'en approche), le contre-exemple est **réel** (marge ≤ 0, dans la
  boîte), et — étant exact — il est **≥ la borne inférieure (saine) de DeepPoly** partout et
  **strictement plus serré** à certains rayons ; déterministe. Distinct du branch-and-bound (#26),
  complet **à tolérance près** : MILP est exact (tranche même la frontière de mesure nulle).
- **Branch-and-bound — vérification *complète*** (`nn::ibp::verify_robustness`/`BabResult`,
  GCP-CROWN, Zhang et al. 2022, roadmap #26) : là où IBP/CROWN/DeepPoly donnent **une** borne
  *saine mais incomplète*, le branch-and-bound **décide**. Il borne les **marges** par classe
  (`logitₜ − logitⱼ`, fusionnées dans une dernière couche pour que DeepPoly suive la corrélation)
  sur la boîte d'entrée ; si toutes les bornes inférieures sont `> 0` la boîte est **prouvée
  robuste** ; sinon il sonde le **centre** de la boîte pour un **contre-exemple concret**, et à
  défaut **scinde** la boîte le long de son axe le plus large et récurse. Comme les sous-boîtes
  rétrécissent, la relaxation ReLU de DeepPoly devient exacte, si bien que la recherche **tranche**
  (jusqu'à une tolérance) — prouvant des cas qu'une borne unique ne peut pas, et renvoyant un
  contre-exemple réel quand la classe peut effectivement changer. Oracle honnête : `Robust` est
  **sain** (5000 points échantillonnés bien classés) ; le **rayon ℓ∞ certifié dépasse strictement**
  celui de DeepPoly seul (et la région supplémentaire est échantillonnée robuste) ; `Unsafe`
  renvoie un **vrai** contre-exemple (mal classé, dans la boîte) ; déterministe. Exposé dans la CLI
  `certify`. (Le branchement est sur le **domaine d'entrée** ; le split des ReLU instables et les
  plans coupants de GCP-CROWN ne sont pas implémentés.) Couronne la pile de vérification (IBP #1,
  CROWN #2, zonotopes #29, DeepPoly #28, CROWN-IBP #30).
- **DeepPoly — domaine abstrait relationnel** (`nn::ibp::deeppoly_certify`/`IbpMlp::certify_deeppoly`,
  Singh et al. 2019, roadmap #28) : un vérificateur de robustesse plus précis qu'IBP. Là où IBP
  traite chaque neurone par un simple intervalle (perdant toute corrélation), DeepPoly garde pour
  chaque neurone une **borne basse et haute affines en les entrées** du réseau et les **back-
  substitue** couche par couche. La relaxation ReLU est **asymétrique** : pour un pré-activation de
  plage `[l,u]` instable, la borne supérieure est la **corde** `z ≤ (u/(u−l))(y−l)` et la borne
  inférieure `z ≥ λy` avec `λ` choisi pour **minimiser l'aire** de la relaxation (`λ=1` si `u>−l`,
  sinon `0`). Comme les bornes restent affines, les corrélations sont préservées et le résultat est
  plus serré qu'IBP — **à n'importe quelle profondeur** (là où `crown_bounds` était limité à 2
  couches). Oracle honnête : **sain** (4000 points échantillonnés ∈ boîte certifiée, MLP 3 couches)
  + **strictement plus serré qu'IBP** sur `relu(x)+relu(−x)=|x|` sur `x∈[−1,1]` (DeepPoly donne la
  boîte **exacte** [0,1] car le `x` s'annule dans la borne supérieure, vs IBP [0,2]) + déterminisme.
  Exposé dans la CLI `certify` (à côté d'IBP, CROWN, zonotopes, smoothing). Prolonge IBP (#1) /
  CROWN (#2) / zonotopes (#29).
- **CROWN-IBP — entraînement certifié (vérifié)** (`nn::crown_ibp::CrownIbpMlp`, Zhang et al.
  2020, roadmap #30) : l'entraînement ordinaire minimise la perte aux entrées *concrètes* — un
  réseau peut les ajuster parfaitement et pourtant **changer de prédiction** sous une perturbation
  minime. CROWN-IBP entraîne au contraire sur une **borne certifiée de la perte du pire cas** sur
  une boule ℓ∞ autour de chaque entrée, rendant le réseau **prouvablement** robuste. L'idée clé :
  la **propagation par intervalles (IBP) est différentiable**. Pour une couche affine `y=x·W+b`,
  la boîte se transforme en `centre'=centre·W+b`, `rayon'=rayon·|W|` — et `|W|=relu(W)+relu(−W)`,
  donc toute la borne (y compris le `|W|` qui semblait exiger un op `abs` dédié) tourne sur la
  tape N-D ; la ReLU sur un intervalle `[l,u]` devient `[relu(l),relu(u)]`. Les **logits robustes**
  placent la vraie classe à sa borne **inférieure** et les autres à leur borne **supérieure**
  (`zₜ=cₜ−rₜ`, `z_j=c_j+r_j`) : une cross-entropy faible dessus signifie que la vraie classe gagne
  *même dans le pire cas* — le point est **certifié**. Oracle honnête : la propagation IBP sur la
  tape **coïncide** avec le vérificateur de référence `IbpMlp` (plain `f32`) et est **saine**
  (2000 points échantillonnés ∈ boîte certifiée) ; après entraînement certifié, le **rayon ℓ∞
  certifié croît** nettement (réseau robuste-entraîné vs accuracy-only, tous deux classant juste à
  100 %) + déterminisme bit-exact. Prolonge IBP (#1) / CROWN (#2) / zonotopes (#29) vers
  l'entraînement.
- **Sophia — optimiseur de 2e ordre clippé** (`nn::nd_optim::NdSophia`, Liu et al. 2023, roadmap
  #44) : Sophia met à l'échelle le momentum de chaque coordonnée par une estimation de la
  **Hessienne diagonale** et **clippe** le résultat : `θ ← θ − lr·clip(m/max(γ·h,eps),ρ)`. Les
  directions plates (petite courbure `h`) prennent un pas borné de type signe ; les directions
  courbées prennent un pas de type **Newton** `m/h` — d'où une robustesse au mauvais
  conditionnement. La Hessienne diagonale est estimée par un **estimateur de Hutchinson** avec un
  **produit Hessien-vecteur en différences finies** : avec un vecteur de signes `v∈{±1}` seedé,
  `Hv ≈ (∇L(θ+εv) − ∇L(θ))/ε` et `ĥ = v⊙Hv` (pour un quadratique c'est la Hessienne diagonale
  **exacte**, mon ancien blocage « il faut un op `abs` sur la tape » était infondé — le clipping
  se fait dans l'optimiseur en `f32`, pas sur la tape). Comme SAM, cela demande **deux** calculs
  de gradient par pas, donc l'appelant orchestre `probe` (perturbe `θ` de `εv`) puis `step`
  (restaure `θ`, applique la mise à jour) — optimiseur **de bibliothèque hors de la boucle
  `lm --opt`** (à un seul gradient). Oracle honnête : **converge sur un quadratique mal
  conditionné** (courbures 4 vs 0,25, conditionnement 16) là où le pas Newton par coordonnée
  neutralise le conditionnement + déterminisme bit-exact (probe seedé). Rejoint la famille
  d'optimiseurs (Adam, Lion, Muon, Shampoo, SOAP, Adafactor, LAMB, Adan, Prodigy, SAM, …).
- **QuIP# — incohérence Hadamard + codebook lattice E8** (`quantization::quantize_quip`/
  `nearest_e8`/`random_hadamard_transform`, Tseng et al. 2024, roadmap #64) : deux idées. (1) Le
  **traitement d'incohérence** : multiplier les poids par une **transformée de Hadamard
  randomisée** (signes ±1 seedés puis FWHT, *orthogonale*) étale les aberrants à travers les
  coordonnées et **rétrécit la plage dynamique** ; à budget de bits **égal**, les `2^bits`
  niveaux fixes résolvent alors bien mieux le gros des poids (le RTN scalaire devait, lui, étaler
  ses rares niveaux sur toute la plage pour couvrir les aberrants). (2) Le codebook **lattice
  E8** : quantifier les poids tournés par blocs de 8 vers le point le plus proche du **réseau
  E8** (`D8 ∪ (D8+½·1)`, décodeur fermé de Conway-Sloane) — le plus dense en dimension 8, avec un
  **moment quadratique** plus bas que la grille cubique à densité **égale** (gain de packing
  ~14 %). Oracle honnête : la RHT est orthogonale (round-trip exact) et **réduit la plage** d'un
  poids à aberrants ; le décodeur E8 renvoie un point **valide** du réseau (coords toutes
  entières ou toutes demi-entières, somme paire) et quantifie **mieux que la grille cubique en
  moyenne** (gain lattice mesuré sur 4000 vecteurs) ; bout-en-bout, QuIP# reconstruit **mieux que
  le RTN** scalaire à budget 2-bit sur des poids à aberrants épars + déterminisme bit-exact. (Le
  grand Hadamard global et le codebook E8P curé de QuIP# sont simplifiés ici en un Hadamard par
  bloc de 8 et le réseau E8 nu.) Complète la famille de quantification (AQLM, GPTQ, AWQ, NF4,
  SqueezeLLM, SpQR, KVQuant, LLM.int8, OmniQuant, BitNet).
- **AQLM — quantification additive multi-codebook** (`quantization::quantize_aqlm`/`AqlmResult`,
  Egiazarian et al. 2024, roadmap #70) : au lieu de quantifier chaque poids **scalairement**, AQLM
  découpe les poids en **groupes** de dimension `g` et approxime chaque groupe par la **somme**
  d'un mot de code tiré de chacun de `M` codebooks appris (de `K` mots chacun). Les codebooks sont
  initialisés par **k-means résiduel** puis affinés par **optimisation alternée** : ré-encoder
  chaque groupe (assignation résiduelle gloutonne à travers les `M` codebooks) puis ré-ajuster
  chaque codebook par moindres carrés sachant la contribution des autres (la beam search d'AQLM
  est ici simplifiée en assignation gloutonne — documenté). Comme les mots de code sont des
  **vecteurs**, la quantification additive capte la **structure inter-dimensions** que le
  round-to-nearest scalaire ignore, d'où une bien meilleure reconstruction à bas budget. Oracle
  honnête : erreur **< 0,7× RTN** scalaire à budget ~2-bit **égal** (`M·log₂K/g`) sur des poids
  structurés (groupes bâtis sur quelques directions prototypes) + round-trip exact (longueur non
  divisible, padding zéro) + déterminisme bit-exact. Rejoint la famille de quantification (GPTQ,
  AWQ, NF4, SqueezeLLM, SpQR, KVQuant, LLM.int8, OmniQuant, BitNet).
- **S5 — SSM MIMO + scan associatif parallèle** (`nn::nd_layers::s5_scan`/`s5_parallel_scan`/
  `NdS5`, Smith et al. 2023, roadmap #52) : contrairement aux SSM **SISO par canal** de S4D
  (chaque canal son propre état indépendant), S5 pilote un **unique état partagé** de dimension
  `n` avec **toutes** les entrées via une matrice `B`, et lit `m` sorties via `C` (d'où *MIMO*) :
  `hₜ=Ā⊙hₜ₋₁+xₜB`, `yₜ=hₜC`. La récurrence étant linéaire, elle se calcule par un **scan
  associatif** : l'élément `(aₜ,uₜ)` représente la carte affine `h↦aₜ⊙h+uₜ`, et ces cartes se
  composent par l'opérateur **associatif** `(a₁,u₁)∘(a₂,u₂)=(a₂⊙a₁, a₂⊙u₁+u₂)`. Un scan
  inclusif de **Hillis-Steele** (ordre de doublage `log₂ seq` fixe ⇒ **déterministe**) produit
  tous les états préfixes en parallèle. Oracle honnête : le **scan parallèle ≡ la récurrence
  séquentielle** — testé avec `aₜ` **variable dans le temps** (un vrai scan associatif, pas le
  cas trivial constant), ce qui prouve l'associativité qui licencie la parallélisation ;
  `s5_scan` sur la tape ≡ référence MIMO écrite à la main (valide le câblage `B`/`C`) ;
  **gradient check** (x, Ā, B, C) ; `NdS5` entraîne (MSE↓) + déterminisme bit-exact. Complète la
  famille espace-d'états (Mamba, Mamba-2/SSD, S4).
- **Mamba-2 / SSD — dualité espace-d'états ↔ attention** (`nn::nd_layers::ssd_dual`/`NdMamba2`,
  Dao & Gu 2024, roadmap #50) : Mamba-2 restreint la matrice d'état du SSM à une **décroissance
  scalaire** `aₜ` par pas (au lieu du `A` diagonal par canal de Mamba). Cette restriction rend
  la récurrence linéaire `Hₜ=aₜHₜ₋₁+xₜBₜᵀ` (état `d×n`), `yₜ=HₜCₜ` **exactement égale** à une
  unique forme quadratique masquée de type attention — la **dualité** : `Y=(L⊙CBᵀ)X` avec
  `L[i,j]=∏_{j<k≤i}aₖ` pour `i≥j`. Calculée sur la tape : le log-décroissance cumulé
  `cumlogᵢ=Σ_{k≤i}a_logₖ` est une **préfixe-somme** (matmul avec une matrice triangulaire de
  uns), `L=exp(cumlogᵢ−cumlogⱼ)` masquée causale, `Y=(L⊙CBᵀ)X`. `a_log=log a` est le paramètre
  (en Mamba-2 `a_logₜ=Δₜ·A`), donc **aucun op `log`** n'est requis ; le masque est appliqué
  **avant** l'`exp` (`diff⊙mask`, puis `exp`, puis `⊙mask`) pour garder l'exposant borné dans le
  triangle supérieur (évite `inf·0=NaN`) et y annuler exactement. Oracle honnête : la **forme
  duale ≡ la récurrence séquentielle** écrite à la main (c'est littéralement la dualité du
  papier) ; **gradient check** (x, B, C, a_log) ; `NdMamba2` entraîne (MSE↓) + déterminisme
  bit-exact. Rejoint Mamba/S4/RWKV/RetNet/GLA/HGRN/DeltaNet/xLSTM/Hyena.
- **FNO — opérateur neuronal de Fourier** (`nn::fno::FnoSpectralConv1d`/`NdFno`, Li et al.
  2021, roadmap #75) : un opérateur neuronal apprend une application entre **fonctions** (p.ex.
  condition initiale ↦ solution de PDE), pas entre vecteurs de taille fixe. FNO réalise
  l'intégrale de noyau **globale** dans le **domaine de Fourier** : transformer le signal
  échantillonné, garder les `modes` plus basses fréquences, multiplier chaque mode par un
  **poids complexe appris** `R_k=Ar_k+iAi_k` (matrice `width×width`, mélange de canaux), puis
  transformer en sens inverse. La DFT réelle et son inverse sont des **matrices cosinus/sinus
  fixes** : tout le transform est un matmul ordinaire (déterministe) que la tape N-D dérive
  directement — **sans FFT, sans type complexe, sans nouvel op** ; les poids par mode sont
  appliqués par un matmul **par lots** (`bmm`) sur les modes. Bloc FNO complet :
  `σ(SpectralConv(v)+W·v)`. Oracle honnête : reconstruction **exacte** d'un signal band-limité
  aux modes gardés (DFT⁻¹∘DFT, valide les matrices + l'inverse unilatéral facteur-2) ;
  **gradient check** par différences finies (signal, Ar, Ai) ; comme la dérivation est
  diagonale en Fourier (`d/dx↔×ik`), une seule conv spectrale **apprend l'opérateur de
  dérivation** `sin(ωx+φ)↦ω cos(ωx+φ)` et **généralise à une phase non vue** (MSE test <0,02,
  ajustement convexe) ; déterminisme bit-exact. Rejoint la famille calcul scientifique
  (Neural ODE, PINN, DeepONet, KAN).
- **Hyena — convolutions longues implicites + gating** (`nn::nd_layers::hyena_long_conv`/
  `NdHyena`, Poli et al. 2023, roadmap #56) : un mélangeur de tokens **sans attention**. La
  portée longue vient d'une **convolution causale** dont le filtre n'est pas stocké tap par
  tap mais **généré** par un petit MLP à partir d'un encodage positionnel fixe, puis fenêtré
  par une décroissance exponentielle apprenable `exp(−γ·t̄)` par canal — c'est ce qui permet
  des filtres **longs à peu de paramètres** (le coeur de Hyena). L'équivalent du rôle de
  l'attention (la dépendance aux données) est fourni par un **gating multiplicatif** :
  `z=x1⊙(h1*v)` puis `z=x2⊙(h2*z)` (ordre 2). La convolution causale par canal
  `y[t,c]=Σ_τ h[τ,c]·u[t−τ,c]` est exprimée sur la tape comme `Σ_τ h[τ,:]⊙(Sτ·u)` avec des
  **matrices de décalage constantes** `Sτ` (distribuer le matmul sur les taps apprenables ⇒
  différentiable en `u` et `h` sans op scatter). Oracle honnête : conv ≡ référence causale
  écrite à la main ; **gradient check** par différences finies (`u`, `h`) ; entraînement
  `NdHyena` (MSE↓) + déterminisme bit-exact. Rejoint la famille de modèles de séquence.
- **xLSTM — sLSTM scalaire + mLSTM matriciel** (`nn::nd_layers::slstm_scan`/`mlstm_scan`/
  `NdXlstm`, Beck et al. 2024, roadmap #57) : le LSTM étendu remplace les portes sigmoïdes
  de l'entrée par une **porte exponentielle** `iₜ=exp(ĩₜ)` accompagnée d'un **état
  normaliseur** `nₜ=fₜnₜ₋₁+iₜ`, la sortie étant `hₜ=oₜ⊙(cₜ/nₜ)`. Comme `cₜ/nₜ` est une
  moyenne pondérée positive de `zₜ=tanh∈(−1,1)`, la sortie reste bornée dans (−1,1) : la
  récurrence est **stable sans le stabilisateur log** (omis, c'est un pur dispositif
  numérique qui s'annule dans le ratio). `tanh` est construit à partir du seul op `sigmoid`
  via l'identité exacte `tanh(x)=2σ(2x)−1`. La variante **mLSTM** porte une mémoire
  covariance `d×d` mise à jour par produits externes `vₜᵀkₜ`, lue par requête, avec le
  dénominateur stabilisant `max(|nₜ·qₜ|,1)` reconstruit **exactement** via `|a|=relu(a)+
  relu(−a)` et `max(a,1)=relu(a−1)+1` (aucun nouvel op, garde fidèle). Oracle honnête :
  mLSTM ≡ récurrence de référence écrite à la main (dénominateur actif) ; **gradient check**
  par différences finies (sLSTM : 4 portes ; mLSTM : q,k,v,iₜ,fₜ, régime lisse) ;
  entraînement `NdXlstm` (MSE↓) + déterminisme bit-exact. Rejoint la famille de modèles de
  séquence (Mamba, S4, RWKV, RetNet, GLA, HGRN, DeltaNet).
- **OmniQuant — clipping de poids apprenable** (`quantization::omniquant_quantize`, Shao
  et al. 2024, roadmap #65) : le round-to-nearest quantifie chaque canal sur sa plage
  **complète** `[−max|w|, max|w|]` — avec des poids à queue lourde, la plupart des niveaux
  de code sont gaspillés sur de rares aberrants. OmniQuant apprend un **facteur de coupe**
  `γ∈(0,1]` par canal qui **rétrécit** la plage à `γ·max|w|`, échangeant un peu d'erreur de
  coupe sur les aberrants contre des pas bien plus fins sur le gros des poids — trouvé ici
  par une recherche déterministe sur une grille qui **inclut `γ=1`** (RTN pur). Oracle
  honnête : erreur de reconstruction **< RTN** sur poids à queue lourde (≥1 canal coupe
  réellement) + **jamais pire** que RTN (γ=1 est candidat) + déterminisme bit-exact.
  Rejoint la famille de quantification (GPTQ, AWQ, NF4, SqueezeLLM, SpQR, KVQuant, LLM.int8).
- **S4 (S4D) — espace d'états structuré diagonal** (`nn::nd_layers::s4_scan`/`NdS4`,
  Gu et al. 2022, roadmap #51) : SSM **linéaire invariant dans le temps** (contrairement
  au `selective_scan` de Mamba dont les matrices dépendent de l'entrée) — `A` diagonal,
  `B`/`C`/`Δ` sont des **paramètres fixes** ; discrétisation `Ā=exp(Δ⊙A)`, `B̄=Δ⊙B`,
  récurrence `h_t=Ā⊙h_{t−1}+B̄⊙x_t` (état `(d,n)`) déroulée sur la tape, lecture
  `y_t=Σ_n C⊙h_t`. Init **HiPPO** diagonale (S4D-Lin) `A[:,j]=−(j+1)`, `A<0` contractif.
  La couche `NdS4` ajoute projections d'entrée/sortie + skip gaté `D⊙x`. Oracle :
  **gradient check** (différences finies vs analytique sur x, a_log, B, C, log_dt) +
  entraînement (MSE↓ vers une cible) + déterminisme bit-exact. Couche de bibliothèque.
- **AI² / zonotopes — domaine abstrait pour la vérification** (`nn::ibp::Zonotope`/
  `IbpMlp::certify_zonotope`, Gehr et al. 2018, roadmap #29) : propagation par
  **zonotopes** (centre + générateurs, `{c+Σεᵢgᵢ : εᵢ∈[−1,1]}`) — les couches affines
  sont **exactes**, la ReLU est relaxée façon **DeepZ** (`y=λx+μ±μ`, `λ=u/(u−l)`,
  `μ=−λl/2`, un générateur frais par neurone instable). Les `εᵢ` partagés capturent les
  **corrélations** linéaires que les intervalles perdent. Oracle honnête : affine exacte
  (= forward intervalle) + **soundness** (des milliers de points échantillonnés dans la
  boîte d'entrée tombent dans la boîte zonotope d'un MLP ReLU 3 couches) + **plus serré
  qu'IBP sous corrélation** (réseau `relu(x)−relu(x)` ≡ 0 : zonotope `[−0,5;0,5]` vs IBP
  `[−1;1]`, les deux sains). Étend `nn::ibp` (IBP #1, CROWN #2) ; affiché dans la CLI
  `certify` à côté d'IBP et CROWN.
- **EAGLE — décodage spéculatif au niveau features** (`nn::nd_decoder::EagleHead`/
  `generate_eagle`, Li et al. 2024, roadmap #62) : là où Medusa prédit des *tokens*
  futurs, EAGLE brouillonne au niveau **feature** — une tête légère mappe
  `(feature_t, embed(token_{t+1})) → feature_{t+1}`, et la tête LM **gelée** transforme
  la feature prédite en token ; chaînée, elle donne un brouillon **autorégressif**
  vérifié par une passe (préfixe accepté + correction greedy). `NdDecoderLM` expose
  `token_embedding`/`head_logits`/`d_model` ; `EagleHead::train` ajuste la tête par MSE
  sur les features du modèle gelé. Oracle honnête : sortie **exactement = greedy** pour
  une tête **quelconque** (vérification) + déterminisme + tête **entraînée** ⇒ ≥1 bloc
  accepte >1 token (forwards < 2·n) en restant exact. Couche de bibliothèque.
- **Medusa — décodage à têtes multiples** (`nn::nd_decoder::MedusaHeads`/`generate_medusa`,
  Cai et al. 2024, roadmap #61) : accélère le décodage en attachant au modèle de base
  des **têtes supplémentaires** (tête `j` prédit le token à `+j+2` depuis l'état caché),
  qui produisent un **brouillon multi-token d'un seul forward** ; une passe de
  vérification accepte le plus long préfixe correspondant à l'argmax du modèle puis
  commet un token de correction/bonus. `NdDecoderLM` expose désormais
  `forward_hidden`/`forward_with_hidden` (état caché post-LayerNorm) ; `MedusaHeads::train`
  entraîne les têtes sur les états cachés du modèle **gelé**. Oracle honnête : sortie
  **exactement = greedy** pour des têtes **quelconques** (même aléatoires — la vérification
  garantit l'exactitude) + déterminisme + têtes **entraînées** ⇒ au moins un bloc accepte
  >1 token (forwards < 2·n) tout en restant exact. Couche de bibliothèque.
- **PagedAttention — KV-cache paginé** (`nn::paged_attention::PagedKvCache`, Kwon et al.
  / vLLM 2023, roadmap #63) : le cache clés/valeurs du décodage est découpé en **blocs**
  de taille fixe tirés d'un pool partagé, adressés indirectement par une **table de
  blocs** (comme la pagination mémoire) ⇒ quasi zéro fragmentation. `append` remplit les
  blocs à la demande, `gather_keys/values` reconstruit le cache contigu, et `attention`
  fait le produit scalaire softmax en indexant clés/valeurs **à travers la table**.
  Oracle honnête : avec des blocs **leurres** interleavés (layout physique non
  séquentiel), le gather est **bit-identique** aux vecteurs insérés et l'attention
  paginée est **bit-identique** à l'attention sur cache contigu (même ordre
  arithmétique) — la pagination est prouvée sans coût numérique ; + comptabilité des
  blocs (`⌈len/bloc⌉`) et cas vide + déterminisme. Couche de bibliothèque (nouveau module).
- **DoRA — adaptation low-rank décomposée poids** (`nn::dora::DoraLinear`, Liu et al.
  2024, roadmap #73) : PEFT qui décompose un poids gelé `W₀` en **magnitude** (vecteur
  par colonne `m`) × **direction** (normalisée), la direction étant pilotée par une
  mise à jour low-rank LoRA `BA` : `W' = m ⊙ (W₀+BA)/‖W₀+BA‖_col`. Seuls `m`, `A`, `B`
  s'entraînent. Backward de la normalisation par colonne en **forme close** (`u=V/‖V‖`,
  `∂L/∂V=(m/‖V‖)(gw−u·s)`, `∂L/∂m=s`). Oracle honnête : init `B=0, m=‖W₀‖_col` ⇒ poids
  effectif **= W₀ exactement** (l'adaptation part de la fonction pré-entraînée) +
  **gradient check** (différences finies centrales vs analytique, params génériques) +
  récupère une cible générée par DoRA (perte ÷100 par descente de gradient) +
  déterminisme bit-exact. Couche de bibliothèque (nouveau module).
- **GaLore — projection low-rank des gradients** (`nn::nd_optim::NdGalore`/
  `galore_subspace`, Zhao et al. 2024, roadmap #48) : optimiseur à **mémoire
  réduite** — pour un paramètre matriciel, le gradient `G` est projeté sur son
  propre sous-espace dominant rang-`r` `P` (top-`r` vecteurs singuliers gauches via
  `jacobi_eigenvectors`, rafraîchi tous les `update_gap` pas), Adam tourne sur le
  petit gradient projeté `PᵀG` puis l'update est remonté par `P`. Les états passent
  de `m×n` à `rank×max(m,n)` ; les vecteurs retombent sur Adam. Oracle honnête :
  `P` **orthonormal** (`PᵀP=I`) et projection **orthogonale optimale** (identité de
  Pythagore `‖G−PPᵀG‖²=‖G‖²−‖PᵀG‖²`, erreur décroissante en `r`, nulle au rang
  plein) + gradient **bas-rang reconstruit exactement** (sous-rang ⇒ résidu) +
  **convergence sur une cible bas-rang** avec état compressé `2×4` (≠ `4×4`) +
  sous-rang ne l'atteint pas + déterminisme bit-exact. Rejoint la famille
  d'optimiseurs ; CLI `lm --opt galore`.
- **YaRN — extension de contexte RoPE** (`nn::yarn`, Peng et al. 2023, roadmap #60) :
  étend le contexte utilisable d'un modèle RoPE d'un facteur `s` par interpolation
  **NTK-by-parts** — `yarn_frequencies` garde intactes les dimensions **haute
  fréquence** (`r_p>β` ⇒ ordre local préservé), interpole pleinement les **basses
  fréquences** (`r_p<α` ⇒ `θ_p→θ_p/s`), avec une rampe linéaire entre les deux
  (`θ'_p=θ_p·((1−γ)/s+γ)`). `rope_apply_freqs`/`rope_yarn` appliquent la rotation
  (convention emboîtée identique à la RoPE existante de `autodiff::nd`) ;
  `yarn_attention_scale` donne la température `0.1·ln(s)+1`. Oracle honnête :
  **propriété de position relative** `⟨rope(q,m),rope(k,n)⟩=g(m−n)` préservée malgré
  les fréquences modifiées + l'angle d'une dimension basse fréquence à la longueur
  **étendue** `s·L` revient **exactement** à sa valeur d'entraînement à `L` (alors
  que la RoPE simple explose) + bornes NTK-by-parts (haute fréquence inchangée, basse
  = `θ/s`, rampe monotone) + `scale=1` ≡ RoPE simple + déterminisme. Couche de
  bibliothèque (primitive positionnelle, pas de CLI).
- **Learn then Test (LtT)** (`nn::conformal::learn_then_test`/`hoeffding_pvalue`,
  Angelopoulos et al. 2021, roadmap #37) : contrôle **distribution-free** de
  **risques multiples arbitraires** (non emboîtés) par tests d'hypothèses. Chaque
  configuration `λ` d'une grille devient une **p-value de Hoeffding** pour
  `H₀: R(λ) > α` (`p = exp(−2n(α−R̂)₊²)`, super-uniforme sous le null), puis
  correction **familiale de Bonferroni** au niveau `δ` : on retient les `λ` avec
  `p ≤ δ/m`. Garantit que, avec proba `≥ 1−δ`, **toute** config retenue vérifie
  `R(λ) ≤ α` (FWER `≤ δ`) — **sans** hypothèse de monotonie (contrairement à RCPS
  #36). Oracle honnête : FWER vérifié **par simulation** (toutes les configs sur
  la frontière `R=α` ⇒ FWER mesuré `≤ δ`, vs sélection naïve qui échoue ~toujours)
  + puissance (les configs sûres sont retenues, les non-sûres rejetées) +
  déterminisme. Couche de bibliothèque.
- **Comptable RDP (Rényi DP)** (`dp::gaussian_rdp`/`rdp_to_dp`/`rdp_gaussian_epsilon`,
  Mironov 2017, roadmap #78) : comptabilité de budget de confidentialité par
  **Rényi-DP**, plus serrée et plus principielle que la composition `(ε,δ)` naïve.
  RDP du mécanisme gaussien `RDP(α)=α/(2σ²)` (additif en composition), conversion
  Mironov `ε=RDP(α)+ln(1/δ)/(α−1)` (le `α−1` est ce qui la rend serrée), optimisée
  sur une grille d'ordres α. Renforce le DP-SGD existant (#19). Oracle : RDP et
  conversion exactes (formes closes) + `ε` **bien en dessous** de la composition
  linéaire basique (qui paie une pénalité ~√étapes) + monotonie (plus d'étapes ⇒ ε
  plus grand ; plus de bruit ⇒ ε plus petit). Couche de bibliothèque.
- **Watermark pour LLM** (`nn::watermark`, Kirchenbauer et al. 2023, roadmap #79) :
  filigrane statistique rendant le texte généré **auditable sans accès au modèle**.
  Le token précédent seede une partition du vocabulaire en liste **verte** (fraction
  γ) / rouge ; `apply_green_bias` ajoute `δ` aux logits verts pour orienter la
  génération. Le détecteur, qui ne connaît que le seed et γ, recompte les tokens
  verts : un texte filigrané en contient bien plus que la fraction γ attendue par
  hasard, ce qu'un **test z** `(g−γn)/√(nγ(1−γ))` (`detect_z`) signale par une
  p-value minuscule, tandis que le texte naturel score `z≈0`. Tout est un hash
  déterministe de `(seed, prev, token)`. Oracle : fraction verte ≈ γ + biais
  appliqué aux seuls tokens verts + texte filigrané détecté (z≫8) vs naturel (z≈0)
  + un **mauvais seed ne détecte pas** (pas de fausse provenance) + déterminisme.
  Couche de bibliothèque.
- **DeepONet — apprentissage d'opérateurs** (`nn::deeponet::DeepONet`, Lu et al.
  2021, roadmap #76) : apprend un **opérateur** `G : u ↦ G(u)` (fonction →
  fonction) via une factorisation **branch × trunk** `G(u)(y) ≈ Σ_k b_k(u)·t_k(y)`
  — la branch encode la fonction d'entrée `u` (échantillonnée à des capteurs
  fixes), la trunk encode la position `y`. Variante **POD-DeepONet** (trunk cosinus
  **fixe** `cos(kπy)` + branch **linéaire**) ⇒ ajustement **convexe**, exact pour
  les opérateurs linéaires comme l'**antidérivée** `∫₀^y u`. Oracle : entraîné sur
  certaines fonctions, il approxime l'antidérivée sur des fonctions **non vues** à
  MSE test < 0,01 (≪ prédicteur constant) — la propriété d'apprentissage
  d'opérateurs — + déterminisme. Couche de bibliothèque.
- **Deep Ensembles** (`nn::ensemble::DeepEnsemble`, Lakshminarayanan, Pritzel &
  Blundell 2017, roadmap #40) : incertitude prédictive par **ensemble seedé**. N
  petits MLP ReLU (`1→hidden→1`) entraînés sur la tape N-D avec `NdAdam`, chacun
  seedé différemment ; `predict(x)` renvoie `(moyenne, écart-type)` — l'estimation
  ponctuelle et son **incertitude épistémique** (désaccord entre membres). Oracle :
  la MSE de la moyenne d'ensemble est ≤ la MSE moyenne des membres (Jensen) +
  l'écart-type est **bien plus grand hors-distribution** (loin de la plage
  d'entraînement) qu'en-distribution + déterminisme bit-exact. Couche de
  bibliothèque.
- **LLM.int8()** (`quantization::int8_mixed_matmul`, Dettmers et al. 2022, roadmap
  #71) : matmul mixte int8/fp32. Les activations des transformeurs ont quelques
  **colonnes de features outliers** de très grande magnitude ; les quantifier en
  int8 avec le reste gonfle l'échelle et écrase la résolution des features
  normales. LLM.int8() garde ces colonnes (et les lignes de W correspondantes) en
  **pleine précision** et quantifie le reste en **int8** :
  `X·W = X_normal·W_normal (int8) + X_outlier·W_outlier (fp32)`. Une colonne est
  outlier si un `|X[i,j]|` dépasse le seuil (défaut 6.0). Oracle : sur des
  activations à colonnes outliers, l'erreur vs fp est **< 0,5×** celle de l'int8
  simple ; sans outliers, se réduit à l'int8 pur ; déterminisme. Couche de
  bibliothèque.
- **RCPS — Risk-Controlling Prediction Sets** (`nn::conformal::hoeffding_ucb` +
  `rcps_select`, Bates et al. 2021, roadmap #36) : là où le conformal contrôle la
  *couverture*, RCPS contrôle un **risque borné quelconque** (perte dans [0,1] :
  taux de faux négatifs, non-couverture, …) avec une garantie **haute probabilité
  (PAC)**. Pour une famille de prédicteurs `C_λ` à risque non-croissant en λ, RCPS
  choisit le plus petit `λ̂` dont la **borne de concentration de Hoeffding** sur le
  risque est ≤ α (pour λ̂ et tout λ plus grand) ⇒ `R(λ̂) ≤ α` avec proba ≥ 1−δ.
  Oracle : la borne dépasse la moyenne du bon écart + sélection exacte (cas
  calculé) + sur données fraîches le risque empirique reste ≤ α (borne
  conservatrice). Couche de bibliothèque.
- **Prodigy** (`nn::nd_optim::NdProdigy` + `ProdigyConfig`, Mishchenko & Defazio
  2023, roadmap #46) : un Adam **sans learning-rate** (« parameter-free »). Il
  estime en ligne la distance `d ≈ ‖x₀ − x*‖` à la solution — via la corrélation
  globale `⟨g, x₀ − x⟩` accumulée — et l'utilise comme taux effectif, partant d'un
  `d₀ = 1e-6` minuscule qui croît jusqu'à l'échelle du problème. `d`, le numérateur
  `r` et la norme du dénominateur sont des scalaires **globaux** sur tous les
  paramètres. Oracle : `d` s'adapte à l'échelle de la distance (sans réglage de lr)
  + la perte quadratique chute fortement + déterminisme bit-exact. CLI :
  `scirust lm --opt prodigy` (8 langues).
- **KVQuant** (`quantization::kvquant_kv`, Hooper et al. 2024, roadmap #68) :
  quantification du **KV-cache** d'attention à la granularité qui épouse sa
  structure d'outliers — **clés per-canal** (les outliers des clés se concentrent
  par colonne de features) et **valeurs per-token** (par ligne). Bien plus fidèle
  qu'une échelle per-tensor unique, qu'une poignée de gros canaux de clés
  domineraient (écrasant la résolution de tous les autres). Oracle : sur des clés à
  outliers de canal, l'erreur de la sortie d'attention vs fp est **< 0,6×** celle
  de la quantif per-tensor ; le per-canal résout les petites colonnes (<0,1× erreur)
  là où le per-tensor échoue ; déterminisme. Couche de bibliothèque.
- **ALiBi — Attention with Linear Biases** (`nn::nd_layers::alibi_slopes` +
  `alibi_bias` + `NdMultiHeadAttention::with_alibi`, Press, Smith & Lewis 2022,
  roadmap #59) : remplace les positions apprises/rotatives par un **biais statique
  linéaire en distance** ajouté aux scores d'attention — pour la requête `i` et la
  clé `j ≤ i`, `−penteₕ·(i−j)`, avec des pentes par tête en suite géométrique
  `2^(−8h/H)`. Aucune position apprise ⇒ **extrapolation en longueur**. Branché dans
  `NdMultiHeadAttention` (builder `with_alibi`, inclut le masque causal). Oracle :
  pentes géométriques (ratio `2^(−8/H)`) + biais linéaire/causal/Toeplitz + poids
  softmax décroissant avec la distance (exactement `∝ exp(−pente·dist)`) + forward
  d'attention déterministe.
- **ACI — Adaptive Conformal Inference** (`nn::conformal::AdaptiveConformal`, Gibbs
  & Candès 2021, roadmap #38) : conformal **en ligne** robuste à la **dérive de
  distribution**. Le conformal statique perd silencieusement sa couverture sous
  changement de distribution ; ACI suit un niveau effectif `αₜ` et le corrige après
  chaque observation par rétroaction `αₜ₊₁ = αₜ + γ(α − errₜ)`, ce qui pilote le
  taux d'erreur long-terme vers `α` (couverture vers `1−α`) pour **tout** flux de
  scores. Avec une fenêtre glissante de scores récents, la couverture reste ≈ 1−α
  à travers les changements là où le conformal statique s'effondre. Oracle : règle
  de mise à jour de `αₜ` exacte (cas calculé) + couverture ≈ 1−α maintenue sous
  changement de variance (vs conformal statique qui chute) + déterminisme. Couche
  de bibliothèque. Complète CQR/APS/RAPS dans le pilier conformal.
- **KAN — Kolmogorov-Arnold Networks** (`nn::kan::KanLayer`, Liu et al. 2024 ;
  base RBF de FastKAN, Li 2024 ; roadmap #77) : activations **apprenables sur les
  arêtes** plutôt que sur les nœuds — `y_j = Σᵢ φᵢⱼ(xᵢ)` avec chaque `φ` une somme
  de RBF gaussiennes (grille fixe) + un terme de base `SiLU`. La sortie est
  **linéaire dans les coefficients**, donc l'ajustement est un problème de moindres
  carrés **convexe** résolu par descente de gradient déterministe. Oracle : une
  seule couche KAN ajuste la cible additive non-linéaire `sin(2x₀)+x₁²` à MSE<0,02
  — bien en dessous du meilleur modèle linéaire (qui ne peut représenter sin/carré)
  ; base RBF localisée ; déterminisme bit-exact. Couche de bibliothèque (variante
  RBF/FastKAN, pas les B-splines du papier original).
- **RWKV time-mixing (WKV)** (`nn::nd_layers::rwkv_wkv` + `NdRwkv`, Peng et al.
  2023, roadmap #53) : opérateur **WKV** — attention linéaire récurrente à
  **décroissance temporelle exponentielle par canal** `decay ∈ (0,1)` plus un
  **bonus** pour le token courant, normalisée (numérateur/dénominateur), déroulée
  en temps linéaire sur la tape. A nécessité un nouvel op autograd **`div`**
  (division élémentaire, gradient `∂a=g/b`, `∂b=−g·a/b²`, gradient-checké). La
  couche `NdRwkv` ajoute une **réception** `r=σ(W_r·x)` qui gate la sortie, avec
  decay/bonus par canal apprenables. Oracle : la récurrence sur tape **≡ la
  formule de somme pondérée explicite** + gradient check (k, v, decay, bonus) +
  entraînement (MSE↓) + déterminisme bit-exact. CLI : `scirust rwkv` (8 langues).
- **GloRo — robustesse certifiée par Lipschitz** (`nn::lipschitz`, Leino, Wang &
  Fredrikson 2021, roadmap #32) : `spectral_norm` (norme spectrale par power
  iteration déterministe), `spectral_normalize` (couche **1-Lipschitz** contrainte)
  et `GloroClassifier` (classifieur linéaire à **rayon de robustesse L2 prouvé**
  `marge/(√2·‖W‖₂)`, sans recherche ni échantillonnage ; le `√2` vient de la
  Lipschitz `≤ √2·L` de la marge `f_A−f_B`). Oracle : normes spectrales connues
  (diagonale, rectangulaire) ; norme ≈ 1 après normalisation ; rayon **sain** (la
  pire perturbation à ce rayon ne bascule pas la prédiction) **et conservateur**
  (≤ distance exacte à la frontière la plus proche) ; déterminisme. Couche de
  bibliothèque. Complète le pilier certifiable : IBP, CROWN, smoothing, GloRo.
- **Randomized Smoothing — robustesse L2 certifiée** (`nn::smoothing::SmoothedClassifier`
  + `clopper_pearson_lower` + `inv_normal_cdf`, Cohen, Rosenfeld & Kolter 2019,
  roadmap #27) : transforme tout classifieur en un classifieur **lissé** sous bruit
  gaussien `N(0,σ²I)`, avec un **rayon de robustesse L2 prouvé** `σ·Φ⁻¹(pₐ)`. La
  probabilité de la classe top `pₐ` est minorée par **Clopper-Pearson** (beta
  incomplète régularisée `betai`/`lgamma`, exacte) ; `Φ⁻¹` par l'approximation
  rationnelle d'Acklam. Oracle : pour un classifieur **demi-espace** le rayon
  certifié **égale la distance exacte à la frontière** (indépendant de σ) +
  soundness/abstention au bord + déterminisme + valeurs repères de `Φ⁻¹`/`betai`/
  Clopper-Pearson. CLI : `scirust certify` affiche désormais IBP/CROWN
  (déterministe) **et** smoothing (probabiliste).
- **SpQR — Sparse-Quantized Representation** (`quantization::SpqrOutliers`,
  Dettmers et al. 2023, roadmap #67) : l'erreur de quantification est à **queue
  lourde** — une petite fraction de poids « outliers » concentre l'essentiel de
  l'erreur. SpQR garde cette fraction (les plus grosses erreurs de quantif dense)
  en **pleine précision** (canal creux) et quantifie le reste densément, donc ~1 %
  d'outliers retire une grande part de l'erreur pour un faible surcoût mémoire.
  Oracle : sur poids gaussiens + outliers injectés, garder 1 % des poids divise
  l'erreur quadratique par > 3 ; reconstruction exacte des outliers ; déterminisme.
  Couche de bibliothèque (les scales groupés bi-niveaux du papier sont orthogonaux).
- **SqueezeLLM** (`quantization::SqueezeLlmCodebook` + `weighted_quant_error`, Kim
  et al. 2023, roadmap #66) : quantification **non-uniforme** des poids par
  **k-means pondéré par la sensibilité** (proxy de la diagonale de la Hessienne)
  — un codebook de `2^bits` centroïdes placés là où ils réduisent le plus la
  *perte*, et non là où les poids sont denses. Init déterministe (quantiles) +
  itérations de Lloyd pondérées. Oracle : erreur de quantification pondérée
  **strictement < round-to-nearest uniforme** (poids gaussiens, 3 bits, < 0,85×) +
  round-trip exact sur les valeurs du codebook + déterminisme. Couche de
  bibliothèque (la branche « sparse » outliers n'est pas modélisée).
- **APS / RAPS — ensembles de prédiction adaptatifs** (`nn::conformal::AdaptivePredictionSets`,
  Romano, Sesia & Candès 2020 ; Angelopoulos et al. 2021 ; roadmap #34/#35) :
  conformal **classification** par score cumulatif `s(x,c)` = masse de toutes les
  classes au moins aussi probables que `c`. Set `{c : s(x,c) ≤ q̂}` ⇒ couverture
  marginale sans distribution ≥ 1−α avec **taille adaptative** (entrée confiante →
  petit ensemble, ambiguë → grand). **RAPS** ajoute `λ·max(0, rang−k_reg)` au
  score (`calibrate_raps`) pour rogner les classes peu probables et produire des
  ensembles **plus petits** à couverture égale. Oracle : score cumulatif exact
  (cas main) + couverture sur données fraîches + adaptativité (facile vs ambigu) +
  RAPS < APS en taille moyenne + déterminisme. Couche de bibliothèque (comme
  `ConformalClassifier`).
- **CQR — Conformalized Quantile Regression** (`nn::conformal::ConformalQuantileRegressor`,
  Romano, Patterson & Candès 2019, roadmap #33) : conformalise un régresseur de
  **quantiles** pour produire des intervalles **adaptatifs** (hétéroscédastiques)
  à couverture garantie. Score signé `Eᵢ = max(q_lo(xᵢ)−yᵢ, yᵢ−q_hi(xᵢ))`,
  correction finie `Q` (quantile conformal des `Eᵢ`, réutilise `conformal_quantile`),
  intervalle `[q_lo(x)−Q, q_hi(x)+Q]` — largeur **variable selon x** là où le
  split-conformal symétrique est de largeur constante (`Q` peut être négatif et
  resserrer une bande trop large). Oracle : sémantique exacte du score (cas
  calculé à la main) + couverture marginale ≥ 1−α sur données fraîches +
  **adaptativité** (intervalles bien plus larges en région à fort bruit) +
  déterminisme. CLI : `scirust conformal` montre désormais split **et** CQR.
- **SAM — Sharpness-Aware Minimization** (`nn::nd_optim::NdSam` + `SamConfig`,
  Foret et al. 2021, roadmap #47) : optimiseur **à deux phases** qui minimise la
  perte du *pire cas* dans une boule de rayon ρ (biais vers les minima plats).
  `ascent` perturbe les poids vers `θ + ρ·g/‖g‖` (norme **globale** du gradient) ;
  `descent` restaure θ et fait un pas SGD avec le gradient **au point perturbé**.
  Deux gradients par pas ⇒ hors de la boucle `lm --opt` à gradient unique (couche
  de bibliothèque). Oracle : perturbation = `ρ·g/‖g‖` avec `‖ε‖ = ρ` + convergence
  sur quadratique (bande ∝ lr·ρ) + déterminisme.
- **Shampoo** (`nn::nd_optim::NdShampoo` + `ShampooConfig` + `inverse_pth_root`,
  Gupta/Koren/Singer 2018, roadmap #41) : préconditionneur **Kronecker** structuré
  — pour une matrice de poids, maintient les deux facteurs `L = E[GGᵀ]`,
  `R = E[GᵀG]` et avance par l'update préconditionné
  `W ← W − lr·L^(−1/4) G R^(−1/4)`. Les racines inverses des matrices viennent
  d'une décomposition de Jacobi (`inverse_pth_root`, réutilise
  `jacobi_eigenvectors`), cachées et rafraîchies tous les `precond_freq` pas.
  Paramètres non-matriciels : Adagrad diagonal. Oracle : `A^(−1/2)²·A ≈ I` +
  convergence sur quadratique matricielle + repli Adagrad + déterminisme. CLI :
  `scirust lm --opt shampoo` (11e valeur `--opt`).
- **Adafactor** (`nn::nd_optim::NdAdafactor` + `AdafactorConfig`, Shazeer & Stern
  2018, roadmap #42) : optimiseur à **moments du 2e ordre factorisés** — pour une
  matrice de poids, ne stocke que les sommes **ligne** et **colonne** du carré du
  gradient (`rows + cols` nombres au lieu de `rows·cols`) et reconstruit la rank-1
  `V[i,j] = R[i]·C[j]/ΣR` (mémoire sous-linéaire). Update `G/√V` **clippé en RMS** ;
  planning `β2ₜ = 1 − t^(−0.8)`. Paramètres non-matriciels : 2e moment complet
  (RMSProp). Oracle : reconstruction rang-1 **exacte** quand `G²` est rang-1 +
  convergence (bande) + chemin matriciel factorisé qui réduit `½‖W−T‖²` +
  déterminisme. CLI : `scirust lm --opt adafactor` (10e valeur `--opt`).
- **NF4** (`quantization::nf4_quantize`/`nf4_dequantize` + `NF4_LEVELS`, QLoRA,
  Dettmers et al. 2023, roadmap #74) : type 4-bit **NormalFloat** — 16 niveaux qui
  sont les **quantiles d'une normale** (échelle absmax par bloc). Optimal pour des
  poids gaussiens. Oracle : erreur de reconstruction **< int4 uniforme** sur des
  poids gaussiens (Box-Muller seedé) + round-trip exact sur les niveaux +
  déterminisme. Couche de bibliothèque.
- **BitNet b1.58** (`quantization::ternary_quantize` + `ternary_matmul`, Ma et al.
  2024, roadmap #69) : quantification **ternaire** des poids vers `{−1,0,+1}`
  (échelle absmean, ~1,58 bit/poids, ~20× plus compact) ; **matmul sans
  multiplication** (addition / soustraction / skip selon le signe). Oracle :
  `ternary_matmul` = la forme somme-de-signes **bit-exact** et = le produit
  déquantifié à la réassociation flottante près. CLI : `scirust bitnet` (en
  direct : max erreur 1,4e-6 vs déquant, 986/4096 poids nuls). Déterministe.
- **HGRN** (`nn::nd_layers::hgrn` + `NdHgrn`, Qin et al. 2023, roadmap #58) : RNN
  linéaire à intégration leaky par canal (`hₜ = fₜ⊙h_{t-1} + (1−fₜ)⊙cₜ`), porte
  d'oubli **bornée inférieurement** `f = lb + (1−lb)·σ(·)` (la borne `lb` fixe
  l'horizon mémoire minimal). Pas d'état matriciel ; déroulé sur la tape. Tests :
  match référence + gradient check (c,f) + entraînement + déterminisme. CLI :
  `scirust hgrn` (en direct : MSE 27.37 → 4.59).
- **GLA — Gated Linear Attention** (`nn::nd_layers::gated_linear_attention` +
  `NdGla`, Yang et al. 2024, roadmap #55) : attention linéaire **gatée** — porte
  d'oubli par canal **dépendante de l'entrée** `αₜ=σ(·)`
  (`S_t = diag(αₜ)·S_{t-1} + kₜᵀvₜ`, `o_t = q_t·S_t`), déroulée sur la tape.
  Tests : match d'une référence Vec + gradient check (q,k,v,α) + entraînement +
  déterminisme. CLI : `scirust gla` (en direct : MSE 27.16 → 0.0000).
- **RetNet** (`nn::nd_layers::retention` + `NdRetention`, Sun et al. 2023,
  roadmap #54) : couche de **rétention** — attention linéaire récurrente à
  décroissance `γ` (`S_t = γ·S_{t-1} + kₜᵀvₜ`, `o_t = q_t·S_t`), déroulée sur la
  tape. **Oracle de dualité** : la forme récurrente **égale** la forme parallèle
  `(QKᵀ⊙D)V` (`D_{nm}=γ^{n-m}`), testé ; + gradient check (q,k,v) + entraînement
  + déterminisme. CLI : `scirust retnet` (en direct : MSE 24.63 → 0.0002).
- **LAMB** (`nn::nd_optim::NdLamb`, You et al. 2020, roadmap #43) : Adam à
  **confiance par couche** — direction Adam `r` remise à l'échelle par
  `‖θ‖/‖r‖` par tenseur. CLI `lm --opt lamb`. Tests : convergence (bande ∝ lr,
  car la norme de pas ≈ lr·‖θ‖) + déterminisme.
- **Adan** (`nn::nd_optim::NdAdan`, Xie et al. 2022, roadmap #49) : momentum de
  **Nesterov adaptatif** — 3 EMA (gradient `m`, différences `v`, terme
  look-ahead au carré `n`) ; `θ ← (θ − η⊙(m+(1−β2)v))/(1+lr·wd)`. CLI
  `lm --opt adan`. Tests : convergence quadratique + déterminisme.
- **LoRA** (`nn::nd_layers::LoraLinear`, Hu et al. 2022, roadmap #72) : adaptation
  **low-rank** — poids de base `W` **gelé** + mise à jour `ΔW = (α/r)·A·B` ; seuls
  `A` (`in×r`) et `B` (`r×out`) sont entraînés (`r·(in+out)` paramètres au lieu de
  `in·out`). `B=0` à l'init ⇒ la couche **vaut exactement la base**. Couche de la
  tape N-D. Tests : init = base, **gradient check** sur `A` et `B`, `parameters()`
  n'expose que `A`,`B`.
- **Temperature scaling / calibration** (`nn::calibration`, Guo et al. 2017,
  roadmap #39) : `temperature_scale` (recherche golden-section sur la NLL) +
  `expected_calibration_error` + `nll`. Recalibration post-hoc des probabilités
  **sans changer l'accuracy** (l'argmax est invariant à `T>0`). Déterministe. CLI :
  `scirust calibrate` (en direct : ECE 0.29 → 0.004, −98,5 %, T=2,70). Tests : ECE
  baisse + accuracy inchangée + déterminisme.
- **Lookahead** (`nn::nd_optim::NdLookahead`, Zhang et al. 2019, roadmap #45) :
  optimiseur **wrapper** poids lents/rapides autour d'Adam — `k` pas rapides puis
  `φ ← φ + α(θ − φ) ; θ ← φ`. Déterministe. CLI : `scirust lm --opt lookahead`.
  Tests : convergence quadratique, déterminisme bit-à-bit. (1er du pool de
  candidats Tier 8-14.)
- **PINN** (`nn::pinn` : `Pinn1D`, `solve_harmonic`, Raissi et al. 2019,
  roadmap #17) : réseau **physics-informed** — la **physique est dans la loss**
  via un résidu de PDE aux points de collocation + conditions aux limites.
  Résout le problème aux limites `u'' = −u`, `u(0)=0`, `u(π/2)=1` (solution
  exacte `sin x`) ; la dérivée seconde `u''` est prise par **différences finies
  dans l'entrée** (les évaluations `u(x±h)` passent par les *mêmes* paramètres
  dans un seul graphe forward), donc le gradient par rapport aux paramètres reste
  exact (autodiff inverse) et déterministe. Vérifié contre la solution analytique
  (erreur max ≈ 0,004). CLI : `scirust pinn`.
- **Mamba** (`nn::nd_layers::selective_scan` + `NdMamba`, Gu & Dao 2023,
  roadmap #18) : **selective scan** S6 — état-espace à matrice `A` diagonale et
  paramètres `Δ, B, C` **dépendants de l'entrée** (sélectifs) ; discrétisation
  par maintien d'ordre zéro `Ā = exp(Δ·A)`, `B̄x = Δ·B·x` ; récurrence
  déterministe linéaire-temps `h_t = Ā_t ⊙ h_{t-1} + B̄x_t`, `y_t = h_t·C_t`,
  déroulée sur la tape N-D. Nouvel op autograd `NdVar::exp` (gradient-checké).
  Init S4D-réelle (`A[:,j] = −(j+1)`), saut `D⊙x`. Tests : `selective_scan` match
  une référence Vec, gradient check (x, Δ, A, B, C), couche entraîne (MSE↓) +
  déterminisme. CLI : `scirust mamba`.
- **DeltaNet** (`nn::nd_layers::delta_rule` + `NdDeltaNet`, Yang et al. 2024,
  roadmap #25) : couche d'**attention linéaire récurrente** à règle delta
  (`S_t = S_{t-1} + β_t(v_t − S_{t-1}k_t)k_tᵀ`, `o_t = S_t q_t`) — mémoire à poids
  rapides, temps linéaire, causale. La récurrence est **déroulée sur la tape N-D**
  (nouvel op autograd `NdVar::cat0` : concaténation axe 0 + backward par découpe,
  **gradient-checké**), donc les gradients sont exacts et vérifiés par différences
  finies (q, k, v, β). Tests : correspondance avec une référence Vec, gradient
  check, entraînement (MSE↓) + déterminisme bit-à-bit. CLI : `scirust deltanet`.
- **SOAP** (`nn::nd_optim::NdSoap` + `jacobi_eigenvectors`, Vyas et al. 2024,
  roadmap #24) : optimiseur qui exécute **Adam dans la base propre de Shampoo**.
  Pour chaque matrice de poids : facteurs `L = E[GGᵀ]`, `R = E[GᵀG]` (moyenne
  mobile) ; rotation du gradient dans leur base propre (`Ĝ = Q_Lᵀ G Q_R`), Adam
  dans cette base, puis rotation inverse de la mise à jour. Base propre par
  **eigensolveur de Jacobi cyclique** déterministe (`jacobi_eigenvectors`),
  rafraîchie tous les `precond_freq` pas (moments tournés dans la nouvelle base).
  Repli Adam pour les paramètres non matriciels. Déterministe. CLI :
  `scirust lm --opt soap`. Tests : Jacobi diagonalise (orthogonalité +
  reconstruction), convergence sur quadratique matricielle, déterminisme bit-à-bit.
- **AWQ** (`quantization::awq_quantize` + `awq_act_scale` + `AwqResult`, Lin et al.
  2023, roadmap #15) : quantification int8 **consciente des activations** par
  recherche d'échelle. Importance par canal d'entrée `a_j = moyenne|x_:,j|` ;
  facteurs `s_j = a_j^alpha` (normalisés à moyenne géométrique unité) appliqués
  aux poids avant la quantification int8 per-canal, l'équivalence étant préservée
  côté activations ; `alpha` choisi par **grille** sur `[0,1]` (`alpha=0` =
  round-to-nearest) en minimisant l'erreur de sortie pondérée par la calibration.
  CLI : `scirust awq [--seed N] [--samples S] [--grid G]`. Tests : protège les
  canaux saillants → erreur < round-to-nearest (`alpha>0` choisi) + déterminisme
  bit-à-bit. **Complète le volet quantification #15** (SmoothQuant + GPTQ + AWQ).
- **GPTQ** (`quantization::quantize_gptq` + `gptq_hessian`, Frantar et al. 2022,
  roadmap #15) : quantification int8 des poids par **feedback d'erreur d'ordre 2**.
  Hessienne proxy `H = XᵀX` sur des activations de calibration ; inverse par
  Cholesky (en f64, déterministe) ; pour chaque canal de sortie, quantification
  séquentielle des poids d'entrée avec propagation de l'erreur (OBQ/GPTQ, ordre
  naturel) et complément de Schur. Scale symétrique par canal de sortie. CLI :
  `scirust gptq [--seed N] [--samples S] [--damp D]`. Tests : **erreur de
  reconstruction pondérée par la calibration < round-to-nearest** (≈ −85 % sur
  données corrélées) + soundness (jamais pire) + déterminisme bit-à-bit. Complète
  le volet quantification (#15) avec SmoothQuant et l'int8 per-canal.
- **CROWN** (`nn::ibp::crown_bounds`, Zhang et al. 2018, roadmap #2) : bornes de
  sortie d'un MLP ReLU à 1 couche cachée par **relaxation linéaire** +
  back-substitution sur une boîte L∞. Relaxation par neurone : exacte pour les
  neurones stables, chorde supérieure / pente inférieure adaptative pour les
  instables. **Plus serrée qu'IBP** (prouvé par test). CLI : `scirust certify`
  affiche désormais IBP **et** CROWN côte à côte (CROWN certifie la robustesse
  là où IBP échoue). Tests : soundness (échantillonnage de la boîte) + largeur
  CROWN ≤ largeur IBP par sortie.
- **AdEMAMix** (`nn::nd_optim::NdAdEMAMix`, Pagliardini et al. 2024, roadmap #23) :
  Adam à **deux EMA** du gradient (rapide β1 + lente β3 à longue mémoire, mélangées
  par α) ; déterministe. CLI : `scirust lm --opt ademamix`. Tests : convergence
  quadratique (bande), déterminisme bit-à-bit.

### Nettoyé
- Suppression de `scirust-core/src/nn/.legacy/` (**2363 lignes** de code mort) :
  répertoire non câblé dans l'arbre de modules (dotfile, zéro référence),
  superposé par les implémentations réelles `nn::conv2d`/`batch_norm`/`layer_norm`/
  `pool`/`loss`/`transformer`. Conforme au fondamental « code sous src/ câblé et
  testé, sinon archivé ».

### Ajouté — campagne « faire grandir scirust » (suite)
- **Schedule-Free** (`nn::nd_optim::NdScheduleFree`, Defazio et al. 2024, roadmap
  #22) : optimiseur **sans planning de learning-rate** — séquence de base `z`
  (descente), moyenne de Polyak `x` (**point d'évaluation**), gradient pris en
  `y = (1−β)z + βx`. Déterministe. CLI : `scirust lm --opt schedule-free`
  (le point d'éval `x` est chargé avant la prédiction). Tests : convergence sur
  quadratique, déterminisme bit-à-bit.
- **Conformal prediction** (`nn::conformal`, Angelopoulos & Bates 2021, roadmap
  #21) : `conformal_quantile`, `ConformalRegressor`, `ConformalClassifier` —
  ensembles/intervalles de prédiction à **couverture garantie sans hypothèse de
  distribution** (`≥ 1 − α`). Tests : la couverture empirique atteint la cible
  sur des données fraîches (régression *et* classification). CLI : `scirust
  conformal [--seed N] [--alpha A]` (couverture mesurée en direct, ex. 90,8 %
  pour une cible de 90 %). CLI : 41 → 42 commandes.
- **Lot 3 recherche → fonctions** (testées, 8 gates verts ; **14 des 20** items
  de [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md)) :
  - **Muon** (`nn::nd_optim`, Jordan et al. 2024) : optimiseur matriciel —
    momentum puis **orthogonalisation Newton–Schulz** (quintique, sans SVD) de
    la mise à jour des matrices 2-D ; `newton_schulz_orthogonalize` exposé.
    Déterministe. Tests : orthogonalité (déviation ‖A·Aᵀ−I‖ s'effondre), perte
    matricielle, déterminisme.
  - **Wanda** (`pruning::prune_wanda`, Sun et al. 2023) : élagage one-shot par
    `|W|·‖X‖` (poids × norme d'activation), par ligne de sortie — diffère de
    l'élagage par magnitude sur les canaux à activations aberrantes.
  - **SmoothQuant** (`quantization::smoothquant_scales`/`apply_smoothquant`,
    Xiao et al. 2022) : lissage par canal d'entrée qui migre les valeurs
    aberrantes d'activation vers les poids ; **préserve `X·W`**.
- **Lot 2 recherche → fonctions** (3 features de plus, testées, 8 gates verts ;
  **11 des 20** items de [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md)) :
  - **RoPE** (`autodiff::nd`, Su et al. 2021) : op `rope` (rotation par paires,
    backward = rotation inverse) ; gradient-checkée, conservation de norme et
    **propriété de position relative** testées ; branchée via
    `NdMultiHeadAttention::with_rope`.
  - **GQA / MQA** (`nn::nd_layers`, Ainslie et al. 2023) :
    `NdMultiHeadAttention::new_gqa(num_kv_heads, …)` — têtes K/V partagées via le
    broadcast `bmm` (aucune nouvelle op) ; gradient-checkée (GQA et MQA).
  - **Neural ODE** (`nn::neural_ode`, Chen et al. 2018) : `rk4_integrate` +
    `NeuralOde` — backprop **à travers** le solveur RK4 sur la tape N-D (fusion
    solveurs + autograd). RK4 validé (`dy/dt=y → e`), gradient-check à travers
    le solveur, et la dynamique **apprend** (Adam).
- **Feuille de route recherche → fonctions** ([`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md)) :
  20 papers réels traduits en fonctions concrètes, avec statut et effort. Premier
  lot **livré cette session** (testé, 8 gates verts) :
  - **IBP — bornes de sortie certifiées** (`nn::ibp`, Gowal et al. 2018) :
    propagation d'intervalles dans un MLP ReLU → boîte de sortie **prouvée** ;
    `certified_robust` transforme la borne en garantie de classe. Soundness
    testée par échantillonnage (4000 points ∈ boîte certifiée). *Le* pilier « IA
    certifiable » rendu concret.
  - **Réductions reproductibles** (`reproducible`, Demmel & Nguyen) :
    `reproducible_sum`/`_mean`/`_dot` **bit-identiques quel que soit l'ordre /
    le nombre de threads** (tri canonique + expansion exacte de Shewchuk) ;
    survit à l'annulation catastrophique.
  - **Couches LLaMA N-D** (`nn::nd_layers`) : `NdRmsNorm`, `NdSwiGLU` (+ ops
    `rmsnorm`/`sigmoid` gradient-checkées) et `NdLlamaBlock` (Pre-RMSNorm +
    attention causale + SwiGLU) — entraînables, Adam-ready.
  - **Décodage spéculatif exact** (`nn::nd_decoder`, Leviathan/Chen 2023) :
    `generate_speculative` produit **exactement** la sortie greedy de la cible
    pour n'importe quel brouillon, avec moins de forwards ; + `generate_greedy`.
  - **Optimiseurs** (`nn::nd_optim`) : **AdamW** (weight-decay découplé) et
    **Lion** (sign-momentum, déterministe).
- **Commande CLI `lm`** : entraîne un petit LM décodeur causal (tape N-D + Adam)
  sur une séquence de tokens et rapporte la courbe de perte + le rappel exact —
  `scirust lm ["t0,t1,.."] [--seed N] [--steps S] [--lr R]`. Déterministe par
  graine ; expose toute la pile N-D (embeddings, attention causale, gather,
  cross-entropy, Adam) en une commande. CLI : 39 → 40 commandes.
- **Optimiseur Adam N-D, réutilisable et déterministe** (`nn::nd_optim`) :
  `NdAdam` (Kingma & Ba) sur un ensemble ordonné de paramètres. Chaque couche
  expose `parameters() -> Vec<NdParam>` (vue `&mut` des valeurs + index du
  gradient issu de `backward`) ; la composition remonte l'arbre
  (`NdLinear`/`NdEmbedding`/`NdLayerNorm` → attention → bloc → `NdDecoderLM`),
  donc **un seul `opt.step()` met à jour tout le modèle**. Arithmétique f32 en
  ordre fixe ⇒ **bit-à-bit déterministe**. Tests : convergence sur quadratique
  (oracle), déterminisme bit-à-bit, et **le LM décodeur entraîné par Adam via
  `parameters()`** (< 10 % de perte en 150 pas vs 300 en SGD, prédictions
  exactes).
- **Modèle de langage décodeur causal bout-en-bout** (`nn::nd_decoder`) :
  `NdDecoderLM` de style GPT entièrement sur la tape N-D — embedding de tokens
  + embedding positionnel appris → N blocs transformer Pre-LN **causals** →
  LayerNorm final → tête linéaire vers les logits de vocabulaire, entraîné par
  cross-entropy au token suivant. Test phare : **le LM sur-apprend une séquence
  et la reprédit exactement** à chaque position (preuve bout-en-bout que toute
  la pile apprend) ; forward déterministe par graine. `NdEmbedding` (table
  adossée à `gather`) ajoutée comme couche réutilisable.
- **Ops N-D `gather` + `cross_entropy`** (`autodiff::nd`) : `gather(indices)`
  (lookup d'embedding `(vocab, dim) → (n, dim)`, backward scatter-add — les
  indices répétés s'accumulent, les lignes jamais vues gardent un gradient nul)
  et `cross_entropy(targets)` (softmax + NLL moyen **fusionné**, log-sum-exp
  stable, backward `(softmax − onehot)/n`). Gradient-checkées ; sanity
  `logits uniformes → ln(vocab)`.
- **Attention causale N-D** (`NdMultiHeadAttention { causal }`, propagée à
  `NdTransformerBlock`) : masque triangulaire additif (`-1e9` au-dessus de la
  diagonale) avant le softmax — aucune nouvelle op d'autograd. Test de
  **causalité** : perturber le dernier token d'entrée laisse **chaque** sortie
  antérieure bit-à-bit inchangée, tandis que la sortie perturbée bouge.
- **Bloc transformer N-D complet et entraînable** (`nn::nd_layers`) :
  `NdLinear`, `NdMultiHeadAttention`, `NdLayerNorm` (affine γ/β) et
  `NdTransformerBlock` (Pre-LN : `x + Attn(LN(x))`, `x₁ + FFN(LN(x₁))`) sur la
  tape N-D, tous **entraînables** (`sgd_step`). Tests : gradient check
  entrée/couche d'attention/LayerNorm, **un MLP N-D qui apprend** ET **un bloc
  transformer N-D complet qui apprend** (perte < 70 % de l'initiale). Ops
  N-D ajoutées : `bmm`, `softmax`, `transpose_last2`, `reshape`, `permute`,
  `layernorm` — toutes gradient-checkées.
- **`MiniLLM::generate_sampled(&str)`** : génération publique à partir d'une
  chaîne, sampling seedé sur le KV-cache, déterministe ; greedy reproduit
  `generate`.
- **Attention N-D gradient-checkée** : `autodiff::nd` exprime une **attention
  multi-tête complète** `softmax(Q·Kᵀ/√d)·V` sur `(têtes, seq, d)` (ops
  `bmm`/`transpose_last2`/`softmax`/`mul`/`add`/`sub`/`relu`/`sum`), validée
  par gradient check. La tape N-D devient le sur-ensemble capable ; la 2D
  reste le défaut par choix d'architecture (coexistence, cf. GROWTH_PLAN).
- **Sampling seedé** (`nn::sampling`) : température / top-k / top-p pilotés par
  `PcgEngine` seedé → déterministe. `MiniLLM::generate_ids_cached_sampled`
  (génération O(n) à KV-cache avec sampling). Greedy reproduit le chemin argmax.
- **BPE byte-level** (`ByteBpeTokenizer`, style GPT-2) : vocab de base = 256
  octets ⇒ **aucun OOV**, round-trip **lossless** sur tout UTF-8 (accents,
  emoji, scripts inconnus). Déterministe. Exposé en CLI via `bpe --bytes`.
- **LLM bout-en-bout** : décodage KV-cache O(n) (`MiniLLM::generate_ids_cached`,
  `TransformerBlock/Encoder::infer_step`, `PositionalEncoding::encoding_at`)
  **prouvé équivalent** au recalcul complet ; génération découplée du tokenizer
  (`MiniLLM::generate_ids`) → un BPE peut piloter la génération (test
  d'intégration dans `scirust-learning`). Décodage glouton (sampling à venir).
- **CLI `bpe`** : entraîne un tokenizer BPE déterministe sur un corpus
  (documents séparés par `;`), encode/decode, rapporte la taille de vocab et le
  round-trip. Adossé à `scirust-learning` (38 → 39 commandes ; nouveau groupe
  NLP).
- **Matmul par lots N-D** (`NdVar::bmm`) : `(…,m,k)·(…,k,n)→(…,m,n)` avec axes
  batch broadcastés — la capacité que la tape 2D ne sait pas exprimer
  (scores d'attention par tête). Forward + backward gradient-checkés.
- **Autograd N-D (MVP, P2.4)** : `autodiff::nd` — `NdTape`/`NdVar` sur
  `TensorND` (add/mul broadcastés, matmul 2D, relu, sum), à côté de la tape 2D
  de production. Validé par un **gradient check numérique** (différences
  finies vs backward) sur `sum(relu(X·W+b)·V)`.
- **Ops GPU élargies** : kernel elementwise wgpu (add/mul/relu) ; une couche
  entière (matmul → +biais → relu) reste **résidente en VRAM**, validée contre
  l'oracle CPU sur lavapipe.
- **ONNX import** : `import_onnx_json` + `OnnxGraph::weights` — les poids
  font un aller-retour export→import **bit-exact** (format de checkpoint).
- **KV-cache vérifié** : test prouvant que le décodage incrémental
  (`MultiHeadAttention::infer_step`) donne le même dernier token que le forward
  complet — décodage O(n) désormais testé.
- **BPE déterministe** : tie-break par paire (`(count, Reverse(pair))`) — le
  `max_by_key(count)` dépendait de l'ordre d'itération du HashMap ; +5 tests.

### Réparé
- **Revue de code (max-effort) — durcissement** : (1) chemin GPU résident
  (`GpuChain`) : les dimensions dégénérées (`m`/`n`/`k == 0`) faisaient paniquer
  wgpu (buffers de taille nulle) ; gardes ajoutées (placeholder 4 octets,
  dispatch sauté, `download` court-circuité) + test. (2) `scirust ode` :
  `h = 0` provoquait un dépassement de capacité (panique, code 101), `t1 ≤ t0`
  renvoyait silencieusement `y0` (code 0, réponse fausse) et dopri5/rk4
  divergeaient sur de mauvaises bornes ; garde unifiée (`t1 > t0`, `h > 0` fini
  → code 2) + tests. Les autres axes de revue (maths GEMM/transpose, routage
  des gradients Conv2d, `matmul_gpu` av/ar, déterminisme de la réduction
  threadée, restructuration cfg SIMD) ont été tracés à la main : corrects.
- **`scirust-rustc-driver` recompile (P2.3, infra)** : le driver (exclu du
  workspace, `rustc_private`) ne compilait plus sur la nightly courante
  (`get_attrs` renvoie un itérateur, plus un slice). Réparé + warning-clean ;
  job CI informatif `rustc-driver` (continue-on-error) pour rendre la dérive
  d'API future visible ; `scirust-rustc-driver/target/` retiré du suivi git
  (artefacts de build) et ignoré.

### Ajouté
- **Primitives d'inférence de forme N-D (P2.4, fondation)** : `TensorND`
  gagne `broadcast_shape`, `matmul_shape` (matmul batché, broadcast des axes
  batch) et `broadcast_to` (matérialisation numpy) — les briques d'inférence
  de forme « au-delà de rows/cols » que la future tape/IR N-D utilisera, avec
  le pont `from/to_tensor_2d` existant. 3 tests. (La fusion de la tape 2D
  elle-même reste le gros chantier, à faire par incréments testés.)
- **Entraînement data-parallèle à déterminisme certifié (P2.1)** :
  `DataParallelTrainer::train_batch_threaded(n_threads, ..)` exécute les
  workers sur N threads OS (vol de tâches via compteur atomique) mais réduit
  les gradients dans un ordre fixe (worker 0,1,…,n-1), indépendant de
  l'ordonnanceur. L'addition flottante n'étant pas associative, le résultat
  est **bit-identique pour 1/2/4/8 threads** et identique au séquentiel —
  garantie testée que les frameworks grand public n'offrent pas. Trois tests
  CI : contributions sensibles à l'ordre (±1e16), vrai backward autograd, et
  une **boucle SGD multi-pas complète** dont la trajectoire de poids est
  bit-identique pour 1/2/4 threads (l'invariance se compose sur l'entraînement).

## [0.14.0] — 2026-06-13

### Réparé
- **`scirust-gpu` honnête (P2.2, étape « trancher »)** : les backends
  `WgpuBackend`/`CudaBackend` renvoyaient `vec![0.0; m*n]` — des résultats
  **fabriqués** (zéros) sous une étiquette « wgpu »/« cuda », en violation
  de la politique « 100 % câblé/testé, zéro sur-promesse ». Remplacés par
  un vrai backend CPU de référence **testé** (oracle GEMM bit-déterministe)
  et des chemins device qui signalent honnêtement `BackendError::Unavailable`
  (jamais de sortie inventée), à l'image de `scirust_core::compute_backend`.
  Crate passée de 0 à 6 tests. (Le câblage wgpu réel a suivi dans une étape
  séparée — voir « Ajouté » : GEMM WGSL testé sur Vulkan logiciel.)
- **`docs/GPU.md` honnête** : la page décrivait une API GPU en une ligne
  (`GpuContext::try_init`, `ConvGpuPipelines`, `Conv2d::on_gpu`…) qui
  n'existe pas (modules archivés ; `--features wgpu` ne compile rien).
  Réécrite en page de statut + roadmap honnête (ce qui existe = backend CPU
  de référence testé ; pourquoi le GPU n'est pas revendiqué ; plan P2.2).
- Régression de merge cassant la compilation sur toutes architectures
  (sgemv AVX2/SSE2/NEON, champ slab arena).
- CI rendue réalisable : retrait de `--all-features` (features BLAS
  mutuellement exclusives), `deny.toml` réécrit (TOML invalide),
  cross-check aarch64 ajouté ; 6 gates verts localement.
- Fusion d'opérateurs du graphe lazy : les chaînes pointwise fusionnent
  réellement (chaque maillon devenait sa propre chaîne de longueur 1).
- `RandomCrop` écrivait son résultat dans le vide (no-op silencieux).
- 22 warnings rustdoc ; warnings rustc/clippy ramenés à zéro
  (`-D warnings` tenable sur tous les targets).

### Changé
- **Statut GPU** retiré du tableau des features livrées du README (il
  listait du non-câblé) → remplacé par une note honnête « Not included
  yet » pointant la roadmap P2.2.
- **Augmentation de données 100 % déterministe** : RNG `PcgEngine`
  injecté, flux par échantillon indépendant de l'ordre, `with_seed`
  effectif, vrai bruit gaussien (Box-Muller).
- README aligné sur le code : statut GPU requalifié « Archived — not
  wired », compte de tests mesuré.
- `publish = false` sur les 51 manifestes (deps par chemin, licence
  non commerciale).

### Ajouté
- **GPU wgpu réel et testé (P2.2, étape « recâbler »)** : vrai GEMM `f32`
  en WGSL (`C = A·B`) derrière la feature `wgpu`, exécuté sur adaptateur
  Vulkan/Metal/DX12/GL via wgpu 0.20. **Validé contre l'oracle CPU**
  (tolérance flottante documentée, l'accumulation GPU n'étant pas
  bit-identique) et **testé en CI** sur Vulkan logiciel Mesa lavapipe
  (`llvmpipe`) — aucun GPU matériel requis, « pas de claim sans test »
  respecté. `cargo deny` passe sur l'arbre de deps wgpu ; dépendance
  optionnelle (les 8 gates par défaut ne la compilent pas). Nouveau job CI
  `GPU (wgpu / lavapipe)`.
- **GPU wgpu branché dans la tape autograd (P2.2, étape « tape »)** :
  `WgpuEngine` implémente le hook `GpuEngine` du `Tape` (kernel GEMM
  général `C = α·op(A)·op(B) + β·C` avec transposition). `Var::matmul_gpu`
  exécute **forward ET backward** (`dA = g·Bᵀ`, `dB = Aᵀ·g`) sur le GPU,
  device/pipeline mis en cache, repli CPU si un dispatch échoue. Validé
  bout-en-bout contre la tape CPU (forward + 2 gradients, tolérance) sur
  lavapipe. Opt-in (feature + `matmul_gpu`) → garantie bit-exacte par
  défaut intacte.
- **Conv2d GPU (P2.2, étape « Conv2d »)** : les GEMM im2col de Conv2d
  (forward `W·col`, backward `dW = dout·colᵀ` et `dInput = Wᵀ·dout`) passent
  par l'engine via le nouvel helper `Tape::gemm_ab` (chemin transpose natif),
  quand un `WgpuEngine` est attaché. Validé bout-en-bout contre la Conv2d CPU
  sur lavapipe (forward + dInput + dWeight, tolérance). Repli CPU
  bit-identique sans engine (aucune régression). im2col/col2im restent CPU.
- **Activations résidentes en VRAM (P2.2, étape « résidence »)** : API
  `GpuChain` — upload des entrées une fois, chaîne de `matmul` sur des
  handles `GpuMatrix`, un intermédiaire reste en mémoire GPU et alimente le
  GEMM suivant sans aller-retour CPU ; seul le résultat final est téléchargé.
  Validé contre l'oracle CPU sur lavapipe (chaîne 2 GEMM + transpose). La
  résidence transparente dans la tape (DeviceTensor matérialisé paresseusement
  en GPU) reste un chantier futur — sans bénéfice mesurable hors GPU matériel.
- **SBOM CycloneDX + automatisation de release** : SBOM CycloneDX 1.5
  reproductible (`docs/sbom/scirust.cdx.json`, horodatage figé via
  `SOURCE_DATE_EPOCH`, sans serial aléatoire → octet-identique pour une
  source donnée), généré par `./scripts/generate-sbom.sh`. Nouveau job CI
  `sbom` (artefact à chaque build) et workflow `release.yml` (sur tag `v*` :
  rejoue les gates, génère le SBOM, crée la release et y attache le SBOM).
  Section SBOM dans `SECURITY.md`, `docs/sbom/README.md` (provenance).
- **CLI : 5e vague** — `tt` (compression tensor-train TT-SVD d'une matrice,
  `scirust-tn` ; rapporte cœurs, rangs de liaison, ratio de compression et
  erreur de reconstruction, sortie 1 si `--max-err` dépassé), `solve-system`
  (système non-linéaire F(x)=0 par Broyden, `scirust-solvers`), `inverse`
  (inverse de matrice LU), `fem-heat` (chaleur 1D −u″=source par éléments
  finis linéaires), et méthode `dopri5` (Dormand–Prince adaptatif) pour `ode`.
  `FemSolver1D` était non testé : 2 tests ajoutés (oracle parabolique
  −u″=f exact aux nœuds + symétrie). Nouveau groupe TENSOR NETWORKS.
  `reconstruct_matrix` réexporté depuis `scirust-tn` (paire de
  `tt_decompose_matrix`). `newton_system` non exposé (closure `Fn(&[Dual])`
  comme `bfgs`).
- **CLI : 4e vague** — `trig` (identités trigonométriques), `patterns`
  (tendance d'une série), `qr` (décomposition QR), `cg` (gradient
  conjugué SPD). `bfgs` délibérément non exposé (closure `Fn(&[Dual])`
  non constructible depuis une expression symbolique évaluée en f64).
- **CLI : 3e vague** — `symreg` (régression symbolique par programmation
  génétique, `scirust-symreg`), `sat` (satisfiabilité DPLL,
  `scirust-neuro-symbolic`), et deux méthodes de plus pour `root`
  (`secant`, `newton` via dérivée symbolique). Nouveau groupe LOGIC.
- **CLI : 2e vague de commandes** (29 → toutes testées) : `integrate
  --method simpson|gauss`, `root --method bisection`, `optimize`
  (Nelder–Mead multi-variable), `lstsq` (moindres carrés QR), `cholesky`,
  `prove` (équivalence symbolique), `gradient` (numérique 1–2 var). Les
  commandes à expression réutilisent `scirust-symbolic::eval`.
- **CLI massivement étoffée** (19 commandes, toutes adossées à du code
  testé) : ajout de `cmaes` ; maths symboliques `to-rust`, `regress` ;
  solveurs numériques `integrate` (Romberg), `root`/`minimize` (Brent,
  via dérivée symbolique), `linsolve`/`det` (LU), `polyroots`,
  `ode` (RK4). Les commandes pilotées par expression utilisent
  `scirust-symbolic::eval` comme pont vers les solveurs `scirust-solvers`.
  +10 tests CLI ; bug d'ordre (intercept,slope) de `regress` corrigé et
  épinglé par un test.
- **CLI `scirust` étoffée** (niveau industriel) : nouvelles commandes
  groupées et documentées — `som train` (modèle d'ownership, accuracy vs
  baseline), `evo` (optimiseur génétique seedé), `diff`/`simplify`/`eval`/
  `solve` (maths symboliques), `info` (garanties). `scirust help` les
  liste par thème. Chaque commande est adossée à du code déjà testé.
- **Flash Attention réellement testé** : 4 tests dans
  `nn/transformer/flash_attention.rs` (forward vs oracle d'attention
  dense, masque causal, déterminisme bit-exact, gradients finis) — la
  ligne de statut passe de revendiquée à vérifiée.
- **CLI unifiée `scirust`** (`scirust-cli`) : point d'entrée unique et
  découvrable (`scirust help`) regroupant `quickstart` (démo MLP 2→8→2
  bit-déterministe, 4/4), `analyze` (ownership, délègue à som-cli),
  `verify` (certificats, délègue à `proofcli`), `version`. Logique verify
  factorisée dans `scirust_runtime::proofcli` (zéro duplication ;
  `scirust-verify` délègue désormais). Quickstart du README réécrit
  autour de la CLI (plus de copier-coller de 40 lignes d'API), exemple
  bibliothèque corrigé pour l'API réelle.
- **Support Rust stable** : `#![feature(portable_simd)]` rendu réellement
  optionnel (`cfg_attr`), fallback scalaire du tiling ; les 683 tests
  passent sur stable ; job CI `build-test-stable`. La feature nightly
  `portable-simd` (cassée par la migration d'API std::simd) est réparée.
- **`scirust-verify`** : certificats d'inférence `SCIRUST-PROOF-1`
  fichier-à-fichier (emit/verify, exit codes), détection d'altération
  artefact/certificat testée, ré-émission bit-identique.
- **`cargo som` + `--sarif`** : le linter d'ownership en sous-commande
  cargo avec sortie SARIF 2.1.0 pour le code scanning CI.
- **SOM opérationnel sur du vrai Rust** : frontend `syn`
  (`scirust-som-frontend`), oracle d'ownership **type-aware**
  (Copy/move exact, E0382/E0502/E0503-style), CLI `som-analyze`,
  pipeline Transformer entraîné/évalué contre l'oracle (ownership
  87,3 % vs baseline 33,1 % sur held-out), bit-déterminisme testé.
- Modules recâblés et réparés : `core::lazy` (fusion), 
  `core::tensor::{broadcast,device}`, `scirust_symbolic::prelude`.
- `archive/` : sources historiques retirées du build avec état documenté
  (GPU non câblé, NEON/SVE dupliqués, brouillon quant incorrect).
- Docs industrielles : `docs/REFERENCE.md` (commandes/binaires/API
  exhaustifs), `CONTRIBUTING.md`, `SECURITY.md`, audit
  `scirust_complete_audit_report.md`.
