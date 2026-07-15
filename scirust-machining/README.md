# scirust-machining

**Bibliothèque de calcul pour l'ingénierie mécanique et la productique — pur Rust, déterministe, sans dépendance d'exécution.**

`scirust-machining` regroupe **458 modules** (~1800 fonctions publiques, 2863 tests) couvrant la chaîne de calcul du génie mécanique et de la fabrication : de la cinématique de coupe au dimensionnement des éléments de machines, de la mécanique des fluides à la thermique, en passant par les vibrations, la métrologie, la qualité et l'énergétique.

La documentation est en **français**, les identifiants (fonctions, types, paramètres) en **anglais**.

```toml
[dependencies]
scirust-machining = { path = "…" }   # aucune dépendance d'exécution
```

---

## Philosophie : des modèles d'ingénieur honnêtes

Chaque module suit trois principes stricts, vérifiables dans le code :

1. **Aucune constante physique inventée.** Les grandeurs matériau / procédé / installation — coefficient de coupe `kc1.1`, exposant de Taylor `n`, facteur de forme de Lewis `Y`, coefficient de frottement `µ`, indice de travail de Bond `Wi`, coefficient de Seebeck… — sont **fournies par l'appelant** d'après un catalogue, une norme ou des essais. La crate calcule leurs *conséquences* ; elle n'introduit jamais de valeur « par défaut » qui serait invérifiable. Les seules constantes exposées sont **universelles** et nommées (π via `core::f64::consts`, constante de Wien, limite de Betz 16/27, μ₀…).

2. **Une section « Limite honnête » par module.** L'en-tête `//!` de chaque fichier énonce explicitement les hypothèses et le domaine de validité du modèle (régime permanent, élasticité linéaire, écoulement incompressible, corps gris, petites oscillations…) — ce que la formule *ne* dit pas.

3. **Des tests d'identités physiques.** Chaque module vérifie des réciprocités, des cas limites, des proportionnalités et un cas chiffré réaliste — pas des nombres magiques — plus un test de panique sur entrée invalide. Les fonctions gardent leurs entrées par `assert!` avec messages en français.

```rust
use scirust_machining::{spindle_speed_rpm, mrr_turning_cm3_min};

// Chariotage d'un acier Ø80 mm à Vc = 200 m/min, ap = 3 mm, f = 0,25 mm/tr.
let n = spindle_speed_rpm(200.0, 80.0);          // ≈ 796 tr/min
let q = mrr_turning_cm3_min(200.0, 3.0, 0.25);   // 150 cm³/min
```

Tous les identifiants sont ré-exportés à plat depuis la racine de la crate ; ils sont aussi accessibles par leur module (`scirust_machining::kinematics::spindle_speed_rpm`).

---

## Domaines couverts

Les familles ci-dessous sont **illustratives** (un échantillon par thème), pas exhaustives — voir la rustdoc pour la liste complète.

### Coupe & usinage
Cinématique de coupe (`Vc↔N`, `Vf`, MRR en tournage/fraisage/perçage), effort et puissance par le modèle de **Kienzle**, durée de vie outil par **Taylor**, économie d'usinage de **Gilbert**, temps de coupe, rugosité théorique, angle de cisaillement de **Merchant**, température de coupe, géométrie de foret, brochage, taillage à la fraise-mère, alésage, tournage conique, moletage.

### Formage & fabrication
Emboutissage, pliage (développé, facteur K), roulage, laminage à plat, forgeage (Hollomon), extrusion, tréfilage, fluotournage (loi du sinus), découpage/poinçonnage, roulage de filet, rétreint, compactage de poudre.

### Assemblage & procédés spéciaux
Soudage (apport de chaleur, préchauffage, équivalent carbone, dilution, refroidissement, cordon d'angle, groupe de cordons), CND (ultrasons, courants de Foucault, radiographie), fonderie (Chvorinov, masselottes, retrait), moulage par injection, EDM, ECM, découpe laser/jet d'eau, brasage, collage, revêtement (électrolytique, anodisation).

### Éléments de machines
Engrenages (droits/hélicoïdaux/coniques/roue-vis, **Lewis**, **ISO 6336**, épicycloïdal, jeu de denture, grippage, usure et charge dynamique de **Buckingham**), roulements & paliers (L10 **ISO 281**, statique **ISO 76**, frottement de Palmgren, hydrostatique, PV), ressorts (hélicoïdaux, Belleville, coniques, ondulés, à force constante, spiral, de torsion), courroies & chaînes, embrayages et freins (disque, cône, centrifuge, à bande, à courants de Foucault), accouplements (hydrodynamique, magnétique, convertisseur de couple), arbres, clavettes, goupilles, vis, boulonnerie (précharge, serrage à l'angle, bride ASME).

### Résistance des matériaux & structure
Poutres (réactions, flèches, charges réparties, poutre continue par les trois moments, sur fondation élastique de Winkler), flambement (Euler, Rankine, plaque, coque, poteau-poutre), sections (modules, centres de cisaillement, flux de Jourawski), contraintes (Mohr, von Mises, combinées, concentration, Hertz), plasticité (Ramberg-Osgood, flexion plastique), énergie (Castigliano, Strain energy).

### Dynamique, vibration & fatigue
Cinématique/dynamique du solide, mécanismes (quatre barres, bielle-manivelle, cames, croix de Malte, genouillère), équilibrage, vitesses critiques, vibrations (1 et 2 ddl, forcées, isolation, Coulomb, amortisseur accordé, balourd, **ISO 10816**), fatigue (Goodman/Soderberg/Gerber, Coffin-Manson, Paris, Weibull, endurance).

### Mécanique des fluides & hydraulique
Bernoulli, pertes de charge (Darcy, Colebrook), débitmétrie (Venturi, diaphragme, Pitot, rotamètre), surface libre (déversoirs, canal, ressaut, vanne de fond, Parshall), cavitation/NPSH, coup de bélier, cheminée d'équilibre, sédimentation de Stokes, écoulement compressible (tuyère bloquée, isentropique), pipeline de gaz (Weymouth), siphon, éjecteur.

### Machines à fluide
Pompes (centrifuges, à engrenages, à palettes, péristaltique, vitesse spécifique, lois d'affinité, NPSH), ventilateurs, compresseurs (alternatif, surpresseur Roots), turbines hydrauliques, éoliennes (limite de Betz), vérins hydrauliques/pneumatiques, presse hydraulique, palier hydrostatique.

### Thermique & énergétique
Conduction (permanente/transitoire, résistances, ailettes, isolation critique), convection, rayonnement (Stefan-Boltzmann, facteurs de forme, réseau de résistances, écrans), échangeurs (DTLM, NTU, encrassement), diphasique (condensation de Nusselt, ébullition de Rohsenow/Zuber), cycles thermodynamiques, réfrigération, pompe à chaleur, psychrométrie, combustion, chaudière, tour de refroidissement, thermoélectricité (Peltier, Seebeck).

### Procédés & équipements
Silos (Janssen), broyage (Bond/Rittinger/Kick), cyclones, filtration sur gâteau, agitation, transport pneumatique, convoyeurs (bande, vis, godets).

### Métrologie & qualité
Contrôle dimensionnel (barre sinus, cales, erreur d'Abbe/cosinus, verre étalon, planéité, battement), GD&T, incertitude, MSA, cartes de contrôle, échantillonnage, capabilité, Six Sigma (DPMO), Taguchi, AMDEC (RPN).

### Production, économie & ergonomie
Ordonnancement (Johnson, CPM, PERT), équilibrage de ligne, Little's law, takt time, EOQ, courbe d'apprentissage, prévision, SMED, OEE, coût machine, rentabilité ; ergonomie (NIOSH, WBGT, vibrations main-bras et corps entier).

### Instrumentation & mécatronique
Jauges de déformation et rosettes, thermocouple, actionneurs (solénoïde à réluctance, bobine mobile), piézoélectricité, moteurs (asynchrone, CC, triphasé, variateur V/f, démarrage), asservissement (PID, premier/second ordre, Bode), analyse vibratoire (ordres, fréquences de défaut de roulement).

---

## Conventions d'unités

SI cohérent, avec les conventions usuelles des fiches techniques rappelées par chaque fonction : `Vc` en m/min, longueurs/diamètres en mm, régimes en tr/min, avances en mm, efforts en N, puissances en kW ou W, couples en N·m, pressions en Pa, températures en K (sauf mention °C), angles en radians pour les fonctions trigonométriques.

## Exemples exécutables

```bash
cargo run -p scirust-machining --example atelier
```

`examples/atelier.rs` enchaîne les modules sur un cas concret de chariotage : choix du régime de coupe → vérification de la puissance broche → durée de vie outil → optimum économique → temps et coût → état de surface → tolérance → vérification d'un pignon.

## Tests

```bash
cargo test  -p scirust-machining          # 2863 tests d'identités physiques
cargo clippy -p scirust-machining --all-targets -- -D warnings
```

## Positionnement dans SciRust

Cette crate complète les autres briques mécaniques de l'écosystème : `scirust-tolerance` (tolérancement inertiel/statistique, ISO 286/1101), `scirust-metrology` (incertitude GUM), `scirust-fatigue` (rainflow, Palmgren-Miner) et `scirust-fab` (contrôle de procédé). Elle en constitue le **cœur de calcul déterministe**.

## Licence

Voir le dépôt racine `scirust`.
