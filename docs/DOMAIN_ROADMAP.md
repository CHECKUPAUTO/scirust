# SciRust — Domaines industriels à ouvrir (feuille de route de marché)

Complément de `INDUSTRIAL_ROADMAP.md` (go-to-market) et `INDUSTRIAL_VERTICALS.md`
(implémentation des verticaux déjà en cours : PdM, estimation, sûreté OT). Ici :
le résultat d'une recherche de marché ciblée sur les secteurs **régulés** où le
déterminisme bit-exact et l'auditabilité totale de SciRust constituent un
avantage *mesurable*, pas seulement un argument marketing — et qui ne sont
**pas** déjà couverts par les crates existantes (`scirust-signal`,
`scirust-opcua`, `scirust-mqtt`, `scirust-pdm`, `scirust-mlops`,
`scirust-func-safety`, `scirust-estimation`, `scirust-nav`, `scirust-water`,
`scirust-ids`, `scirust-hvac`, `scirust-bms`, `scirust-biomed`, `scirust-grid`,
`scirust-shm`, `scirust-spc`, `scirust-robotics`, `scirust-metrology`,
`scirust-reliability`).

## Doctrine (identique à `INDUSTRIAL_VERTICALS.md`)

1. Rust pur, zéro FFI, déterminisme bit-exact (PRNG germé, ordre de réduction fixe).
2. Aucune affirmation sans test — un oracle honnête, pas un stub.
3. Le différenciateur est toujours une **garantie** : reproductibilité,
   traçabilité hash-chaînée, borne certifiée, conformité à une norme nommée.
4. Chaque nouveau domaine doit passer par le connecteur unique décrit en fin de
   document (`scirust-mcp`) — un domaine ajouté devient immédiatement pilotable
   par un agent (le SLM `scirust-sciagent`, un LLM externe, ou un script), sans
   glue code spécifique.

## Pourquoi ces secteurs : le point commun documenté

La littérature de chaque secteur régulé documente le **même** point de
friction avec l'outillage dominant (Python/NumPy/SciPy, MATLAB/Simulink, ML
« boîte noire ») : la non-associativité du flottant, le threading BLAS non
déterministe et le manque de traçabilité cassent la reproductibilité exigée
par leurs propres normes.

- Intel documente l'absence de garantie bit-exacte même à nombre de threads
  fixe (non-associativité + FMA + réordonnancement compilateur).
- MathWorks documente que la correspondance simulation ⇄ code généré
  (SIL/PIL) n'est garantie qu'« à une tolérance près », d'où l'obligation
  ISO 26262 de tests MIL/SIL/PIL/HIL redondants.
- Des bugs de threading OpenBLAS ont produit des résultats silencieusement
  faux (issue publique openblas#1844) ; scikit-learn documente lui-même ne
  pas contrôler le déterminisme du threading BLAS sous-jacent.
- DO-178C (aéronautique) et IEC 62304 Edition 2 (dispositifs médicaux) butent
  explicitement sur le ML entraîné : la traçabilité exigée « suppose un
  comportement déterministe » qu'un réseau entraîné ne garantit pas — d'où
  les nouveaux cadres EASA (AI Concept Paper) et FDA/IMDRF (GMLP, PCCP) qui
  demandent explicitement de la transparence algorithmique.

C'est exactement le créneau que `docs/GROWTH_PLAN.md` revendique déjà
(« IA certifiable, reproductible et auditable ») — les domaines ci-dessous
sont où ce créneau a la demande la plus documentée.

## Domaines classés par force de la preuve trouvée

### D1 · Sûreté fonctionnelle des procédés (IEC 61511/61508 — SIS)
- **Client** : pétrochimie, chimie fine, raffinage — systèmes instrumentés de
  sécurité (SIS).
- **Pourquoi maintenant** : l'attaque Triton/Trisis (2017) a reflashé la
  logique de sécurité d'un Triconex Schneider sans être détectée avant un
  déclenchement accidentel — le cas d'école pour une logique de sécurité
  *non auditable*.
- **Algorithmes** : calcul PFDavg/SIL par architecture de vote (1oo2, 2oo3),
  intervalles de test de preuve, matrices cause-à-effet, journal
  hash-chaîné de la logique de vote (extension directe du modèle
  `scirust-func-safety::audit`).
- **Taille** : petite à moyenne — le chemin le plus rapide vers un produit
  différenciant « audit-grade ».

### D2 · Protection réseau électrique & estimation d'état (IEC 61850, NERC CIP, IEEE C37.118)
- **Pourquoi maintenant** : le rapport post-mortem du blackout nord-américain
  de 2003 pointe un estimateur d'état dont la défaillance n'a pas pu être
  reconstituée ; les protocoles GOOSE/Sampled Values de l'IEC 61850 sont
  démontrés usurpables (littérature académique citée).
  `scirust-grid` existe déjà mais reste à approfondir sur cet axe précis.
  `scirust-estimation` (Kalman/UD) est directement réutilisable ici.
- **Algorithmes** : estimation d'état par moindres carrés pondérés (WLS),
  détection de mauvaises données (largest normalized residual), logique de
  relais de distance/différentielle, traitement de synchrophaseurs (PMU).
- **Taille** : moyenne — s'appuie sur l'estimation déjà présente.

### D3 · Dispositifs médicaux à boucle fermée (IEC 62304 Ed.2, FDA SaMD/GMLP/PCCP)
- **Pourquoi maintenant** : la future édition d'IEC 62304 ajoute un cycle de
  vie dédié à l'IA/ML précisément parce qu'un modèle adaptatif ne rentre pas
  dans le modèle déterministe historique de la norme.
  `scirust-biomed` existe (traitement de signal) ; l'ouverture ici est le
  **contrôle** (dosage, ventilation) avec certificat de preuve d'inférence
  (`scirust-runtime`), pas seulement l'analyse de signal.
- **Algorithmes** : PID/MPC certifié à bornes IBP/CROWN, contrôleur de repli
  Simplex (déjà en I5 de `INDUSTRIAL_VERTICALS.md`), traçabilité PCCP.
- **Taille** : moyenne à grande (exigences réglementaires lourdes).

### D4 · Aéronautique — lois de commande de vol & fatigue structurelle (DO-178C/DO-333)
- **Pourquoi maintenant** : la traçabilité DO-178C suppose un comportement
  déterministe entrée→sortie ; c'est documenté comme rompu par la
  non-associativité flottante et par tout composant ML embarqué.
- **Algorithmes** : comptage rainflow pour la durée de vie en fatigue,
  numérique de lois de commande à virgule fixe déterministe (réutilise le
  pipeline int8 déjà validé de SciRust), bornes certifiées pour tout
  composant appris.
- **Taille** : grande — expertise de certification aéronautique nécessaire ;
  à traiter comme un partenariat plutôt qu'un sprint solo.

### D5 · Maritime autonome & classification DNV (IMO MASS Code 2026, DNV AROS, IACS UR E26/E27)
- **Pourquoi maintenant** : le nouveau code MASS (obligatoire, 2026) exige que
  les décisions autonomes restent « explicables et auditables » sans méthode
  de vérification encore consensuelle dans l'industrie — une fenêtre
  d'opportunité pour poser un standard de référence.
- **Algorithmes** : boucles de positionnement dynamique (DP), géométrie
  d'évitement de collision COLREGs, stabilité/tenue à la mer.
- **Taille** : moyenne.

### D6 · Contrôle « run-to-run » en fabrication de semi-conducteurs (SEMI E10/E58/E116)
- **Pourquoi maintenant** : le contrôleur R2R réinjecte directement la sortie
  du contrôle statistique de procédé (FDC/métrologie virtuelle) dans la
  recette du run suivant — une dérive numérique silencieuse coûte des
  plaquettes ; les normes d'audit y sont proches du 21 CFR Part 11 déjà géré
  par `scirust-func-safety::golden_batch`.
- **Algorithmes** : contrôle EWMA run-to-run, SPC/FDC multivarié, T² de
  Hotelling / PCA (réutilise le futur SVD général — voir plus bas).
- **Taille** : grande (surface statistique large).

### D7 · Agriculture de précision — conformité & traçabilité (ISO 25119, ISO 18497, ISOBUS/ISO 11783)
- **Pourquoi maintenant** : un cas documenté montre que les mêmes données de
  rendement, passées dans QGIS / Agro-Map / Farm Works, produisent des cartes
  de rendement *différentes* — une rupture de reproductibilité concrète et
  publiée. Les registres phytosanitaires et le MRV carbone exigent de plus en
  plus une trace horodatée inviolable.
- **Algorithmes** : recalcul déterministe de cartes de rendement, journal
  hash-chaîné de traitement phytosanitaire, sûreté fonctionnelle ISO 25119
  pour l'guidage/automatisation.
- **Taille** : moyenne.

### D8 · Nucléaire — protection de réacteur (IEC 61513/60880/62138)
- **Pourquoi maintenant** : l'AIEA et la littérature académique citent la
  défaillance de cause commune logicielle entre canaux redondants comme un
  point de licensing non résolu ; aucune plateforme ouverte de ce niveau
  n'existe aujourd'hui.
- **Algorithmes** : logique de vote 2-sur-4, calcul de seuils de flux,
  diversité fonctionnelle démontrable.
- **Taille** : LOC modeste, mais expertise de licensing très élevée — à
  n'aborder qu'en partenariat avec un exploitant/intégrateur qualifié.

*(Le ferroviaire EN 50128/50716 et les mines ISO 17757 ont aussi été
étudiés : la douleur documentée y est la complexité de vérification/model
checking, pas la reproductibilité numérique — moins spécifiquement alignée
avec l'ADN déterminisme/auditabilité de SciRust ; à revisiter si un
partenaire sectoriel se présente.)*

## Ce qui rend tous ces domaines exécutables sans explosion de code

Le point commun de ces huit domaines n'est pas un algorithme unique : c'est
qu'ils exigent tous (a) une brique numérique solide (moindres carrés,
eigen/SVD, optimisation sous contraintes, filtrage), déjà renforcée dans
cette itération (`scirust-solvers` — voir `CHANGELOG.md`), et (b) un moyen
standard de brancher ces briques sur l'infrastructure réelle d'un client
(capteurs, automates, historiens) et sur un agent qui orchestre le tout.
C'est le rôle des deux nouvelles crates de cette itération :

- **`scirust-mcp`** — expose toute capacité SciRust (solveur, PdM, signal,
  discovery) comme outil [Model Context Protocol](https://modelcontextprotocol.io)
  standard, appelable par `scirust-sciagent` ou par n'importe quel agent
  externe, avec schéma JSON et journal d'audit hash-chaîné par appel. Un
  nouveau domaine n'a qu'à enregistrer ses outils dans le registre existant.
- **`scirust-discovery`** — trouve, de façon sûre et consentie (modèle de
  zones/conduits IEC 62443, découverte native aux protocoles plutôt que scan
  générique), le matériel industriel réellement présent sur le réseau d'un
  client, pour que l'agent sache *à quoi* connecter ces outils.

Voir `scirust-mcp/README.md` et `scirust-discovery/README.md` pour le détail
technique et les sources citées.
