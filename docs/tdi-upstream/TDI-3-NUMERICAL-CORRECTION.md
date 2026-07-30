# TDI-3 — Correction numérique postérieure au gel

Date de détection : 12 juillet 2026.

## Exécution concernée

La première exécution du protocole TDI-3 a été lancée sur le commit gelé :

`25b6146764b2ee87bbffda559dd9dd0559213360`

Elle s’est interrompue avant la production de métriques ou de verdicts.

## Erreur observée

L’échec est reproductible avec :

- largeur : 5 ;
- graine : `20_000_000` ;
- horizon : 6 ;
- distribution : référence ;
- état signalé : bits `15` ;
- erreur : `ProbabilityOverflow`.

Message original :

`ReferenceDistribution(ProbabilityOverflow { state: State { bits: 15, width: 5 }, depth: 6 })`

## Cause

`ExactRatio` utilisait deux entiers `u128`.

La propagation restait rationnellement exacte, mais l’addition de
probabilités issues de chemins possédant des dénominateurs différents pouvait
produire un dénominateur commun réduit supérieur à la capacité de `u128`.

Il ne s’agit donc pas :

- d’un résultat négatif de TDI-3 ;
- d’une modification des données ;
- d’une instabilité flottante ;
- d’un dépassement provenant de la cible statistique.

Il s’agit d’une limite de représentation du moteur rationnel exact.

## Correction

`ExactRatio` utilise désormais `BigUint`, une représentation entière
arbitrairement grande en Rust pur.

Les opérations suivantes restent exactes :

- réduction par PGCD ;
- addition rationnelle ;
- division par le nombre local de successeurs ;
- comparaison rationnelle ;
- calcul du recouvrement distributionnel.

La conversion en `f64` reste limitée à l’affichage et à la modélisation
statistique, conformément au préenregistrement.

## Contrôle de non-altération

La correction ne modifie pas :

- les populations ;
- les plages de graines ;
- les horizons 2 et 6 ;
- la perturbation ;
- les caractéristiques ;
- la baseline ;
- le modèle ridge ;
- `lambda = 1` ;
- le bootstrap ;
- les critères TDI-3A et TDI-3B.

Un test de régression vérifie que la largeur 5, graine `20_000_000`, peut
désormais être analysée sans débordement. Ce test ne publie ni cible, ni
prédiction, ni métrique.

## Preuves conservées

Les preuves locales de l’échec initial ont été conservées dans :

`/tmp/tdi3-first-run-overflow`

Hashes observés :

- log partiel :
  `28c43a599ba94afd4c74a1bed2f11e9050a7c4e774847e71e2c8c82e28e99893` ;
- console complète :
  `00f0a9d6ce5102ecdf1630c60149c885bce3616a30cc3096024fc9b106465b42`.

## Nouveau gel

Après validation complète, un nouveau hash de l’évaluateur et un manifeste
SHA-256 de l’ensemble du code scientifique seront produits avant la nouvelle
exécution intégrale.
