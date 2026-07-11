# Assistant de diagnostic automobile OBD2 (exemple SciRust)

Une **petite IA** (un réseau de neurones MLP) spécialisée dans le diagnostic
automobile : elle lit un **code défaut OBD2** + quelques symptômes et propose la
**cause racine** la plus probable, classe toutes les hypothèses par
probabilité, et suggère l'action de réparation.

C'est le même moteur que la démo `quickstart_v2` (autograd `Tape`, couches
`Linear`/`ReLU`, optimiseur `Adam`), mais **spécialisé** pour un métier.

## Lancer

**Version simple** (25 cas, 5 causes) — démo pédagogique :

```bash
cargo run -p obd2_diagnostic --release
```

**Version massive** (10 000 cas synthétiques, 10 causes) — entraînement production :

```bash
cargo run -p obd2_diagnostic --release --bin obd2_massive
```

**Version ultra-massive** (100 000 cas synthétiques, 10 causes, modèle très profond) — défi à grande échelle :

```bash
cargo run -p obd2_diagnostic --release --bin obd2_ultra
```

**Version MÉGAVERSE** (1 000 000 cas synthétiques, 1000 causes, classification extrême) — le défi ultime :

```bash
cargo run -p obd2_diagnostic --release --bin obd2_megaverse       # 8 epochs
cargo run -p obd2_diagnostic --release --bin obd2_megaverse -- 3  # nb epochs au choix
```

## Ce que fait le programme

1. **Encode** chaque situation d'atelier en 7 nombres (le code défaut + des
   relevés : correction carburant long terme, débit d'air MAF, ralenti…).
2. **S'entraîne** sur 25 cas de réparations « validées ».
3. **Diagnostique** des cas nouveaux : hypothèses classées par % + action.

## L'idée clé : désambiguïser la cause racine

Un même code (`P0171`, mélange trop pauvre) peut avoir **plusieurs** causes.
L'IA apprend que le **débit d'air (MAF)** départage les deux plus fréquentes :

| Code | Correction carburant | Débit d'air MAF | → Cause prédite |
|------|----------------------|-----------------|-----------------|
| P0171 | +21 % (élevée) | **normal** | Prise d'air / fuite de dépression |
| P0171 | +18 % (élevée) | **bas** | Capteur MAF défectueux |

Même code, mêmes symptômes principaux — un seul relevé change le diagnostic.

## L'adapter à VOS données

- **Ajouter une cause racine** : ajoutez une entrée dans `CAUSES` et `ACTIONS`,
  augmentez `N_CLASSES`, et fournissez des exemples dans `training_data()`.
- **Ajouter un symptôme** (ex. régime moteur, température) : augmentez
  `N_FEATURES` et ajoutez la colonne à chaque ligne.
- **Vraies données** : remplacez `training_data()` par votre historique de
  réparations validées (un cas = features + cause confirmée).

## Version massive (entraînement production)

Le binary `obd2_massive` inclut :
- **10 000+ cas synthétiques** répartis en train/val/test
- **10 causes racines** (au lieu de 5)
- **Bruit réaliste** (~2 % pendant l'entraînement, ~8 % au test)
- **Modèle plus profond** : 10 → 64 → 32 → 10
- **Métriques de performance** : train/val/test accuracy

Résultats sur données synthétiques :
- Précision train : 99.88 %
- Meilleure précision validation : 79.80 %
- Précision test : 56.60 % (566 / 1000 cas bruités)

Le 56.6 % sur 10 classes reflète la séparabilité réelle des patterns générés.
Avec de vraies données d'atelier (signatures causales plus fortes), les
résultats seraient meilleurs.

## Version MÉGAVERSE (1M cas × 1000 causes)

Le binary `obd2_megaverse` pousse le framework à l'échelle :
- **1 000 000 cas synthétiques** (800K train / 100K val / 100K test)
- **1000 causes racines**, chacune avec une signature unique de 8 capteurs
  anormaux (haut/bas) parmi 20 — unicité vérifiée à la génération
- **Mini-batches de 256** via le support multi-batch natif (matmul batché +
  CrossEntropy à labels entiers) : 3 125 graphes d'autodiff par epoch au
  lieu de 800 000
- **Shuffle Fisher-Yates** de l'ordre des exemples à chaque epoch
- **Bruit** : ±0.03 à l'entraînement, ±0.05 au test (plus dur)

Résultats mesurés (modèle 20 → 256 → 128 → 1000, ~167K paramètres,
Adam lr=0.001, seed 42) :

| Métrique | Valeur |
|----------|--------|
| Génération des 1M cas | 0.07 s |
| Entraînement (3 epochs) | 157 s (~52 s/epoch) |
| Validation | **100.00 %** dès l'epoch 1 |
| **Test (100 000 cas)** | **100.00 %** (100000/100000) |
| Baseline aléatoire | 0.10 % |

Le 100 % s'explique : chaque cause possède une signature de capteurs
**unique et bien séparée** du bruit (écart signal ~0.3-0.45 vs bruit ±0.05).
Le réseau n'a « plus qu'à » apprendre 1000 régions de décision dans un
espace à 20 dimensions — ce que 800K exemples rendent possible. C'est une
démonstration de **capacité et de passage à l'échelle du framework**
(1M cas, 1000 classes, minutes de calcul), pas une mesure de difficulté
du diagnostic réel.

La v1 de ce binary plafonnait à ~0.1 % : signatures en collision
(périodicité modulo 20 → 20 signatures pour 1000 causes), données jamais
mélangées (oubli catastrophique) et un graphe d'autodiff par exemple
(~9 h par epoch). Le commentaire d'en-tête de `main_megaverse.rs` détaille
les trois corrections.

## Honnêteté sur les limites

Le jeu de données est **synthétique**, même en version massive. L'IA apprend
les patterns parfaits. Avec de vraies données d'atelier (bruitées,
contradictoires, avec des cas limites), les probabilités seraient plus
**nuancées** — et c'est justement là que le classement des hypothèses devient
utile pour le mécanicien. Cet exemple est **pédagogique & d'entraînement**,
pas un outil de diagnostic homologué.
