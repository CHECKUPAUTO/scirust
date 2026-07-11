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

## Honnêteté sur les limites

Le jeu de données est **synthétique**, même en version massive. L'IA apprend
les patterns parfaits. Avec de vraies données d'atelier (bruitées,
contradictoires, avec des cas limites), les probabilités seraient plus
**nuancées** — et c'est justement là que le classement des hypothèses devient
utile pour le mécanicien. Cet exemple est **pédagogique & d'entraînement**,
pas un outil de diagnostic homologué.
