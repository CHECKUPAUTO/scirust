# Assistant de diagnostic automobile OBD2 (exemple SciRust)

Une **petite IA** (un réseau de neurones MLP) spécialisée dans le diagnostic
automobile : elle lit un **code défaut OBD2** + quelques symptômes et propose la
**cause racine** la plus probable, classe toutes les hypothèses par
probabilité, et suggère l'action de réparation.

C'est le même moteur que la démo `quickstart_v2` (autograd `Tape`, couches
`Linear`/`ReLU`, optimiseur `Adam`), mais **spécialisé** pour un métier.

## Lancer

```bash
cargo run -p obd2_diagnostic --release
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

## Honnêteté sur les limites

Le jeu de données est **synthétique et propre**, donc l'IA affiche des
certitudes de 100 %. Avec de vraies données d'atelier (bruitées,
contradictoires), les probabilités seraient plus nuancées — et c'est justement
là que le classement des hypothèses devient utile pour le mécanicien. Cet
exemple est **pédagogique**, pas un outil de diagnostic homologué.
