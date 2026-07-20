# Architecture CayleyFilter

## Couches

- `scalar` : oracle sédénion `f64` ;
- `operator` : matrice réelle 16 × 16 ;
- `analysis` : rang, noyau et résidus ;
- `projector` / `soft` : filtres SVD ;
- `search` : diviseurs de zéro vérifiés ;
- `selection` : train, dev et porte de sécurité ;
- `temporal` / `spectral` : adaptations des signaux.

## Invariants

- Rust pur, sans `unsafe` ;
- résultats déterministes ;
- aucune sélection sur les données de test ;
- activation uniquement si `dev_loss < 1.0` ;
- sinon, sortie identité.

## Validation

Les noyaux Cayley fonctionnent sur les sous-espaces synthétiques alignés.

Sur VoiceBank, MIT-BIH ECG et CWRU, la porte sélectionne `Identity` afin de
préserver le signal ou la signature diagnostique.
