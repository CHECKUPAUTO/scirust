# scirust-cayley-filter

Filtrage expérimental déterministe en Rust fondé sur les sédénions.

## Règle de sécurité

Le filtre Cayley est activé uniquement si :

`development_loss < 1.0`

Sinon, la sortie reste identique à l’entrée.

## Résultats

- succès sur les bruits synthétiques alignés ;
- abstention sur VoiceBank, MIT-BIH ECG et CWRU ;
- 71 tests validés ;
- Clippy strict validé ;
- code unsafe interdit.

Voir `docs/ARCHITECTURE.md`.
