# Contribuer à SciRust

## Contrat qualité (non négociable)

Toute contribution doit passer les gates exécutés par la CI
(`.github/workflows/ci.yml`) — les mêmes commandes en local :

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace --all-targets
cargo test --workspace
cargo check --workspace --all-targets --target aarch64-unknown-linux-gnu
cargo deny check
cargo doc --workspace --no-deps   # zéro warning exigé
```

`RUSTFLAGS="-D warnings"` est actif en CI : un warning est une erreur.

## Règles du dépôt

1. **100 % du code sous `*/src/` est compilé, câblé et testé.** Pas de
   fichier hors module-tree, pas de crate placeholder, pas de feature
   Cargo vide. Le code retiré du build vit dans `archive/` avec son état
   documenté.
2. **Déterminisme par construction.** Tout aléa passe par un
   `PcgEngine` seedé injecté par l'appelant (`thread_rng`,
   `rand::random`, `from_entropy` sont interdits dans le framework ;
   seule exception : les horodatages d'observabilité dans `logging.rs`
   et le binaire de démo `openclaw-u`, hors framework).
3. **Validation par oracle.** Toute primitive numérique nouvelle est
   acceptée avec un test la comparant à une référence (scalaire, valeur
   connue, ou oracle symbolique) — bit-exact quand le contrat le promet.
4. **Véracité des claims.** README/doc ne décrivent que ce qu'un test
   exécute. Un module non câblé ne peut pas être annoncé « ✅ ».
5. **Pas de résolution de conflit de merge sans relancer les gates.**
   (Une résolution manuelle a déjà cassé master sur toutes les
   architectures — voir `scirust_complete_audit_report.md` §8.)
6. Pour toute fonctionnalité touchant la tape autodiff : inclure une
   comparaison de gradients contre une référence numérique.

## Style

`rustfmt.toml` et `clippy.toml` font foi (`cargo fmt` avant commit).
Commentaires FR ou EN acceptés ; rustdoc obligatoire sur les API
publiques nouvelles.
