# archive/ — code retiré de l'arbre de compilation

Politique du dépôt : **100 % du code sous `*/src/` est compilé, câblé et
testé**. Ce répertoire conserve, hors build, des sources historiques qui
ne satisfont pas ce contrat, avec leur état exact. Rien ici n'est compilé
par le workspace ; tout y est récupérable (et l'historique git fait foi).

| Origine | Contenu | État constaté | Pour le faire revivre |
|---|---|---|---|
| `scirust-gpu/src/*.rs` (8 fichiers) | kernels WGSL (wgpu), matmul cuBLAS (cudarc), tenseur GPU, quant GPU | jamais déclarés en `mod` ; dépendances `wgpu`/`cudarc` absentes du workspace ; API `scirust-core` a dérivé depuis | ajouter les deps optionnelles, déclarer les modules derrière `cfg(feature)`, réaligner sur l'API actuelle, valider contre l'oracle CPU |
| `scirust-simd/neon.rs` | kernels NEON | duplicat abandonné — les kernels NEON actifs vivent dans `scirust-simd/src/dispatch.rs` (testés sur aarch64) | n/a (préférer dispatch.rs) |
| `scirust-simd/sve.rs` | kernels SVE | utilise des intrinsics `sv*` indisponibles dans `std::arch` ; ne compile pas | attendre la stabilisation SVE de Rust, ou passer en asm inline comme `scirust_simd::sve::sve_vector_length_elements` |
| `scirust-core/quant/` (« Pilier 5 ») | int8/int4/bf16 SIMD | brouillon non câblé contenant des kernels **incorrects** (masque `0x7F` sur valeurs signées, recombinaison de lanes erronée, corruption du bit de signe bf16) ; duplique le chemin int8 **validé** (`scirust-core/src/quantization.rs` + binaires d'audit `scirust-runtime`) | repartir du chemin validé ; toute reprise exige des tests d'équivalence bit-exacte contre le scalaire |

Décision documentée dans `scirust_complete_audit_report.md` (mise à jour
fiabilisation) : conformité « tout est câblé » + interdiction de la
duplication non validée.
