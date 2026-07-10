# Agent prompt — drive an RSI loop with `scirust-rsi`

Ready-to-paste prompt for a Claude Code session **scoped to `Memorithm/RSI`**
(this scirust session does not have access to that repo). It makes `RSI` consume
the `scirust-rsi` engine to generate and then recursively improve algorithms,
with the safety contract enforced and verified.

```text
Contexte
--------
Le dépôt Memorithm/scirust contient désormais un crate `scirust-rsi` : un moteur
pur-Rust, déterministe et BORNÉ d'auto-amélioration récursive (boucle élitiste
« propose → évalue → garde si STRICTEMENT meilleur → répète »).
Doc d'intégration : scirust-rsi/INTEGRATION.md sur la branche master.
Il expose les traits RefineTask / BootstrapTask / ExpertIterationTask / PbtTask,
le pilote OnePlusLambda, la primitive `ascend` et le garde-fou `Guard`
(max_iters, patience, target, min_delta) avec `Report::is_monotone()`.

Objectif
--------
Faire du dépôt Memorithm/RSI un CONSOMMATEUR de scirust-rsi : un agent qui
GÉNÈRE des algorithmes candidats puis les AMÉLIORE en boucle, sans jamais
régresser. Développe sur une nouvelle branche, ne pousse PAS sur main directement.

Étapes
------
1. Inspecte l'état actuel de Memorithm/RSI (structure, Cargo.toml, ce qui existe).
   Lis aussi scirust-rsi/INTEGRATION.md et scirust-rsi/src/lib.rs dans scirust.
2. Ajoute la dépendance git :
     scirust-rsi = { git = "https://github.com/Memorithm/scirust", branch = "master" }
   (et scirust-algogen / scirust-synthesis si tu veux un vrai générateur de code).
3. Implémente le trait adapté à la tâche de RSI (par défaut RefineTask) où :
     - `score`  = ÉVALUATEUR : compile/teste le candidat, renvoie une Fitness
                  (ex. fraction de tests passés − pénalité de complexité) ;
     - `refine` = GÉNÉRATEUR : produit une révision critiquée du candidat.
   Si aucun générateur LLM n'est câblé, commence par un générateur déterministe
   (mutations symboliques via scirust-algogen) pour que tout soit testable et reproductible.
4. Pilote la boucle avec un Guard explicite
   (ex. Guard::new().max_iters(50).patience(8).target(...)) et conserve le Report.
5. VÉRIFICATION (obligatoire, ne pas sauter) :
     - `cargo build` et `cargo test` passent ;
     - un test prouve que `report.is_monotone()` est vrai (non-régression) ;
     - un test prouve la terminaison (itérations ≤ max_iters) ;
     - exécute un petit exemple `cargo run --example ...` qui montre une Fitness
       qui s'améliore puis se stabilise, et logge le Report.
     - `cargo clippy` propre.
6. Documente dans le README de RSI : comment lancer l'agent, le contrat de sûreté
   (bornes, non-régression, sandbox de l'évaluateur, graine reproductible),
   et le fait que tout code généré est exécuté dans TON sandbox, pas par le moteur.
7. Commit avec messages clairs, pousse sur la branche, ouvre une PR vers main.
   NE FUSIONNE PAS sans mon accord. Donne-moi le lien de la PR et le résumé des
   vérifications (sorties de tests réelles).

Garde-fous
----------
- Ne génère/exécute aucun code en dehors d'un sandbox que TU contrôles dans RSI.
- La boucle doit rester bornée et élitiste : aucune régression ne doit pouvoir
  être adoptée. Si tu ne peux pas le garantir, arrête-toi et explique pourquoi.
- Rends compte fidèlement : si un test échoue, montre la sortie ; ne prétends pas
  que c'est vert si ça ne l'est pas.
```
