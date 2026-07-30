# CCOS — Causal Context Operating System
### La mémoire de travail des agents IA : le bon contexte, au bon moment, à coût minimal — et auditable.

> *Document de présentation — destiné à un lecteur non‑technique comme à un investisseur.
> Tous les chiffres cités sont **mesurés** sur du vrai code, pas estimés. Ce qui relève de
> l'hypothèse est étiqueté comme tel.*

---

## 1. En une phrase

**CCOS est une « mémoire externe » pour les IA qui écrivent du code : au lieu de noyer le
modèle (ou le développeur) sous des milliers de lignes, il sélectionne automatiquement *la
poignée de fichiers réellement liés au problème*, dans un budget minuscule, de façon
déterministe et traçable.**

Pour une IA, c'est un copilote qui « sait où regarder ». Pour un humain, c'est un bouclier
qui masque le bruit et pointe la cause racine d'un bug.

---

## 2. Le problème, expliqué simplement

Une IA qui code (Copilot, Cursor, Claude Code, Devin…) ne « voit » qu'une petite fenêtre de
texte à la fois — sa **fenêtre de contexte**. C'est sa mémoire de travail. Trois douleurs en
découlent :

- **Elle est trop petite pour un vrai projet.** Un dépôt fait des dizaines de milliers de
  lignes ; la fenêtre n'en tient que quelques milliers. Que mettre dedans ?
- **La remplir coûte cher.** Aujourd'hui on y déverse « le fichier ouvert + des bouts trouvés
  par recherche ». C'est cher (chaque jeton d'entrée se paie), lent, et souvent **on rate la
  vraie cause** — qui est dans *un autre fichier*.
- **C'est une boîte noire.** Quand un agent autonome dérape, impossible de rejouer *pourquoi*
  son attention est partie au mauvais endroit.

Le même problème touche **l'humain** : un test échoue, 50 lignes de trace s'affichent, on ouvre
le fichier du symptôme, on se perd dans les dossiers, et la cause est deux sauts plus loin.

> **L'analogie clé.** Un processeur ne laisse pas un programme fouiller toute la RAM au hasard :
> une puce dédiée, la **MMU** (Memory Management Unit), lui présente *juste la bonne page de
> mémoire*, à la demande. **CCOS est la MMU cognitive des agents IA** : il pagine le code par
> *pertinence causale*, pas par chance.

---

## 3. La solution : CCOS

CCOS lit le code, en construit un **graphe causal** (qui dépend de qui), puis répond à la
question « je travaille sur le fichier X, que dois‑je avoir sous les yeux ? » en remontant les
dépendances réelles — et s'arrête tout seul à la **région causale**, sans réglage à tâtonner.

Quatre piliers, tous fonctionnels et testés :

| Pilier | Ce que ça veut dire | Pourquoi c'est rare |
| ------ | ------------------- | ------------------- |
| **Contexte causal frugal** | Le voisinage de dépendances du fichier courant, borné à un petit budget. | La plupart des outils *remplissent* le budget ; CCOS *se borne* à la région utile. |
| **Déterminisme** | Mêmes entrées → même sortie, rejouable bit‑à‑bit. | Les piles RAG sont probabilistes ; CCOS est reproductible. |
| **Auditabilité** | Chaque opération est journalisée dans un registre infalsifiable (chaîne de hachage) + un débogueur « voyage dans le temps ». | Personne, dans l'écosystème agent, ne propose un journal cognitif rejouable. |
| **Local & frugal** | Tourne entièrement sur la machine (le code ne part jamais dans le cloud), en Rust léger, **jusque sur un edge (Jetson)**. | Confidentialité + coût + souveraineté, sans dépendre d'une API distante. |

---

## 4. Le produit concret — le « bouclier attentionnel »

Le premier produit livrable, `ccos focus`, illustre la valeur en une image. Vous lancez vos
tests, ça plante. Au lieu de la trace brute, CCOS affiche :

```
⚡ CCOS focus — 3 fichiers → 3 affichés (~120 jetons)
  panic à src/lib.rs:4

  ▸ src/filter.rs   ◀ cause probable          ← LA CAUSE, en premier
      pub const MIN_SCORE: f64 = 0.0;   // BUG : devrait être 0.5
  ▸ src/lib.rs      · symptôme
  ▸ src/reader.rs   · lié
```

Le symptôme était dans un fichier ; **la cause dans un autre**. CCOS la remonte et la met en
tête, masque le reste. Le même flux alimente :

- **un agent IA** (via le standard MCP) — CCOS devient sa mémoire de travail ;
- **un humain** (extensions Neovim / VS Code) — un volet qui pointe la cause et masque le bruit.

*L'humain (ou l'agent) reste le raisonneur ; CCOS lui apporte le bon contexte.*

---

## 5. Les gains — **constatés** (mesurés sur du vrai code)

| Axe | Résultat mesuré | Mesuré sur |
| --- | --------------- | ---------- |
| **Couverture du bon contexte** | **81–100 %** des dépendances d'un fichier tiennent dans un budget de 2048 jetons — contre **0–2 %** si on ouvre juste le fichier. | 3 vrais dépôts (`syn`, `serde_json`, CCOS), 183 dépendances réelles |
| **Frugalité** | Le voisinage causal complet = **~4 % du coût** d'envoyer tout le dépôt (**25–30× moins de jetons**). | crate `syn` (191 000 jetons) |
| **Résolution de bugs multi‑fichiers** | Un LLM local **corrige la cause racine** là où un « dump » de même budget échoue (**3/3 vs 0/3**), sur 2 familles de modèles. | bugs contrôlés, qwen3‑coder 30B & DeepSeek‑V2‑Lite |
| **Latence sur edge** | **~10 ms** pour rafraîchir le contexte sur un Jetson (mise à jour incrémentale O(Δ)). | matériel cible ARM |
| **Fiabilité / audit** | Rejeu déterministe, journal infalsifiable, 364 tests verts en intégration continue. | base de code CCOS |

> **À lire honnêtement :** la **couverture** (donner le bon contexte, frugalement) est l'acquis
> large et solide. Le **gain de résolution** est réel mais **étroit** : il se manifeste sur les
> bugs dont la cause est dans un fichier différent du symptôme — environ **1–2 %** des correctifs
> réels dans les projets analysés — et il dépend d'un modèle assez capable pour exploiter le
> contexte. Nous le disons sans fard : c'est ce qui rend les autres chiffres crédibles.

---

## 6. Les gains — **possibles** (logique économique, à valider à l'échelle)

- **Réduction directe du coût des agents.** Dans une boucle agentique, **les jetons d'entrée
  dominent la facture**. Remplacer un « dump » par une fenêtre causale ~25–30× plus petite *à
  couverture égale* réduit mécaniquement le coût d'API et la latence. *(Hypothèse à chiffrer par
  client : le gain net dépend de la boucle.)*
- **Inférence locale plus accessible.** Moins de jetons = moins de VRAM et plus de vitesse →
  des modèles plus petits / du matériel moins cher deviennent utilisables.
- **Productivité humaine.** Le bouclier réduit le temps de localisation d'un bug multi‑fichiers
  (ouvrir‑chercher‑se perdre → la cause d'emblée).
- **Conformité.** Pour les agents autonomes en production, un journal déterministe et
  infalsifiable de *quel contexte a été fourni à chaque décision* est un actif de gouvernance.

---

## 7. Ce que CCOS apporte à l'univers IA

La course actuelle pousse deux leviers chers : **des modèles plus gros** et **des fenêtres de
contexte plus grandes**. Tous deux sont probabilistes et coûteux. Or, à mesure que les modèles
se banalisent, **le facteur différenciant devient la gestion du contexte** : *quoi* mettre dans
la fenêtre, *à quel coût*, *de façon vérifiable*.

CCOS propose la couche qui manque : **une hiérarchie mémoire déterministe, frugale et auditable**
pour les agents — l'équivalent de la MMU/du cache que les CPU ont, et que la pile IA n'a pas
encore. Il déplace une part de la valeur de « plus gros / plus large » (cher, opaque) vers
« le bon contexte, bon marché, déterministe, traçable ». C'est un **changement d'axe** : on
n'optimise plus seulement le cerveau, on optimise *ce qu'il a sous les yeux*.

---

## 8. Les débouchés (marchés visés)

1. **Outillage développeur (le bouclier).** Extensions IDE (VS Code, Neovim) qui pointent la
   cause racine. Population colossale de développeurs ; entrée par le bas (open‑source / gratuit),
   monétisation par les équipes (collaboration, gros monorepos, on‑prem).
2. **Infrastructure d'agents.** CCOS comme **couche mémoire** des agents de codage autonomes (via
   MCP, le standard d'intégration). Le marché du *coding agent* est l'un des segments logiciels
   qui croissent le plus vite ; CCOS s'y branche comme substrat frugal et auditable, sans
   concurrencer les modèles.
3. **Gouvernance & audit de l'IA.** Le journal déterministe et infalsifiable répond au besoin
   montant de **traçabilité des agents** dans les secteurs régulés. *(À cadrer honnêtement : CCOS
   audite la couche d'assemblage de contexte, pas la « pensée » du modèle.)*
4. **IA embarquée / souveraine.** Tourne en local sur edge (Jetson), code jamais exfiltré →
   pertinent pour l'industrie, la défense, la santé, et tout acteur soucieux de confidentialité
   ou de coût d'API.

**Modèles de revenus possibles** (à instruire) : open‑core (noyau libre + fonctions équipe
payantes), licence d'infrastructure pour plateformes d'agents, offre conformité/audit en SaaS,
support/intégration on‑prem.

---

## 9. Différenciation & « douve » (pour l'investisseur)

CCOS **n'est pas un meilleur moteur de recherche** (un RAG bien réglé peut être aussi précis).
Sa douve est **structurelle**, sur des axes que la pile probabiliste ne peut pas copier sans se
réinventer :

- **se borne tout seul** (pas de « top‑k » à régler) ;
- **déterministe** (rejouable, testable, certifiable) ;
- **auditable** (journal infalsifiable, voyage dans le temps) ;
- **frugal et local** (coût, confidentialité, edge).

C'est la **combinaison** — frugalité + déterminisme + auditabilité + local — qui est rare et
défendable, pas un seul de ces axes isolément.

**Pourquoi maintenant :** les agents passent du gadget à la production ; les codebases explosent ;
le coût des jetons d'entrée devient le poste dominant ; et la régulation de l'IA autonome arrive.
CCOS est exactement à cette intersection.

---

## 10. Maturité, risques & honnêteté

*(Section volontairement franche — c'est un gage de sérieux, pas un aveu de faiblesse.)*

- **Stade :** prototype de recherche solide, en Rust (édition 2021), 364 tests + lint stricts en
  CI. **Pré‑produit, pré‑revenu, pré‑traction.** Pas de clients à ce jour.
- **Ce qui est prouvé :** la couverture frugale (large), le déterminisme/audit, le bouclier validé
  de bout en bout *sur le vrai matériel Jetson*.
- **Ce qui reste à prouver :** que le gain de *résolution* se généralise au‑delà du créneau étroit
  mesuré ; l'adoption produit (l'UI éditeur reste à éprouver en usage réel) ; le passage à
  l'échelle multi‑langage (le parseur est calibré Rust ; les autres langages demandent du travail).
- **Limites assumées :** le graphe est **structurel** (qui dépend de qui), pas sémantique (pas de
  flot de données fin) ; CCOS rend la dérive d'un agent *auditable*, il ne prétend pas l'*empêcher*.

---

## 11. Feuille de route (de l'actif actuel vers le produit)

1. **Productiser le bouclier** (extensions IDE finalisées) — la valeur la plus tangible et la plus
   proche, qui s'appuie sur la force prouvée (couverture).
2. **Couche infra agents** — packager le serveur MCP comme mémoire de référence pour les
   plateformes d'agents.
3. **Multi‑langage** — étendre le parseur (Python, JS/TS, Go…) pour sortir du périmètre Rust.
4. **Offre audit/gouvernance** — exposer le journal déterministe comme service de traçabilité.

---

## 12. Synthèse — la thèse en trois lignes

- **Le besoin est universel et croissant** : donner à une IA (ou un humain) *le bon contexte de
  code*, frugalement, dans un monde de codebases géantes et d'agents autonomes.
- **CCOS y répond sur un axe rare** : frugal **et** déterministe **et** auditable **et** local —
  prouvé par des chiffres mesurés (81–100 % de couverture vs 0–2 %, 25–30× moins de jetons, ~10 ms
  sur edge).
- **Le potentiel dépasse l'outil** : c'est la **couche mémoire manquante** de l'âge des agents,
  avec des débouchés en outillage dev, infra d'agents, gouvernance IA et edge.

*CCOS ne vend pas du rêve : il vend du contexte juste, bon marché et traçable — et il le mesure.*
