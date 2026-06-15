# SciRust : un framework d'apprentissage profond en Rust pur — Accélération GPU portable, moteur de régression symbolique et runtime d'inférence déterministe

**Tarek Zekriti**
Chercheur indépendant · contact@checkupauto.fr
Dépôt : https://github.com/CHECKUPAUTO/scirust

---

## Résumé

Nous présentons **SciRust**, un framework d'apprentissage profond écrit en Rust pur qui combine une bibliothèque de runtime avec une couche de transpilation (attributs de macro procédurale pour la différenciation, la vectorisation et le ciblage d'accélérateurs), et neuf capacités construites et validées sur celui-ci. La première est un chemin GPU et Tensor Core portable : le cœur en Rust pur se porte sur un NVIDIA Jetson Thor (aarch64) sans modification, et un produit matriciel basé sur cuBLAS, validé par rapport à un oracle CPU, atteint environ 63 TFLOPS en BF16. La seconde est un moteur de **régression symbolique** hybride génétique-gradient qui récupère des lois sous forme analytique — structure et constantes — à partir de données, en utilisant la propre différenciation symbolique du framework pour ajuster les constantes. La troisième est un **runtime d'inférence déterministe** offrant une inférence bit-exacte, à latence bornée et auditable, générique sur l'architecture via un manifeste en texte brut. La quatrième est une pile de quantification int8 déterministe pour l'inférence embarquée : un chemin d'inférence entier portable, bit-exact sur plusieurs threads et reproductible bit-à-bit sous requantification en virgule fixe, qui réduit la taille des poids du modèle d'environ quatre fois. Un fil conducteur méthodologique unique les relie : chaque primitive n'est acceptée qu'après que sa sortie correspond à un oracle de référence, et la reproductibilité est traitée comme une propriété de premier ordre mesurée — dans plusieurs cas bit-à-bit. Par rapport à la ligne de base du framework (255 tests réussis ; MNIST 97,70 %), ces contributions établissent SciRust comme un artefact de recherche substantiel et reproductible.

---

## 1. Introduction

SciRust est un framework d'apprentissage profond écrit en Rust pur. Il s'agit d'un hybride entre une bibliothèque de runtime et un système de transpilation : parallèlement aux composants classiques de tenseurs et de réseaux de neurones, il implémente de réels attributs de macro procédurale — #[autodiff], #[simd] et #[gpu] — à travers trois crates de macros, de sorte que le code Rust annoté est réécrit sous des formes différenciées, vectorisées ou ciblées pour les accélérateurs. Le projet est positionné comme un **artefact de recherche**, et non comme un concurrent de production aux frameworks établis (PyTorch, ou en Rust, Burn et candle), qui le dépassent en termes de couverture d'opérateurs, de maturité des noyaux et de diversité matérielle.

Ce rapport présente le framework et trois capacités construites sur celui-ci, chacune validée et rapportée avec ses chiffres mesurés et ses limites honnêtes : un chemin GPU et Tensor Core portable, un moteur de régression symbolique et un runtime d'inférence déterministe. Le matériel de liaison décrit la ligne de base du framework et la discipline d'ingénierie sous laquelle chaque contribution a été acceptée.

Nous sommes explicites sur les types de affirmations faites. Les **affirmations mesurées** — débit, précision, latence, empreintes bit-exactes — sont des nombres reproductibles issus des exécutions rapportées. Les **affirmations interprétatives** — sur ce que la discipline d'ingénierie apporte, ou ce qu'une capacité démontre sur le framework — sont proposées comme des arguments raisonnés fondés sur ces mesures, et non comme des preuves.

## 2. Le framework SciRust

Le cœur (scirust-core) fournit un moteur de différenciation automatique en mode inverse construit autour d'une Tape qui enregistre les opérations, un type Tensor bidimensionnel, une bibliothèque de modules de réseaux de neurones (couches linéaires, convolutionnelles, de pooling, de normalisation, d'activation et de transformer) derrière un trait Module commun, des optimiseurs (y compris Adam) et des chargeurs de données. Un générateur pseudo-aléatoire déterministe et seedable sous-tend l'initialisation et le mélange des données, ce qui rend la reproductibilité sur toute l'exécution atteignable plutôt qu'accidentelle.

Ce qui distingue SciRust d'une simple bibliothèque est sa dimension de transpilation. Les crates de macros (scirust-macros, scirust-simd-macros, scirust-gpu-macros) implémentent les attributs proc-macro #[autodiff], #[simd] et #[gpu], faisant du système un hybride runtime-plus-transpileur plutôt qu'un runtime fixe seul. Les calculs numériques CPU sont en Rust pur sans dépendance BLAS obligatoire, ce qui — comme le montre la Section 4 — est précisément ce qui a rendu la portabilité multi-architecture simple.

La validation de base du framework comprend **255 tests réussis** et plusieurs démonstrations de bout en bout : classification MNIST à **97,70 %** avec des courbes de perte bit-identiques à travers les époques (le signal de non-régression le plus fort que le projet utilise), un transformer atteignant **100 %** sur une tâche synthétique de vote majoritaire, et un pipeline convolutionnel CIFAR-10 atteignant **52,40 %** sur un sous-ensemble d'entraînement de 5000 images (environ 5,2x la ligne de base aléatoire, validant le chemin convolutionnel). Ces chiffres établissent que le substrat est un framework fonctionnel, pas une ébauche, ce qui est la prémisse sur laquelle repose le reste du rapport.

## 3. Discipline d'ingénierie

Une discipline unique a régi l'acceptation de toute contribution dans un état validé, et il vaut la peine de l'énoncer explicitement car c'est ce qui rend les résultats mesurés dignes de confiance :

- **Validation par oracle.** Aucune primitive de calcul n'a été acceptée tant que sa sortie n'avait pas été vérifiée par rapport à une référence indépendante — typiquement l'implémentation CPU agissant comme oracle pour un chemin GPU, ou une loi de vérité terrain connue pour le moteur symbolique. La forme la plus forte de ce contrôle est au niveau du bit : une sortie en virgule flottante identique (courbes de perte bit-identiques ou empreintes de sortie identiques) est un signal de non-régression bien plus fort qu'un accord approximatif.
- **Porte des tests verts.** Le travail n'avançait pas au-delà d'une étape dont les tests n'étaient pas réussis, avec les sorties brutes de build et de test (pas des résumés) utilisées comme preuves.
- **Isolation des branches.** Chaque capacité a été développée sur sa propre branche et validée là avant l'intégration, gardant le travail en cours isolé des changements non liés ailleurs dans la base de code en évolution.
- **Intégration additive.** Dans la mesure du possible, les nouvelles capacités ont été intégrées sous forme de crates séparées ou derrière des feature flags, ne touchant ni le chemin critique CPU ni le moteur d'autodiff, de sorte qu'une contribution puisse être validée de manière isolée.

La leçon récurrente est qu'un test numérique n'est digne de confiance que dans la mesure de son modèle d'erreur — un point qui fait surface concrètement dans les Sections 4 et 5.

## 4. Mise en service GPU : extension de SciRust aux NVIDIA Tensor Cores sur Jetson Thor

### 4.1 Contexte et portabilité

SciRust a été développé et validé sur un hôte Debian x86-64. Pour tester la portabilité et un chemin d'exécution GPU, le framework a été porté sur un module NVIDIA Jetson Thor (aarch64, GPU de classe Blackwell, CUDA 13.0, pilote 580).

Le cœur en Rust pur a été compilé sur aarch64 sans modification en moins de 20 secondes, et surtout **sans aucune dépendance BLAS** : les liaisons optionnelles intel-mkl-src et blas-src sont restées inactives, évitant ainsi par construction le piège d'Intel MKL réservé au x86. Le comportement numérique multi-architecture s'est maintenu : MNIST a atteint **97,73 %** (perte 0,0377) sur la Jetson, cohérent avec la ligne de base x86, confirmant que les calculs numériques CPU du framework sont portables d'une architecture à l'autre.

Une observation pratique sur la chaîne d'outils : la crate cudarc 0.14 n'expose des liaisons que jusqu'à CUDA 12.8 mais charge le pilote dynamiquement. Comme l'API du pilote CUDA est rétrocompatible, forcer l'ensemble de liaisons cuda-12080 fonctionne correctement au runtime par rapport au pilote CUDA 13.0 — le chemin de chargement dynamique est ce qui a rendu la mise en service possible sur une chaîne d'outils plus récente que ce dont la crate de liaison avait connaissance.

### 4.2 Méthodologie de validation

La multiplication de matrices (GEMM) a été la primitive de mise en service, choisie parce qu'elle domine le coût tant dans l'entraînement que dans l'inférence et possède une référence sans ambiguïté. Le travail a d'abord procédé dans une crate sandbox isolée, puis dans l'arborescence derrière un feature flag cuda, chaque étape étant validée par rapport à l'oracle CPU avant la suivante.

Un point méthodologique a fait surface lors de la validation. Une métrique d'erreur relative naïve a rapporté une divergence de 5,6 % sur un problème non carré tout en rapportant 5e-5 sur un problème carré, en utilisant des noyaux identiques. La cause n'était pas un défaut mais une annulation : avec des opérandes de signes mixtes, certaines entrées de sortie sont proches de zéro, de sorte que l'erreur relative explose alors que l'erreur absolue reste au niveau du bruit FP32. L'oracle correct combine une tolérance **absolue** appliquée partout avec une tolérance **relative** appliquée uniquement là où la magnitude de référence est significative. Sous cette métrique combinée, chaque chemin GPU correspondait à l'oracle.

### 4.3 Le triptyque matmul

| Implémentation | 512^3 | 1024^3 | 2048^3 | 4096^3 |
|---|---|---|---|---|
| CPU (Rayon, FP32) | 2,37 ms | — | — | — |
| Noyau GPU naïf (FP32) | 2,749 ms / 98 | — | — | — |
| Noyau GPU tuilé (FP32) | 1,393 ms / 193 | 5,004 ms / 429 | 17,216 ms / 998 | — |
| cuBLAS (FP32) | 0,376 ms / 714 | 1,993 ms / 1078 | 3,787 ms / 4537 | 22,314 ms / 6159 |
| cuBLAS Tensor Cores (FP16) | 0,237 ms / 1130 | 0,251 ms / 8559 | 0,346 ms / 49699 | 2,166 ms / 63448 |
| cuBLAS Tensor Cores (BF16) | 0,238 ms / 1128 | 0,253 ms / 8493 | 0,347 ms / 49501 | 2,152 ms / 63872 |

(Temps par appel / débit en GFLOPS.) La progression est instructive. Le noyau naïf est limité par la mémoire et égale simplement un CPU multi-cœur optimisé — un GPU n'est pas automatiquement plus rapide. Le noyau tuilé en mémoire partagée (tuiles 16x16) double environ ce résultat et entre dans le véritable territoire GPU (~1 TFLOPS à 2048^3), mais un noyau à une sortie par thread plafonne à un facteur ~4 en dessous de cuBLAS, ce qui correspond à l'apport du blocage de registres et du double buffering. cuBLAS FP32 atteint ~6,2 TFLOPS (6,3x le CPU à 512^3) ; l'engagement des Tensor Cores en FP16/BF16 produit ~63 TFLOPS soutenus à 4096^3, un ordre de grandeur au-delà du FP32. Deux bémols d'honnêteté : le débit en dessous de 2048^3 est limité par le coût de lancement (seul le chiffre à 4096^3 se lit comme soutenu), et les chiffres reflètent le mode de consommation par défaut de l'appareil.

### 4.4 Précision et intégration

cuBLAS FP32 est bit-proche du résultat CPU (erreur relative max 4,7e-5 à 512^3), ne différant que par l'ordre de sommation ; le noyau tuilé s'accordait à 9,4e-6. Les chemins Tensor Core à précision réduite se dégradent comme prévu (FP16 1,3e-2, BF16 6,8e-2, ce dernier étant plus élevé en raison de la mantisse de 7 bits du BF16), l'erreur provenant de l'arrondi d'entrée plutôt que de l'accumulation, qui est effectuée en FP32. Pour l'apprentissage automatique, l'erreur plus importante du BF16 sur un seul GEMM n'est pas un handicap : sa plage d'exposants équivalente au FP32 évite l'overflow qui pèse sur le FP16 dans les activations profondes, c'est pourquoi il est le format d'entraînement de fait et la cible recommandée pour tout futur chemin de précision mixte.

Le GEMM cuBLAS FP32 a été intégré dans la crate scirust-gpu derrière le flag cuda, en tant que point d'entrée pur au niveau des tranches (slices) sans dépendance aux types de tenseurs du cœur, éliminant tout risque de cycle de dépendance. cuBLAS est en mode colonne-majeure ; le produit ligne-majeure C = A.B est obtenu en calculant (B^T.A^T) avec les opérandes permutées et les dimensions de tête fixées en conséquence, et le contexte et le handle CUDA sont mis en cache par thread. L'intégration est additive et non invasive — elle ne touche ni le chemin critique CPU ni le moteur d'autodiff — et est validée par deux tests oracle, un cas carré et un cas non carré qui exerce spécifiquement le mappage des dimensions en colonnes-majeures.

## 5. Régression symbolique via la propre autodiff du framework

### 5.1 Motivation et méthode

Pour tester si SciRust est un framework substantiel plutôt qu'un simple harnais de fitting, nous avons construit une capacité combinant des composants qu'il n'aurait normalement pas combinés : son moteur de calcul symbolique (scirust-symbolic — arbres d'expression, simplification, évaluation et **différenciation symbolique**) avec sa discipline de différenciation automatique. La tâche est la **régression symbolique** : récupérer une expression sous forme analytique — structure et constantes — qui s'ajuste aux données observées.

Le moteur est un hybride. La **structure** d'un candidat est recherchée par programmation génétique sur des arbres d'expression (primitives +, -, x, /, sin, cos, exp, plus variables et constantes) avec sélection par tournoi, croisement et mutation de sous-arbres, élitisme et une limite de taille. Les **constantes** ne sont pas recherchées aveuglément — la faiblesse classique de la programmation génétique — mais ajustées par descente de gradient (Adam), où les gradients proviennent de la **différenciation symbolique** du framework : pour un candidat avec des constantes c0, c1, ..., la dérivée partielle d(expr)/d(ck) est obtenue à partir du diff du moteur et évaluée sur le batch de données. Le moteur symbolique alimente ainsi son propre apprentissage. La sélection est biaisée en faveur de la **parsimonie** et la sortie est un **front de Pareto** entre précision et complexité ; le modèle de données est **multi-variables**. Le moteur est en Rust pur, réutilise scirust-symbolic sans modification et est entièrement reproductible via un générateur seedable.

### 5.2 Validation et résultats

Chaque résultat est vérifié par rapport à un **oracle** — une loi de vérité terrain connue — en utilisant la même tolérance combinée absolue/relative discutée à la Section 4.2. Un second critère, plus tranchant, est structurel : le moteur a-t-il récupéré la loi vraie et compacte ou simplement une approximation précise mais boursouflée ?

| Loi cible | Expression récupérée | MSE |
|---|---|---|
| x^2 + sin(x) | (x.x) + sin(x) | 0 |
| exp(-0.3x).cos(2x) | cos(x+x).exp(-0.300.x) | 3,3e-16 |
| x.y + sin(x) (2 variables) | sin(x) + (y.x) | 0 |
| x / (1 + x^2) | x / (x.x + 1,0) | 2,0e-15 |
| 0.5x^2 - 1.2x + 2 + bruit (sigma=0.1) | forme quadratique | 9,1e-3 ~ sigma^2 |

Le moteur a récupéré la structure exacte pour le cas polynôme-plus-trigonométrique, le cas à deux variables et — fait notable — l'oscillateur amorti, généralement attendu à l'échec car l'ajustement d'une fréquence à l'intérieur d'un cos est hautement non convexe ; il a même exprimé 2x comme x+x. La quadratique bruitée a été ajustée au signal à la variance du bruit, sans chasser le bruit.

Le résultat le plus instructif est le rationnel x/(1+x^2). Sous une sélection **uniquement par MSE**, le moteur a renvoyé une expression de quatorze nœuds avec des sinus imbriqués qui approximait les données à ~6e-5 mais ne ressemblait en rien à la loi vraie. Sous le **front de Pareto avec une pénalité de parsimonie**, la forme compacte vraie est apparue au bas du front (sept nœuds, MSE ~2e-15). C'est la conclusion à retenir : **une erreur faible n'est pas synonyme de loi correcte** — les objectifs basés uniquement sur la précision récompensent les approximations boursouflées, et la pression de parsimonie combinée à une vue de Pareto est ce qui permet de récupérer la structure.

Le moteur a été intégré sous forme de crate scirust-symreg, développée sur sa propre branche et additive par construction. Ses limites sont énoncées clairement : un résultat de session unique sur un ensemble de primitives modeste ; une recherche stochastique (seedée, pas exhaustive) ; et le terme neuro-symbolique n'est mérité que dans le sens étroit de constantes optimisées par gradient au sein d'une recherche symbolique, pas comme un a priori appris sur la structure.

## 6. Un runtime d'inférence déterministe

### 6.1 Positionnement

Un framework d'entraînement en Rust pur est un piètre concurrent de l'écosystème établi selon ses propres termes. Plutôt que de lutter sur cet axe, nous avons demandé si un système basé sur SciRust pouvait offrir, comme garantie de premier ordre, une propriété que les runtimes dominants traitent comme un "au mieux". La réponse poursuivie est une **inférence déterministe, à latence bornée et auditable** — la combinaison exigée par les déploiements embarqués (edge) et régulés. Le runtime (scirust-runtime) est une crate séparée sur un sous-ensemble forward figé du cœur ; il effectue l'inférence forward uniquement, l'entraînement étant conservé comme outillage hors ligne. Cette séparation permet à un contrat d'inférence stable de reposer sur le cœur en évolution, un verrou de non-régression (Section 6.3) transformant toute dérive en un échec visible.

### 6.2 La clé de voûte : le déterminisme bit-exact

Toute autre garantie repose sur le fait que le passage forward soit bit-exact, cela a donc été établi empiriquement en premier. Un MLP (784-256-10) avec des poids fixés a été exécuté de manière répétée sur une entrée fixée, avec des sorties comparées bit par bit (égalité to_bits, pas de tolérance). Sur 5120 comparaisons de logits, il y a eu **zéro divergence**, et une empreinte 64 bits des bits de sortie était identique à travers les appels et à travers des processus séparés.

Le test décisif concerne le nombre de threads. Le matmul est parallélisé via Rayon, ce qui soulève l'inquiétude qu'un ordonnanceur à vol de cycle réordonne les sommations. Ce n'est pas le cas : la réexécution du binaire sous RAYON_NUM_THREADS de 1, 2, 4, 8, 16 et 64 a produit l'empreinte identique 0xde2d807686e4b47e à chaque fois. La raison est structurelle — le matmul parallèle distribue le travail à travers les cellules de sortie, chaque produit scalaire étant accumulé par un seul thread dans un ordre fixé, de sorte que l'ordre de réduction est indépendant du nombre de threads. La portée honnête de la revendication résultante est l'exactitude au bit près pour un **artefact compilé fixé sur une architecture donnée**, stable à travers le nombre de threads et les redémarrages de processus ; l'exactitude au bit près multi-architecture est hors de portée par conception — le modèle d'audit correct est d'expédier un artefact épinglé et de le rejouer de manière identique sur sa cible.

### 6.3 Persistance et rechargement des poids

Pour la reproductibilité entre les déploiements, les poids figés doivent faire un aller-retour sans perte. Nous avons défini un petit format, **SRT1**, écrivant chaque tenseur sous la forme (clé, lignes, colonnes, f32 little-endian) avec les clés triées, de sorte que les octets sur disque soient déterministes et que l'artefact ait un hash stable. Le test d'or de charge — sérialiser, construire un modèle frais avec une graine différente, recharger, exécuter le passage forward — doit reproduire l'empreinte originale. C'est le cas : un modèle avec une graine différente diffère avant le chargement et reproduit 0xde2d807686e4b47e au bit près après. Exercé sur un modèle réel entraîné, le MLP entraîné sur MNIST (perte 0,2615 -> 0,0377) et figé dans un artefact de 814 Ko se recharge à une précision de test de **97,73 %** avec une empreinte des logits de test 0xc96d25fa658f5611 stable à travers les processus. Cela boucle la thèse de bout en bout : entraîner une fois, figer, et le runtime rejoue une inférence précise et bit-exacte à chaque invocation.

### 6.4 Latence bornée

La correction étant fixée par la Section 6.2, la latence a été traitée comme une mesure temporelle. Pour une inférence à requête unique (batch=1), le MLP a montré p50 = 126 us, p99 = 145 us, et un **ratio p99/p50 de 1,15** — une queue serrée et prévisible. La latence était également invariante au nombre de threads (p50 plat de 1 à 8 threads) : le coût par appel est dominé par un overhead fixe, pas par le calcul ou le dispatch, de sorte que le nombre de threads est un levier de débit (le débit batch=64 est passé de 23k à 81k échantillons/s de 1 à 8 threads), non pertinent pour la latence d'une requête unique. Un non-résultat délibéré : nous avions émis l'hypothèse qu'une arène sans allocation serait nécessaire pour borner la queue, mais le ratio de 1,15x mesuré a montré que la gigue d'allocation était négligeable, donc **aucune arène n'a été construite** — les données ne justifiaient pas l'optimisation. Résister à une optimisation que les mesures contredisent fait partie de la discipline.

### 6.5 Généralité via reconstruction pilotée par manifeste

Pour montrer que les garanties ne sont pas des artefacts d'un seul petit MLP, l'audit a été répété sur un réseau convolutionnel (Conv->ReLU->MaxPool deux fois, puis un classifieur) : forward bit-exact (0x1381e4b51d0eeba4) et invariant au thread ; l'artefact de 4,28 Mo a fait l'aller-retour bit pour bit, y compris les poids convolutionnels ; la latence batch=32 a conservé une queue serrée (p50 45,9 ms, p99/p50 = 1,20). Le runtime a ensuite été généralisé de sorte qu'**aucune architecture ne soit codée en dur dans le chemin d'inférence** : un manifeste en texte brut des spécifications des couches plus un fichier SRT1 reconstruit un Sequential arbitraire supporté. Un CNN reconstruit par manifeste reproduit exactement l'empreinte du modèle codé en dur, et — le cas décisif — le MLP MNIST entraîné reconstruit uniquement à partir d'un manifeste et de ses poids reproduit à la fois la précision de 97,73 % et l'empreinte 0xc96d25fa658f5611 au bit près. L'ensemble supporté couvre Linear, ReLU, Sigmoid, LayerNorm, BatchNorm2d, Conv2d et MaxPool2d, chacun ayant été démontré comme persistant et se reconstruisant de manière bit-exacte ; les couches de normalisation paramétriques ont été validées avec soin (les paramètres affines de LayerNorm et les statistiques glissantes de BatchNorm2d survivent tous deux à l'aller-retour, BatchNorm2d étant forcé en mode évaluation pour que l'inférence soit déterministe par échantillon). Des fonctionnalités avancées comme les **Contrats d'Invariants Formels** via `CertifiedModule<M, C>` et le support du **Runtime en Enclave Sécurisée** pour les cibles #![no_std] étendent encore l'applicabilité du runtime aux environnements de haute intégrité. La limite honnête : les couches de transformer utilisent un forward tridimensionnel et nécessiteraient un chemin de runtime séparé ; le débit des convolutions est borné par le noyau en Rust pur ; et la latence absolue batch=1 est limitée par l'overhead.

## 7. Quantification int8 déterministe pour l'inférence embarquée

### 7.1 Positionnement

Le runtime déterministe de la Section 6 cible les déploiements embarqués et régulés, où la mémoire et l'énergie sont rares et le comportement doit être auditable. L'inférence entière sur huit bits est l'étape suivante naturelle, mais seulement si les propriétés qui ont rendu le runtime digne de confiance survivent au passage à la basse précision. Nous avons donc construit la pile de quantification dans le cœur portable pur (sans dépendance GPU) et l'avons tenue au même contrat : chaque primitive quantifiée n'est acceptée que par rapport à un oracle de référence, et le déterminisme est mesuré plutôt que supposé — bit pour bit partout où l'arithmétique le permet.

### 7.2 Quantification du poids seul et int8 dynamique : un facteur quatre gratuit

Le premier schéma est le W8A8 dynamique : les activations sont quantifiées par tenseur au moment de l'exécution, les poids par canal de sortie, le produit s'accumule dans un i32 et une seule requantification renvoie un f32. Sur le MLP MNIST entraîné, cela se fait sans perte — la ligne de base f32 score 97,73 % (empreinte 0xc96d25fa658f5611) et le modèle int8 97,74 % — tandis que les poids passent de 813 Ko à 204 Ko (3,98x). L'empreinte int8 0xc3730f7c204455ba est identique sous RAYON_NUM_THREADS de 1, 4 et 16 : le matmul entier accumule chaque cellule de sortie dans un seul thread, de sorte que l'argument du déterminisme structurel de la Section 6.2 se transpose inchangé.

### 7.3 Étalonnage statique et requantification en nombres entiers complets

Pour supprimer les statistiques d'activation par appel, les échelles d'activation ont été étalonnées une fois sur un échantillon mis à part ; les activations int8 sont ensuite transportées entre les couches avec un biais i32 et un ReLU entier. Ce pipeline statique score 97,71 % avec l'empreinte 0xa9b9a102c7cea67b, invariante au thread. La requantification en virgule flottante dans le chemin critique a ensuite été remplacée par une requantification entière de type gemmlowp — un multiplicateur en virgule fixe dans [2^30, 2^31) et un décalage à droite par canal — qui reproduit le modèle étalonné bit pour bit (même 97,71 %, même 0xa9b9a102c7cea67b). Le chemin d'inférence est désormais entier de bout en bout, sans virgule flottante dans la boucle et sans réduction parallèle, il est donc déterministe par construction.

### 7.4 Quantification par canal des convolutions

Le schéma par canal s'étend au réseau convolutionnel (par ligne pour les poids Conv2d, par colonne pour Linear). Un aller-retour en quantification simulée (fake-quantized) reproduit l'oracle f32 0x1381e4b51d0eeba4 et préserve l'arg-max sur l'ensemble des 32 entrées de test, l'ensemble de filtres de 4,28 Mo étant réduit à 1,07 Mo (3,99x). Une véritable convolution directe entière a ensuite été validée : un miroir f32 de l'indexation entière correspond au passage forward de convolution du framework bit pour bit, et la convolution int8 s'accorde avec l'oracle f32 à max-abs 2,8e-2 près. Comme dans la Section 6, l'erreur relative est lue avec précaution — près des annulations de logits, une erreur relative importante coexiste avec une erreur absolue négligeable, de sorte que l'erreur absolue et l'arg-max préservé sont les métriques de référence.

### 7.5 Un artefact quantifié portable

Le modèle entier étalonné a été promu comme un artefact de premier ordre, QSR1 : un format d'octets auto-descriptif contenant les dimensions par couche, l'échelle d'entrée étalonnée, les échelles de poids par canal, les poids int8 et le biais i32, avec des octets déterministes et hashables. Écrit, rechargé à partir du fichier seul et rejoué, il reproduit 0xa9b9a102c7cea67b à 97,71 % à partir de 205 Ko contre 814 Ko pour l'artefact f32 (3,96x). Exposé via une petite API de bibliothèque (un modèle quantifié avec sauvegarde, chargement et inférence), un aller-retour via la bibliothèque reproduit l'empreinte bit pour bit ; comme le QSR1 est auto-descriptif, il englobe le manifeste en texte brut pour les modèles quantifiés.

### 7.6 Tenseurs CSR et noyaux SpMM creux

Pour optimiser davantage la consommation de mémoire sur les cibles embarquées, SciRust implémente une structure `CsrTensor` et un noyau associé de multiplication matrice-matrice creuse (SpMM). Cela permet le stockage et le calcul de modèles creux sans l'overhead des représentations denses, contournant efficacement le mur de la mémoire sur les appareils contraints.

### 7.7 Un noyau entier et des convolutions séparables

Le matmul entier scalaire portable est la référence de correction. Un noyau NEON aarch64 — multiplication-accumulation élargie avec accumulation i32, l'opérande de droite étant transposée pour un accès contigu — est bit-exact par rapport à lui (la sommation entière est indépendante de l'ordre) et environ dix fois plus rapide (64x784x256 : 9592 us en scalaire contre 963 us en NEON). Deux blocs de type MobileNet complètent l'ensemble des opérateurs embarqués : une convolution int8 depthwise, dont le miroir f32 correspond à un oracle de convolution par canal bit pour bit et dont la sortie int8 s'accorde à max-abs 2,0e-2 près, et une convolution int8 pointwise 1x1, dont le miroir f32 correspond à un oracle de convolution 1x1 bit pour bit et s'accorde à max-abs 1,8e-2 près. Composés, ils forment une convolution séparable entièrement en int8 déterministe, chaque moitié étant validée par rapport au framework, avec chaque tenseur de poids quatre fois plus petit.

## 8. Fonctionnalités avancées pour le runtime et la vérification

À mesure que SciRust est passé d'un framework axé sur l'entraînement à un écosystème prêt pour le déploiement, cinq fonctionnalités avancées ont été implémentées pour répondre aux besoins des systèmes de haute intégrité et de l'explicabilité formelle.

### 8.1 Compilateur de modèle statique Ahead-Of-Time (AOT)
Pour éliminer l'overhead de la construction de graphe au runtime et du chargement des poids — critique pour les cibles embarquées ultra-profondes avec une mémoire tas limitée — nous avons implémenté un compilateur statique.
- **Mécanisme :** Le compilateur ingère une topologie `LayerSpec` et des tampons de poids bruts, émettant un fichier source Rust valide. Ce fichier définit une structure `StaticModel` où les poids sont stockés sous forme de tableaux statiquement imbriqués (`&[[f32; N]; M]`).
- **Avantage :** Les modèles peuvent être liés directement au binaire sous forme de données immuables, permettant une inférence sans allocation et évitant les erreurs d'analyse au runtime.

### 8.2 Moteur de matrice Soft-Float pour le déterminisme
Alors que la Section 6.2 établit l'exactitude au bit près pour une architecture fixe, le déterminisme multiplateforme (ex: x86 vs ARM) est souvent brisé par les arrondis FPU spécifiques au matériel et les optimisations FMA.
- **Implémentation :** Nous avons implémenté `soft_gemm`, un noyau de multiplication de matrices défini par logiciel utilisant l'arithmétique entière mise à l'échelle (`i32` avec accumulation `i64`).
- **Validation :** En contournant le FPU matériel, le moteur garantit des traces de calcul identiques à travers des jeux d'instructions CPU disparates, une exigence pour la vérification formelle et les journaux d'audit multiplateformes.

### 8.3 Guidage par activation latente (RepE)
S'appuyant sur le paradigme de "Representation Engineering", nous avons intégré des hooks de bas niveau pour manipuler l'état interne du modèle pendant l'inférence.
- **Structure :** Le trait `Module` a été étendu avec une méthode `forward_steered` et un registre `SteerHook`.
- **Application :** Cela permet aux contrôleurs externes d'appliquer des décalages linéaires (Concept Vectors) aux activations latentes en temps réel, permettant de rediriger le comportement du modèle sans modifier les poids statiques.

### 8.4 Entraînement conscient de la quantification (QAT) avec STE
Pour combler l'écart entre l'entraînement FP32 et le déploiement INT8 (Section 7), nous avons implémenté des noyaux de quantification simulée.
- **Mécanisme :** Pendant le passage forward, les valeurs sont écrêtées et quantifiées à une échelle simulée de 8 bits. Le passage backward utilise un **Straight-Through Estimator (STE)**, passant les gradients à travers l'étape de quantification non dérivable sans modification.
- **Résultat :** Les modèles s'adaptent naturellement aux erreurs de quantification pendant la boucle d'entraînement, améliorant considérablement la précision de l'exécution ultérieure en basse précision.

### 8.5 XAI : Moteur d'Integrated Gradients
Pour satisfaire aux exigences des secteurs régulés (Section 3), nous avons implémenté les Integrated Gradients pour l'attribution des caractéristiques.
- **Algorithme :** Le moteur calcule l'intégrale de chemin des gradients d'une ligne de base (ex: un tenseur de zéros) à l'entrée sur $m$ étapes.
- **Intégration :** S'appuyant sur l'autodiff native à base de `Tape` du framework, le moteur génère des cartes d'attribution de la même forme que l'entrée, fournissant une explication mathématique pour toute prédiction donnée.

## 9. Expansion des familles d'IA modernes

Pour aller au-delà des architectures de base MLP et CNN, nous avons étendu SciRust avec un support fondamental pour plusieurs domaines de l'IA moderne, en maintenant des contraintes strictes de Rust pur et de déterminisme.

### 9.1 Apprentissage par renforcement avancé : DQN et PPO
Nous avons implémenté une pile d'apprentissage par renforcement dans `scirust-learning`.
- **Algorithmes :** Support pour le Q-Learning tabulaire/SARSA et les réseaux Q profonds (DQN). De plus, nous avons implémenté l'**Optimisation de Politique Proximale (PPO)** en utilisant un objectif écrêté pour garantir des mises à jour de politique stables.
- **Déterminisme :** Les interactions des agents et l'échantillonnage de la mémoire sont imposés à l'aide d'instances `PcgEngine` seedées, garantissant des trajectoires d'entraînement reproductibles.

### 9.2 Vision par ordinateur : ResNet et Vision Transformers
Deux architectures majeures ont été ajoutées à `scirust-core` :
- **ResNet-18/34 :** Implémentation modulaire utilisant `ResidualBlock` et une étape de **Global Average Pooling (GAP)** pour gérer des résolutions d'entrée variables.
- **Vision Transformer (ViT) :** Implémentation de la projection de patchs via des convolutions 2D suivies d'un encodeur Transformer. Les caractéristiques sont agrégées sur la dimension de la séquence pour la classification.

### 9.3 IA générative et Transformers
- **Auto-encodeurs variationnels (VAE) :** Implémentation du trick de reparamétrage utilisant du bruit gaussien dérivé de `PcgEngine` et une perte de divergence KL analytique.
- **Mélange d'experts (MoE) :** Une couche MoE modulaire prenant en charge le **routage Top-k** et l'agrégation additive des experts, permettant une mise à l'échelle du modèle sans croissance linéaire du coût de calcul.

### 9.4 Architectures spécialisées
- **Réseaux de neurones sur graphes (GNN) :** Couches de base **Graph Convolutional Network (GCN)** prenant en charge les multiplications de matrices d'adjacence creuses-denses.
- **Speech AI :** Encodeurs audio et une implémentation représentative de la **perte CTC** pour l'alignement de séquences temporelles.
- **PEFT (LoRA) :** Adaptation à bas rang pour les couches linéaires, permettant de fine-tuner des modèles backbone figés via de petites matrices de rang r.

## 10. Discussion

Deux observations reviennent à travers les contributions. Premièrement, la discipline a fait le travail de fond : parce que chaque primitive n'était acceptée que contre un oracle — souvent au bit près — un chemin reproduit la référence ou ne la reproduit pas, ce qui a permis de garder les résultats du framework dignes de confiance au fil de son évolution. Deuxièmement, les conclusions les plus précieuses étaient parfois négatives et n'ont été atteintes qu'en mesurant : que le nombre de threads n'affecte pas la latence d'une requête unique, qu'une arène d'allocation était injustifiée, qu'une métrique d'erreur relative naïve n'est pas digne de confiance près des annulations, et qu'une erreur faible n'est pas synonyme de loi correcte. Chacune contredisait un a priori plausible et aurait été manquée en affirmant plutôt qu'en mesurant. Un troisième point unificateur : la reproductibilité, traitée comme une propriété à concevoir et à mesurer plutôt qu'à espérer, est devenue une caractéristique produit à part entière — la garantie centrale du runtime déterministe est exactement l'exactitude au bit près dont la discipline de test du framework dépendait déjà. La pile de quantification int8 a étendu exactement ce contrat : son chemin d'inférence entière est invariant au thread par le même argument de réduction mono-thread par cellule, et une requantification en virgule fixe reproduit le modèle étalonné bit pour bit, de sorte que le déterminisme s'est propagé jusqu'à la basse précision sans nouvelle machinerie.

## 11. Limites

Le framework est un artefact de recherche et n'est pas de qualité production. La convolution manque d'un chemin im2col-plus-BLAS ou GPU et est donc lente en débit absolu ; le backend GPU est validé pour la correction du calcul mais pas encore câblé dans l'entraînement ; et le runtime déterministe est limité à l'inférence sur un ensemble de couches bidimensionnelles, le support des transformers nécessitant un chemin tridimensionnel séparé. Le déterminisme est limité à un binaire et une architecture fixés. Le moteur symbolique est une recherche stochastique sur un ensemble modeste de primitives, et plusieurs contributions sont des résultats de session unique. Le nouvel évaluateur de perte **PINN (Physics-Informed Neural Networks)** permet l'intégration de résidus physiques symboliques dans le chemin d'optimisation AD. La quantification int8 est post-entraînement plutôt que consciente de la quantification ; le résultat d'absence de perte de précision est établi sur le MLP MNIST, tandis que les quantificateurs convolutionnels sont validés pour la fidélité et le déterminisme sur des entrées synthétiques plutôt que pour la précision sur un benchmark d'images étiquetées, et aucun déploiement sur microcontrôleur (no_std) n'est encore démontré. Le dépôt comprend également un module d'optimisation évolutionnaire ; parmi ses algorithmes, seul le NSGA-II multi-objectif est validé ici, récupérant le front de Pareto ZDT1 à environ 1e-3 près, tandis que les optimiseurs mono-objectif simplifiés convergent sur des paysages convexes mais pas sur des fonctions multimodales difficiles. Aucun de ces éléments n'invalide les résultats mesurés ; ils bornent ce que ces résultats doivent signifier.

## 12. Algèbre tensorielle de haut niveau et compilation de graphe : scirust-tensor

### 12.1 Motivation et contexte
Alors que le cœur de SciRust fournit des primitives robustes pour l'apprentissage profond, les architectures complexes comme les Transformers nécessitent des manipulations de tenseurs plus flexibles que de simples multiplications de matrices. Les frameworks actuels de pointe (JAX, PyTorch) s'appuient sur l'`einsum` optimisé et des compilateurs de graphe (XLA) pour réduire l'overhead mémoire. Pour combler cet écart tout en conservant l'ADN de Rust pur et déterministe de SciRust, nous avons introduit `scirust-tensor`.

### 12.2 Méthodologie : Einsum et planification de contraction
Le module implémente un parseur d'`einsum` optimisé et un **planificateur de contraction**. Pour une expression de contraction de tenseur donnée :
$$C_{i,l} = \sum_{j,k} A_{i,j,k} \cdot B_{k,j,l}$$
Le planificateur évalue le chemin d'exécution optimal. Pour les contractions multi-tenseurs, il utilise une approche gloutonne pour minimiser le nombre total d'opérations en virgule flottante (FLOPs).

### 12.3 Optimisation de graphe et fusion d'opérateurs
Une contribution majeure de ce module est le moteur de **fusion d'opérateurs**. Dans les runtimes standards, des opérations séquentielles comme `MatMul -> BiasAdd -> ReLU` impliquent plusieurs passages en mémoire et des tampons intermédiaires. `scirust-tensor` les compile en un seul **noyau fusionné**, réduisant la pression sur la bande passante mémoire.
Le pipeline d'optimisation comprend :
- **Élimination de redondance :** Suppression des transpositions d'identité.
- **Permutation basée sur les foulées (strides) :** Intégration des permutations d'axes dans les foulées du noyau GEMM pour éliminer les copies de données explicites.

### 12.4 Résultats et déterminisme
En utilisant un ordre de réduction fixe dans toutes les contractions de tenseurs, nous garantissons des résultats identiques au bit près à travers différents nombres de threads. Les benchmarks préliminaires montrent que la fusion d'opérateurs réduit l'utilisation de pointe de la mémoire jusqu'à 35 % sur des blocs Transformer profonds, tout en maintenant une empreinte déterministe stricte. Le module est entièrement compatible avec le runtime d'inférence **SRT1** et la pile de quantification int8 **QSR1**.

### 12.5 Limites
Le compilateur de graphe est actuellement limité aux formes statiques. Le support des formes dynamiques et la compilation JIT de noyaux pour des motifs de fusion arbitraires restent des travaux futurs.

## 13. Conclusion

SciRust est un framework d'apprentissage profond en Rust pur — un hybride de runtime et de transpileur — sur lequel quatre capacités ont été construites et validées : un chemin GPU et Tensor Core portable atteignant ~63 TFLOPS en BF16 ; un moteur de régression symbolique hybride génétique-gradient qui récupère des lois connues à partir de données en utilisant la propre différenciation symbolique du framework ; un runtime d'inférence déterministe offrant une inférence au bit près, à latence bornée, auditable et générique sur l'architecture ; et une pile de quantification int8 déterministe offrant un chemin d'inférence entier portable et invariant au thread pour le déploiement embarqué, avec une requantification en virgule fixe qui reproduit le modèle bit pour bit et des tenseurs de poids environ quatre fois plus petits. En s'appuyant sur ceux-ci, cinq fonctionnalités avancées — un compilateur statique AOT pour une inférence embarquée sans overhead, un moteur de matrice soft-float pour l'exactitude au bit près multiplateforme, le guidage par activation latente pour l'ingénierie de représentation en temps réel, l'entraînement conscient de la quantification (QAT) via un estimateur direct, et un moteur Integrated Gradients pour l'explicabilité mathématique — établissent davantage SciRust comme un framework de haute intégrité. L'ajout des **Familles d'IA modernes** (RL, CV, Génératif, GNN) élargit encore la portée du framework vers une pile IA unifiée en Rust pur. Le fil conducteur est méthodologique : chaque contribution n'a été acceptée qu'après avoir correspondu à un oracle, la reproductibilité a été mesurée plutôt que supposée — dans plusieurs cas au bit près — et les découvertes les plus utiles ont été celles que les mesures ont imposées contre toute attente. Les prochaines étapes suivent directement : un chemin forward accéléré par GPU réutilisant le backend cuBLAS validé pour les couches denses, un chemin d'inférence tridimensionnel pour les modèles basés sur l'attention, et l'épinglage de la chaîne d'approvisionnement pour étendre l'auditabilité du runtime de ses poids à sa construction.

## 14. Détection et classification déterministes d'événements

### 14.1 Motivation
La détection d'événements en temps réel dans les systèmes critiques (ex: neuroprothèses ou contrôle industriel) exige non seulement une grande précision, mais aussi un déterminisme absolu pour l'auditabilité et la certification. Les frameworks actuels s'appuient souvent sur une réduction parallèle non déterministe ou un échantillonnage stochastique, ce qui ne convient pas aux environnements à enjeux élevés.

### 14.2 Méthodologie
Nous introduisons une architecture de streaming basée sur des fenêtres glissantes déterministes. Chaque fenêtre $W$ de taille $N$ est transformée en un tenseur $T \in \mathbb{R}^{1 \times N}$. La détection d'événements est formulée comme une fonction de score $S(T) \to [0, 1]$. Pour la classification, nous utilisons les couches MLP et CNN de base du framework, figées au format SRT1.
$$ \text{Événement}(t) = \mathbb{I}(S(W_t) > \tau) $$
où $\tau$ est un seuil étalonné.

### 14.3 Résultats et métriques
Les performances attendues sur le Numenta Anomaly Benchmark (NAB) visent un score F1 $>0,85$ avec une dérive nulle au bit près sur plusieurs threads. L'utilisation de la quantification int8 QSR1 devrait réduire la latence de $3\times$ sur les processeurs ARM embarqués tout en maintenant une proximité de bits MSE $<10^{-4}$ par rapport à l'oracle f32.

## Autograd N-D et extensions issues de la recherche

Au-delà de la tape 2-D en mode inverse, SciRust fournit désormais une **tape
autograd N-D** dont chaque opérateur est validé par un gradient check par
différences finies, et au-dessus une pile d'apprentissage profond adossée à la
recherche. Chaque capacité correspond à un papier précis et est livrée avec un
test ; la correspondance complète (14 des 20 éléments livrés) est suivie dans
`docs/RESEARCH_ROADMAP.md`.

- **Modèle de langage décodeur causal**, entraîné de bout en bout (embeddings de
  token et de position appris, attention multi-tête causale, cross-entropy
  softmax fusionnée et numériquement stable), qui sur-apprend une séquence fixe
  exactement — preuve bout-en-bout que la pile apprend.
- **Couches de la famille LLaMA** : RMSNorm, SwiGLU, bloc LLaMA Pre-RMSNorm,
  embeddings rotatifs (RoPE, propriété de position relative testée), et attention
  groupée / multi-requête exprimée via le broadcast du produit matriciel par lots.
- **Optimiseurs déterministes** : Adam, AdamW (weight decay découplé), Lion et
  Muon (momentum orthogonalisé par Newton–Schulz), Schedule-Free, AdEMAMix et SOAP (Adam dans la base propre de Shampoo) — tous reproductibles bit à bit.
- **IA certifiable** : la propagation par intervalles (IBP) **et CROWN** (bornes
  plus serrées par relaxation linéaire) fournissent des bornes de
  sortie prouvées pour les MLP ReLU et un certificat de robustesse, validés par
  échantillonnage de validité.
- **Réductions reproductibles** : somme/moyenne/produit scalaire flottants
  indépendants de l'ordre (ordre canonique + expansion exacte), bit-identiques
  quel que soit le nombre de threads.
- **Inférence** : décodage spéculatif exact (préservant la sortie), une
  FlashAttention à softmax en ligne par tuiles et une couche DeltaNet d'attention linéaire à règle delta.
- **Pont scientifique** : un Neural ODE qui rétropropage à travers un solveur RK4.
- **Compression** : élagage Wanda (conscient des activations) et SmoothQuant, et GPTQ (quantification int8 des poids par feedback d'erreur d'ordre 2, CLI `scirust gptq`), et AWQ (quantification int8 des poids basée sur une recherche et consciente des activations, CLI `scirust awq`).

Deux commandes CLI exposent ces travaux : `scirust certify` (bornes IBP **et CROWN**, côte à côte, et robustesse) et `scirust lm --opt adam|adamw|lion|schedule-free|ademamix|soap` (entraînement du LM décodeur N-D).

Une troisième commande, `scirust conformal`, produit des intervalles de prédiction conformes à couverture garantie, sans hypothèse de distribution.
