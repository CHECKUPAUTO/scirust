# Visualisations — `scirust-tolerance`

Pages HTML autonomes (aucune dépendance réseau, thème clair/sombre) qui
rendent visibles les concepts du tolérancement inertiel.

## `inertia_cone.html` — Le cône d'inertie

Outil interactif du **cône d'inertie** de Pillet. Un lot est jugé par sa
distance à la cible dans le plan `(δ, σ)` — l'inertie `I = √(δ² + σ²)` — et
non par sa marge à l'intervalle de tolérance.

- **Plan (δ, σ)** — la carte d'acceptation (vue de dessus du cône) : le
  demi-disque d'inertie `I ≤ I_max` (Cpi ≥ 1) superposé au triangle
  d'acceptation `Cpk ≥ 1` de la méthode classique. On y voit d'un coup d'œil
  qu'un lot très précis mais décentré, ou centré mais dispersé, peut sortir du
  cône alors que le Cpk le tolérerait.
- **Cône 3D** — le graphe `z = I(δ, σ)` est un cône de révolution ; accepter
  `I ≤ I_max` revient à le couper par un plan horizontal. Glisser pour pivoter.
- **Distribution** — la densité `N(μ, σ)` du lot vis-à-vis de `[LSL, USL]` et
  de la cible, queues hors-spec ombrées.
- **Lecture directe** — `I`, `I_max`, `Cpi`, `Cpm`, `Cp`, `Cpk` et la
  non-conformité en ppm, recalculés en glissant le point de lot ou les
  curseurs (IT, Cp visé, μ, σ).

Les formules reprennent exactement celles de la crate : `InertiaCone`,
`Inertia`, `capability::cpi` — voir `scirust-tolerance/src/`.

Ouvrir le fichier dans un navigateur (`file://…/inertia_cone.html`), aucune
compilation ni serveur requis.
