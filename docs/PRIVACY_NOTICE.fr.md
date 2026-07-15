# SciRust — notice de confidentialité (modèle)

> **Modèle — ne constitue pas un avis juridique.** Cette notice ne couvre que les
> données personnelles traitées par les mécanismes de **licence et de provenance**
> de SciRust. Renseignez chaque `<champ>`, faites-la relire par votre délégué à la
> protection des données (DPO) et par un conseil qualifié, et publiez-la à l'URL
> référencée au § 5 de [LICENSING.fr.md](../LICENSING.fr.md). Elle est rédigée pour
> satisfaire aux exigences de transparence des articles 13 et 14 du règlement (UE)
> 2016/679 (**RGPD**). Version anglaise : [PRIVACY_NOTICE.md](PRIVACY_NOTICE.md).

## 1. Responsable de traitement

- **Responsable de traitement :** `<dénomination sociale>`, `<adresse>`,
  `<n° d'immatriculation>`.
- **Contact :** `<courriel>` · `<adresse postale>`.
- **Délégué à la protection des données (DPO) / contact vie privée :**
  `<nom / courriel, ou « non désigné — contacter ci-dessus »>`.
- **Représentant dans l'UE** (si le responsable est établi hors UE, art. 27) :
  `<nom / contact, ou « sans objet »>`.

## 2. Champ d'application

Cette notice ne concerne **que** les fonctions de licence et de provenance. Le
calcul réalisé par SciRust ne traite **pas** vos données de charge de travail, vos
entrées ni vos résultats : il n'y a ni télémétrie, ni connexion sortante, et la
vérification est effectuée entièrement hors-ligne. Elle ne s'applique pas à un
site web, à une activité de vente ou de support distincts, couverts par
`<lien vers votre politique de confidentialité générale>`.

## 3. Données personnelles traitées

| Donnée | Emplacement | Remarques |
|---|---|---|
| Identité du licencié (nom / organisation) et identifiant de licence | dans le fichier de licence délivré | fournies par vous lors de la délivrance d'une licence |
| **Numéro de série** de la signature à usage unique (« leaf ») | fichier de licence et artefacts signés | numéro par délivrance, utilisé pour l'authenticité et l'attribution des fuites |
| **Empreinte machine pseudonymisée** (SHA-256), *uniquement si vous optez pour le verrouillage machine* | dans le fichier de licence | condensé salé d'un identifiant machine que **vous** fournissez ; l'identifiant brut n'est jamais collecté ni stocké |

Aucune donnée de catégorie particulière (art. 9) n'est traitée. Aucune donnée
comportementale, d'usage ou de charge de travail n'est collectée.

## 4. Finalités et bases légales (art. 6)

| Finalité | Base légale |
|---|---|
| Délivrer et vérifier les licences ; accorder les droits d'usage achetés | **Exécution du contrat**, art. 6, §1, b) |
| Authenticité des marques de provenance et **attribution des copies fuitées / contrefaisantes** (anti-piratage) | **Intérêt légitime**, art. 6, §1, f) — protéger la propriété intellectuelle du responsable ; mis en balance avec vos intérêts, et limité à un numéro de série et à un condensé salé (cf. § 3) |
| Respecter des obligations légales (comptabilité, réponse aux demandes légales) | **Obligation légale**, art. 6, §1, c) |

Lorsque la base est l'intérêt légitime, vous pouvez **vous opposer** (art. 21) ;
voir § 7.

## 5. Destinataires et transferts hors UE

- **Destinataires :** `<aucun / vos sous-traitants, p. ex. hébergeur ou CRM — à
  lister>`. Tout sous-traitant agit dans le cadre d'un contrat écrit (art. 28).
- **Transferts hors UE :** `<aucun / le cas échéant, la garantie utilisée —
  décision d'adéquation ou outil de l'art. 46 (clauses types)>`.
- La vérification s'exécute hors-ligne sur vos systèmes ; le responsable ne reçoit
  aucune donnée de votre déploiement en fonctionnement.

## 6. Durées de conservation

- Données de licence (identité, identifiant de licence, numéro de série) :
  `<p. ex. durée de la licence + <n> ans>` à des fins contractuelles, de garantie
  et de défense des droits de PI.
- Les données qui ne sont plus nécessaires sont supprimées ou anonymisées.
  `<Fixez des durées concrètes.>`

## 7. Vos droits

Dans les conditions prévues par le RGPD, vous pouvez demander l'**accès**
(art. 15), la **rectification** (art. 16), l'**effacement** (art. 17), la
**limitation** (art. 18), la **portabilité** (art. 20), et vous **opposer** au
traitement fondé sur l'intérêt légitime (art. 21). Pour les exercer, contactez
`<contact vie privée du § 1>`. Vous avez également le droit d'introduire une
réclamation auprès d'une autorité de contrôle — en France, la **CNIL**
(www.cnil.fr) — ou auprès de l'autorité de votre lieu de résidence habituelle.

## 8. La fourniture de ces données est-elle obligatoire ?

La fourniture de l'identité du licencié est une **exigence contractuelle** pour
délivrer et vérifier une licence ; sans elle, aucune licence ne peut être
accordée. Le verrouillage machine est **facultatif** — si vous n'y optez pas,
aucune empreinte machine n'est traitée.

## 9. Décision automatisée

La vérification de licence est un contrôle de validité déterministe qui accorde ou
refuse une capacité logicielle et qui est **entièrement réversible** par
l'installation d'une licence valide. Le responsable ne met en œuvre aucune décision
automatisée produisant des effets juridiques ou vous affectant de manière
significative au sens de l'**art. 22**.

## 10. Source des données (art. 14)

Lorsque des données personnelles (p. ex. l'identité de votre contact désigné)
parviennent au responsable via le licencié plutôt que directement auprès de la
personne concernée, leur source est l'**organisation licenciée** ayant demandé la
licence.

## 11. Modifications

`<Version / date>`. Nous mettrons cette notice à jour si nécessaire et indiquerons
ici la date de la dernière révision.
