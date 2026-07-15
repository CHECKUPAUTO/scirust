# Licence de SciRust

_Version française de [LICENSING.md](LICENSING.md). En cas de contrat avec un
consommateur résidant en France, la version française prévaut (Loi n° 94-665 du
4 août 1994, dite « Toubon »)._

SciRust est à sources ouvertes (« source-available ») sous les termes complets de
la **PolyForm Noncommercial License 1.0.0** figurant dans [LICENSE.md](LICENSE.md).
Conservez ce fichier et sa ligne `Required Notice:` lors de la distribution de
toute copie ou version modifiée.

L'usage commercial n'est pas accordé par les termes PolyForm Noncommercial. Un
contrat commercial distinct peut être obtenu auprès du titulaire des droits :

- Tarek Zekriti
- zekrititarek@gmail.com

Un contrat commercial est distinct de la licence du dépôt et n'est valable que
s'il est accordé par écrit par le titulaire des droits. Ce résumé est fourni à
titre d'orientation ; en cas de contradiction avec [LICENSE.md](LICENSE.md), les
termes complets de la licence prévalent.

---

## Provenance, marquage d'intégrité et licence — information (UE)

> **Modèle — ne constitue pas un avis juridique.** Cette clause informe des
> mécanismes d'intégrité/provenance et de licence présents dans SciRust et est
> rédigée pour l'**Union européenne** (les transpositions françaises sont
> indiquées, le titulaire des droits étant établi dans l'UE). Faites-la relire et
> adapter par un avocat PI/IT UE-français qualifié, et alignez le § 5 avec votre
> délégué à la protection des données (DPO), avant de vous en prévaloir. Les
> renvois aux directives sont indicatifs ; vérifiez les transpositions en vigueur.
> La mise en œuvre opérationnelle figure dans
> [docs/PROVENANCE_OPERATIONS.md](docs/PROVENANCE_OPERATIONS.md).

**1. Les marques de provenance = informations électroniques relatives au régime
des droits.** Les artefacts générés par SciRust, et les binaires construits à
partir de celui-ci, peuvent porter des informations de provenance cryptographiques
— une signature à base de hachage (Lamport/Merkle) intégrée au code source généré,
et des marqueurs de chemin d'exécution neutres sur le résultat — identifiant le
moteur d'origine et la version. Il s'agit d'**informations électroniques relatives
au régime des droits** au sens de l'**article 7 de la directive 2001/29/CE**
(« InfoSoc »), transposé en France dans le Code de la propriété intellectuelle
(CPI, art. L.331-11 et s., assorti de sanctions pénales). Vous ne devez pas, en
connaissance de cause, les supprimer, altérer, masquer ou falsifier, ni distribuer
des copies dont elles auraient été retirées ou modifiées, dès lors que vous savez
ou avez des raisons valables de penser que cela induit, permet, facilite ou
dissimule une atteinte à un droit.

**2. Vos droits impératifs sur le logiciel sont préservés.** Rien dans cette
clause ne restreint, et elle ne saurait être interprétée comme restreignant, les
droits que vous détenez au titre de la **directive 2009/24/CE** concernant la
protection juridique des programmes d'ordinateur et qui ne peuvent être écartés
par contrat — en particulier le droit d'**observer, étudier ou tester** le
programme afin d'en déterminer les idées et principes sous-jacents (art. 5, §3) et
le droit de **décompilation à des fins d'interopérabilité** (art. 6) ; toute clause
contractuelle contraire est **nulle et non écrite** (art. 8 ; en France, CPI
art. L.122-6-1). Les sections 1 et 5 portent sur l'intégrité des informations de
provenance et sur les données personnelles et s'appliquent **sans préjudice** de
ces exceptions impératives.

**3. Intégrité numérique et reproductibilité.** Les marques de provenance sont
**neutres sur le résultat** : elles ne modifient pas les résultats numériques
calculés par SciRust, et les chemins numériques bit-à-bit sont protégés par des
tests de reproductibilité. Une **build sans marquage** est disponible (la
configuration par défaut, marquage désactivé), de sorte que le marquage
d'intégrité ne compromet jamais la reproductibilité scientifique ni votre capacité
à vérifier les résultats de manière indépendante.

**4. Application de la licence — gracieuse, hors-ligne, proportionnée.** Certaines
capacités à forte valeur (par ex. le module d'accélération GPU) peuvent exiger une
licence valide et signée pour s'exécuter. L'application est **gracieuse et non
destructrice** : une build non licenciée refuse d'armer la capacité concernée et
renvoie une erreur claire — elle ne corrompt jamais de données, ne dégrade pas les
résultats, n'endommage pas le matériel et n'altère pas silencieusement la sortie.
La vérification est effectuée **entièrement hors-ligne** ; SciRust ne se connecte à
aucun serveur (« phone home ») et ne transmet aucune télémétrie ni donnée d'usage.
Une licence peut, en option, être liée à une machine que vous désignez ; vous
pouvez la réactiver sur une autre machine en y installant le fichier de licence qui
vous a été délivré, sans contacter le fournisseur. Pour tout licencié ayant la
qualité de **consommateur**, cette application s'exerce sans préjudice des droits à
la conformité et aux remèdes prévus par la **directive (UE) 2019/770** (contenus et
services numériques) et de la protection contre les clauses abusives prévue par la
**directive 93/13/CEE** (France : Code de la consommation, art. L.212-1 et
L.224-25-1 et s.) ; toute clause qui serait abusive, ou qui priverait un
consommateur d'un droit impératif, ne lui est pas opposable.

**5. Protection des données (RGPD).** Les mécanismes des §§ 1 à 4 ne collectent ni
ne transmettent **aucune** donnée personnelle ni aucun contenu de charge de
travail ; il n'y a ni télémétrie ni connexion sortante. Un fichier de licence
contient l'identité du licencié et l'identifiant de licence et — lorsque vous
optez pour le verrouillage à la machine — une **empreinte pseudonymisée et salée**
d'un identifiant machine que vous fournissez (une valeur SHA-256 ; l'identifiant
brut n'est jamais stocké), traitée conformément au **règlement (UE) 2016/679
(RGPD)**. Lorsque ces données se rapportent à une personne physique identifiable,
le fournisseur les traite en qualité de responsable de traitement sur la base
légale de l'**exécution du contrat (art. 6, §1, b)**, dans la limite de ce qui est
nécessaire à l'authenticité, aux droits d'usage et à l'attribution des fuites
(**minimisation des données, art. 5, §1, c**). Les droits des personnes concernées
(art. 12 à 22) ainsi que l'identité complète et les coordonnées du responsable de
traitement figurent dans la notice de confidentialité distincte :
[docs/PRIVACY_NOTICE.fr.md](docs/PRIVACY_NOTICE.fr.md) (à compléter et publier).

**6. Réserve au titre de la fouille de textes et de données / entraînement d'IA.**
Le titulaire des droits **réserve expressément** les droits sur le code source de
SciRust et sur les artefacts générés à l'égard de la fouille de textes et de
données, y compris l'usage visant à entraîner des systèmes d'apprentissage
automatique ou d'IA générative, en application de l'**article 4, §3 de la directive
(UE) 2019/790** (DAMUN/« DSM »). La présente clause vaut réserve ; dans la mesure
du possible, une réserve lisible par machine (par ex. `robots.txt` / un fichier de
métadonnées de réservation TDM) accompagne les distributions publiées.

**7. Langue, loi applicable et juridiction.** Pour un contrat avec un consommateur
résidant en France, la version en **langue française** prévaut (Loi n° 94-665 du
4 août 1994, « Toubon »). La loi applicable et la juridiction compétente sont
`<à définir — typiquement le droit français>`, sans préjudice des règles
impératives de protection des consommateurs du pays de résidence habituelle du
consommateur (règlement (CE) n° 593/2008 « Rome I », art. 6 ; règlement (UE)
n° 1215/2012 « Bruxelles I bis »).
