//! # scirust-machining — productique mécanique (usinage)
//!
//! Primitives déterministes et pures-Rust de calcul en **productique
//! mécanique**, couvrant la chaîne de décision d'une opération d'usinage —
//! celle qu'outillent en pratique les calculateurs de fabricants et les
//! toolboxes du domaine :
//!
//! - [`kinematics`] — cinématique de coupe : conversions `Vc ↔ N`, vitesse
//!   d'avance `Vf`, débit de copeaux (MRR) en tournage/fraisage/perçage.
//! - [`forces`] — effort et puissance de coupe par le modèle de **Kienzle**
//!   (`kc = kc1.1·h^(-mc)`), puissance moteur et couple de broche.
//! - [`toollife`] — durée de vie de l'outil par la loi de **Taylor**
//!   (`Vc·T^n = C`) et sa forme étendue en avance et profondeur.
//! - [`economics`] — économie d'usinage de **Gilbert** : vitesses de coupe de
//!   production maximale et de coût minimal.
//! - [`time`] — temps de coupe (temps machine) des opérations élémentaires.
//! - [`roughness`] — rugosité théorique de l'état de surface en tournage.
//! - [`gears`] — engrenages cylindriques droits et **hélicoïdaux** : géométrie
//!   de la denture à développante, flexion en pied de dent (**Lewis**), dents
//!   minimales anti-interférence et rapport de conduite.
//! - [`iso6336`] — résistance des engrenages au flanc (**ISO 6336-2**, pitting) :
//!   contrainte de contact, facteur d'élasticité (via `hertz`) et sécurité.
//! - [`threads`] — filetages métriques ISO : diamètres primitif/noyau, section
//!   résistante (**ISO 898-1**), pas et angle d'hélice.
//! - [`hertz`] — mécanique du contact de **Hertz** : pression et dimensions de
//!   contact linéaire/ponctuel (engrenages, roulements, cames).
//! - [`bearings`] — durée de vie des roulements (**ISO 281**, L10), charge
//!   dynamique équivalente et durée corrigée en fiabilité.
//! - [`springs`] — ressorts hélicoïdaux de compression (**EN 13906**) : raideur,
//!   flèche et cisaillement corrigé (facteur de Wahl).
//! - [`shafts`] — arbres de transmission : torsion/flexion des sections
//!   circulaires, contrainte équivalente de **von Mises** et angle de torsion.
//! - [`keys`] — clavetages (clavette parallèle, **ISO 773**) : cisaillement de
//!   la clavette et pression de matage sur les flancs.
//! - [`belts`] — transmissions par courroie (**Euler-Eytelwein**) : rapport des
//!   tensions plate/trapézoïdale, angle d'enroulement et puissance transmise.
//! - [`power_screws`] — vis de transmission : couple de montée/descente,
//!   rendement et condition d'irréversibilité (filets carré/trapézoïdal).
//! - [`torseurs`] — torseurs de la mécanique du solide (statique/cinématique) :
//!   transport du moment, invariants, axe central, comoment (puissance).
//! - [`liaisons`] — les 11 liaisons mécaniques normalisées : degrés de liberté,
//!   mobilité et inconnues statiques (dualité avec [`torseurs`]).
//! - [`hyperstatism`] — isostatisme/hyperstatisme d'un mécanisme : boucles
//!   indépendantes, mobilité et degré d'hyperstaticité `h = m + 6μ − Ic`.
//! - [`friction`] — frottement sec de **Coulomb** : adhérence/glissement, angle
//!   et cône d'adhérence, arc-boutement sur plan incliné.
//! - [`dynamics`] — dynamique du solide en rotation : énergie cinétique, moments
//!   d'inertie usuels, théorème de **Huygens**, PFD (`M = J·α`) et puissance.
//! - [`cams`] — cames à disque : lois de mouvement du suiveur (MHS, cycloïdale),
//!   déplacement/vitesse/accélération.
//! - [`vibrations`] — vibrations à **1 ddl** (masse-ressort-amortisseur) :
//!   pulsation propre, amortissement, décrément logarithmique.
//! - [`beams`] — RDM flexion des poutres : moments quadratiques, contrainte de
//!   flexion et flèches des cas de charge usuels.
//! - [`buckling`] — flambage des colonnes comprimées (**Euler**) : charge
//!   critique, longueur de flambement, élancement et domaine de validité.
//! - [`mohr`] — état de contrainte plan (cercle de **Mohr**) : contraintes
//!   principales, cisaillement maximal, critères de **von Mises** et **Tresca**.
//! - [`trusses`] — treillis : contrainte axiale, allongement des barres et
//!   équilibre d'un nœud par la méthode des nœuds.
//! - [`torsion_profiles`] — torsion des profils non circulaires : tubes minces
//!   fermés (**Bredt**) et sections minces ouvertes (Saint-Venant).
//! - [`stress_concentration`] — concentration de contrainte : facteur `Kt`,
//!   contrainte de pointe sur section nette et facteur de fatigue `Kf`.
//! - [`forced_vibrations`] — vibrations **forcées** à 1 ddl : amplification
//!   dynamique, phase, transmissibilité et réponse au balourd tournant.
//! - [`balancing`] — équilibrage des rotors : force centrifuge de balourd,
//!   correction en un plan et balourd résiduel admissible (**ISO 1940-1**).
//! - [`critical_speed`] — vitesses critiques des arbres tournants : critique d'un
//!   disque, **Rankine** (flèche statique) et combinaison de **Dunkerley**.
//! - [`flywheel`] — volant d'inertie : coefficient de fluctuation, énergie à
//!   emmagasiner et inertie requise pour régulariser la vitesse.
//! - [`impact`] — chocs et charges dynamiques : restitution, choc direct de deux
//!   masses et facteur d'amplification (charge subite ou tombant d'une hauteur).
//! - [`slider_crank`] — mécanisme bielle-manivelle : course, vitesse et
//!   accélération du piston selon l'angle de manivelle.
//! - [`fourbar`] — quadrilatère articulé : critère de **Grashof** et
//!   classification (manivelle-balancier, double-manivelle…).
//! - [`epicyclic`] — trains épicycloïdaux : équation de **Willis**, vitesses
//!   soleil/couronne/porte-satellites et rapport de réduction.
//! - [`universal_joint`] — joint de **Cardan** : irrégularité de transmission,
//!   rapport de vitesses instantané et bornes de fluctuation.
//! - [`geneva`] — croix de Malte : indexeur intermittent, angle de la roue menée,
//!   rapport de vitesses et angles d'indexage/repos.
//! - [`bernoulli`] — mécanique des fluides : pression hydrostatique/dynamique,
//!   charge de **Bernoulli**, continuité, **Torricelli** et **Reynolds**.
//! - [`pipe_flow`] — pertes de charge : **Darcy-Weisbach**, facteur de frottement
//!   (**Colebrook**, Swamee-Jain) et pertes singulières.
//! - [`pumps`] — pompes centrifuges : puissances, **NPSH** disponible, lois
//!   d'affinité et vitesse spécifique.
//! - [`thermo_cycles`] — cycles thermodynamiques : rendements **Carnot**, Otto,
//!   Diesel et COP des machines frigorifiques/pompes à chaleur.
//! - [`heat_exchanger`] — échangeurs : **DTLM** et méthode **NUT-efficacité**
//!   (ε-NTU) co-courant/contre-courant.
//! - [`bolted_joints`] — assemblages boulonnés précontraints (**VDI 2230**) :
//!   précharge, facteur de charge et répartition de l'effort extérieur.
//! - [`fastener_groups`] — groupes de boulons/rivets sous charge excentrée :
//!   cisaillements primaire et secondaire, effort résultant.
//! - [`welds`] — soudures : gorge des cordons d'angle, cisaillement direct et
//!   cisaillement d'un groupe de cordons sous moment.
//! - [`riveted_joints`] — assemblages rivés : modes de ruine (rivets, matage,
//!   déchirure) et rendement du joint.
//! - [`interference_fit`] — frettage : pression de contact (Lamé), couple/effort
//!   transmissible et échauffement de montage.
//! - [`brakes`] — freins et embrayages : couple d'un embrayage à disques
//!   (usure/pression uniforme) et couple de freinage d'un frein à sangle.
//! - [`journal_bearings`] — paliers lisses : charge unitaire, nombre de
//!   **Sommerfeld**, frottement de **Petroff** et film minimal.
//! - [`bevel_worm_gears`] — engrenages coniques et roue-vis : angles de cône,
//!   rapport de réduction, angle d'hélice, rendement et irréversibilité.
//! - [`pulley_systems`] — poulies et moufles : avantage mécanique, effort à
//!   fournir, rapport de vitesses et rendement.
//! - [`hydraulic_cylinders`] — vérins hydrauliques : efforts sortie/rentrée,
//!   vitesse de tige, débit et puissance fluide.
//! - [`fatigue_mean_stress`] — fatigue à contrainte moyenne : critères de
//!   **Goodman**, **Soderberg** et **Gerber** (diagramme de Haigh).
//! - [`endurance_limit`] — limite d'endurance : facteurs de **Marin** et droite
//!   **S-N** (Basquin) à nombre fini de cycles.
//! - [`creep`] — fluage : paramètre de **Larson-Miller** et vitesse de fluage
//!   secondaire (**Norton**).
//! - [`hardness`] — dureté : essais **Brinell** et **Vickers**, estimation de
//!   la résistance à la traction.
//! - [`fracture`] — mécanique de la rupture : facteur d'intensité de contrainte,
//!   taille critique de fissure et contrainte de **Griffith**.
//! - [`thermal`] — thermique : dilatation, conduction (**Fourier**), convection,
//!   chaleur sensible et contrainte thermique.
//! - [`tolerancing`] — systèmes de tolérancement de dessin : tolérances
//!   générales **ISO 2768** (parties 1 et 2) et catalogue des normes **GPS**.
//! - [`dimension_chains`] — chaînes de cotes : cumul pire cas et statistique
//!   (RSS), cote de fermeture et jeux extrêmes.
//! - [`msa`] — MSA/**Gage R&R** : répétabilité, reproductibilité, %R&R et
//!   nombre de catégories distinctes.
//! - [`sheet_bending`] — développé de pliage : allongement au pli, retrait et
//!   longueur à plat (facteur `K`).
//! - [`process_time`] — temps de gamme : temps de série, temps par pièce et
//!   cadence de production.
//! - [`oee`] — taux de rendement synthétique (**TRS/OEE**) : disponibilité,
//!   performance, qualité et leur produit.
//! - [`torsion_springs`] — ressorts de torsion : raideur angulaire, rotation et
//!   contrainte de flexion du fil.
//! - [`extension_springs`] — ressorts de traction : raideur, tension initiale et
//!   effort/flèche au décollement des spires.
//! - [`leaf_springs`] — ressorts à lames : contrainte, flèche et raideur d'un
//!   empilage en console.
//! - [`belleville_washers`] — rondelles **Belleville** : loi effort-flèche non
//!   linéaire (Almen-László) et charge d'aplatissement.
//! - [`elastomer_mounts`] — plots élastomères : facteur de forme, module
//!   apparent et raideurs compression/cisaillement.
//! - [`fins`] — ailettes : paramètre d'ailette, efficacité, efficience et flux
//!   évacué (bout adiabatique).
//! - [`radiation`] — rayonnement (**Stefan-Boltzmann**) : émittance, échange net
//!   avec l'environnement et coefficient linéarisé.
//! - [`transient_conduction`] — conduction transitoire : nombres de **Biot** et
//!   **Fourier**, capacité thermique localisée.
//! - [`convection_correlations`] — convection : **Prandtl**, **Nusselt** → `h`,
//!   **Dittus-Boelter** et **Rayleigh**.
//! - [`thermal_network`] — réseaux de résistances thermiques : convection,
//!   série/parallèle et coefficient global d'échange.
//! - [`drag_lift`] — traînée et portance : efforts aérodynamiques, puissance de
//!   traînée et vitesse limite de chute.
//! - [`flow_meters`] — débitmètres déprimogènes (orifice, **Venturi**) : débit,
//!   perte de pression et facteur de vitesse d'approche.
//! - [`open_channel`] — écoulement à surface libre : rayon hydraulique, formules
//!   de **Manning** et de **Chézy**.
//! - [`water_hammer`] — coup de bélier : célérité de l'onde, surpression de
//!   **Joukowsky** et durée critique de manœuvre.
//! - [`pump_system`] — point de fonctionnement pompe-réseau : intersection des
//!   caractéristiques pompe et réseau.
//! - [`centroids`] — centroïdes de surfaces composées : aire totale et position
//!   du centre de gravité (évidements négatifs).
//! - [`area_moments`] — moments quadratiques composés : théorème de **Huygens**,
//!   rayon de giration et axes perpendiculaires.
//! - [`wedge`] — coins : effort d'entrée, avantage mécanique idéal et condition
//!   d'auto-blocage avec frottement.
//! - [`distributed_loads`] — charges réparties : résultante et position
//!   (uniforme, triangulaire, trapézoïdale).
//! - [`cables`] — câbles paraboliques : tension horizontale/maximale et longueur
//!   développée (ponts suspendus).
//! - [`gear_trains`] — trains d'engrenages ordinaires : rapports simple/composé,
//!   vitesse et couple de sortie, effet des roues folles.
//! - [`rack_pinion`] — pignon-crémaillère : rayon primitif, vitesse linéaire,
//!   course par tour et effort.
//! - [`chain_drive`] — transmissions par chaîne à rouleaux : diamètre primitif,
//!   vitesse, rapport et longueur de chaîne.
//! - [`couplings`] — accouplements à plateaux : couple transmissible par les
//!   boulons et conversion puissance ↔ couple.
//! - [`splines`] — cannelures : couple transmissible par matage des flancs.
//! - [`elasticity_relations`] — constantes élastiques isotropes : relations entre
//!   `E`, `G`, `K`, `ν` et coefficient de **Lamé**.
//! - [`hookes_law_3d`] — loi de **Hooke** généralisée 3D : déformations
//!   triaxiales, cisaillement, déformation volumique.
//! - [`composites`] — règle des mélanges : bornes de **Voigt** et **Reuss**,
//!   masse volumique et résistance longitudinale.
//! - [`true_stress_strain`] — grandeurs vraies (rationnelles) et loi
//!   d'écrouissage de **Hollomon**.
//! - [`strain_energy`] — énergie de déformation élastique : densités, module de
//!   résilience et énergie totale.
//! - [`beam_reactions`] — RDM : réactions d'appui et moments maximaux de poutres
//!   isostatiques (charge excentrée, répartie, console).
//! - [`combined_stress`] — RDM : sollicitations composées (traction+flexion,
//!   flexion+torsion), moments idéaux d'arbre.
//! - [`pressure_vessels`] — RDM : réservoirs sous pression, parois minces
//!   (cylindre/sphère) et cylindres épais (**Lamé**).
//! - [`deflection_cases`] — RDM : flèches et pentes complémentaires et
//!   **superposition** des cas de charge.
//! - [`castigliano`] — RDM : énergie de déformation des éléments (traction,
//!   flexion, torsion) et principe de **Castigliano**.
//! - [`upsetting`] — mise en forme : refoulement (forgeage) — déformation vraie,
//!   effort avec frottement et travail.
//! - [`wire_drawing`] — mise en forme : tréfilage — réduction, contrainte et
//!   effort d'étirage, réduction maximale.
//! - [`extrusion`] — mise en forme : extrusion — rapport, déformation, pression
//!   de **Johnson** et effort.
//! - [`rolling`] — mise en forme : laminage à plat — réduction, longueur de
//!   contact, effort et couple.
//! - [`press_brake`] — mise en forme : pliage à la presse — effort de pliage et
//!   retour élastique (springback).
//! - [`merchant`] — usinage : coupe orthogonale (**Merchant**) — rapport de
//!   coupe, angle de cisaillement et déformation.
//! - [`milling_chip`] — usinage : fraisage — avance par dent, angle d'engagement
//!   et épaisseur de copeau.
//! - [`drilling_ops`] — usinage : perçage — couple, puissance, pénétration et
//!   poussée d'un foret.
//! - [`chvorinov`] — fonderie : solidification — module thermique, règle de
//!   **Chvorinov** et masselotte.
//! - [`gating`] — fonderie : remplissage — vitesse de descente (**Torricelli**),
//!   temps de coulée et profil anti-aspiration.
//! - [`reliability`] — fiabilité : modèle **exponentiel** (taux constant) —
//!   fiabilité, probabilité de défaillance, MTBF et estimation par essai.
//! - [`weibull`] — fiabilité : distribution de **Weibull** — fiabilité, taux de
//!   hasard et durées de vie `Bx` (mortalité infantile, usure).
//! - [`system_reliability`] — fiabilité des systèmes : série, parallèle
//!   (redondance active) et redondance `k`-sur-`n`.
//! - [`maintenance`] — maintenance : MTBF, MTTR, disponibilité intrinsèque et
//!   maintenabilité.
//! - [`accelerated_life`] — essais de vie accélérée : facteur d'**Arrhenius**,
//!   règle des « 10 °C » et déclassement (derating).
//! - [`pneumatic_cylinders`] — vérins pneumatiques : effort théorique/utile
//!   (rendement) et consommation d'air libre par course.
//! - [`valve_sizing`] — dimensionnement de vannes : coefficients `Kv`/`Cv` et
//!   débit liquide.
//! - [`air_flow`] — écoulement d'air comprimé : régime **bloqué** (sonique),
//!   rapport critique, vitesse du son et débit-masse.
//! - [`compressed_air`] — air comprimé : gaz parfait, travail de compression
//!   isotherme/adiabatique et température de refoulement.
//! - [`vacuum_gripping`] — préhension par le vide : effort de ventouse, charge
//!   admissible, diamètre requis et nombre de ventouses.
//! - [`ball_screw`] — vis à billes : conversion rotation ↔ translation et
//!   couple ↔ effort axial (rendement catalogue).
//! - [`reflected_inertia`] — inertie ramenée à l'arbre moteur : réducteur, vis à
//!   billes, inertie propre de vis et ratio d'inertie.
//! - [`motion_profile`] — profil de mouvement trapézoïdal/triangulaire : temps et
//!   distances de rampe, vitesse de crête, durée totale.
//! - [`motor_torque`] — couple moteur : couple d'accélération (`J·α`), couple
//!   total et couple **efficace (RMS)** sur un cycle.
//! - [`stepper_motor`] — moteur pas à pas : résolution angulaire, cadence
//!   d'impulsions, vitesse et résolution linéaire.
//! - [`first_order_response`] — automatique : système du **premier ordre** —
//!   réponse indicielle, constante de temps et fréquence de coupure.
//! - [`second_order_response`] — automatique : système du **second ordre** —
//!   pulsation amortie, dépassement, temps de pic et de réponse.
//! - [`bode_first_order`] — automatique : diagramme de **Bode** d'un premier
//!   ordre — gain (dB), phase et pulsation de coupure.
//! - [`pid_control`] — automatique : régulateur **PID** — sortie parallèle,
//!   forme standard et réglage de **Ziegler-Nichols**.
//! - [`feedback_loop`] — automatique : boucle de contre-réaction — gain en boucle
//!   fermée, sensibilité et erreur statique.
//! - [`viscosity`] — tribologie : viscosité dynamique/cinématique, unités et
//!   dépendance en température (**Andrade**).
//! - [`film_lubrication`] — tribologie : nombre de **Hersey**, régimes de
//!   **Stribeck** et rapport de film `λ`.
//! - [`archard_wear`] — tribologie : usure — loi d'**Archard**, taux spécifique,
//!   profondeur et distance de glissement.
//! - [`frictional_heating`] — tribologie : échauffement par frottement — puissance
//!   dissipée, densité de flux et élévation de température.
//! - [`rolling_resistance`] — tribologie : résistance au roulement — coefficient,
//!   effort résistant, puissance et pente.
//! - [`o_ring_seals`] — joints toriques (étanchéité statique) : taux d'écrasement,
//!   d'étirement et de remplissage de gorge.
//! - [`gasket_seating`] — joints plats sous brides (**ASME VIII**, facteurs `m`/`y`) :
//!   charges d'assise, de service, poussée de fond et boulonnerie requise.
//! - [`adhesive_lap_joint`] — collage à simple recouvrement : cisaillement moyen,
//!   capacité du joint et longueur de recouvrement requise.
//! - [`snap_fit_cantilever`] — encliquetage à poutre cantilever : déformation
//!   maximale, effort de déflexion et effort d'emmanchement.
//! - [`bolt_circle`] — cercle de perçage (PCD) : angle/position des trous et corde.
//! - [`taper`] — cônes : conicité, angle inclus et diamètre courant.
//! - [`sine_bar`] — barre sinus : conversion réciproque hauteur de cales ↔ angle.
//! - [`three_wire_thread`] — mesure de filetage aux trois piges : pige optimale et
//!   cote sur piges.
//! - [`gear_span_measurement`] — cote sur `k` dents (Wildhaber) : fonction
//!   développante et base tangent length.
//! - [`planetary_constraints`] — trains épicycloïdaux : coaxialité, denture
//!   satellite et condition d'assemblage.
//! - [`vbelt_design`] — courroies trapézoïdales : puissance de dimensionnement,
//!   puissance corrigée par brin et nombre de courroies.
//! - [`timing_belt`] — courroies crantées : dents de courroie, dents en prise et
//!   longueur primitive.
//! - [`wire_rope`] — câbles de levage : effort de rupture, charge d'utilisation et
//!   diamètre de poulie minimal.
//! - [`sling_tension`] — élingues : tension de brin, facteur de charge et effort
//!   horizontal selon l'angle.
//! - [`lifting_lug`] — oreille de levage : matage, section nette et cisaillement
//!   double de l'axe.
//! - [`blanking_force`] — découpage/poinçonnage : effort de découpe, dévêtisseur et
//!   travail.
//! - [`deep_drawing`] — emboutissage : rapport limite (LDR), effort de poinçon et
//!   serre-flan.
//! - [`tube_bending`] — cintrage de tube : déformation de fibre externe, rayon
//!   minimal et fibre neutre.
//! - [`injection_cooling`] — injection plastique : temps de refroidissement (1D),
//!   température au cœur et épaisseur admissible.
//! - [`hydraulic_accumulator`] — accumulateur hydropneumatique (**Boyle**) : volume
//!   de gaz, fluide utile et volume de pré-charge.
//! - [`clutch_engagement`] — embrayage à friction : énergie de glissement, durée de
//!   synchronisation et échauffement adiabatique.
//! - [`brake_thermal`] — échauffement de frein : énergie cinétique dissipée,
//!   élévation de température du disque et puissance.
//! - [`hydraulic_motor`] — moteur hydraulique volumétrique : couple, vitesse,
//!   puissances et rendement global.
//! - [`gear_pump`] — pompe à engrenages : cylindrée, débit théorique et réel.
//! - [`torsional_vibration`] — vibration de torsion à deux disques : raideur, nœud
//!   et fréquence propre.
//! - [`governor_flyball`] — régulateur centrifuge (**Watt**/**Porter**) : hauteur de
//!   cône, force centrifuge et régime d'équilibre.
//! - [`toggle_mechanism`] — genouillère : rapport d'amplification et effort de
//!   sortie au point mort.
//! - [`scissor_lift`] — table élévatrice à ciseaux : effort du vérin, avantage
//!   mécanique et hauteur.
//! - [`belt_conveyor`] — bande transporteuse : tension effective, débit et
//!   puissances d'entraînement/élévation.
//! - [`screw_conveyor`] — vis sans fin : débits volumétrique et massique.
//! - [`carburizing`] — cémentation : profondeur en √t (**Harris**) et durée de cycle.
//! - [`weld_heat_input`] — soudage à l'arc : puissance d'arc et apport linéique
//!   `η·U·I/v`.
//! - [`surface_grinding`] — rectification plane : débits, épaisseur de copeau
//!   équivalente et rapport `G`.
//! - [`broaching`] — brochage : dents en prise et effort de coupe maximal.
//! - [`tapping_torque`] — taraudage : couple mécaniste/empirique et puissance.
//! - [`coil_spring_surge`] — résonance (surge) d'un ressort hélicoïdal : premier
//!   mode fixe-fixe/fixe-libre.
//! - [`spring_nest`] — ressorts concentriques : raideur combinée et répartition de
//!   charge.
//! - [`cam_pressure_angle`] — angle de pression d'une came à suiveur translatant.
//! - [`disc_spring_stack`] — empilage de rondelles Belleville : série/parallèle,
//!   raideur et nombre de rondelles.
//! - [`rotating_ring_stress`] — jante mince en rotation : contrainte
//!   circonférentielle et vitesse d'éclatement.
//! - [`niosh_lifting`] — ergonomie : équation **NIOSH** — poids limite recommandé
//!   et indice de levage.
//! - [`eoq_inventory`] — gestion de stock : quantité économique (**Wilson**),
//!   coût total, point de commande.
//! - [`break_even`] — seuil de rentabilité : quantité/CA d'équilibre, marge sur
//!   coût variable et marge de sécurité.
//! - [`machine_hour_rate`] — coût horaire machine : amortissement, énergie et
//!   taux horaire global.
//! - [`learning_curve`] — courbe d'apprentissage (**Wright**) : temps unitaire,
//!   cumulé et moyen.
//! - [`forward_kinematics_2r`] — robotique : cinématique directe d'un bras
//!   planaire **2R** (position TCP/coude).
//! - [`jacobian_2r`] — robotique : jacobien d'un bras **2R**, vitesse TCP et
//!   singularités.
//! - [`workspace_2r`] — robotique : espace de travail d'un bras **2R** (portées,
//!   atteignabilité, aire).
//! - [`die_casting`] — fonderie sous pression : vitesse de porte, temps de
//!   remplissage et effort de verrouillage.
//! - [`sand_casting_shrinkage`] — fonderie sable : surdimensionnement du modèle
//!   (retrait, usinage, dépouille).
//! - [`plastic_shrinkage`] — injection : retrait au moulage (dimension de cavité).
//! - [`mold_cooling_channel`] — canaux de refroidissement : Reynolds, débit et
//!   extraction thermique du fluide.
//! - [`gdt_position_tolerance`] — GD&T : tolérancement de position (zone
//!   diamétrale, bonus MMC).
//! - [`total_runout`] — métrologie : battement total/circulaire par relevé.
//! - [`flatness_from_readings`] — métrologie : planéité/rectitude par l'étendue
//!   min-max.
//! - [`cosine_error`] — métrologie : erreur cosinus d'un instrument désaligné.
//! - [`thread_milling`] — usinage : fraisage de filet (interpolation hélicoïdale,
//!   correction d'avance).
//! - [`drill_point_geometry`] — usinage : géométrie de pointe de foret (longueur
//!   de pointe, course supplémentaire).
//! - [`tap_drill_size`] — usinage : diamètre de foret d'avant-trou de taraudage
//!   (filet ISO).
//! - [`countersink_depth`] — usinage : profondeur d'un lamage conique
//!   (fraisure).
//! - [`electroplating`] — galvanoplastie : loi de **Faraday** (masse déposée,
//!   épaisseur, durée d'électrolyse).
//! - [`anodizing`] — anodisation : croissance de couche d'oxyde (épaisseur,
//!   durée, densité de courant).
//! - [`shot_peening`] — grenaillage : couverture (Avrami) et intensité **Almen**.
//! - [`spot_welding`] — soudage par points : chaleur de **Joule** `I²·R·t` et
//!   densité de chaleur du noyau.
//! - [`laser_cutting`] — découpe laser : vitesse de coupe, puissance requise et
//!   densité de puissance.
//! - [`waterjet_cutting`] — jet d'eau abrasif : vitesse du jet (**Bernoulli**),
//!   indice de vitesse et puissance.
//! - [`edm_machining`] — électroérosion : débit d'enlèvement, usure d'électrode et
//!   surcoupe.
//! - [`cotter_joint`] — clavette transversale : cisaillement, matage et traction.
//! - [`knuckle_joint`] — assemblage à chape et œil : cisaillement d'axe, matage et
//!   traction.
//! - [`coupling_misalignment`] — désalignement d'accouplement : angle, effort de
//!   réaction et résultante.
//! - [`shaft_alignment`] — alignement d'arbres au comparateur (rim-and-face) :
//!   angulaire, offset et cales.
//! - [`gear_efficiency`] — rendement d'engrenage droit : perte par glissement,
//!   puissance perdue/transmise.
//! - [`retaining_ring`] — circlip : capacité axiale (matage de gorge, cisaillement
//!   de l'anneau).
//! - [`bearing_preload`] — précharge de roulement : flèche, rigidité effective et
//!   effort de calage.
//! - [`vbelt_length`] — longueur de courroie (montage ouvert/croisé) et angles
//!   d'enroulement.
//! - [`belt_slip`] — glissement de courroie : perte de vitesse, rapport effectif
//!   et fluage.
//! - [`bevel_gear_forces`] — efforts sur engrenage conique : tangentiel, radial,
//!   axial et résultant.
//! - [`acceptance_sampling`] — contrôle de réception : probabilité d'acceptation
//!   (plan simple binomial) et AOQ.
//! - [`shewhart_xbar`] — cartes de **Shewhart** X̄/R : limites de contrôle (A2, D3,
//!   D4) et test sous contrôle.
//! - [`rolled_throughput_yield`] — qualité multi-étapes : FPY, RTY, rendement
//!   normalisé et DPU.
//! - [`takt_time`] — Lean : temps takt, cadence requise et nombre d'opérateurs.
//! - [`line_balancing`] — équilibrage de ligne : postes minimaux, efficacité,
//!   retard d'équilibrage et lissage.
//! - [`littles_law`] — loi de **Little** : en-cours, débit et temps de passage.
//! - [`johnson_scheduling`] — flow-shop 2 machines : makespan d'une séquence et
//!   temps d'inactivité.
//! - [`capacity_planning`] — capacité disponible/requise, utilisation et goulot.
//! - [`smed_setup`] — SMED : temps de changement, arrêt machine et gain
//!   interne→externe.
//! - [`helical_gear_forces`] — engrenage hélicoïdal : efforts tangentiel, radial,
//!   axial, normal et résultant.
//! - [`worm_gear_forces`] — roue et vis : décomposition tangentiel/axial/séparateur
//!   et identités vis ↔ roue.
//! - [`internal_gear`] — engrenage intérieur : entraxe, rapport et diamètre de tête
//!   de couronne.
//! - [`gearbox_efficiency`] — réducteur multi-étages : rendement global, puissances
//!   et nombre d'étages.
//! - [`roll_bending`] — roulage 3 rouleaux : rayon cintré, retour élastique et
//!   rayon minimal.
//! - [`stretch_forming`] — formage par étirage : déformation vraie, effort et
//!   épaisseur après.
//! - [`ironing`] — repassage : réduction d'épaisseur, déformation vraie et effort.
//! - [`blank_nesting`] — mise en bande : pas, utilisation matière, pièces et chute.
//! - [`hydraulic_flow_velocity`] — vitesse de fluide en conduite, diamètre et
//!   plafond de vitesse.
//! - [`pressure_intensifier`] — multiplicateur de pression : rapport, pression et
//!   débit de sortie.
//! - [`air_receiver`] — réservoir d'air comprimé : volume tampon, air utile et
//!   temps de remplissage.
//! - [`cutting_temperature`] — température de coupe : élévation adiabatique,
//!   interface et indice de **Cook**.
//! - [`corrosion_rate`] — corrosion uniforme : taux de pénétration et conversion
//!   **Faraday** courant ↔ vitesse.
//! - [`machining_cost`] — coût d'usinage par pièce : coupe, mise en train amortie
//!   et outillage.
//! - [`scotch_yoke`] — mécanisme à coulisse : mouvement harmonique simple pur.
//! - [`quick_return`] — mécanisme à retour rapide : rapport aller/retour et
//!   fraction de coupe.
//! - [`ratchet_pawl`] — rochet à cliquet : pas angulaire, couple de maintien,
//!   dents minimales.
//! - [`differential_screw`] — vis différentielle : avance nette `p1−p2` et
//!   avantage mécanique.
//! - [`capillary_rise`] — ascension capillaire (**Jurin**) et saut de Laplace.
//! - [`surface_tension_laplace`] — surpression de **Laplace** (goutte, bulle) et
//!   longueur capillaire.
//! - [`nozzle_thrust`] — tuyère : débit-masse, poussée idéale et vitesse d'éjection.
//! - [`paris_law`] — propagation de fissure (**Paris**) : `ΔK`, `da/dN = C·ΔK^m`.
//! - [`thermal_shock_resistance`] — paramètres de **Kingery** `R`/`R'` et `ΔT` max.
//! - [`bimetal_strip`] — bilame thermique : courbure, rayon et flèche.
//! - [`stress_relaxation`] — relaxation (**Maxwell**) : `σ(t)=σ0·e^{−t/τ}`.
//! - [`ceramic_weibull`] — rupture fragile (**Weibull**) : survie, effet d'échelle.
//! - [`curved_beam`] — poutre fortement courbe (**Winkler-Bach**) : axe neutre,
//!   contrainte hyperbolique.
//! - [`circular_plate`] — plaque circulaire sous pression : rigidité, flèche max.
//! - [`taper_pin`] — goupille conique : section moyenne, cisaillement simple/double.
//! - [`woodruff_key`] — clavette Woodruff : cisaillement et matage arbre-moyeu.
//! - [`abbe_error`] — métrologie : erreur d'**Abbe** `e = d·tan α`.
//! - [`gauge_temperature_correction`] — correction thermique des mesures à 20 °C
//!   (**ISO 1**).
//! - [`powder_compaction`] — métallurgie des poudres : densité crue, retrait de
//!   frittage.
//! - [`additive_manufacturing`] — impression 3D : couches, débit de dépôt, temps.
//! - [`ultrasonic_testing`] — CND ultrasons : longueur d'onde, profondeur de
//!   défaut, champ proche, atténuation.
//! - [`eddy_current`] — CND courants de Foucault : profondeur standard, déphasage.
//! - [`radiographic_testing`] — CND radiographie : atténuation (**Beer-Lambert**),
//!   couche de demi-atténuation, flou géométrique.
//! - [`carbon_equivalent`] — soudabilité : équivalent carbone **IIW** et classe de
//!   risque.
//! - [`weld_preheat`] — préchauffage (**Séférian**) : température et nécessité.
//! - [`weld_cooling_rate`] — refroidissement soudure (**Rosenthal**) : `t8/5`.
//! - [`weld_dilution`] — taux de dilution et composition du dépôt.
//! - [`vibration_isolation`] — isolation par plots : fréquence propre (flèche
//!   statique), transmissibilité, efficacité.
//! - [`sound_pressure_level`] — acoustique : niveaux de pression (dB), somme,
//!   atténuation en distance.
//! - [`sound_power_level`] — acoustique : niveau de puissance `Lw` et conversion
//!   puissance ↔ pression.
//! - [`noise_exposure`] — exposition au bruit (**ISO 1999**) : dose, durée admise,
//!   `L_eq`.
//! - [`refrigeration_cycle`] — cycle frigorifique : effet frigorifique, travail,
//!   COP, débit réfrigérant.
//! - [`psychrometrics`] — air humide : teneur en eau, humidité relative, enthalpie.
//! - [`ventilation_rate`] — ventilation : renouvellement d'air (ACH), débit de
//!   dilution.
//! - [`gear_hobbing`] — taillage à la fraise-mère : rapport de génération, avance,
//!   temps.
//! - [`dividing_head`] — appareil diviseur : division simple, angulaire.
//! - [`air_spring`] — ressort pneumatique : raideur polytropique, effort, fréquence.
//! - [`thread_engagement`] — longueur d'engagement anti-arrachement de filet.
//! - [`plate_buckling`] — voilement de plaque comprimée : contrainte/charge
//!   critiques.
//! - [`shell_buckling`] — flambage de coque cylindrique : axial et pression externe.
//! - [`induction_motor`] — moteur asynchrone triphasé : vitesse de synchronisme,
//!   glissement, vitesse et fréquence rotoriques.
//! - [`dc_motor`] — moteur à courant continu idéal : f.c.é.m., couple, tension, vitesse.
//! - [`three_phase_power`] — puissances triphasées équilibrées : active, apparente,
//!   réactive, courant de ligne.
//! - [`motor_efficiency`] — rendement et pertes globales d'un moteur, couple à l'arbre.
//! - [`vfd_volts_per_hertz`] — variateur V/f : rapport V/Hz, tension avec boost, flux relatif.
//! - [`bearing_defect_frequencies`] — fréquences cinématiques de défaut de roulement
//!   (BPFO, BPFI, BSF, FTF).
//! - [`iso10816_vibration`] — sévérité vibratoire ISO 10816 : vitesse efficace, zones A/B/C/D.
//! - [`order_analysis`] — analyse d'ordres : fréquence/ordre, rééchantillonnage angulaire.
//! - [`bearing_static_safety`] — sécurité statique de roulement (ISO 76) : charge
//!   équivalente, facteur s0, charge admissible.
//! - [`grease_relubrication`] — regraissage : quantité de graisse, facteur n·dm,
//!   réduction de durée de vie en température.
//! - [`chain_polygon_effect`] — effet polygonal de chaîne : diamètre primitif,
//!   ondulation de vitesse, vitesse moyenne.
//! - [`hydrostatic_bearing`] — palier hydrostatique : capacité de charge, débit,
//!   puissance de pompage, raideur.
//! - [`conical_spring`] — ressort de compression conique : raideurs mini/maxi, hauteur bloquée.
//! - [`wave_spring`] — ressort ondulé plat (Smalley) : raideur, flèche, effort, contrainte.
//! - [`constant_force_spring`] — ressort à force constante : effort de déroulement,
//!   contrainte, rayon naturel.
//! - [`combustion_air`] — air de combustion stœchiométrique, excès d'air, O2 des fumées.
//! - [`compressed_air_cost`] — coût de l'air comprimé et fuites : débit d'orifice,
//!   puissance spécifique, coût/énergie annuels.
//! - [`shear_center`] — centre de cisaillement des sections minces ouvertes (profilé en U).
//! - [`beam_column`] — poteau-poutre : amplification P-delta, moment et flèche amplifiés.
//! - [`venturi_meter`] — débitmètre à Venturi : rapport β, débit, perte de pression.
//! - [`orifice_plate`] — débitmètre à diaphragme (ISO 5167) : β, facteur d'approche, débit.
//! - [`pitot_tube`] — tube de Pitot : vitesse, pression dynamique/d'arrêt, débit.
//! - [`stokes_settling`] — sédimentation d'une sphère (Stokes) : vitesse limite, Re, traînée.
//! - [`weir_flow`] — déversoir mince : rectangulaire (Poleni), en V (Thomson), charge amont.
//! - [`hydraulic_jump`] — ressaut hydraulique : Froude, profondeur conjuguée, perte d'énergie.
//! - [`cavitation_number`] — cavitation et NPSH : indice σ, NPSHa, marge.
//! - [`pump_affinity`] — lois d'affinité des pompes : débit, hauteur, puissance, roue.
//! - [`lmtd`] — écart de température logarithmique moyen : contre/co-courant, flux U·A·F·ΔT.
//! - [`fin_efficiency`] — ailette droite : paramètre m, rendement, efficience, flux évacué.
//! - [`thermal_expansion`] — dilatation thermique : allongement, volume, contrainte de bridage.
//! - [`torsion_bar`] — barre de torsion : inertie polaire, raideur, contrainte, rotation.
//! - [`keyway_stress`] — clavette parallèle : effort tangentiel, cisaillement, matage, longueur.
//! - [`gyroscopic_couple`] — couple gyroscopique et précession d'un rotor rapide.
//! - [`hall_petch`] — renforcement par taille de grain (Hall-Petch) : limite d'élasticité.
//! - [`larson_miller`] — paramètre de Larson-Miller : fluage/rupture temps-température.
//! - [`calorific_value`] — pouvoir calorifique : PCS de Dulong, PCI, énergie dégagée.
//! - [`valve_flow_coefficient`] — coefficient de débit de vanne (Kv/Cv) en liquide.
//! - [`standard_atmosphere`] — atmosphère standard (troposphère ISA) : pression, température, densité.
//! - [`centrifugal_separation`] — séparation centrifuge : RCF, vitesse de sédimentation.
//! - [`riser_design`] — masselotte de fonderie : méthode des modules, rendement d'alimentation.
//! - [`thermal_resistance`] — mur composite : conduction/convection, série/parallèle, U, flux.
//! - [`view_factor`] — rayonnement : facteurs de forme (réciprocité, fermeture), échange net.
//! - [`faraday_corrosion`] — corrosion (loi de Faraday) : perte de masse, vitesse de pénétration.
//! - [`beam_deflection`] — flèche de poutres (Euler-Bernoulli) : cas console et sur appuis.
//! - [`section_modulus`] — module de flexion Z = I/c et contrainte σ = M/Z.
//! - [`bend_allowance`] — développé de tôle pliée : facteur K, allongement, déduction de pli.
//! - [`turning_roughness`] — rugosité théorique en tournage : Ra, Rz depuis avance/bec.
//! - [`elastic_constants`] — relations E, G, K, ν, λ (matériau isotrope linéaire).
//! - [`restitution`] — choc direct : coefficient de restitution, vitesses, énergie dissipée.
//! - [`projectile_motion`] — balistique du vide : portée, hauteur, temps de vol, angle optimal.
//! - [`spring_combination`] — association de ressorts : raideurs série/parallèle.
//! - [`gear_lewis`] — flexion de denture (Lewis) : contrainte, effort admissible, facteur de Barth.
//! - [`gear_contact_ratio`] — rapport de conduite d'un engrenage droit à développante.
//! - [`bolt_torque`] — serrage de boulon : couple-tension T = K·d·F, précharge.
//! - [`fillet_weld`] — cordon d'angle : gorge, cisaillement, effort admissible, longueur.
//! - [`crank_effort`] — moment moteur sur vilebrequin (bielle-manivelle, obliquité).
//! - [`friction_brake`] — frein à sabot : couple de freinage, puissance dissipée, temps d'arrêt.
//! - [`shaft_sizing`] — arbre en torsion pure : puissance/couple, diamètre requis.
//! - [`motor_starting`] — démarrage moteur : couple accélérateur, temps, courant DOL/étoile-triangle.
//! - [`choked_flow`] — tuyère compressible : rapports critiques, débit massique bloqué.
//! - [`duct_sizing`] — gaine HVAC : diamètre hydraulique/équivalent, vitesse, perte de charge.
//! - [`forging_force`] — forgeage : refoulement, matriçage, contrainte d'écoulement (Hollomon).
//! - [`rolling_force`] — laminage à plat : arc de contact, effort, couple, puissance.
//! - [`bolt_group_shear`] — groupe de boulons excentré : cisaillement direct + torsion.
//! - [`weld_group`] — groupe de cordons excentré : module polaire, contrainte résultante.
//! - [`rotating_unbalance`] — balourd tournant : force centrifuge, amplitude, ISO 1940.
//! - [`pendulum`] — pendule simple et composé : période, longueur synchrone, giration.
//! - [`flat_belt`] — courroie plate (Euler) : rapport des tensions, puissance, tension centrifuge.
//! - [`honing`] — rodage : angle de croisillon, vitesse périphérique, temps d'enlèvement.
//! - [`lapping`] — rodage libre (Preston) : taux d'enlèvement, matière enlevée, pression.
//! - [`disc_clutch`] — embrayage à disques : couple (usure/pression uniforme), effort axial.
//! - [`centrifugal_clutch`] — embrayage centrifuge : force de patin, régime d'engagement, couple.
//! - [`hydraulic_press`] — presse hydraulique (Pascal) : multiplication d'effort, multiplicateur.
//! - [`magnetic_holding`] — force de maintien magnétique (traction de Maxwell).
//! - [`vacuum_lifting`] — levage par ventouse : force de préhension, aire requise.
//! - [`reaming`] — alésage à l'alésoir : vitesse, avance, temps d'usinage.
//! - [`taper_turning`] — tournage conique : conicité, décalage de contre-poupée, angle de chariot.
//! - [`gear_backlash`] — jeu de denture : jeu circonférentiel/angulaire, entraxe requis.
//! - [`bucket_elevator`] — élévateur à godets : débit, puissance de levage et moteur.
//! - [`pump_specific_speed`] — vitesse spécifique de pompe : adimensionnelle, aspiration, classement.
//! - [`hydraulic_turbine`] — turbine hydraulique : puissance, vitesse spécifique, type Pelton/Francis.
//! - [`plastic_bending`] — flexion plastique : module plastique, facteur de forme, moment plastique.
//! - [`rankine_column`] — flambement Rankine-Gordon (poteaux courts à moyens).
//! - [`set_screw`] — vis de pression : effort de maintien, couple transmissible, nombre de vis.
//! - [`kanban_sizing`] — boucle Kanban : nombre de cartes, point de recommande, stock de sécurité.
//! - [`helmholtz_resonator`] — résonateur de Helmholtz : fréquence propre, correction d'embouchure.
//! - [`tuned_mass_damper`] — amortisseur à masse accordée (Den Hartog) : accord et amortissement optimaux.
//! - [`film_condensation`] — condensation en film laminaire (Nusselt) : coefficient d'échange.
//! - [`nucleate_boiling`] — ébullition nucléée : flux de Rohsenow, flux critique de Zuber.
//! - [`cone_clutch`] — embrayage à cône : couple, effort axial, largeur de portée.
//! - [`crane_stability`] — stabilité au basculement : moments, facteur, charge admissible.
//! - [`parshall_flume`] — canal jaugeur Parshall : débit en écoulement libre, noyage.
//! - [`isentropic_flow`] — écoulement isentropique : rapports d'arrêt, section A/A*.
//! - [`gear_planetary_torque`] — répartition des couples d'un train épicycloïdal.
//! - [`buoyancy_stability`] — stabilité d'un corps flottant : poussée, hauteur métacentrique.
//! - [`siphon`] — siphon : vitesse de Torricelli, débit, hauteur maximale avant cavitation.
//! - [`spiral_spring`] — ressort spiral plan : couple de rappel, contrainte, énergie.
//! - [`labyrinth_seal`] — joint labyrinthe : fuite de gaz (Martin), report cinétique.
//! - [`hydrostatic_force`] — poussée sur surface immergée : force résultante, centre de poussée.
//! - [`strain_gauge`] — jauge de déformation : pont de Wheatstone, quart/plein pont.
//! - [`strain_rosette`] — rosette 0/45/90 : déformations principales, orientation, cisaillement max.
//! - [`wien_law`] — loi de déplacement de Wien : pic d'émission du corps noir, pyrométrie.
//! - [`differential_gear`] — différentiel automobile : relation cinématique, répartition du couple.
//! - [`harmonic_drive`] — réducteur à onde de déformation : rapport, vitesse et couple de sortie.
//! - [`phase_fraction`] — règle des bras de levier (diagramme de phases binaire).
//! - [`diffusion_fick`] — lois de Fick : flux, profil erfc, coefficient d'Arrhenius.
//! - [`beam_shear_flow`] — cisaillement transverse de Jourawski VQ/(I·t), pas des connecteurs.
//! - [`cpm_schedule`] — chemin critique (CPM) : dates au plus tôt/tard, marges.
//! - [`pert_estimate`] — PERT : durée espérée, variance, score normal réduit.
//! - [`taguchi_loss`] — fonction de perte de Taguchi : coefficient, perte moyenne.
//! - [`dpmo_sigma`] — Six Sigma : DPMO, DPU, rendement de Poisson.
//! - [`wbgt_index`] — indice WBGT de stress thermique (ISO 7243).
//! - [`hand_arm_vibration`] — exposition main-bras HAV : A(8), dose partielle, durée limite.
//! - [`duty_cycle_torque`] — couple efficace (RMS) d'un cycle de service, facteur de marche.
//! - [`brazing_joint`] — joint brasé : cisaillement à recouvrement, pression capillaire.
//! - [`metal_spinning`] — fluotournage conique : loi du sinus, amincissement.
//! - [`arch_thrust`] — arc à trois articulations : poussée horizontale, réactions.
//! - [`cooling_tower`] — tour de refroidissement : plage, approche, efficacité, évaporation.
//! - [`venturi_ejector`] — éjecteur à Venturi : rapport d'entraînement, vide généré.
//! - [`gear_scuffing`] — grippage d'engrenage : température éclair de Blok, sécurité.
//! - [`gear_wear_load`] — charge limite d'usure d'engrenage (Buckingham).
//! - [`gear_dynamic_load`] — charge dynamique de denture (Buckingham, Barth).
//! - [`bearing_friction`] — couple de frottement de roulement (Palmgren), puissance dissipée.
//! - [`pv_limit`] — facteur PV des paliers lisses : pression, vitesse, PV admissible.
//! - [`torque_angle`] — serrage à l'angle (tour-d'écrou) : précharge depuis l'angle.
//! - [`gas_pipeline`] — écoulement de gaz : équation de Weymouth.
//! - [`critical_insulation_radius`] — rayon critique d'isolation (cylindre/sphère).
//! - [`degree_days`] — degrés-jours : chauffage/refroidissement, énergie déperdie.
//! - [`reverberation_time`] — RT60 : Sabine, Eyring, aire d'absorption.
//! - [`sound_transmission_loss`] — affaiblissement d'une paroi : loi de masse, composite.
//! - [`optical_flat`] — verre étalon plan : franges d'interférence, écart de planéité.
//! - [`heat_pump_cop`] — COP de pompe à chaleur : chauffage/froid, borne de Carnot.
//! - [`boiler_efficiency`] — rendement de chaudière : méthodes directe/indirecte, purge.
//! - [`flange_bolt`] — boulonnage de bride (ASME) : efforts, aire de boulons requise.
//! - [`surge_tank`] — cheminée d'équilibre : oscillation en masse, aire de Thoma.
//! - [`steam_quality`] — titre de vapeur : propriétés du mélange saturé.
//! - [`forecasting`] — prévision : lissage exponentiel, moyenne mobile, MAD.
//! - [`ecm_machining`] — usinage électrochimique (Faraday) : débit enlevé, jeu d'équilibre.
//! - [`thermocouple`] — thermocouple (Seebeck) : f.é.m., compensation soudure froide.
//! - [`radiation_network`] — rayonnement : méthode du réseau de résistances (surface, espace).
//! - [`radiation_shield`] — écrans anti-rayonnement : facteur 1/(N+1), flux réduit.
//! - [`coulomb_damping`] — frottement sec : décroissance linéaire, bande morte, arrêt.
//! - [`coffin_manson`] — fatigue oligocyclique : courbe déformation-durée, transition.
//! - [`ramberg_osgood`] — loi élasto-plastique : déformation totale/élastique/plastique.
//! - [`whole_body_vibration`] — WBV (ISO 2631) : A(8), axe dominant, somme vectorielle.
//! - [`fmea_rpn`] — AMDEC : indice de priorité de risque RPN = S·O·D, criticité.
//! - [`silo_pressure`] — silo (Janssen) : pression verticale/pariétale, asymptote.
//! - [`comminution`] — broyage : lois de Bond, Rittinger, Kick, rapport de réduction.
//! - [`mixing_power`] — agitation : puissance turbulente/laminaire, Reynolds d'agitation.
//! - [`cyclone_separator`] — cyclone (Lapple) : diamètre de coupure, rendement, perte de charge.
//! - [`filtration`] — filtration sur gâteau : flux de Darcy, équation de Ruth.
//! - [`sluice_gate`] — vanne de fond : débit soutiré, veine contractée, poussée.
//! - [`doppler`] — effet Doppler : décalage de fréquence, vélocimétrie par réflexion.
//! - [`lighting_lumen`] — éclairagisme : méthode des lumens, indice du local, flux requis.
//! - [`pneumatic_conveying`] — transport pneumatique dilué : taux de charge, Froude de saltation.
//! - [`heat_exchanger_fouling`] — encrassement : résistance Rf, coefficient encrassé, propreté.
//! - [`air_mixing`] — mélange adiabatique d'air humide : température, teneur en eau, enthalpie.
//!
//! ## Positionnement
//!
//! Cette crate complète les briques mécaniques existantes de SciRust :
//! `scirust-tolerance` (tolérancement inertiel/statistique, ajustements ISO 286,
//! ISO 1101 numérique), `scirust-metrology` (incertitude GUM), `scirust-fatigue`
//! (comptage rainflow, Palmgren-Miner) et `scirust-fab` (contrôle de procédé).
//! Elle apporte le **cœur calcul de la coupe** qui manquait.
//!
//! ## Unités
//!
//! Convention des fiches outil : `Vc` en m/min, longueurs et diamètres en mm,
//! `N` en tr/min, avances en mm (par tour ou par dent), efforts en N,
//! puissances en kW, couples en N·m, rugosités en µm. Chaque fonction rappelle
//! ses unités.
//!
//! **Limite honnête** : ce sont des **modèles d'ingénieur** (Kienzle, Taylor,
//! Gilbert, rugosité géométrique). Leurs constantes — `kc1.1`, `mc`, `n`, `C` —
//! sont des données du couple outil/matière que l'appelant fournit d'après un
//! catalogue ou des essais ; la crate calcule leurs conséquences, elle
//! n'invente aucune valeur « par défaut » qui serait invérifiable.
//!
//! ## Exemple
//!
//! ```
//! use scirust_machining::kinematics::{spindle_speed_rpm, mrr_turning_cm3_min};
//! use scirust_machining::forces::{KienzleModel, cutting_power_kw};
//!
//! // Chariotage d'un acier Ø80 mm à Vc = 200 m/min, ap = 3 mm, f = 0,25 mm/tr.
//! let n = spindle_speed_rpm(200.0, 80.0);              // ≈ 796 tr/min
//! let q = mrr_turning_cm3_min(200.0, 3.0, 0.25);       // 150 cm³/min
//!
//! // Effort et puissance de coupe (outil couteau, κr = 90°).
//! let steel = KienzleModel { kc11: 1700.0, mc: 0.25 };
//! let fc = steel.cutting_force_turning(3.0, 0.25, 90.0);
//! let pc = cutting_power_kw(fc, 200.0);                // kW à la coupe
//! assert!(n > 795.0 && n < 797.0);
//! assert!((q - 150.0).abs() < 1e-9);
//! assert!(pc > 0.0);
//! ```

pub mod accelerated_life;
pub mod air_flow;
pub mod archard_wear;
pub mod area_moments;
pub mod balancing;
pub mod ball_screw;
pub mod beam_reactions;
pub mod beams;
pub mod bearings;
pub mod belleville_washers;
pub mod belts;
pub mod bernoulli;
pub mod bevel_worm_gears;
pub mod bode_first_order;
pub mod bolted_joints;
pub mod brakes;
pub mod buckling;
pub mod cables;
pub mod cams;
pub mod castigliano;
pub mod centroids;
pub mod chain_drive;
pub mod chvorinov;
pub mod combined_stress;
pub mod composites;
pub mod compressed_air;
pub mod convection_correlations;
pub mod couplings;
pub mod creep;
pub mod critical_speed;
pub mod deflection_cases;
pub mod dimension_chains;
pub mod distributed_loads;
pub mod drag_lift;
pub mod drilling_ops;
pub mod dynamics;
pub mod economics;
pub mod elasticity_relations;
pub mod elastomer_mounts;
pub mod endurance_limit;
pub mod epicyclic;
pub mod extension_springs;
pub mod extrusion;
pub mod fastener_groups;
pub mod fatigue_mean_stress;
pub mod feedback_loop;
pub mod film_lubrication;
pub mod fins;
pub mod first_order_response;
pub mod flow_meters;
pub mod flywheel;
pub mod forced_vibrations;
pub mod forces;
pub mod fourbar;
pub mod fracture;
pub mod friction;
pub mod frictional_heating;
pub mod gating;
pub mod gear_trains;
pub mod gears;
pub mod geneva;
pub mod hardness;
pub mod heat_exchanger;
pub mod hertz;
pub mod hookes_law_3d;
pub mod hydraulic_cylinders;
pub mod hyperstatism;
pub mod impact;
pub mod interference_fit;
pub mod iso6336;
pub mod journal_bearings;
pub mod keys;
pub mod kinematics;
pub mod leaf_springs;
pub mod liaisons;
pub mod maintenance;
pub mod merchant;
pub mod milling_chip;
pub mod mohr;
pub mod motion_profile;
pub mod motor_torque;
pub mod msa;
pub mod oee;
pub mod open_channel;
pub mod pid_control;
pub mod pipe_flow;
pub mod pneumatic_cylinders;
pub mod power_screws;
pub mod press_brake;
pub mod pressure_vessels;
pub mod process_time;
pub mod pulley_systems;
pub mod pump_system;
pub mod pumps;
pub mod rack_pinion;
pub mod radiation;
pub mod reflected_inertia;
pub mod reliability;
pub mod riveted_joints;
pub mod rolling;
pub mod rolling_resistance;
pub mod roughness;
pub mod second_order_response;
pub mod shafts;
pub mod sheet_bending;
pub mod slider_crank;
pub mod splines;
pub mod springs;
pub mod stepper_motor;
pub mod strain_energy;
pub mod stress_concentration;
pub mod system_reliability;
pub mod thermal;
pub mod thermal_network;
pub mod thermo_cycles;
pub mod threads;
pub mod time;
pub mod tolerancing;
pub mod toollife;
pub mod torseurs;
pub mod torsion_profiles;
pub mod torsion_springs;
pub mod transient_conduction;
pub mod true_stress_strain;
pub mod trusses;
pub mod universal_joint;
pub mod upsetting;
pub mod vacuum_gripping;
pub mod valve_sizing;
pub mod vibrations;
pub mod viscosity;
pub mod water_hammer;
pub mod wedge;
pub mod weibull;
pub mod welds;
pub mod wire_drawing;

// Lot massif (vol. 24) — assemblage/étanchéité, métrologie d'atelier,
// sélection de transmissions, levage, mise en forme, fluide/plastique.
pub mod adhesive_lap_joint;
pub mod blanking_force;
pub mod bolt_circle;
pub mod deep_drawing;
pub mod gasket_seating;
pub mod gear_span_measurement;
pub mod hydraulic_accumulator;
pub mod injection_cooling;
pub mod lifting_lug;
pub mod o_ring_seals;
pub mod planetary_constraints;
pub mod sine_bar;
pub mod sling_tension;
pub mod snap_fit_cantilever;
pub mod taper;
pub mod three_wire_thread;
pub mod timing_belt;
pub mod tube_bending;
pub mod vbelt_design;
pub mod wire_rope;

// Lot massif (vol. 25) — transmission de puissance, thermique de fabrication,
// dynamique des machines et éléments de ressort.
pub mod belt_conveyor;
pub mod brake_thermal;
pub mod broaching;
pub mod cam_pressure_angle;
pub mod carburizing;
pub mod clutch_engagement;
pub mod coil_spring_surge;
pub mod disc_spring_stack;
pub mod gear_pump;
pub mod governor_flyball;
pub mod hydraulic_motor;
pub mod rotating_ring_stress;
pub mod scissor_lift;
pub mod screw_conveyor;
pub mod spring_nest;
pub mod surface_grinding;
pub mod tapping_torque;
pub mod toggle_mechanism;
pub mod torsional_vibration;
pub mod weld_heat_input;

// Lot massif (vol. 26) — économie de production, robotique planaire,
// fonderie/injection, GD&T/métrologie et usinage étendu.
pub mod break_even;
pub mod cosine_error;
pub mod countersink_depth;
pub mod die_casting;
pub mod drill_point_geometry;
pub mod eoq_inventory;
pub mod flatness_from_readings;
pub mod forward_kinematics_2r;
pub mod gdt_position_tolerance;
pub mod jacobian_2r;
pub mod learning_curve;
pub mod machine_hour_rate;
pub mod mold_cooling_channel;
pub mod niosh_lifting;
pub mod plastic_shrinkage;
pub mod sand_casting_shrinkage;
pub mod tap_drill_size;
pub mod thread_milling;
pub mod total_runout;
pub mod workspace_2r;

// Lot massif (vol. 27) — traitement de surface, procédés d'assemblage/découpe,
// éléments d'assemblage, alignement/transmission et qualité/SPC.
pub mod acceptance_sampling;
pub mod anodizing;
pub mod bearing_preload;
pub mod belt_slip;
pub mod bevel_gear_forces;
pub mod cotter_joint;
pub mod coupling_misalignment;
pub mod edm_machining;
pub mod electroplating;
pub mod gear_efficiency;
pub mod knuckle_joint;
pub mod laser_cutting;
pub mod retaining_ring;
pub mod rolled_throughput_yield;
pub mod shaft_alignment;
pub mod shewhart_xbar;
pub mod shot_peening;
pub mod spot_welding;
pub mod vbelt_length;
pub mod waterjet_cutting;

// Lot massif (vol. 28) — gestion de production, engrenages spéciaux, mise en
// forme avancée, hydraulique de puissance, thermique de coupe/corrosion/coût.
pub mod air_receiver;
pub mod blank_nesting;
pub mod capacity_planning;
pub mod corrosion_rate;
pub mod cutting_temperature;
pub mod gearbox_efficiency;
pub mod helical_gear_forces;
pub mod hydraulic_flow_velocity;
pub mod internal_gear;
pub mod ironing;
pub mod johnson_scheduling;
pub mod line_balancing;
pub mod littles_law;
pub mod machining_cost;
pub mod pressure_intensifier;
pub mod roll_bending;
pub mod smed_setup;
pub mod stretch_forming;
pub mod takt_time;
pub mod worm_gear_forces;

// Lot massif (vol. 29) — mécanismes, fluides/tension superficielle, rupture/
// thermique matériaux, éléments de machine, métrologie, fabrication.
pub mod abbe_error;
pub mod additive_manufacturing;
pub mod bimetal_strip;
pub mod capillary_rise;
pub mod ceramic_weibull;
pub mod circular_plate;
pub mod curved_beam;
pub mod differential_screw;
pub mod gauge_temperature_correction;
pub mod nozzle_thrust;
pub mod paris_law;
pub mod powder_compaction;
pub mod quick_return;
pub mod ratchet_pawl;
pub mod scotch_yoke;
pub mod stress_relaxation;
pub mod surface_tension_laplace;
pub mod taper_pin;
pub mod thermal_shock_resistance;
pub mod woodruff_key;

// Lot massif (vol. 30) — CND, métallurgie du soudage, isolation/acoustique,
// réfrigération/HVAC, fabrication d'engrenages, éléments/structure.
pub mod air_spring;
pub mod carbon_equivalent;
pub mod dividing_head;
pub mod eddy_current;
pub mod gear_hobbing;
pub mod noise_exposure;
pub mod plate_buckling;
pub mod psychrometrics;
pub mod radiographic_testing;
pub mod refrigeration_cycle;
pub mod shell_buckling;
pub mod sound_power_level;
pub mod sound_pressure_level;
pub mod thread_engagement;
pub mod ultrasonic_testing;
pub mod ventilation_rate;
pub mod vibration_isolation;
pub mod weld_cooling_rate;
pub mod weld_dilution;
pub mod weld_preheat;

// Lot massif (vol. 31) — machines électriques & entraînements, surveillance
// vibratoire des roulements, tribologie, ressorts spéciaux, énergie/fluides,
// structure.
pub mod beam_column;
pub mod bearing_defect_frequencies;
pub mod bearing_static_safety;
pub mod chain_polygon_effect;
pub mod combustion_air;
pub mod compressed_air_cost;
pub mod conical_spring;
pub mod constant_force_spring;
pub mod dc_motor;
pub mod grease_relubrication;
pub mod hydrostatic_bearing;
pub mod induction_motor;
pub mod iso10816_vibration;
pub mod motor_efficiency;
pub mod order_analysis;
pub mod shear_center;
pub mod three_phase_power;
pub mod vfd_volts_per_hertz;
pub mod wave_spring;

// Lot massif (vol. 32) — mécanique des fluides (débitmétrie, surface libre,
// cavitation, similitude), transfert thermique, éléments de machines,
// métallurgie et énergie.
pub mod calorific_value;
pub mod cavitation_number;
pub mod centrifugal_separation;
pub mod fin_efficiency;
pub mod gyroscopic_couple;
pub mod hall_petch;
pub mod hydraulic_jump;
pub mod keyway_stress;
pub mod larson_miller;
pub mod lmtd;
pub mod orifice_plate;
pub mod pitot_tube;
pub mod pump_affinity;
pub mod standard_atmosphere;
pub mod stokes_settling;
pub mod thermal_expansion;
pub mod torsion_bar;
pub mod valve_flow_coefficient;
pub mod venturi_meter;
pub mod weir_flow;

// Lot massif (vol. 33) — fonderie, transfert thermique/rayonnement, corrosion,
// structure (poutres, sections), tôlerie, usinage, dynamique du choc/vol,
// éléments de machines et entraînements.
pub mod beam_deflection;
pub mod bend_allowance;
pub mod bolt_torque;
pub mod crank_effort;
pub mod elastic_constants;
pub mod faraday_corrosion;
pub mod fillet_weld;
pub mod friction_brake;
pub mod gear_contact_ratio;
pub mod gear_lewis;
pub mod motor_starting;
pub mod projectile_motion;
pub mod restitution;
pub mod riser_design;
pub mod section_modulus;
pub mod shaft_sizing;
pub mod spring_combination;
pub mod thermal_resistance;
pub mod turning_roughness;
pub mod view_factor;

// Lot massif (vol. 34) — écoulement compressible/HVAC, procédés de mise en forme,
// assemblages excentrés, dynamique/vibration, embrayages & presse, manutention,
// usinage et éléments d'engrenage.
pub mod bolt_group_shear;
pub mod bucket_elevator;
pub mod centrifugal_clutch;
pub mod choked_flow;
pub mod disc_clutch;
pub mod duct_sizing;
pub mod flat_belt;
pub mod forging_force;
pub mod gear_backlash;
pub mod honing;
pub mod hydraulic_press;
pub mod lapping;
pub mod magnetic_holding;
pub mod pendulum;
pub mod reaming;
pub mod rolling_force;
pub mod rotating_unbalance;
pub mod taper_turning;
pub mod vacuum_lifting;
pub mod weld_group;

// Lot massif (vol. 35) — turbomachines & écoulement compressible, plasticité &
// flambement, maintien/fixation, production tirée, acoustique/vibration,
// transfert thermique diphasique, transmissions, hydrostatique.
pub mod buoyancy_stability;
pub mod cone_clutch;
pub mod crane_stability;
pub mod film_condensation;
pub mod gear_planetary_torque;
pub mod helmholtz_resonator;
pub mod hydraulic_turbine;
pub mod hydrostatic_force;
pub mod isentropic_flow;
pub mod kanban_sizing;
pub mod labyrinth_seal;
pub mod nucleate_boiling;
pub mod parshall_flume;
pub mod plastic_bending;
pub mod pump_specific_speed;
pub mod rankine_column;
pub mod set_screw;
pub mod siphon;
pub mod spiral_spring;
pub mod tuned_mass_damper;

// Lot massif (vol. 36) — extensométrie & instrumentation, rayonnement, transmissions
// spéciales, métallurgie/diffusion, structure, ordonnancement de projet, qualité
// Six Sigma, ergonomie, dimensionnement moteur, assemblage/mise en forme, thermique.
pub mod arch_thrust;
pub mod beam_shear_flow;
pub mod brazing_joint;
pub mod cooling_tower;
pub mod cpm_schedule;
pub mod differential_gear;
pub mod diffusion_fick;
pub mod dpmo_sigma;
pub mod duty_cycle_torque;
pub mod hand_arm_vibration;
pub mod harmonic_drive;
pub mod metal_spinning;
pub mod pert_estimate;
pub mod phase_fraction;
pub mod strain_gauge;
pub mod strain_rosette;
pub mod taguchi_loss;
pub mod venturi_ejector;
pub mod wbgt_index;
pub mod wien_law;

// Lot massif (vol. 37) — transmissions (grippage/usure/dynamique), tribologie,
// fixation, réseaux de fluides, thermique du bâtiment, acoustique du bâtiment,
// métrologie optique, énergétique, appareils à pression, production, procédés
// spéciaux et instrumentation.
pub mod bearing_friction;
pub mod boiler_efficiency;
pub mod critical_insulation_radius;
pub mod degree_days;
pub mod ecm_machining;
pub mod flange_bolt;
pub mod forecasting;
pub mod gas_pipeline;
pub mod gear_dynamic_load;
pub mod gear_scuffing;
pub mod gear_wear_load;
pub mod heat_pump_cop;
pub mod optical_flat;
pub mod pv_limit;
pub mod reverberation_time;
pub mod sound_transmission_loss;
pub mod steam_quality;
pub mod surge_tank;
pub mod thermocouple;
pub mod torque_angle;

// Lot massif (vol. 38) — rayonnement thermique, fatigue/plasticité & vibration,
// qualité/risque, équipements de procédé et manutention de vrac, hydraulique,
// acoustique, éclairagisme, thermique.
pub mod air_mixing;
pub mod coffin_manson;
pub mod comminution;
pub mod coulomb_damping;
pub mod cyclone_separator;
pub mod doppler;
pub mod filtration;
pub mod fmea_rpn;
pub mod heat_exchanger_fouling;
pub mod lighting_lumen;
pub mod mixing_power;
pub mod pneumatic_conveying;
pub mod radiation_network;
pub mod radiation_shield;
pub mod ramberg_osgood;
pub mod silo_pressure;
pub mod sluice_gate;
pub mod whole_body_vibration;

pub use accelerated_life::{
    BOLTZMANN_EV_K, arrhenius_acceleration_factor, derated_value, ten_degree_rule_factor,
};
pub use air_flow::{choked_mass_flow, critical_pressure_ratio, is_choked, speed_of_sound};
pub use archard_wear::{sliding_distance_for_depth, specific_wear_rate, wear_depth, worn_volume};
pub use area_moments::{
    composite_second_moment, parallel_axis as parallel_axis_area, polar_second_moment,
    radius_of_gyration as area_radius_of_gyration,
};
pub use balancing::{
    centrifugal_force, correction_mass, permissible_eccentricity_um, permissible_unbalance_g_mm,
    unbalance,
};
pub use ball_screw::{axial_force_from_torque, drive_torque, linear_speed, rotational_speed_rpm};
pub use beam_reactions::{
    cantilever_point_load_moment, cantilever_udl_moment, ss_point_load_max_moment,
    ss_point_load_reactions, ss_udl_reaction,
};
pub use beams::{
    bending_stress as beam_bending_stress, deflection_cantilever_end_load,
    deflection_simply_supported_center_load, deflection_simply_supported_udl,
    moment_cantilever_end_load, moment_simply_supported_center_load, moment_simply_supported_udl,
    second_moment_circle, second_moment_rectangle,
};
pub use bearings::{
    BearingType, Reliability, adjusted_rating_life, basic_rating_life_hours,
    basic_rating_life_revs, equivalent_dynamic_load,
};
pub use belleville_washers::{flatten_load, k1_factor, load as belleville_load};
pub use belts::{
    belt_speed_m_s, slack_tension, tension_ratio_flat, tension_ratio_vbelt, transmissible_power_w,
    wrap_angle_small_pulley_rad,
};
pub use bernoulli::{
    continuity_velocity, dynamic_pressure, hydrostatic_pressure, reynolds_number,
    torricelli_velocity, total_head,
};
pub use bevel_worm_gears::{
    bevel_pitch_angle_gear, bevel_pitch_angle_pinion, worm_efficiency, worm_gear_ratio,
    worm_lead_angle, worm_self_locking,
};
pub use bode_first_order::{
    corner_frequency_rad as bode_corner_frequency_rad, decibels, magnitude_db, phase_deg,
};
pub use bolted_joints::{
    bolt_working_load, load_factor, preload_from_torque, residual_clamp_load, separation_load,
};
pub use brakes::{
    band_brake_torque, band_tension_ratio, disc_clutch_torque_uniform_pressure,
    disc_clutch_torque_uniform_wear,
};
pub use buckling::{
    EndCondition, critical_load, critical_stress, effective_length, is_euler_valid,
    limiting_slenderness, radius_of_gyration, slenderness_ratio,
};
pub use cables::{horizontal_tension, max_tension, parabolic_length};
pub use cams::{
    cycloidal_acceleration, cycloidal_displacement, cycloidal_velocity, shm_acceleration,
    shm_displacement, shm_velocity,
};
pub use castigliano::{axial_energy, bending_energy, torsion_energy, total_energy};
pub use centroids::{composite_centroid, total_area};
pub use chain_drive::{
    chain_length_pitches, chain_velocity, sprocket_pitch_diameter, sprocket_speed_ratio,
};
pub use chvorinov::{casting_modulus, riser_modulus, solidification_time};
pub use combined_stress::{
    combined_axial_bending, equivalent_bending_moment, equivalent_twisting_moment,
    von_mises_bending_torsion,
};
pub use composites::{
    longitudinal_strength, reuss_modulus, rule_of_mixtures_density, voigt_modulus,
};
pub use compressed_air::{
    adiabatic_outlet_temperature, adiabatic_work, compression_ratio, ideal_gas_density,
    isothermal_work,
};
pub use convection_correlations::{
    convection_coefficient, dittus_boelter, prandtl_number, rayleigh_number,
};
pub use couplings::{bolt_force_from_torque, power_to_torque, torque_from_bolts};
pub use creep::{larson_miller_parameter, norton_creep_rate, rupture_time_from_lmp};
pub use critical_speed::{
    critical_speed_from_deflection_rad, critical_speed_rad, dunkerley_critical_speed_rad,
    rad_to_rpm,
};
pub use deflection_cases::{
    cantilever_end_slope, cantilever_udl_deflection, fixed_fixed_center_deflection,
    simply_supported_center_slope, superpose_deflections,
};
pub use dimension_chains::{
    closing_max, closing_min, closing_nominal, rss_tolerance, worst_case_tolerance,
};
pub use distributed_loads::{
    trapezoidal_resultant, triangular_centroid_from_zero, triangular_resultant, uniform_centroid,
    uniform_resultant,
};
pub use drag_lift::{drag_force, drag_power, lift_force, terminal_velocity};
pub use drilling_ops::{drilling_power_kw, drilling_thrust, drilling_torque, penetration_rate};
pub use dynamics::{
    angular_momentum, inertia_hollow_cylinder, inertia_rod_center, inertia_rod_end,
    inertia_solid_cylinder, inertia_solid_sphere, inertia_thin_ring, kinetic_energy_rotation,
    kinetic_energy_translation, parallel_axis, rotational_power, torque_from_angular_accel,
};
pub use economics::MachiningEconomics;
pub use elasticity_relations::{
    bulk_modulus_from_e_nu, lame_first_parameter, poisson_from_e_g, shear_modulus_from_e_nu,
    youngs_modulus_from_g_nu,
};
pub use elastomer_mounts::{
    apparent_compression_modulus, compression_stiffness, deflection as elastomer_deflection,
    shape_factor, shear_stiffness,
};
pub use endurance_limit::{
    corrected_endurance_limit, fatigue_strength_at_cycles, sn_coefficients,
    steel_endurance_estimate,
};
pub use epicyclic::{
    carrier_speed, reduction_ratio_ring_fixed, ring_speed, ring_teeth, sun_speed, willis_ratio,
};
pub use extension_springs::{
    body_shear_stress, deflection as extension_spring_deflection, force_at_deflection,
    rate as extension_spring_rate,
};
pub use extrusion::{extrusion_force, extrusion_pressure, extrusion_ratio, extrusion_true_strain};
pub use fastener_groups::{group_polar_moment, primary_shear, resultant_shear, secondary_shear};
pub use fatigue_mean_stress::{
    gerber_safety_factor, goodman_safety_factor, mean_stress, soderberg_safety_factor,
    stress_amplitude, stress_ratio,
};
pub use feedback_loop::{
    closed_loop_gain, closed_loop_gain_with_feedback, sensitivity, steady_state_error_step,
};
pub use film_lubrication::{LubricationRegime, hersey_number, lambda_ratio, regime_from_lambda};
pub use fins::{fin_effectiveness, fin_efficiency, fin_heat_rate, fin_parameter};
pub use first_order_response::{
    corner_frequency_rad as first_order_corner_frequency_rad, cutoff_frequency_hz, step_response,
    time_to_fraction,
};
pub use flow_meters::{beta_ratio, flow_rate, pressure_drop_for_flow};
pub use flywheel::{
    coefficient_of_fluctuation, energy_fluctuation, mean_speed, required_inertia, stored_energy,
};
pub use forced_vibrations::{
    frequency_ratio, magnification_factor, phase_lag_rad, resonance_peak_ratio,
    rotating_unbalance_response, transmissibility,
};
pub use forces::{KienzleModel, cutting_power_kw, motor_power_kw, spindle_torque_nm};
pub use fourbar::{FourBarType, classify, is_grashof};
pub use fracture::{
    critical_crack_length, fracture_safety_factor, griffith_stress, stress_intensity,
};
pub use friction::{
    angle_of_repose_deg, friction_angle_deg, incline_self_locking, is_sliding, kinetic_friction,
    max_static_friction, within_adhesion_cone,
};
pub use frictional_heating::{friction_force, friction_power, heat_flux, temperature_rise};
pub use gating::{choke_area, pouring_time, sprue_exit_velocity, sprue_taper_ratio};
pub use gear_trains::{
    compound_train_ratio, gear_pair_speed_ratio, is_direction_reversed, output_speed, output_torque,
};
pub use gears::{
    HelicalGear, SpurGear, center_distance, gear_ratio, lewis_bending_stress,
    minimum_teeth_no_undercut, pitch_line_velocity_m_s, tangential_force_from_power,
    tangential_force_from_torque, transverse_contact_ratio,
};
pub use geneva::{
    center_distance_ratio, crank_ratio, driven_angle, dwell_crank_angle, indexing_crank_angle,
    velocity_ratio as geneva_velocity_ratio,
};
pub use hardness::{brinell_hardness, tensile_strength_from_brinell, vickers_hardness};
pub use heat_exchanger::{
    actual_heat_transfer, capacity_ratio, effectiveness_counterflow, effectiveness_parallel_flow,
    heat_duty_lmtd, lmtd, ntu,
};
pub use hertz::{
    effective_modulus, effective_radius, line_contact_half_width, line_contact_max_pressure,
    point_contact_max_pressure, point_contact_radius,
};
pub use hookes_law_3d::{axial_strain, hydrostatic_stress, shear_strain, volumetric_strain};
pub use hydraulic_cylinders::{
    bore_area, extend_force, fluid_power, piston_speed, retract_force, rod_side_area,
};
pub use hyperstatism::{
    degree_of_hyperstaticity, independent_loops, is_isostatic, kinematic_unknowns, static_unknowns,
};
pub use impact::{
    direct_impact_velocities, energy_lost, falling_load_factor, suddenly_applied_factor,
};
pub use interference_fit::{
    assembly_temperature_rise, contact_pressure_same_material, holding_axial_force, holding_torque,
    hub_hoop_stress,
};
pub use iso6336::{
    contact_stress, elasticity_factor_ze, nominal_contact_stress, safety_factor_pitting,
};
pub use journal_bearings::{
    minimum_film_thickness, petroff_friction_coefficient, petroff_torque, sommerfeld_number,
    unit_load,
};
pub use keys::{
    key_bearing_pressure, key_shear_stress, required_length_for_bearing, tangential_force,
};
pub use kinematics::{
    cutting_speed_m_min, feed_per_rev_milling, feed_velocity_mm_min, mrr_drilling_mm3_min,
    mrr_milling_mm3_min, mrr_turning_cm3_min, spindle_speed_rpm,
};
pub use leaf_springs::{
    bending_stress as leaf_spring_bending_stress, deflection as leaf_spring_deflection,
    rate as leaf_spring_rate,
};
pub use liaisons::{LIAISONS, Liaison};
pub use maintenance::{inherent_availability, maintainability, mtbf, mttr};
pub use merchant::{
    chip_thickness_ratio, merchant_shear_angle, shear_angle, shear_strain as merchant_shear_strain,
};
pub use milling_chip::{chip_thickness_at_angle, engagement_angle, feed_per_tooth, teeth_in_cut};
pub use mohr::{
    max_in_plane_shear, mohr_radius, normal_stress_rotated, principal_angle_rad,
    principal_stresses, safety_factor, shear_stress_rotated, tresca_plane, von_mises_plane,
    von_mises_principal,
};
pub use motion_profile::{
    accel_distance, accel_time, is_triangular, trapezoidal_total_time, triangular_peak_velocity,
};
pub use motor_torque::{acceleration_torque, angular_acceleration, rms_torque, total_torque};
pub use msa::{gage_rr, number_distinct_categories, percent_rr, total_variation};
pub use oee::{availability, oee, performance, quality};
pub use open_channel::{chezy_velocity, hydraulic_radius, manning_flow, manning_velocity};
pub use pid_control::{
    derivative_gain, integral_gain, pid_output, ziegler_nichols_kp, ziegler_nichols_td,
    ziegler_nichols_ti,
};
pub use pipe_flow::{
    colebrook_friction, darcy_head_loss, laminar_friction_factor, minor_loss, swamee_jain_friction,
};
pub use pneumatic_cylinders::{
    extend_force as pneumatic_extend_force, free_air_per_stroke,
    retract_force as pneumatic_retract_force, useful_force,
};
pub use power_screws::{
    efficiency, is_self_locking, lead_angle_deg, lowering_torque_nm, raising_torque_nm,
};
pub use press_brake::{bending_force, springback_ratio};
pub use pressure_vessels::{
    thick_cylinder_hoop_inner, thick_cylinder_hoop_outer, thin_cylinder_hoop,
    thin_cylinder_longitudinal, thin_sphere,
};
pub use process_time::{batch_time, stations_required, throughput_per_hour, time_per_piece};
pub use pulley_systems::{
    actual_mechanical_advantage, efficiency as pulley_efficiency, effort_required,
    velocity_ratio as pulley_velocity_ratio,
};
pub use pump_system::{operating_flow, operating_head, pump_head, system_head};
pub use pumps::{
    affinity_flow, affinity_head, affinity_power, hydraulic_power, npsh_available, shaft_power,
    specific_speed,
};
pub use rack_pinion::{
    force_from_torque, linear_velocity, pinion_pitch_radius, travel_per_revolution,
};
pub use radiation::{
    STEFAN_BOLTZMANN, blackbody_emissive_power, gray_body_emissive_power,
    net_radiation_to_surroundings, radiation_coefficient,
};
pub use reflected_inertia::{
    ballscrew_load_inertia, inertia_ratio, inertia_through_gear, screw_inertia_solid,
};
pub use reliability::{
    exponential_reliability, failure_rate_from_mtbf, mtbf_from_failure_rate, mtbf_from_test,
    probability_of_failure,
};
pub use riveted_joints::{
    bearing_strength, joint_efficiency, rivet_shear_strength, solid_plate_strength,
    tearing_strength,
};
pub use rolling::{contact_length, draft, max_draft, roll_force, roll_torque};
pub use rolling_resistance::{
    coefficient_from_arm, resistance_on_grade, rolling_power, rolling_resistance_force,
};
pub use roughness::{
    feed_for_target_ra, theoretical_ra_turning, theoretical_rt_sharp, theoretical_rt_turning,
};
pub use second_order_response::{damped_frequency, overshoot, peak_time, settling_time_2pct};
pub use shafts::{
    angle_of_twist_deg, bending_stress, polar_section_modulus_hollow, polar_section_modulus_solid,
    section_modulus_hollow, section_modulus_solid, torsional_shear_stress, von_mises_solid,
};
pub use sheet_bending::{bend_allowance, bend_deduction, developed_length, outside_setback};
pub use slider_crank::{
    obliquity_ratio, piston_acceleration_approx, piston_displacement, piston_velocity,
};
pub use splines::{mean_radius, torque_capacity};
pub use springs::HelicalSpring;
pub use stepper_motor::{
    linear_resolution, pulse_rate_for_speed, speed_from_pulse_rate, step_angle_deg,
    steps_per_revolution,
};
pub use strain_energy::{
    axial_strain_energy_density, modulus_of_resilience, shear_strain_energy_density,
    total_strain_energy,
};
pub use stress_concentration::{
    fatigue_stress_concentration, nominal_stress_plate_with_hole, peak_stress,
};
pub use system_reliability::{k_out_of_n_reliability, parallel_reliability, series_reliability};
pub use thermal::{
    conduction_heat_flow, convection_heat_flow, linear_expansion, sensible_heat,
    thermal_resistance, thermal_stress,
};
pub use thermal_network::{
    convection_resistance, heat_flow, overall_heat_transfer_coefficient, parallel_resistance,
    series_resistance,
};
pub use thermo_cycles::{
    carnot_efficiency, cop_heat_pump_carnot, cop_refrigerator_carnot, diesel_efficiency,
    otto_efficiency, thermal_efficiency,
};
pub use threads::MetricThread;
pub use time::{
    drilling_time_min, milling_time_min, number_of_passes, pass_time_min, turning_time_min,
};
pub use tolerancing::{
    GPS_CATALOGUE, GeneralClass, GeometricalClass, GpsStandard, general_angular_tolerance,
    general_circular_runout, general_linear_tolerance, general_perpendicularity,
    general_straightness_flatness, general_symmetry,
};
pub use toollife::{ExtendedTaylor, taylor_cutting_speed, taylor_tool_life};
pub use torseurs::Torseur;
pub use torsion_profiles::{
    bredt_shear_stress, bredt_twist_rate, rectangular_max_shear, rectangular_torsion_constant,
    thin_strip_max_shear, thin_strip_torsion_constant,
};
pub use torsion_springs::{
    angular_deflection, angular_rate, bending_stress as torsion_spring_bending_stress,
};
pub use transient_conduction::{
    biot_number, fourier_number, lumped_capacitance_valid, lumped_temperature, time_constant,
};
pub use true_stress_strain::{hollomon_stress, true_strain, true_stress};
pub use trusses::{axial_stress, member_elongation, two_member_joint};
pub use universal_joint::{
    max_velocity_ratio as cardan_max_velocity_ratio,
    min_velocity_ratio as cardan_min_velocity_ratio, output_angle,
    velocity_ratio as cardan_velocity_ratio,
};
pub use upsetting::{forming_work, upsetting_force, upsetting_true_strain};
pub use vacuum_gripping::{
    number_of_cups, required_diameter, theoretical_holding_force, working_load,
};
pub use valve_sizing::{cv_from_kv, kv_from_cv, liquid_flow, required_kv};
pub use vibrations::{
    critical_damping, damped_frequency_rad, damping_ratio, log_decrement, natural_frequency_hz,
    natural_frequency_rad, quality_factor,
};
pub use viscosity::{
    andrade_viscosity, dynamic_from_kinematic, kinematic_from_dynamic, pa_s_from_centipoise,
};
pub use water_hammer::{critical_time, joukowsky_surge, wave_speed_elastic, wave_speed_rigid};
pub use wedge::{driving_force, extraction_force, ideal_mechanical_advantage, self_locking};
pub use weibull::{weibull_b_life, weibull_hazard_rate, weibull_reliability};
pub use welds::{
    butt_weld_stress, fillet_direct_shear_stress, fillet_throat_area, throat_thickness,
    weld_group_torsional_shear,
};
pub use wire_drawing::{
    MAX_REDUCTION_IDEAL, area_reduction, drawing_force, drawing_stress, drawing_true_strain,
};

// Lot massif (vol. 24) — ré-exports à plat.
pub use adhesive_lap_joint::{
    adhesive_average_shear_stress, adhesive_joint_strength, required_overlap_length,
};
pub use blanking_force::{blanking_force, blanking_work, stripping_force};
pub use bolt_circle::{bolt_hole_angle_rad, bolt_hole_position, chord_between_holes};
pub use deep_drawing::{
    blank_holder_force, drawing_force as deep_drawing_force, limiting_draw_ratio,
};
pub use gasket_seating::{
    gasket_min_seating_load, gasket_operating_load, hydrostatic_end_force, required_bolt_load,
};
pub use gear_span_measurement::{base_tangent_length, involute_function};
pub use hydraulic_accumulator::{gas_volume_at_pressure, required_gas_volume, usable_fluid_volume};
pub use injection_cooling::{
    center_temperature, cooling_fourier_number, cooling_time, wall_thickness_for_time,
};
pub use lifting_lug::{
    lug_double_shear_stress, lug_net_section_stress, lug_pin_area, lug_pin_bearing_stress,
};
pub use o_ring_seals::{gland_fill_percent, oring_squeeze_ratio, oring_stretch_ratio};
pub use planetary_constraints::{
    assembly_condition, planet_teeth_from_sun_ring, ring_teeth_from_sun_planet,
    sun_planet_center_distance,
};
pub use sine_bar::{sine_bar_angle_rad, sine_bar_gauge_height};
pub use sling_tension::{sling_horizontal_force, sling_leg_tension, sling_load_factor};
pub use snap_fit_cantilever::{snap_deflection_force, snap_mating_force, snap_max_strain};
pub use taper::{
    taper_diameter_at_distance, taper_half_angle_rad, taper_included_angle_rad, taper_ratio,
    taper_ratio_from_included_angle,
};
pub use three_wire_thread::{
    best_wire_diameter, measurement_over_wires, pitch_diameter_from_measurement,
};
pub use timing_belt::{belt_pitch_length_two_pulley, belt_teeth_in_mesh, timing_belt_teeth};
pub use tube_bending::{minimum_bend_radius, neutral_axis_length, outer_fiber_strain};
pub use vbelt_design::{corrected_power_per_belt, number_of_belts, vbelt_design_power};
pub use wire_rope::{minimum_breaking_force, minimum_sheave_diameter, wire_rope_working_load};

// Lot massif (vol. 25) — ré-exports à plat.
pub use belt_conveyor::{
    conveyor_drive_power, conveyor_effective_tension, conveyor_lift_power, conveyor_mass_flow,
};
pub use brake_thermal::{brake_dissipated_energy, brake_power, brake_temperature_rise};
pub use broaching::{broaching_force, broaching_force_per_tooth, broaching_teeth_engaged};
pub use cam_pressure_angle::{
    cam_min_base_radius, cam_pitch_radius, cam_pressure_angle_offset_rad, cam_pressure_angle_rad,
    cam_velocity_per_rad,
};
pub use carburizing::{
    carburizing_constant_from_diffusion, carburizing_time_for_depth, case_depth_from_diffusion,
    case_depth_rule_of_thumb,
};
pub use clutch_engagement::{clutch_slip_energy, clutch_slip_time, clutch_temperature_rise};
pub use coil_spring_surge::{spring_surge_frequency_fixed_free_hz, spring_surge_frequency_hz};
pub use disc_spring_stack::{
    stack_combined, stack_deflection_series, stack_load_parallel, stack_stiffness,
    stack_washer_count,
};
pub use gear_pump::{gearpump_actual_flow, gearpump_displacement, gearpump_theoretical_flow};
pub use governor_flyball::{
    GRAVITY, flyball_centrifugal_force, porter_governor_height, porter_governor_speed_rad,
    watt_governor_height,
};
pub use hydraulic_motor::{
    hydromotor_hydraulic_power, hydromotor_output_power, hydromotor_overall_efficiency,
    hydromotor_speed, hydromotor_torque,
};
pub use rotating_ring_stress::{rim_hoop_stress, rim_speed_from_rpm, rotating_burst_speed_rad};
pub use scissor_lift::{scissor_actuator_force, scissor_height, scissor_mechanical_advantage};
pub use screw_conveyor::{screw_conveyor_mass_flow, screw_conveyor_volumetric_flow};
pub use spring_nest::{
    nested_spring_combined_rate, nested_spring_deflection, nested_spring_load_share_inner,
    nested_spring_load_share_outer,
};
pub use surface_grinding::{
    equivalent_chip_thickness, grinding_material_removal_rate, grinding_ratio,
    specific_removal_rate,
};
pub use tapping_torque::{tapping_power, tapping_torque_cutting, tapping_torque_empirical};
pub use toggle_mechanism::{toggle_force_ratio, toggle_output_force};
pub use torsional_vibration::{
    torsional_stiffness, two_disc_equivalent_inertia, two_disc_natural_frequency_hz,
    two_disc_natural_frequency_rad, two_disc_node_position,
};
pub use weld_heat_input::{weld_arc_power, weld_energy_per_length, weld_heat_input};

// Lot massif (vol. 26) — ré-exports à plat.
pub use break_even::{
    break_even_quantity, break_even_revenue, margin_contribution, margin_of_safety,
};
pub use cosine_error::{
    alignment_max_angle_for_error, cosine_error, cosine_true_value_from_reading,
};
pub use countersink_depth::{countersink_depth, countersink_diameter_from_depth};
pub use die_casting::{diecast_fill_time, diecast_gate_velocity, diecast_locking_force};
pub use drill_point_geometry::{
    drill_point_extra_travel, drill_point_length, drill_point_lip_length,
};
pub use eoq_inventory::{
    eoq_economic_order_quantity, eoq_number_of_orders, inventory_reorder_point,
    inventory_total_cost,
};
pub use flatness_from_readings::{
    flatness_error, flatness_is_within, straightness_error, straightness_is_within,
};
pub use forward_kinematics_2r::{fk2r_elbow_position, fk2r_reach_distance, fk2r_tcp_position};
pub use gdt_position_tolerance::{
    gdt_position_bonus_tolerance, gdt_position_diametral_deviation, gdt_position_is_within,
    gdt_position_total_tolerance,
};
pub use jacobian_2r::{jac2r_determinant, jac2r_is_singular, jac2r_jacobian, jac2r_tip_velocity};
pub use learning_curve::{
    learning_curve_average_time, learning_curve_cumulative_time, learning_curve_exponent,
    learning_curve_unit_time,
};
pub use machine_hour_rate::{
    machine_depreciation_per_hour, machine_hour_rate, machine_power_cost_per_hour,
};
pub use mold_cooling_channel::{
    COOLANT_TURBULENT_REYNOLDS, coolant_flow_rate, coolant_is_turbulent, coolant_reynolds,
    coolant_velocity_from_flow_rate, mold_heat_removal_rate, mold_mass_flow_for_heat_removal,
};
pub use niosh_lifting::{
    LOAD_CONSTANT, lifting_index, lifting_recommended_weight_limit, niosh_asymmetry_multiplier,
    niosh_distance_multiplier, niosh_horizontal_multiplier, niosh_vertical_multiplier,
};
pub use plastic_shrinkage::{
    cavity_dimension, cavity_shrinkage_compensation, plastic_actual_part_dimension,
    plastic_shrinkage_rate,
};
pub use sand_casting_shrinkage::{
    casting_shrinkage_ratio, pattern_dimension, pattern_draft_added_dimension,
    pattern_full_dimension, pattern_machining_allowance_added,
};
pub use tap_drill_size::{
    TAP_DRILL_HEIGHT_FACTOR_ISO60, tap_drill_diameter, thread_engagement_percent,
};
pub use thread_milling::{
    thread_mill_helical_revolutions, thread_mill_pass_time,
    thread_mill_peripheral_feed_compensation, thread_mill_time,
};
pub use total_runout::{runout_circular, runout_is_within, tir_total_indicated_runout};
pub use workspace_2r::{ws2r_is_reachable, ws2r_max_reach, ws2r_min_reach, ws2r_workspace_area};

// Lot massif (vol. 27) — ré-exports à plat.
pub use acceptance_sampling::{
    average_outgoing_quality, probability_of_acceptance_binomial,
    probability_of_rejection_binomial, sampling_binomial_coefficient, sampling_binomial_pmf,
};
pub use anodizing::{
    anodizing_current_density_for_thickness, anodizing_growth_factor, anodizing_time_for_thickness,
    oxide_sealed_thickness, oxide_thickness,
};
pub use bearing_preload::{axial_preload_from_offset, preload_deflection, preloaded_stiffness};
pub use belt_slip::{
    belt_slip_creep_from_tension, belt_slip_effective_velocity_ratio, belt_slip_speed_loss,
};
pub use bevel_gear_forces::{
    bevel_axial_force, bevel_radial_force, bevel_resultant_force, bevel_separating_force,
    bevel_tangential_force,
};
pub use cotter_joint::{cotter_crushing_stress, cotter_shear_stress, rod_tensile_stress};
pub use coupling_misalignment::{
    coupling_parallel_offset_reaction, misalign_angular_deg, misalign_combined,
};
pub use edm_machining::{edm_electrode_wear_ratio, edm_material_removal_rate, edm_overcut};
pub use electroplating::{
    FARADAY, plating_deposited_mass, plating_thickness, plating_time_for_thickness,
};
pub use gear_efficiency::{
    gear_eff_output_power, gear_eff_power_loss, gear_eff_spur_efficiency,
    mesh_sliding_factor_from_arcs, mesh_sliding_loss_fraction,
};
pub use knuckle_joint::{
    knuckle_eye_crushing_stress, knuckle_fork_crushing_stress, knuckle_pin_shear_stress,
    knuckle_rod_tensile_stress,
};
pub use laser_cutting::{laser_cutting_speed, laser_power_density, laser_required_power};
pub use retaining_ring::{ring_shear_capacity, ring_thrust_capacity};
pub use rolled_throughput_yield::{
    rty_first_pass_yield, rty_normalized_yield, rty_rolled_throughput_yield,
    rty_total_defects_per_unit, rty_yield_from_defects_per_unit,
};
pub use shaft_alignment::{
    alignment_angular_misalignment, alignment_shim_correction, rim_parallel_offset,
};
pub use shewhart_xbar::{
    rchart_center_line, rchart_lower_control_limit, rchart_upper_control_limit,
    shewhart_process_in_control, xbar_center_line, xbar_lower_control_limit,
    xbar_upper_control_limit,
};
pub use shot_peening::{
    almen_arc_height, almen_saturation_increase_ratio, peening_coverage_from_passes,
    peening_coverage_percent, peening_passes_for_coverage, peening_time_for_coverage,
};
pub use spot_welding::{joule_heat, nugget_heat_density, spot_current_from_heat};
pub use vbelt_length::{
    crossed_belt_length, open_belt_length, open_belt_wrap_angle_large, open_belt_wrap_angle_small,
};
pub use waterjet_cutting::{
    waterjet_cutting_speed_index, waterjet_jet_power, waterjet_jet_velocity,
};

// Lot massif (vol. 28) — ré-exports à plat.
pub use air_receiver::{receiver_pump_up_time, receiver_usable_air, receiver_volume};
pub use blank_nesting::{
    nesting_material_utilization, nesting_part_area_for_utilization, nesting_parts_per_strip,
    nesting_scrap_fraction, nesting_strip_pitch,
};
pub use capacity_planning::{
    capacity_available_capacity, capacity_bottleneck_rate, capacity_required_capacity,
    capacity_utilization,
};
pub use corrosion_rate::{
    CORROSION_FARADAY, corrosion_penetration_rate, corrosion_rate_from_current, cpr_mass_loss,
    faraday_corrosion_current,
};
pub use cutting_temperature::{
    cutting_temp_cook_temperature_index, cutting_temp_cutting_temperature,
    cutting_temp_shear_zone_temperature_rise,
};
pub use gearbox_efficiency::{
    gearbox_output_power, gearbox_overall_efficiency, gearbox_power_loss, gearbox_stages_for_ratio,
};
pub use helical_gear_forces::{
    helical_axial_force, helical_normal_force, helical_radial_force, helical_resultant_force,
    helical_tangential_force,
};
pub use hydraulic_flow_velocity::{
    hydvel_flow_velocity, hydvel_is_velocity_acceptable, hydvel_pipe_diameter_for_velocity,
};
pub use internal_gear::{
    internal_center_distance, internal_gear_ratio, internal_ring_tip_diameter,
};
pub use ironing::{
    ironing_force, ironing_reduction, ironing_true_strain, ironing_true_strain_from_reduction,
    ironing_wall_perimeter,
};
pub use johnson_scheduling::{flowshop_idle_time_machine2, flowshop_makespan_two_machines};
pub use line_balancing::{
    balance_delay, balance_smoothness_index, line_cycle_time, line_efficiency,
    line_theoretical_min_stations,
};
pub use littles_law::{little_cycle_time, little_throughput, little_wip};
pub use machining_cost::{
    machining_cost_per_part, machining_cost_tooling_per_part, machining_cost_total_batch,
};
pub use pressure_intensifier::{
    intensifier_output_flow, intensifier_output_pressure, intensifier_pressure_ratio,
    intensifier_required_input_pressure,
};
pub use roll_bending::{
    roll_bend_minimum_radius, roll_bend_radius_from_geometry, roll_bend_springback_ratio,
};
pub use smed_setup::{
    setup_downtime, setup_economic_batch_from_setup, smed_downtime_after_conversion,
    smed_downtime_reduction_ratio, smed_total_changeover_time,
};
pub use stretch_forming::{stretch_force, stretch_thickness_after, stretch_true_strain};
pub use takt_time::{lean_number_of_operators, takt_required_output_rate, takt_time};
pub use worm_gear_forces::{
    worm_axial_force, worm_normal_force, worm_separating_force, worm_tangential_force,
};

// Lot massif (vol. 29) — ré-exports à plat.
pub use abbe_error::{abbe_error, abbe_error_small_angle, abbe_max_offset_for_error};
pub use additive_manufacturing::{am_build_time, am_deposition_rate, am_number_of_layers};
pub use bimetal_strip::{bimetal_curvature, bimetal_radius, bimetal_tip_deflection};
pub use capillary_rise::{CAPILLARY_GRAVITY, capillary_height, capillary_pressure};
pub use ceramic_weibull::{
    ceramic_weibull_failure_probability, ceramic_weibull_size_effect,
    ceramic_weibull_stress_for_probability, ceramic_weibull_survival_probability,
};
pub use circular_plate::{
    clamped_plate_max_deflection, plate_flexural_rigidity, simply_supported_max_deflection,
};
pub use curved_beam::{
    curvedbeam_neutral_axis_shift, curvedbeam_neutral_radius,
    curvedbeam_rectangular_neutral_radius, curvedbeam_rectangular_section_integral,
    curvedbeam_stress,
};
pub use differential_screw::{
    differential_advance, differential_lead, differential_turns, diffscrew_ideal_axial_force,
    diffscrew_mechanical_advantage,
};
pub use gauge_temperature_correction::{
    GAUGETEMP_REFERENCE_TEMPERATURE_C, dimcorr_thermal_offset, dimcorr_to_reference_length,
    gaugetemp_corrected_length, gaugetemp_differential_expansion_error,
};
pub use nozzle_thrust::{nozzle_exit_velocity_bernoulli, nozzle_mass_flow, nozzle_thrust};
pub use paris_law::{
    paris_crack_growth_rate, paris_delta_k_from_rate, paris_stress_intensity_range,
};
pub use powder_compaction::{
    pm_compaction_pressure_ratio, pm_green_density_ratio, pm_green_porosity, pm_sintered_dimension,
    pm_sintering_linear_shrinkage, pm_sintering_volume_shrinkage,
};
pub use quick_return::{
    quickreturn_cutting_time_fraction, quickreturn_ratio, quickreturn_return_stroke_angle,
    quickreturn_time_ratio_from_angles,
};
pub use ratchet_pawl::{
    ratchet_holding_torque, ratchet_min_teeth_for_angle, ratchet_pitch_radius,
    ratchet_tangential_force, ratchet_tooth_pitch_angle,
};
pub use scotch_yoke::{scotch_acceleration, scotch_displacement, scotch_stroke, scotch_velocity};
pub use stress_relaxation::{
    relaxation_remaining_stress_fraction, relaxation_stress, relaxation_time_constant,
};
pub use surface_tension_laplace::{
    SURFACE_TENSION_STANDARD_GRAVITY, laplace_bubble_pressure, laplace_capillary_length,
    laplace_droplet_pressure,
};
pub use taper_pin::{taperpin_double_shear_stress, taperpin_mean_diameter, taperpin_shear_stress};
pub use thermal_shock_resistance::{
    tsr_max_temperature_difference, tsr_parameter_r, tsr_parameter_r_prime,
};
pub use woodruff_key::{woodruff_bearing_stress, woodruff_shear_stress, woodruff_tangential_force};

// Lot massif (vol. 30) — ré-exports à plat.
pub use air_spring::{airspring_force, airspring_natural_frequency, airspring_rate};
pub use carbon_equivalent::{
    CE_THRESHOLD_DIFFICULT, CE_THRESHOLD_GOOD, WeldabilityClass, carbon_equivalent_iiw,
    ce_weldability_class,
};
pub use dividing_head::{dividing_angular_turns, dividing_hole_count, dividing_simple_turns};
pub use eddy_current::{
    eddy_frequency_for_depth, eddy_phase_lag, eddy_standard_depth_of_penetration,
};
pub use gear_hobbing::{
    hobbing_cutting_time, hobbing_generating_time, hobbing_table_feed, hobbing_work_rpm_from_hob,
};
pub use noise_exposure::{
    noise_dose_percent, noise_equivalent_continuous_level, noise_permitted_time,
};
pub use plate_buckling::{
    plate_buckling_coefficient_simply_supported, plate_buckling_slenderness, plate_critical_load,
    plate_critical_stress,
};
pub use psychrometrics::{
    psychro_humidity_ratio, psychro_partial_pressure_from_humidity,
    psychro_partial_pressure_from_ratio, psychro_relative_humidity,
    psychro_specific_enthalpy_moist_air,
};
pub use radiographic_testing::{
    rt_geometric_unsharpness, rt_half_value_layer, rt_transmitted_intensity,
};
pub use refrigeration_cycle::{
    refrig_compressor_work, refrig_cop_refrigeration, refrig_refrigerant_mass_flow,
    refrig_refrigerating_effect,
};
pub use shell_buckling::{
    shell_axial_critical_stress, shell_critical_axial_load, shell_external_pressure_critical,
};
pub use sound_power_level::{spl_to_swl, swl_from_power, swl_to_power, swl_to_spl};
pub use sound_pressure_level::{
    spl_distance_attenuation, spl_from_pressure, spl_sum_two_sources, spl_to_pressure,
};
pub use thread_engagement::{
    thread_eng_length_for_full_strength, thread_eng_required_length, thread_eng_stripping_area,
};
pub use ultrasonic_testing::{
    ut_attenuation_db, ut_defect_depth, ut_near_field_length, ut_wavelength,
};
pub use ventilation_rate::{
    ach_air_changes_per_hour, ach_required_flow, vent_dilution_flow_for_contaminant,
};
pub use vibration_isolation::{
    iso_efficiency, iso_natural_frequency_from_deflection, iso_transmissibility,
};
pub use weld_cooling_rate::{weldcool_cooling_rate, weldcool_cooling_time_thick_plate};
pub use weld_dilution::{dilution_filler_fraction, dilution_ratio, weld_diluted_composition};
pub use weld_preheat::{
    preheat_combined_thickness, preheat_from_carbon_equivalent, preheat_is_required,
};

// Lot massif (vol. 31) — ré-exports à plat.
pub use beam_column::{
    beam_column_amplification_factor, beam_column_amplified_deflection,
    beam_column_amplified_moment, beam_column_load_ratio,
};
pub use bearing_defect_frequencies::{bearing_bpfi, bearing_bpfo, bearing_bsf, bearing_ftf};
pub use bearing_static_safety::{
    bearing_static_equivalent_load, bearing_static_permissible_load, bearing_static_safety_factor,
};
pub use chain_polygon_effect::{
    chain_mean_speed, chain_pitch_diameter, chain_speed_ripple, chain_tooth_angle,
};
pub use combustion_air::{
    COMBUSTION_AMBIENT_O2_VOLUME_PERCENT, combustion_actual_air, combustion_excess_air_from_o2,
    combustion_stoichiometric_air_fuel_ratio,
};
pub use compressed_air_cost::{
    compressed_air_annual_energy, compressed_air_leak_cost, compressed_air_leak_flow,
    compressed_air_specific_power,
};
pub use conical_spring::{
    conical_spring_max_rate, conical_spring_min_rate, conical_spring_solid_height,
};
pub use constant_force_spring::{
    CONSTANT_FORCE_SPRING_FORCE_CONSTANT, constant_force_spring_force,
    constant_force_spring_natural_radius, constant_force_spring_stress,
};
pub use dc_motor::{dc_back_emf, dc_speed_rad, dc_terminal_voltage, dc_torque};
pub use grease_relubrication::{
    grease_life_reduction_factor, grease_mean_diameter_mm, grease_quantity_grams,
    grease_speed_factor_ndm,
};
pub use hydrostatic_bearing::{
    hydrostatic_flow_rate, hydrostatic_load_capacity, hydrostatic_pumping_power,
    hydrostatic_stiffness,
};
pub use induction_motor::{
    induction_rotor_frequency, induction_rotor_speed_rpm, induction_slip,
    induction_synchronous_speed_rpm,
};
pub use iso10816_vibration::{
    VibrationZone, iso10816_peak_from_rms, iso10816_rms_from_peak,
    iso10816_velocity_rms_from_components, iso10816_zone,
};
pub use motor_efficiency::{
    motor_efficiency, motor_input_power, motor_losses, motor_output_torque,
};
pub use order_analysis::{
    order_frequency, order_max_from_samples_per_rev, order_number,
    order_samples_per_rev_for_max_order, order_shaft_frequency,
};
pub use shear_center::{
    shear_center_channel_axis_inertia, shear_center_channel_offset,
    shear_center_is_on_centroid_for_double_symmetry,
};
pub use three_phase_power::{
    three_phase_active_power, three_phase_apparent_power, three_phase_line_current,
    three_phase_reactive_power,
};
pub use vfd_volts_per_hertz::{vf_flux_ratio, vf_ratio, vf_voltage_for_frequency};
pub use wave_spring::{
    WAVE_SPRING_RATE_CONSTANT, wave_spring_deflection, wave_spring_load, wave_spring_rate,
    wave_spring_stress,
};

// Lot massif (vol. 32) — ré-exports à plat.
pub use calorific_value::{calorific_dulong_hhv, calorific_fuel_energy, calorific_lhv_from_hhv};
pub use cavitation_number::{
    cavitation_margin, cavitation_number, npsh_available as cavitation_npsh_available,
};
pub use centrifugal_separation::{
    CENTRIFUGAL_STANDARD_GRAVITY, centrifugal_rcf, centrifugal_rcf_from_rpm,
    centrifugal_sedimentation_velocity,
};
pub use fin_efficiency::{
    fineff_effectiveness, fineff_efficiency_adiabatic_tip, fineff_heat_rate, fineff_parameter_m,
};
pub use gyroscopic_couple::{
    gyroscopic_angular_momentum, gyroscopic_couple, gyroscopic_precession_rate,
};
pub use hall_petch::{
    hall_petch_grain_size_for_yield, hall_petch_strengthening_coefficient,
    hall_petch_yield_strength,
};
pub use hydraulic_jump::{
    hydraulic_jump_conjugate_depth, hydraulic_jump_energy_loss, hydraulic_jump_froude,
};
pub use keyway_stress::{
    keyway_bearing_stress, keyway_required_length_shear, keyway_shear_stress,
    keyway_tangential_force,
};
pub use larson_miller::{
    larson_miller_parameter_value, larson_miller_rupture_time, larson_miller_temperature_for_life,
};
pub use lmtd::{lmtd_counterflow, lmtd_heat_duty, lmtd_parallelflow};
pub use orifice_plate::{
    orifice_beta_ratio, orifice_flow_rate, orifice_velocity_of_approach_factor,
};
pub use pitot_tube::{
    pitot_dynamic_pressure, pitot_stagnation_pressure, pitot_velocity, pitot_volumetric_flow,
};
pub use pump_affinity::{
    pump_affinity_flow, pump_affinity_head, pump_affinity_impeller_flow, pump_affinity_power,
};
pub use standard_atmosphere::{
    ISA_GAS_CONSTANT, ISA_GRAVITY, ISA_LAPSE_RATE, ISA_MOLAR_MASS_AIR, ISA_SEA_LEVEL_PRESSURE,
    ISA_SEA_LEVEL_TEMPERATURE, ISA_TROPOPAUSE_ALTITUDE, isa_density, isa_pressure, isa_temperature,
};
pub use stokes_settling::{stokes_drag_force, stokes_reynolds_number, stokes_terminal_velocity};
pub use thermal_expansion::{
    thermal_constrained_stress, thermal_free_strain, thermal_linear_expansion,
    thermal_volumetric_expansion,
};
pub use torsion_bar::{
    torsion_bar_angle, torsion_bar_polar_inertia, torsion_bar_rate, torsion_bar_shear_stress,
};
pub use valve_flow_coefficient::{
    VALVE_KV_TO_CV_FACTOR, valve_cv_from_kv, valve_flow_rate_from_kv, valve_kv_from_cv,
    valve_kv_required, valve_pressure_drop_bar,
};
pub use venturi_meter::{venturi_beta_ratio, venturi_flow_rate, venturi_pressure_drop};
pub use weir_flow::{weir_head_from_rectangular_flow, weir_rectangular_flow, weir_vnotch_flow};

// Lot massif (vol. 33) — ré-exports à plat.
pub use beam_deflection::{
    beam_cantilever_end_load, beam_cantilever_udl, beam_simply_supported_center_load,
    beam_simply_supported_udl,
};
pub use bend_allowance::{
    bend_allowance_length, bend_deduction_length, bend_flat_length, bend_neutral_radius,
    bend_outside_setback,
};
pub use bolt_torque::{
    bolt_clamp_force_after_relaxation, bolt_preload_from_torque, bolt_preload_from_yield,
    bolt_tightening_torque,
};
pub use crank_effort::{
    crank_obliquity_ratio, crank_piston_force_to_torque, crank_turning_moment_gas,
};
pub use elastic_constants::{
    elastic_bulk_modulus_from_e_nu, elastic_lame_lambda, elastic_poisson_from_e_g,
    elastic_shear_modulus_from_e_nu, elastic_youngs_from_g_nu,
};
pub use faraday_corrosion::{
    FARADAY_CONSTANT, faraday_corrosion_current_from_rate, faraday_mass_loss_rate,
    faraday_penetration_rate,
};
pub use fillet_weld::{
    fillet_weld_capacity, fillet_weld_required_length, fillet_weld_shear_stress, fillet_weld_throat,
};
pub use friction_brake::{
    friction_brake_double_shoe_torque, friction_brake_heat_power, friction_brake_stopping_time,
    friction_brake_torque,
};
pub use gear_contact_ratio::{
    gear_contact_base_pitch, gear_contact_length_of_action, gear_contact_ratio,
};
pub use gear_lewis::{
    gear_lewis_allowable_force, gear_lewis_bending_stress, gear_lewis_with_velocity_factor,
};
pub use motor_starting::{
    motor_acceleration_torque, motor_dol_starting_current, motor_star_delta_starting_current,
    motor_starting_time,
};
pub use projectile_motion::{
    projectile_max_height, projectile_optimal_angle, projectile_range, projectile_time_of_flight,
};
pub use restitution::{
    restitution_coefficient, restitution_energy_loss, restitution_final_velocity_1,
    restitution_final_velocity_2,
};
pub use riser_design::{
    riser_cooling_modulus, riser_cylinder_volume_for_modulus, riser_feeding_efficiency,
    riser_modulus_criterion,
};
pub use section_modulus::{
    section_bending_stress, section_modulus_hollow_circle, section_modulus_rectangle,
    section_modulus_solid_circle,
};
pub use shaft_sizing::{
    shaft_diameter_from_torque, shaft_power_from_torque, shaft_torque_from_diameter,
    shaft_torque_from_power,
};
pub use spring_combination::{
    spring_deflection_from_rate, spring_parallel_rate, spring_series_rate, spring_series_two,
};
pub use thermal_resistance::{
    thermal_heat_rate, thermal_overall_coefficient, thermal_resistance_conduction,
    thermal_resistance_convection, thermal_resistance_parallel, thermal_resistance_series,
};
pub use turning_roughness::{turning_feed_for_ra, turning_theoretical_ra, turning_theoretical_rz};
pub use view_factor::{
    view_factor_infinite_parallel_plates, view_factor_net_exchange, view_factor_reciprocity,
    view_factor_summation_last,
};

// Lot massif (vol. 34) — ré-exports à plat.
pub use bolt_group_shear::{
    boltgroup_direct_shear, boltgroup_polar_inertia, boltgroup_resultant_shear,
    boltgroup_torsional_shear,
};
pub use bucket_elevator::{
    bucket_capacity, bucket_lifting_power, bucket_motor_power, bucket_spacing_from_pitch,
};
pub use centrifugal_clutch::{
    centrifugal_clutch_engagement_speed, centrifugal_clutch_net_force,
    centrifugal_clutch_shoe_force, centrifugal_clutch_torque,
};
pub use choked_flow::{
    choked_critical_pressure_ratio, choked_critical_temperature_ratio, choked_is_choked,
    choked_mass_flow as choked_nozzle_mass_flow,
};
pub use disc_clutch::{
    disc_clutch_axial_force_uniform_wear, disc_clutch_mean_radius,
    disc_clutch_transmissible_torque_uniform_pressure,
    disc_clutch_transmissible_torque_uniform_wear,
};
pub use duct_sizing::{
    duct_friction_loss, duct_hydraulic_diameter, duct_rectangular_equivalent_diameter,
    duct_velocity, duct_velocity_pressure,
};
pub use flat_belt::{
    flatbelt_centrifugal_tension, flatbelt_power, flatbelt_slack_from_ratio, flatbelt_tension_ratio,
};
pub use forging_force::{forging_closed_die_force, forging_flow_stress, forging_open_die_force};
pub use gear_backlash::{
    gear_backlash_angular, gear_backlash_center_distance_for_backlash, gear_backlash_circular,
    gear_backlash_from_tooth_thinning,
};
pub use honing::{honing_crosshatch_angle, honing_peripheral_speed, honing_stock_removal_time};
pub use hydraulic_press::{
    hydraulic_press_force_from_pressure, hydraulic_press_intensifier_pressure,
    hydraulic_press_output_force, hydraulic_press_output_stroke,
};
pub use lapping::{
    lapping_material_removed, lapping_pressure, lapping_removal_rate, lapping_time_for_stock,
};
pub use magnetic_holding::{
    MU0_VACUUM, magnetic_flux_density_for_force, magnetic_holding_force_two_poles,
    magnetic_maxwell_pull,
};
pub use pendulum::{
    pendulum_compound_period, pendulum_equivalent_length, pendulum_radius_of_gyration,
    pendulum_simple_period,
};
pub use reaming::{
    reaming_cutting_speed, reaming_feed_rate, reaming_machining_time, reaming_spindle_speed,
};
pub use rolling_force::{
    rolling_contact_length, rolling_force, rolling_power as rolling_mill_power,
    rolling_torque_per_roll,
};
pub use rotating_unbalance::{
    unbalance_centrifugal_force, unbalance_forced_amplitude, unbalance_permissible_from_grade,
    unbalance_transmitted_force,
};
pub use taper_turning::{
    taper_compound_rest_angle, taper_half_angle, taper_ratio as taper_turning_ratio,
    taper_tailstock_offset,
};
pub use vacuum_lifting::{
    vacuum_cup_area_from_diameter, vacuum_effective_lift, vacuum_lift_force, vacuum_required_area,
};
pub use weld_group::{
    weldgroup_direct_shear_stress, weldgroup_polar_modulus, weldgroup_resultant_stress,
    weldgroup_torsional_shear_stress,
};

// Lot massif (vol. 35) — ré-exports à plat.
pub use buoyancy_stability::{
    buoyancy_force, buoyancy_is_stable, buoyancy_metacentric_height, buoyancy_righting_moment,
};
pub use cone_clutch::{
    cone_clutch_engagement_force, cone_clutch_face_width, cone_clutch_torque_uniform_pressure,
    cone_clutch_torque_uniform_wear,
};
pub use crane_stability::{
    crane_max_load_at_radius, crane_stability_factor, crane_stabilizing_moment,
    crane_tipping_moment,
};
pub use film_condensation::{
    condensation_film_reynolds, condensation_horizontal_tube_coefficient,
    condensation_vertical_plate_coefficient,
};
pub use gear_planetary_torque::{
    planetary_torque_carrier_reaction, planetary_torque_ratio_carrier_output,
    planetary_torque_ring_from_sun, planetary_torque_tangential_force,
};
pub use helmholtz_resonator::{
    helmholtz_cavity_volume_for_frequency, helmholtz_effective_length, helmholtz_resonant_frequency,
};
pub use hydraulic_turbine::{
    TurbineType, turbine_available_power, turbine_hydraulic_power, turbine_specific_speed,
    turbine_type_from_specific_speed, turbine_unit_speed,
};
pub use hydrostatic_force::{
    submerged_center_of_pressure, submerged_force_on_vertical_rectangle,
    submerged_pressure_at_depth, submerged_resultant_force,
};
pub use isentropic_flow::{
    isentropic_area_ratio, isentropic_density_ratio, isentropic_pressure_ratio,
    isentropic_temperature_ratio,
};
pub use kanban_sizing::{
    kanban_card_count, kanban_max_inventory, kanban_reorder_point, kanban_safety_stock,
};
pub use labyrinth_seal::{
    labyrinth_carry_over_factor, labyrinth_leakage_flow, labyrinth_pressure_drop_per_tooth,
};
pub use nucleate_boiling::{
    boiling_critical_heat_flux_zuber, boiling_excess_temperature, boiling_rohsenow_heat_flux,
};
pub use parshall_flume::{
    flume_head_from_free_flow, flume_parshall_free_flow, flume_submergence_ratio,
};
pub use plastic_bending::{
    plastic_moment, plastic_section_modulus_rectangle, plastic_shape_factor, plastic_yield_moment,
};
pub use pump_specific_speed::{
    PumpSpecificImpellerType, pump_specific_dimensional_ns_rpm, pump_specific_impeller_class,
    pump_specific_speed, pump_specific_suction_speed,
};
pub use rankine_column::{
    rankine_constant_from_euler, rankine_crippling_load, rankine_crippling_stress,
};
pub use set_screw::{
    setscrew_axial_holding_force, setscrew_holding_torque_on_shaft, setscrew_required_count,
    setscrew_seating_torque,
};
pub use siphon::{
    siphon_crest_pressure, siphon_flow_rate, siphon_max_crest_height, siphon_outlet_velocity,
};
pub use spiral_spring::{
    spiral_spring_active_length, spiral_spring_bending_stress, spiral_spring_stored_energy,
    spiral_spring_torque,
};
pub use tuned_mass_damper::{
    tmd_absorber_mass, tmd_absorber_stiffness, tmd_optimal_damping_ratio,
    tmd_optimal_frequency_ratio,
};

// Lot massif (vol. 36) — ré-exports à plat.
pub use arch_thrust::{
    arch_axial_thrust_at_crown, arch_horizontal_thrust_point_center, arch_horizontal_thrust_udl,
    arch_support_reaction_vertical_udl,
};
pub use beam_shear_flow::{
    shearflow_fastener_spacing, shearflow_max_rectangular, shearflow_shear_flow,
    shearflow_transverse_shear_stress,
};
pub use brazing_joint::{
    brazing_capillary_gap_pressure, brazing_joint_shear_strength, brazing_required_overlap,
    brazing_shear_area,
};
pub use cooling_tower::{
    coolingtower_approach, coolingtower_effectiveness, coolingtower_evaporation_loss,
    coolingtower_heat_rejected, coolingtower_range,
};
pub use cpm_schedule::{
    CPM_CRITICALITY_TOLERANCE, cpm_early_finish, cpm_free_float, cpm_is_critical, cpm_late_start,
    cpm_total_float,
};
pub use differential_gear::{
    diffgear_case_speed, diffgear_other_wheel_speed, diffgear_wheel_speed_sum,
    diffgear_wheel_torque,
};
pub use diffusion_fick::{
    fick_diffusion_coefficient_arrhenius, fick_diffusion_length, fick_first_law_flux,
    fick_penetration_concentration,
};
pub use dpmo_sigma::{dpmo_defects_per_unit, dpmo_from_defects, dpmo_yield, dpmo_yield_from_dpu};
pub use duty_cycle_torque::{
    dutycycle_average_power, dutycycle_duty_factor, dutycycle_peak_to_rms_ratio,
    dutycycle_rms_torque,
};
pub use hand_arm_vibration::{
    hav_combined_a8, hav_daily_exposure_a8, hav_partial_exposure, hav_time_to_limit,
};
pub use harmonic_drive::{
    harmonic_drive_output_speed, harmonic_drive_output_torque, harmonic_drive_reduction_ratio,
    harmonic_drive_tooth_difference,
};
pub use metal_spinning::{
    spinning_max_half_angle_for_reduction, spinning_reduction_ratio, spinning_sine_law_thickness,
    spinning_thickness_reduction,
};
pub use pert_estimate::{pert_expected_time, pert_standard_deviation, pert_variance, pert_z_score};
pub use phase_fraction::{
    lever_complementary_fraction, lever_phase_fraction, lever_tie_line_length,
};
pub use strain_gauge::{
    straingauge_full_bridge_output, straingauge_quarter_bridge_output,
    straingauge_resistance_change, straingauge_strain_from_resistance,
};
pub use strain_rosette::{
    rosette_max_shear_strain, rosette_mean_strain, rosette_mohr_radius,
    rosette_principal_angle_rectangular, rosette_principal_strains_rectangular,
};
pub use taguchi_loss::{
    taguchi_average_loss, taguchi_loss_coefficient, taguchi_loss_nominal_best,
    taguchi_loss_smaller_better,
};
pub use venturi_ejector::{
    ejector_entrainment_ratio, ejector_mixed_mass_flow, ejector_throat_velocity,
    ejector_vacuum_pressure,
};
pub use wbgt_index::{wbgt_indoor, wbgt_outdoor, wbgt_with_clothing_adjustment};
pub use wien_law::{
    WIEN_DISPLACEMENT_CONSTANT, WIEN_FREQUENCY_CONSTANT, wien_peak_frequency, wien_peak_wavelength,
    wien_temperature_from_peak,
};

// Lot massif (vol. 37) — ré-exports à plat.
pub use bearing_friction::{
    bearingfric_load_torque, bearingfric_power_loss, bearingfric_total_torque,
    bearingfric_viscous_torque,
};
pub use boiler_efficiency::{
    boiler_blowdown_rate, boiler_dry_flue_gas_loss, boiler_efficiency_direct,
    boiler_efficiency_indirect,
};
pub use critical_insulation_radius::{
    insulation_critical_radius_cylinder, insulation_critical_radius_sphere,
    insulation_cylinder_resistance, insulation_is_below_critical,
};
pub use degree_days::{
    degreeday_annual_from_daily, degreeday_cooling, degreeday_energy_demand, degreeday_heating,
};
pub use ecm_machining::{
    ECM_FARADAY_CONSTANT, ecm_current_from_gap, ecm_equilibrium_gap, ecm_material_removal_rate,
    ecm_penetration_rate,
};
pub use flange_bolt::{
    flangebolt_gasket_load_operating, flangebolt_hydrostatic_end_force,
    flangebolt_operating_bolt_load, flangebolt_required_bolt_area, flangebolt_seating_bolt_load,
};
pub use forecasting::{
    forecast_exponential_smoothing, forecast_mean_absolute_deviation, forecast_moving_average,
    forecast_tracking_signal,
};
pub use gas_pipeline::{
    gaspipe_pressure_drop_squared, gaspipe_weymouth_flow, gaspipe_weymouth_transmission_factor,
};
pub use gear_dynamic_load::{
    geardyn_buckingham_increment, geardyn_dynamic_load, geardyn_service_factor_load,
    geardyn_velocity_factor_barth,
};
pub use gear_scuffing::{
    gearscuff_contact_temperature, gearscuff_flash_temperature, gearscuff_safety_factor,
};
pub use gear_wear_load::{
    gearwear_limiting_load, gearwear_load_stress_factor, gearwear_ratio_factor,
};
pub use heat_pump_cop::{
    heatpump_carnot_cop_heating, heatpump_cop_cooling, heatpump_cop_heating, heatpump_cop_relation,
    heatpump_second_law_efficiency,
};
pub use optical_flat::{
    opticalflat_flatness_error_from_curvature, opticalflat_fringes_from_deviation,
    opticalflat_gap_at_fringe, opticalflat_surface_deviation,
};
pub use pv_limit::{pv_bearing_pressure, pv_factor, pv_max_speed_for_load, pv_sliding_velocity};
pub use reverberation_time::{
    reverb_eyring_time, reverb_mean_absorption, reverb_sabine_time, reverb_total_absorption,
};
pub use sound_transmission_loss::{
    stl_composite_transmission, stl_mass_law, stl_mass_law_field,
    stl_transmission_coefficient_from_loss,
};
pub use steam_quality::{
    steam_mixture_enthalpy, steam_property_from_quality, steam_quality_from_enthalpy,
    steam_wetness_fraction,
};
pub use surge_tank::{
    surgetank_max_surge_amplitude, surgetank_oscillation_period, surgetank_thoma_area,
};
pub use thermocouple::{
    thermocouple_cold_junction_correction, thermocouple_emf_linear, thermocouple_sensitivity,
    thermocouple_temperature_from_emf,
};
pub use torque_angle::{
    torqueangle_axial_advance, torqueangle_combined_stiffness, torqueangle_preload_from_angle,
};

// Lot massif (vol. 38) — ré-exports à plat.
pub use air_mixing::{
    airmix_enthalpy, airmix_humidity_ratio, airmix_mass_flow_out, airmix_temperature,
};
pub use coffin_manson::{
    coffin_elastic_strain_amplitude, coffin_plastic_strain_amplitude,
    coffin_total_strain_amplitude, coffin_transition_reversals,
};
pub use comminution::{
    comminution_bond_work, comminution_kick_energy, comminution_reduction_ratio,
    comminution_rittinger_energy,
};
pub use coulomb_damping::{
    coulomb_amplitude_after_cycles, coulomb_amplitude_loss_per_cycle, coulomb_cycles_to_stop,
    coulomb_dead_band,
};
pub use cyclone_separator::{
    cyclone_collection_efficiency, cyclone_cut_diameter, cyclone_number_of_turns,
    cyclone_pressure_drop,
};
pub use doppler::{
    doppler_observed_moving_observer, doppler_observed_moving_source, doppler_shift_reflected,
    doppler_velocity_from_shift,
};
pub use filtration::{
    filtration_cake_resistance, filtration_darcy_flux, filtration_time_constant_pressure,
    filtration_volume_from_time,
};
pub use fmea_rpn::{
    fmea_criticality, fmea_exceeds_threshold, fmea_normalized_rpn, fmea_risk_priority_number,
};
pub use heat_exchanger_fouling::{
    fouling_area_oversize_factor, fouling_cleanliness_factor, fouling_fouled_overall_coefficient,
    fouling_resistance_from_coefficients,
};
pub use lighting_lumen::{
    lighting_average_illuminance, lighting_flux_required, lighting_luminaires_required,
    lighting_room_index,
};
pub use mixing_power::{
    mixing_power_laminar, mixing_power_turbulent, mixing_reynolds, mixing_tip_speed,
};
pub use pneumatic_conveying::{
    pneuconvey_acceleration_pressure_drop, pneuconvey_gas_mass_flow, pneuconvey_saltation_froude,
    pneuconvey_solids_loading_ratio,
};
pub use radiation_network::{
    radnet_space_resistance, radnet_surface_resistance, radnet_two_gray_parallel_plates,
    radnet_two_gray_surface_exchange,
};
pub use radiation_shield::{
    radshield_flux_one_shield, radshield_flux_with_shields,
    radshield_reduction_factor_equal_emissivity, radshield_two_plate_flux,
};
pub use ramberg_osgood::{
    rambergosgood_elastic_strain, rambergosgood_plastic_strain, rambergosgood_secant_modulus,
    rambergosgood_total_strain,
};
pub use silo_pressure::{
    silo_asymptotic_pressure, silo_hydraulic_radius, silo_janssen_vertical_pressure,
    silo_janssen_wall_pressure,
};
pub use sluice_gate::{
    sluice_contracted_depth, sluice_discharge, sluice_discharge_coefficient, sluice_force_on_gate,
};
pub use whole_body_vibration::{
    wbv_daily_exposure_a8, wbv_dominant_axis_acceleration, wbv_time_to_limit, wbv_vector_sum,
};
