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
pub mod fins;
pub mod first_order_response;
pub mod flow_meters;
pub mod flywheel;
pub mod forced_vibrations;
pub mod forces;
pub mod fourbar;
pub mod fracture;
pub mod friction;
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
pub mod water_hammer;
pub mod wedge;
pub mod weibull;
pub mod welds;
pub mod wire_drawing;

pub use accelerated_life::{
    BOLTZMANN_EV_K, arrhenius_acceleration_factor, derated_value, ten_degree_rule_factor,
};
pub use air_flow::{choked_mass_flow, critical_pressure_ratio, is_choked, speed_of_sound};
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
