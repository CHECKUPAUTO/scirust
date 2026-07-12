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
//! - [`thermal`] — thermique : dilatation, conduction (**Fourier**), convection,
//!   chaleur sensible et contrainte thermique.
//! - [`tolerancing`] — systèmes de tolérancement de dessin : tolérances
//!   générales **ISO 2768** (parties 1 et 2) et catalogue des normes **GPS**.
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

pub mod balancing;
pub mod beams;
pub mod bearings;
pub mod belts;
pub mod bernoulli;
pub mod bolted_joints;
pub mod buckling;
pub mod cams;
pub mod critical_speed;
pub mod dynamics;
pub mod economics;
pub mod epicyclic;
pub mod fastener_groups;
pub mod flywheel;
pub mod forced_vibrations;
pub mod forces;
pub mod fourbar;
pub mod friction;
pub mod gears;
pub mod geneva;
pub mod heat_exchanger;
pub mod hertz;
pub mod hyperstatism;
pub mod impact;
pub mod interference_fit;
pub mod iso6336;
pub mod keys;
pub mod kinematics;
pub mod liaisons;
pub mod mohr;
pub mod pipe_flow;
pub mod power_screws;
pub mod pumps;
pub mod riveted_joints;
pub mod roughness;
pub mod shafts;
pub mod slider_crank;
pub mod springs;
pub mod stress_concentration;
pub mod thermal;
pub mod thermo_cycles;
pub mod threads;
pub mod time;
pub mod tolerancing;
pub mod toollife;
pub mod torseurs;
pub mod torsion_profiles;
pub mod trusses;
pub mod universal_joint;
pub mod vibrations;
pub mod welds;

pub use balancing::{
    centrifugal_force, correction_mass, permissible_eccentricity_um, permissible_unbalance_g_mm,
    unbalance,
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
pub use belts::{
    belt_speed_m_s, slack_tension, tension_ratio_flat, tension_ratio_vbelt, transmissible_power_w,
    wrap_angle_small_pulley_rad,
};
pub use bernoulli::{
    continuity_velocity, dynamic_pressure, hydrostatic_pressure, reynolds_number,
    torricelli_velocity, total_head,
};
pub use bolted_joints::{
    bolt_working_load, load_factor, preload_from_torque, residual_clamp_load, separation_load,
};
pub use buckling::{
    EndCondition, critical_load, critical_stress, effective_length, is_euler_valid,
    limiting_slenderness, radius_of_gyration, slenderness_ratio,
};
pub use cams::{
    cycloidal_acceleration, cycloidal_displacement, cycloidal_velocity, shm_acceleration,
    shm_displacement, shm_velocity,
};
pub use critical_speed::{
    critical_speed_from_deflection_rad, critical_speed_rad, dunkerley_critical_speed_rad,
    rad_to_rpm,
};
pub use dynamics::{
    angular_momentum, inertia_hollow_cylinder, inertia_rod_center, inertia_rod_end,
    inertia_solid_cylinder, inertia_solid_sphere, inertia_thin_ring, kinetic_energy_rotation,
    kinetic_energy_translation, parallel_axis, rotational_power, torque_from_angular_accel,
};
pub use economics::MachiningEconomics;
pub use epicyclic::{
    carrier_speed, reduction_ratio_ring_fixed, ring_speed, ring_teeth, sun_speed, willis_ratio,
};
pub use fastener_groups::{group_polar_moment, primary_shear, resultant_shear, secondary_shear};
pub use flywheel::{
    coefficient_of_fluctuation, energy_fluctuation, mean_speed, required_inertia, stored_energy,
};
pub use forced_vibrations::{
    frequency_ratio, magnification_factor, phase_lag_rad, resonance_peak_ratio,
    rotating_unbalance_response, transmissibility,
};
pub use forces::{KienzleModel, cutting_power_kw, motor_power_kw, spindle_torque_nm};
pub use fourbar::{FourBarType, classify, is_grashof};
pub use friction::{
    angle_of_repose_deg, friction_angle_deg, incline_self_locking, is_sliding, kinetic_friction,
    max_static_friction, within_adhesion_cone,
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
pub use heat_exchanger::{
    actual_heat_transfer, capacity_ratio, effectiveness_counterflow, effectiveness_parallel_flow,
    heat_duty_lmtd, lmtd, ntu,
};
pub use hertz::{
    effective_modulus, effective_radius, line_contact_half_width, line_contact_max_pressure,
    point_contact_max_pressure, point_contact_radius,
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
pub use keys::{
    key_bearing_pressure, key_shear_stress, required_length_for_bearing, tangential_force,
};
pub use kinematics::{
    cutting_speed_m_min, feed_per_rev_milling, feed_velocity_mm_min, mrr_drilling_mm3_min,
    mrr_milling_mm3_min, mrr_turning_cm3_min, spindle_speed_rpm,
};
pub use liaisons::{LIAISONS, Liaison};
pub use mohr::{
    max_in_plane_shear, mohr_radius, normal_stress_rotated, principal_angle_rad,
    principal_stresses, safety_factor, shear_stress_rotated, tresca_plane, von_mises_plane,
    von_mises_principal,
};
pub use pipe_flow::{
    colebrook_friction, darcy_head_loss, laminar_friction_factor, minor_loss, swamee_jain_friction,
};
pub use power_screws::{
    efficiency, is_self_locking, lead_angle_deg, lowering_torque_nm, raising_torque_nm,
};
pub use pumps::{
    affinity_flow, affinity_head, affinity_power, hydraulic_power, npsh_available, shaft_power,
    specific_speed,
};
pub use riveted_joints::{
    bearing_strength, joint_efficiency, rivet_shear_strength, solid_plate_strength,
    tearing_strength,
};
pub use roughness::{
    feed_for_target_ra, theoretical_ra_turning, theoretical_rt_sharp, theoretical_rt_turning,
};
pub use shafts::{
    angle_of_twist_deg, bending_stress, polar_section_modulus_hollow, polar_section_modulus_solid,
    section_modulus_hollow, section_modulus_solid, torsional_shear_stress, von_mises_solid,
};
pub use slider_crank::{
    obliquity_ratio, piston_acceleration_approx, piston_displacement, piston_velocity,
};
pub use springs::HelicalSpring;
pub use stress_concentration::{
    fatigue_stress_concentration, nominal_stress_plate_with_hole, peak_stress,
};
pub use thermal::{
    conduction_heat_flow, convection_heat_flow, linear_expansion, sensible_heat,
    thermal_resistance, thermal_stress,
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
pub use trusses::{axial_stress, member_elongation, two_member_joint};
pub use universal_joint::{
    max_velocity_ratio as cardan_max_velocity_ratio,
    min_velocity_ratio as cardan_min_velocity_ratio, output_angle,
    velocity_ratio as cardan_velocity_ratio,
};
pub use vibrations::{
    critical_damping, damped_frequency_rad, damping_ratio, log_decrement, natural_frequency_hz,
    natural_frequency_rad, quality_factor,
};
pub use welds::{
    butt_weld_stress, fillet_direct_shear_stress, fillet_throat_area, throat_thickness,
    weld_group_torsional_shear,
};
