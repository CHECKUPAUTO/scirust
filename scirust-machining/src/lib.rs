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

pub mod bearings;
pub mod belts;
pub mod economics;
pub mod forces;
pub mod gears;
pub mod hertz;
pub mod keys;
pub mod kinematics;
pub mod power_screws;
pub mod roughness;
pub mod shafts;
pub mod springs;
pub mod threads;
pub mod time;
pub mod tolerancing;
pub mod toollife;
pub mod torseurs;

pub use bearings::{
    BearingType, Reliability, adjusted_rating_life, basic_rating_life_hours,
    basic_rating_life_revs, equivalent_dynamic_load,
};
pub use belts::{
    belt_speed_m_s, slack_tension, tension_ratio_flat, tension_ratio_vbelt, transmissible_power_w,
    wrap_angle_small_pulley_rad,
};
pub use economics::MachiningEconomics;
pub use forces::{KienzleModel, cutting_power_kw, motor_power_kw, spindle_torque_nm};
pub use gears::{
    HelicalGear, SpurGear, center_distance, gear_ratio, lewis_bending_stress,
    minimum_teeth_no_undercut, pitch_line_velocity_m_s, tangential_force_from_power,
    tangential_force_from_torque, transverse_contact_ratio,
};
pub use hertz::{
    effective_modulus, effective_radius, line_contact_half_width, line_contact_max_pressure,
    point_contact_max_pressure, point_contact_radius,
};
pub use keys::{
    key_bearing_pressure, key_shear_stress, required_length_for_bearing, tangential_force,
};
pub use kinematics::{
    cutting_speed_m_min, feed_per_rev_milling, feed_velocity_mm_min, mrr_drilling_mm3_min,
    mrr_milling_mm3_min, mrr_turning_cm3_min, spindle_speed_rpm,
};
pub use power_screws::{
    efficiency, is_self_locking, lead_angle_deg, lowering_torque_nm, raising_torque_nm,
};
pub use roughness::{
    feed_for_target_ra, theoretical_ra_turning, theoretical_rt_sharp, theoretical_rt_turning,
};
pub use shafts::{
    angle_of_twist_deg, bending_stress, polar_section_modulus_hollow, polar_section_modulus_solid,
    section_modulus_hollow, section_modulus_solid, torsional_shear_stress, von_mises_solid,
};
pub use springs::HelicalSpring;
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
