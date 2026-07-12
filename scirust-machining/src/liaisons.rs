//! Liaisons mécaniques normalisées (NF EN ISO 3952 / cinématique du solide) —
//! les **11 liaisons usuelles** entre deux solides, avec leurs degrés de liberté
//! (mobilité) et le nombre d'inconnues de leur torseur d'action (statique).
//!
//! Chaque liaison partage les 6 degrés de liberté de l'espace (3 translations
//! `Tx, Ty, Tz` + 3 rotations `Rx, Ry, Rz`, dans le repère local canonique de la
//! liaison) entre :
//!
//! - la **mobilité** `m` : degrés de liberté laissés libres (composantes non
//!   nulles du **torseur cinématique**) ;
//! - les **inconnues statiques** : composantes transmissibles du **torseur
//!   d'action**.
//!
//! Ces deux ensembles sont **complémentaires** : `mobilité + inconnues = 6`.
//! C'est la dualité statique/cinématique qui relie ce module à [`crate::torseurs`] :
//! le torseur d'action et le torseur cinématique d'une liaison parfaite ont des
//! composantes non nulles sur des directions complémentaires, et leur comoment
//! (puissance des inter-efforts) est **nul**.
//!
//! **Convention** : repère local canonique — l'axe principal d'une liaison de
//! révolution/translation est `x` ; la normale d'un appui/contact est `z` (ou
//! `x` pour l'appui plan), conformément aux conventions d'enseignement usuelles.
//!
//! **Limite honnête** : ce module donne la **structure** (DDL, mobilité,
//! inconnues) des liaisons parfaites. Le cas **hélicoïdal** est particulier :
//! translation et rotation axiales y sont *couplées* par le pas, d'où une
//! mobilité de 1 bien que deux DDL géométriques soient géométriquement permis —
//! c'est pris en compte dans [`Liaison::mobility`].

/// Une des 11 liaisons mécaniques normalisées.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Liaison {
    /// Encastrement (complète) — 0 DDL.
    Encastrement,
    /// Pivot d'axe `x` — 1 DDL (Rx).
    Pivot,
    /// Glissière de direction `x` — 1 DDL (Tx).
    Glissiere,
    /// Pivot glissant d'axe `x` (cylindrique) — 2 DDL (Tx, Rx).
    PivotGlissant,
    /// Hélicoïdale d'axe `x` — 1 DDL (Tx et Rx couplés par le pas).
    Helicoidale,
    /// Sphérique / rotule de centre — 3 DDL (Rx, Ry, Rz).
    Rotule,
    /// Sphérique à doigt (rotule à doigt), doigt d'axe `z` — 2 DDL (Rx, Ry).
    RotuleADoigt,
    /// Appui plan de normale `x` — 3 DDL (Ty, Tz, Rx).
    AppuiPlan,
    /// Linéaire annulaire (sphère-cylindre) d'axe `x` — 4 DDL (Tx, Rx, Ry, Rz).
    LineaireAnnulaire,
    /// Linéaire rectiligne (contact ligne selon `x`, normale `z`) — 4 DDL
    /// (Tx, Ty, Rx, Rz).
    LineaireRectiligne,
    /// Ponctuelle (sphère-plan) de normale `z` — 5 DDL (tout sauf Tz).
    Ponctuelle,
}

impl Liaison {
    /// Nom normalisé de la liaison.
    pub fn name(self) -> &'static str {
        match self
        {
            Liaison::Encastrement => "encastrement",
            Liaison::Pivot => "pivot",
            Liaison::Glissiere => "glissière",
            Liaison::PivotGlissant => "pivot glissant",
            Liaison::Helicoidale => "hélicoïdale",
            Liaison::Rotule => "sphérique (rotule)",
            Liaison::RotuleADoigt => "sphérique à doigt",
            Liaison::AppuiPlan => "appui plan",
            Liaison::LineaireAnnulaire => "linéaire annulaire",
            Liaison::LineaireRectiligne => "linéaire rectiligne",
            Liaison::Ponctuelle => "ponctuelle",
        }
    }

    /// Degrés de liberté **géométriquement** libres, dans l'ordre
    /// `[Tx, Ty, Tz, Rx, Ry, Rz]` (repère local canonique).
    ///
    /// Pour la liaison hélicoïdale, `Tx` et `Rx` apparaissent tous deux `true`
    /// (permis géométriquement) bien qu'ils soient couplés : la **mobilité**
    /// réelle vaut 1 — voir [`Liaison::mobility`].
    pub fn kinematic_dof(self) -> [bool; 6] {
        // Ordre : Tx, Ty, Tz, Rx, Ry, Rz.
        match self
        {
            Liaison::Encastrement => [false, false, false, false, false, false],
            Liaison::Pivot => [false, false, false, true, false, false],
            Liaison::Glissiere => [true, false, false, false, false, false],
            Liaison::PivotGlissant => [true, false, false, true, false, false],
            Liaison::Helicoidale => [true, false, false, true, false, false],
            Liaison::Rotule => [false, false, false, true, true, true],
            Liaison::RotuleADoigt => [false, false, false, true, true, false],
            Liaison::AppuiPlan => [false, true, true, true, false, false],
            Liaison::LineaireAnnulaire => [true, false, false, true, true, true],
            Liaison::LineaireRectiligne => [true, true, false, true, false, true],
            Liaison::Ponctuelle => [true, true, false, true, true, true],
        }
    }

    /// Mobilité `m` : nombre de degrés de liberté **indépendants**.
    ///
    /// Égale au nombre de DDL géométriques, sauf pour l'hélicoïdale où le
    /// couplage pas/rotation ramène la mobilité à 1.
    pub fn mobility(self) -> u8 {
        match self
        {
            Liaison::Helicoidale => 1,
            other => other.kinematic_dof().iter().filter(|&&b| b).count() as u8,
        }
    }

    /// Nombre d'inconnues du torseur d'action (statique) : `6 − mobilité`.
    pub fn static_unknowns(self) -> u8 {
        6 - self.mobility()
    }
}

/// Catalogue des 11 liaisons normalisées, dans un ordre de mobilité croissante.
pub const LIAISONS: [Liaison; 11] = [
    Liaison::Encastrement,
    Liaison::Pivot,
    Liaison::Glissiere,
    Liaison::Helicoidale,
    Liaison::PivotGlissant,
    Liaison::RotuleADoigt,
    Liaison::Rotule,
    Liaison::AppuiPlan,
    Liaison::LineaireAnnulaire,
    Liaison::LineaireRectiligne,
    Liaison::Ponctuelle,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_mobilities_are_correct() {
        assert_eq!(Liaison::Encastrement.mobility(), 0);
        assert_eq!(Liaison::Pivot.mobility(), 1);
        assert_eq!(Liaison::Glissiere.mobility(), 1);
        assert_eq!(Liaison::Helicoidale.mobility(), 1); // couplée
        assert_eq!(Liaison::PivotGlissant.mobility(), 2);
        assert_eq!(Liaison::RotuleADoigt.mobility(), 2);
        assert_eq!(Liaison::Rotule.mobility(), 3);
        assert_eq!(Liaison::AppuiPlan.mobility(), 3);
        assert_eq!(Liaison::LineaireAnnulaire.mobility(), 4);
        assert_eq!(Liaison::LineaireRectiligne.mobility(), 4);
        assert_eq!(Liaison::Ponctuelle.mobility(), 5);
    }

    #[test]
    fn mobility_and_static_unknowns_sum_to_six() {
        for l in LIAISONS
        {
            assert_eq!(l.mobility() + l.static_unknowns(), 6, "{}", l.name());
        }
    }

    #[test]
    fn pivot_frees_only_its_axial_rotation() {
        // Pivot d'axe x : seul Rx libre.
        assert_eq!(
            Liaison::Pivot.kinematic_dof(),
            [false, false, false, true, false, false]
        );
    }

    #[test]
    fn rotule_frees_the_three_rotations() {
        let dof = Liaison::Rotule.kinematic_dof();
        // aucune translation, trois rotations.
        assert_eq!(&dof[0..3], &[false, false, false]);
        assert_eq!(&dof[3..6], &[true, true, true]);
    }

    #[test]
    fn catalogue_is_complete_and_unique() {
        assert_eq!(LIAISONS.len(), 11);
        for (i, a) in LIAISONS.iter().enumerate()
        {
            for b in &LIAISONS[i + 1..]
            {
                assert_ne!(a, b, "doublon dans le catalogue");
            }
            assert!(!a.name().is_empty());
        }
    }

    #[test]
    fn helical_geometric_dof_count_exceeds_its_mobility() {
        // 2 DDL géométriques (Tx, Rx) mais mobilité 1 (couplage).
        let geom = Liaison::Helicoidale
            .kinematic_dof()
            .iter()
            .filter(|&&b| b)
            .count();
        assert_eq!(geom, 2);
        assert_eq!(Liaison::Helicoidale.mobility(), 1);
    }
}
