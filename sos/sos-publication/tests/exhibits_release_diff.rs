//! Exhibit drift, release consistency, semantic diff, and content-addressed
//! sealing.

use std::collections::BTreeMap;

use sos_core::{Author, HashAlgo, ObjectId};
use sos_publication::{
    BindingRole, Claim, ClaimBinding, ColumnDef, ExhibitVerdict, FigureKey, FigureSpec, MediaType,
    Ordering, PolicyId, Publication, RegenPolicy, ReleaseManifest, RendererId, TableKey, TableSpec,
    check_release, diff, seal_publication, seal_release, verify_exhibits,
};

fn oid(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"pub-exhibit", tag)
}

fn paper_with_exhibits() -> Publication {
    let root = oid(b"root");
    let fig_pinned = oid(b"fig-artifact");
    let tbl_pinned = oid(b"tbl-artifact");
    Publication::builder("Exhibits")
        .declared_root(root)
        .claim(Claim::new(
            "C1",
            "x",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, root)],
        ))
        .figure(
            FigureSpec::new(
                "fig-fit",
                "Fit",
                "the fit",
                vec![oid(b"fig-src")],
                RendererId::new("scatter@1"),
                MediaType::Svg,
            )
            .expecting(fig_pinned),
        )
        .table(
            TableSpec::new(
                "tbl-res",
                "Residuals",
                vec![oid(b"tbl-src")],
                vec![
                    ColumnDef::new("planet"),
                    ColumnDef::with_unit("residual", "s"),
                ],
                Ordering::AsProduced,
            )
            .expecting(tbl_pinned),
        )
        .build()
}

#[test]
fn exhibits_reproduce_when_the_rerender_matches_the_pin() {
    let p = paper_with_exhibits();
    let figures: BTreeMap<FigureKey, ObjectId> =
        [(FigureKey::new("fig-fit"), oid(b"fig-artifact"))]
            .into_iter()
            .collect();
    let tables: BTreeMap<TableKey, ObjectId> = [(TableKey::new("tbl-res"), oid(b"tbl-artifact"))]
        .into_iter()
        .collect();
    let report = verify_exhibits(&p, &figures, &tables);
    assert!(report.reproduced());
    assert!(report.first_drift().is_none());
    assert!(
        report
            .exhibits
            .iter()
            .all(|e| e.verdict == ExhibitVerdict::Reproduced)
    );
}

#[test]
fn a_drifted_exhibit_is_localized() {
    let p = paper_with_exhibits();
    // The table re-rendered to a different address than pinned.
    let figures: BTreeMap<FigureKey, ObjectId> =
        [(FigureKey::new("fig-fit"), oid(b"fig-artifact"))]
            .into_iter()
            .collect();
    let tables: BTreeMap<TableKey, ObjectId> = [(TableKey::new("tbl-res"), oid(b"tbl-CHANGED"))]
        .into_iter()
        .collect();
    let report = verify_exhibits(&p, &figures, &tables);
    assert!(!report.reproduced());
    let drift = report.first_drift().unwrap();
    assert_eq!(drift.key, "tbl-res");
    assert_eq!(drift.expected, Some(oid(b"tbl-artifact")));
    assert_eq!(drift.rederived, Some(oid(b"tbl-CHANGED")));
}

#[test]
fn a_checked_exhibit_with_no_rerender_is_missing() {
    let p = paper_with_exhibits();
    let figures: BTreeMap<FigureKey, ObjectId> = BTreeMap::new();
    let tables: BTreeMap<TableKey, ObjectId> = BTreeMap::new();
    let report = verify_exhibits(&p, &figures, &tables);
    assert!(!report.reproduced());
    assert!(
        report
            .exhibits
            .iter()
            .all(|e| e.verdict == ExhibitVerdict::Missing)
    );
}

#[test]
fn a_static_asset_is_not_drift_checked() {
    let root = oid(b"root");
    let p = Publication::builder("Static")
        .declared_root(root)
        .claim(Claim::new(
            "C1",
            "x",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, root)],
        ))
        .figure(
            FigureSpec::new(
                "schematic",
                "Schematic",
                "a hand-drawn schematic",
                Vec::new(),
                RendererId::new("hand@1"),
                MediaType::Png,
            )
            .regeneration(RegenPolicy::StaticAsset),
        )
        .build();
    let report = verify_exhibits(&p, &BTreeMap::new(), &BTreeMap::new());
    // Not checked ⇒ does not count against reproduction.
    assert!(report.reproduced());
    assert_eq!(report.exhibits[0].verdict, ExhibitVerdict::NotChecked);
}

#[test]
fn sealing_is_deterministic_and_content_addressed() {
    let p = paper_with_exhibits();
    let a = seal_publication(p.clone(), Author::human("curator"));
    let b = seal_publication(p.clone(), Author::human("curator"));
    assert!(a.verify_id());
    assert_eq!(a.id, b.id);
    assert_eq!(a.kind.name, "Publication");
    // A different sealing principal changes the object id (provenance is part of
    // identity) even though the document content is the same.
    let c = seal_publication(p, Author::human("someone-else"));
    assert_ne!(a.id, c.id);
}

#[test]
fn release_consistency_detects_a_post_release_edit() {
    let p = paper_with_exhibits();
    let sealed = seal_publication(p.clone(), Author::human("curator"));
    let manifest =
        ReleaseManifest::new(sealed.id, PolicyId::standard(), "v1 release").reviewed(oid(b"root"));

    // Unchanged: consistent.
    let ok = check_release(&sealed, &manifest);
    assert!(ok.is_consistent());
    assert!(ok.matches_publication);

    // Edit the document and re-seal: the id no longer matches the manifest.
    let edited = Publication::builder("Exhibits — revised")
        .declared_root(oid(b"root"))
        .claim(Claim::new(
            "C1",
            "x",
            vec![ClaimBinding::new(
                BindingRole::DirectlySupports,
                oid(b"root"),
            )],
        ))
        .build();
    let resealed = seal_publication(edited, Author::human("curator"));
    let changed = check_release(&resealed, &manifest);
    assert!(!changed.matches_publication);
    assert!(!changed.is_consistent());
}

#[test]
fn a_release_manifest_seals_with_the_publication_as_parent() {
    let sealed = seal_publication(paper_with_exhibits(), Author::human("curator"));
    let manifest = ReleaseManifest::new(sealed.id, PolicyId::standard(), "release");
    let release = seal_release(manifest, Author::human("chair"));
    assert!(release.verify_id());
    assert_eq!(release.kind.name, "ReleaseManifest");
    assert_eq!(release.parents, vec![sealed.id]);
}

#[test]
fn the_semantic_diff_reports_added_removed_and_changed_claims() {
    let root = oid(b"root");
    let e = oid(b"e");
    let base = Publication::builder("Paper")
        .declared_root(root)
        .claim(Claim::new(
            "C1",
            "stable",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, e)],
        ))
        .claim(Claim::new(
            "C2",
            "to be removed",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, e)],
        ))
        .build();
    let revised = Publication::builder("Paper")
        .declared_root(root)
        // C1 reworded (content changes), C2 removed, C3 added.
        .claim(Claim::new(
            "C1",
            "stable but reworded",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, e)],
        ))
        .claim(Claim::new(
            "C3",
            "new",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, e)],
        ))
        .build();

    let d = diff(&base, &revised);
    assert!(!d.is_empty());
    assert_eq!(d.claims_changed, vec![sos_publication::ClaimKey::new("C1")]);
    assert_eq!(d.claims_removed, vec![sos_publication::ClaimKey::new("C2")]);
    assert_eq!(d.claims_added, vec![sos_publication::ClaimKey::new("C3")]);

    // A publication is not different from itself.
    assert!(diff(&base, &base).is_empty());
}
