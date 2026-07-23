//! Rendering: determinism, HTML escaping, JSON round-trip, typed-block
//! resolution, and clean errors for unsupported formats.

use sos_core::{Author, HashAlgo, ObjectId};
use sos_publication::{
    BindingRole, Block, Claim, ClaimBinding, ExternalCitation, FigureSpec, Format, MediaType,
    Publication, PublicationError, Reference, RendererId, Section, SectionKind, render,
};

fn oid(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"pub-render", tag)
}

fn paper() -> Publication {
    let evidence = oid(b"evidence");
    let fig_src = oid(b"fig-src");
    Publication::builder("Kepler, Rederived")
        .subtitle("A machine-checkable replication")
        .author(Author::human("ada"))
        .summary("We rederive the third law.")
        .declared_root(evidence)
        .section(
            Section::new("results", SectionKind::Results, "Results")
                .prose("The fit is tight.")
                .block(Block::Claim(sos_publication::ClaimKey::new("C1")))
                .block(Block::Figure(sos_publication::FigureKey::new("fig-fit")))
                .block(Block::Cite(sos_publication::RefKey::new("kepler"))),
        )
        .claim(Claim::new(
            "C1",
            "Period squared scales with axis cubed.",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, evidence)],
        ))
        .figure(FigureSpec::new(
            "fig-fit",
            "Period vs axis",
            "log-log scatter of period against semi-major axis",
            vec![fig_src],
            RendererId::new("scatter@1"),
            MediaType::Svg,
        ))
        .reference(Reference::internal("kepler", evidence))
        .build()
}

#[test]
fn markdown_is_deterministic_and_binds_claims_to_objects() {
    let p = paper();
    let a = render(&p, Format::Markdown).unwrap();
    let b = render(&p, Format::Markdown).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.format, Format::Markdown);

    let md = &a.content;
    assert!(md.contains("# Kepler, Rederived"));
    assert!(md.contains("A machine-checkable replication"));
    assert!(md.contains("human:ada"));
    // The claim registry surfaces which object supports the claim.
    assert!(md.contains("Period squared scales with axis cubed."));
    assert!(md.contains(&oid(b"evidence").to_string()));
    assert!(md.contains("directly-supports"));
    // The figure surfaces its source.
    assert!(md.contains(&oid(b"fig-src").to_string()));
    // Provenance footer.
    assert!(md.contains("standard@v1"));
}

#[test]
fn html_escapes_every_special_character() {
    let evidence = oid(b"e");
    let p = Publication::builder("A <b>bold</b> & \"quoted\" title")
        .declared_root(evidence)
        .claim(Claim::new(
            "C1",
            "x < y && z > w",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, evidence)],
        ))
        .build();
    let html = render(&p, Format::Html).unwrap();
    assert_eq!(html.format, Format::Html);
    assert!(
        html.content
            .contains("A &lt;b&gt;bold&lt;/b&gt; &amp; &quot;quoted&quot; title")
    );
    assert!(html.content.contains("x &lt; y &amp;&amp; z &gt; w"));
    // The raw markup never leaks.
    assert!(!html.content.contains("<b>bold</b>"));
}

#[test]
fn json_is_deterministic_and_round_trips() {
    let p = paper();
    let a = render(&p, Format::Json).unwrap();
    let b = render(&p, Format::Json).unwrap();
    assert_eq!(a, b);
    let back: Publication = serde_json::from_str(&a.content).unwrap();
    assert_eq!(back, p);
}

#[test]
fn an_external_citation_is_rendered_as_unverified() {
    let evidence = oid(b"e");
    let p = Publication::builder("With external lit")
        .declared_root(evidence)
        .claim(Claim::new(
            "C1",
            "x",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, evidence)],
        ))
        .reference(Reference::external(
            "newton1687",
            ExternalCitation::new("Principia", vec!["Newton".to_owned()])
                .year(1687)
                .identifier("urn:principia"),
        ))
        .build();
    let md = render(&p, Format::Markdown).unwrap();
    assert!(md.content.contains("external, unverified"));
    assert!(md.content.contains("Principia"));
    assert!(md.content.contains("1687"));
}

#[test]
fn a_dangling_block_reference_renders_a_marker_not_a_panic() {
    let evidence = oid(b"e");
    let p = Publication::builder("Dangling")
        .declared_root(evidence)
        .section(
            Section::new("s", SectionKind::Results, "Results").block(Block::Claim(
                sos_publication::ClaimKey::new("does-not-exist"),
            )),
        )
        .claim(Claim::new(
            "C1",
            "x",
            vec![ClaimBinding::new(BindingRole::DirectlySupports, evidence)],
        ))
        .build();
    let md = render(&p, Format::Markdown).unwrap();
    assert!(md.content.contains("[unresolved claim]"));
}

#[test]
fn latex_and_pdf_are_unsupported_but_error_cleanly() {
    let p = paper();
    assert!(matches!(
        render(&p, Format::Latex),
        Err(PublicationError::UnsupportedFormat(Format::Latex))
    ));
    assert!(matches!(
        render(&p, Format::Pdf),
        Err(PublicationError::UnsupportedFormat(Format::Pdf))
    ));
}
