# Third-Party Notices

Dependency license policy is enforced by `deny.toml` and `cargo deny check licenses`. The lockfile is the authoritative version/commit inventory; generated SBOMs contain the complete package list and license metadata for each release. Any dependency with an unknown or conflicting license is a release blocker pending upstream confirmation.

## Legal review record (engineering, not legal advice)

The repository uses a custom `LicenseRef-TarekZekriti-Dual` identifier and
includes PolyForm Noncommercial 1.0.0 text. It must not be described as
OSI-approved open source. Commercial redistribution or use requires separate
terms from the rightsholder.

EU software-copyright context is Directive 2009/24/EC: computer programs are
copyright-protected and reproduction, adaptation, and distribution are
restricted acts subject to the applicable licence ([EUR-Lex](https://eur-lex.europa.eu/eli/dir/2009/24/oj)). Directive (EU) 2019/790 is also relevant to
copyright in the Digital Single Market ([EUR-Lex](https://eur-lex.europa.eu/eli/dir/2019/790/oj)). These are provenance references, not a
determination of enforceability in a particular Member State.

Machine fingerprints and local logs may be personal data when linkable to a
person or device. GDPR Article 4(5) defines pseudonymisation as reversible with
separately protected additional information ([EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/art_4/par_5/)); hashing alone is not
automatically anonymisation. Operators remain responsible for lawful basis,
retention, access controls, and data-subject rights where GDPR applies.

A legal professional should review the dual-license wording, contributor
rights, patent clauses, and customer terms before commercial distribution.

The optional Forge backend is vendored at the exact reviewed commit
`5afe067b6d1223b096c39abcefb935856034ccb9`; its upstream `LICENSING.md` is
retained under `crates/forge/`.
