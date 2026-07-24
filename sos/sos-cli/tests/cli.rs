//! End-to-end tests: populate a real `FileStore` with real engine objects
//! (exactly as other tooling — a library caller, another `sos-*` engine —
//! would), then exercise every `sos` command against it.

use std::fs;
use std::path::PathBuf;

use sos_core::{Author, DeterminismLevel, HashAlgo, Object, ObjectId};
use sos_knowledge::{Relation, seal_edge};
use sos_reasoning::{Derivation, DerivationStep, Soundness};
use sos_registry::{Capability, PluginDescriptor, Role};
use sos_store::{FileStore, ObjectStore, TypedStore};

use sos_cli::args::Args;

fn temp_root(name: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!("sos-cli-test-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn seal_derivation(goal: &str, premises: Vec<ObjectId>) -> Object<Derivation> {
    Object::builder(Derivation::new(
        goal,
        vec![DerivationStep::new("step", Vec::new(), goal)],
        premises,
        Soundness::Proof,
    ))
    .author(Author::human("t"))
    .seal()
}

fn oid(tag: &[u8]) -> ObjectId {
    ObjectId::compute(HashAlgo::default(), b"cli-test", tag)
}

#[test]
fn init_creates_a_store() {
    let root = temp_root("init");
    let msg = sos_cli::init::run(Some(root.to_str().unwrap())).unwrap();
    assert!(msg.contains("Initialized"));
    assert!(root.join("objects").is_dir());
    fs::remove_dir_all(&root).ok();
}

#[test]
fn log_lists_every_object_with_kind_and_level() {
    let root = temp_root("log");
    {
        let mut s = FileStore::open(&root).unwrap();
        let base = seal_derivation("base fact", Vec::new());
        s.put_object(&base).unwrap();
        let derived = Object::builder(Derivation::new(
            "derived fact",
            vec![DerivationStep::new("step", vec![base.id], "derived fact")],
            vec![base.id],
            Soundness::Proof,
        ))
        .author(Author::human("t"))
        .parents(vec![base.id])
        .level(DeterminismLevel::L3)
        .seal();
        s.put_object(&derived).unwrap();
    }

    let out = sos_cli::log::run(Some(root.to_str().unwrap())).unwrap();
    assert!(out.contains("Derivation"));
    assert!(out.contains("L3"));
    assert_eq!(out.lines().count(), 2);
    fs::remove_dir_all(&root).ok();
}

#[test]
fn clone_and_push_copy_objects_blobs_and_refs() {
    let src_root = temp_root("clone-src");
    let dest_root = temp_root("clone-dest");
    let id;
    {
        let mut src = FileStore::open(&src_root).unwrap();
        let obj = seal_derivation("shared fact", Vec::new());
        id = obj.id;
        src.put_object(&obj).unwrap();
        src.put_blob(b"payload");
        src.set_ref("head", id);
    }

    let msg = sos_cli::clone::run(src_root.to_str().unwrap(), dest_root.to_str().unwrap()).unwrap();
    assert!(msg.contains("1 object"));
    assert!(msg.contains("1 blob"));
    assert!(msg.contains("1 ref"));

    let dest = FileStore::open(&dest_root).unwrap();
    assert!(dest.has(id));
    assert!(dest.has_blob(sos_store::BlobRef::of(b"payload")));
    assert_eq!(dest.get_ref("head"), Some(id));

    // A second clone (== push, same operation) is idempotent: nothing new copied.
    let msg2 =
        sos_cli::clone::run(src_root.to_str().unwrap(), dest_root.to_str().unwrap()).unwrap();
    assert!(msg2.contains("0 object"));
    fs::remove_dir_all(&src_root).ok();
    fs::remove_dir_all(&dest_root).ok();
}

#[test]
fn why_lists_the_full_ancestor_chain() {
    let root = temp_root("why");
    let (root_obj, child, grandchild);
    {
        let mut s = FileStore::open(&root).unwrap();
        root_obj = seal_derivation("axiom", Vec::new());
        s.put_object(&root_obj).unwrap();
        child = Object::builder(Derivation::new(
            "lemma",
            Vec::new(),
            vec![root_obj.id],
            Soundness::Proof,
        ))
        .author(Author::human("t"))
        .parents(vec![root_obj.id])
        .seal();
        s.put_object(&child).unwrap();
        grandchild = Object::builder(Derivation::new(
            "theorem",
            Vec::new(),
            vec![child.id],
            Soundness::Proof,
        ))
        .author(Author::human("t"))
        .parents(vec![child.id])
        .seal();
        s.put_object(&grandchild).unwrap();
    }

    let out = sos_cli::why::run(Some(root.to_str().unwrap()), grandchild.id).unwrap();
    assert!(out.contains(&child.id.to_string()));
    assert!(out.contains(&root_obj.id.to_string()));

    let root_out = sos_cli::why::run(Some(root.to_str().unwrap()), root_obj.id).unwrap();
    assert!(root_out.contains("no recorded ancestors"));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn verify_recomputes_the_content_hash_for_a_known_kind() {
    let root = temp_root("verify");
    let obj = seal_derivation("checkable", Vec::new());
    {
        let mut s = FileStore::open(&root).unwrap();
        s.put_object(&obj).unwrap();
    }

    let out = sos_cli::verify::run(Some(root.to_str().unwrap()), obj.id).unwrap();
    assert!(out.contains("Derivation"));
    assert!(out.contains("OK (recomputed address matches)"));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn verify_reports_unrecognized_kinds_honestly() {
    use serde::{Deserialize, Serialize};
    use sos_core::Body;
    use sos_core::canonical::{Canonical, CanonicalEncoder};

    #[derive(Clone, Serialize, Deserialize)]
    struct Custom {
        n: u64,
    }
    impl Canonical for Custom {
        fn encode(&self, e: &mut CanonicalEncoder) {
            e.u64(self.n);
        }
    }
    impl Body for Custom {
        const KIND: &'static str = "CustomThing";
        const SCHEMA_VERSION: u32 = 1;
    }

    let root = temp_root("verify-unknown");
    let obj = Object::builder(Custom { n: 7 })
        .author(Author::human("t"))
        .seal();
    {
        let mut s = FileStore::open(&root).unwrap();
        s.put_object(&obj).unwrap();
    }

    let out = sos_cli::verify::run(Some(root.to_str().unwrap()), obj.id).unwrap();
    assert!(out.contains("CustomThing"));
    assert!(out.contains("not checked (unrecognized kind"));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn diff_reports_ancestor_sets_unique_to_each_root() {
    let root = temp_root("diff");
    let (shared, only_a, only_b, root_a, root_b);
    {
        let mut s = FileStore::open(&root).unwrap();
        shared = seal_derivation("shared ancestor", Vec::new());
        s.put_object(&shared).unwrap();
        only_a = Object::builder(Derivation::new(
            "a-specific",
            Vec::new(),
            vec![shared.id],
            Soundness::Proof,
        ))
        .author(Author::human("t"))
        .parents(vec![shared.id])
        .seal();
        s.put_object(&only_a).unwrap();
        only_b = Object::builder(Derivation::new(
            "b-specific",
            Vec::new(),
            vec![shared.id],
            Soundness::Proof,
        ))
        .author(Author::human("t"))
        .parents(vec![shared.id])
        .seal();
        s.put_object(&only_b).unwrap();
        root_a = Object::builder(Derivation::new(
            "root-a",
            Vec::new(),
            vec![only_a.id],
            Soundness::Proof,
        ))
        .author(Author::human("t"))
        .parents(vec![only_a.id])
        .seal();
        s.put_object(&root_a).unwrap();
        root_b = Object::builder(Derivation::new(
            "root-b",
            Vec::new(),
            vec![only_b.id],
            Soundness::Proof,
        ))
        .author(Author::human("t"))
        .parents(vec![only_b.id])
        .seal();
        s.put_object(&root_b).unwrap();
    }

    let out = sos_cli::diff::run(Some(root.to_str().unwrap()), root_a.id, root_b.id).unwrap();
    assert!(out.contains(&only_a.id.to_string()));
    assert!(out.contains(&only_b.id.to_string()));
    // Only `shared` itself is common to both ancestor closures.
    assert!(out.contains("Shared: 1 object(s)"));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn know_queries_the_edge_graph() {
    let root = temp_root("know");
    let (a, b);
    {
        let mut s = FileStore::open(&root).unwrap();
        a = seal_derivation("phlogiston", Vec::new());
        s.put_object(&a).unwrap();
        b = seal_derivation("oxygen theory", Vec::new());
        s.put_object(&b).unwrap();
        let edge = seal_edge(a.id, b.id, Relation::Contradicts, Author::engine("test"));
        s.put_object(&edge).unwrap();
    }

    let args = Args::parse(&[
        "neighbors".to_owned(),
        a.id.to_string(),
        "contradicts".to_owned(),
    ])
    .unwrap();
    let out = sos_cli::know::run(Some(root.to_str().unwrap()), &args).unwrap();
    assert_eq!(out, b.id.to_string());

    let args = Args::parse(&["related".to_owned(), a.id.to_string(), b.id.to_string()]).unwrap();
    let out = sos_cli::know::run(Some(root.to_str().unwrap()), &args).unwrap();
    assert_eq!(out, "contradicts");

    let args = Args::parse(&["path".to_owned(), a.id.to_string(), b.id.to_string()]).unwrap();
    let out = sos_cli::know::run(Some(root.to_str().unwrap()), &args).unwrap();
    assert!(out.contains(&a.id.to_string()));
    assert!(out.contains(&b.id.to_string()));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn ask_finds_a_contradiction_as_the_top_priority_question() {
    let root = temp_root("ask");
    {
        let mut s = FileStore::open(&root).unwrap();
        let a = seal_derivation("phlogiston", Vec::new());
        s.put_object(&a).unwrap();
        let b = seal_derivation("oxygen theory", Vec::new());
        s.put_object(&b).unwrap();
        let edge = seal_edge(a.id, b.id, Relation::Contradicts, Author::engine("test"));
        s.put_object(&edge).unwrap();
    }

    let args = Args::parse(&[]).unwrap();
    let out = sos_cli::ask::run(Some(root.to_str().unwrap()), &args).unwrap();
    assert!(out.contains("contradiction-hunt"));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn plan_recommends_the_best_candidate() {
    let root = temp_root("plan");
    let candidates_path = root.join("candidates.json");
    fs::create_dir_all(&root).unwrap();
    let candidates = serde_json::json!([
        {
            "experiment": oid(b"cheap-uninformative").to_string(),
            "eig": { "bits_milli": 50, "se_milli": 10, "level": "L3" },
            "cost": { "compute": 1, "time": 1, "samples": 1, "risk": 0 }
        },
        {
            "experiment": oid(b"best").to_string(),
            "eig": { "bits_milli": 900, "se_milli": 50, "level": "L3" },
            "cost": { "compute": 2, "time": 1, "samples": 1, "risk": 0 }
        }
    ]);
    fs::write(&candidates_path, serde_json::to_vec(&candidates).unwrap()).unwrap();

    let args = Args::parse(&[candidates_path.to_str().unwrap().to_owned()]).unwrap();
    let out = sos_cli::plan::run(&args).unwrap();
    assert!(out.contains(&oid(b"best").to_string()));
    assert!(out.contains("Recommend:"));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn publish_seals_and_optionally_stores_and_renders() {
    let root = temp_root("publish");
    let publication_path = root.join("pub.json");
    fs::create_dir_all(&root).unwrap();

    let evidence = oid(b"evidence");
    let publication = sos_publication::Publication::builder("Test Paper")
        .declared_root(evidence)
        .claim(sos_publication::Claim::new(
            "C1",
            "x",
            vec![sos_publication::ClaimBinding::new(
                sos_publication::BindingRole::DirectlySupports,
                evidence,
            )],
        ))
        .build();
    fs::write(&publication_path, serde_json::to_vec(&publication).unwrap()).unwrap();

    let store_path = root.join(".sos");
    let args = Args::parse(&[
        publication_path.to_str().unwrap().to_owned(),
        "--format".to_owned(),
        "md".to_owned(),
        "--store".to_owned(),
        store_path.to_str().unwrap().to_owned(),
    ])
    .unwrap();
    let out = sos_cli::publish::run(&args).unwrap();
    assert!(out.contains("Sealed publication"));
    assert!(out.contains("Stored in"));
    assert!(out.contains("Test Paper"));
    fs::remove_dir_all(&root).ok();
}

#[test]
fn plugins_lists_and_finds_by_role() {
    let root = temp_root("plugins");
    fs::create_dir_all(&root).unwrap();
    let descriptors_path = root.join("descriptors.json");

    let digest = HashAlgo::default().hash(b"cli-test", b"plugin-a");
    let descriptors = vec![
        PluginDescriptor::new(
            "solver-a",
            sos_core::SemVer::new(1, 0, 0),
            digest,
            Role::Reasoning,
        )
        .needs(Capability::Gpu),
        PluginDescriptor::new(
            "retriever-b",
            sos_core::SemVer::new(1, 0, 0),
            digest,
            Role::Memory,
        ),
    ];
    fs::write(&descriptors_path, serde_json::to_vec(&descriptors).unwrap()).unwrap();

    let args = Args::parse(&[descriptors_path.to_str().unwrap().to_owned()]).unwrap();
    let out = sos_cli::plugins::run(&args).unwrap();
    assert!(out.contains("solver-a"));
    assert!(out.contains("retriever-b"));

    let args = Args::parse(&[
        descriptors_path.to_str().unwrap().to_owned(),
        "--role".to_owned(),
        "reasoning".to_owned(),
    ])
    .unwrap();
    let out = sos_cli::plugins::run(&args).unwrap();
    assert!(out.contains("solver-a"));
    assert!(!out.contains("retriever-b"));
    fs::remove_dir_all(&root).ok();
}
