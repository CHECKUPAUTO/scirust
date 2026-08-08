use std::path::PathBuf;

use clap::Parser;
use scirust_sciagent::{
    BpeMergeSemantics, ElasticHardwareIdentity, StoredElasticProfile, VersionedBpeTokenizer,
};

#[derive(Parser)]
#[command(
    name = "tokenizer-profile-verify",
    about = "Verify an ElasticTokenizer profile against canonical tokenizer semantics and local hardware"
)]
struct Args {
    /// Explicitly canonical tokenizer artifact.
    #[arg(short, long)]
    tokenizer: PathBuf,

    /// Persisted ElasticTokenizer execution profile.
    #[arg(short, long)]
    profile: PathBuf,

    /// Stable deployment-local hardware discriminator used during calibration.
    #[arg(long, default_value = "generic")]
    device: String,
}

fn main() {
    let args = Args::parse();
    let tokenizer = VersionedBpeTokenizer::load_json(&args.tokenizer)
        .expect("failed to load tokenizer for profile verification");
    if tokenizer.merge_semantics() != BpeMergeSemantics::CanonicalRankV1
    {
        panic!("tokenizer-profile-verify requires merge_semantics=canonical-rank-v1");
    }
    let canonical = match tokenizer
    {
        VersionedBpeTokenizer::Canonical(tokenizer) => tokenizer,
        VersionedBpeTokenizer::Legacy(_) => unreachable!("semantic guard above rejected legacy"),
    };

    let stored =
        StoredElasticProfile::load(&args.profile).expect("failed to load ElasticTokenizer profile");
    let hardware =
        ElasticHardwareIdentity::new(std::env::consts::ARCH, std::env::consts::OS, args.device);
    stored
        .verify_for(canonical.ordered_merges(), &hardware)
        .expect("ElasticTokenizer profile verification failed");

    println!("elastic-profile-verification=ok");
    println!("merge-semantics={}", canonical.merge_semantics().as_str());
    println!("tokenizer-fingerprint={}", stored.tokenizer_fingerprint);
    println!("hardware-fingerprint={}", hardware.fingerprint());
    println!("thresholds={:?}", stored.profile.thresholds());
    println!("kernels={:?}", stored.profile.kernels());
}
