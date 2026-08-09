use scirust_sciagent::{
    BpeMergeSemantics, ElasticProfile, ElasticTextTokenizer, ElasticThresholds,
};

fn main() {
    let _ = BpeMergeSemantics::CanonicalRankV1;
    let _ = ElasticProfile::reference_only(
        ElasticThresholds::new(16, 64, 256, 1024, 4096)
            .expect("static ElasticTokenizer thresholds are valid"),
    );
    let _ = std::any::TypeId::of::<ElasticTextTokenizer>();
}
