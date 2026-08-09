# ElasticTokenizer semantic versioning

SciAgent keeps historical BPE artifacts on the legacy compatibility path unless a tokenizer explicitly declares `merge_semantics=canonical-rank-v1`.

Canonical artifacts are trained one merge at a time, preserve merge-vector rank order, and are consumed through `ElasticTextTokenizer` / `VersionedBpeTokenizer`. Untagged artifacts remain legacy and unknown future semantics fail closed.

Execution profiles are optimization artifacts only. Changing S/M/L/XL/XXL/XXXL thresholds or kernels must never change token IDs.
