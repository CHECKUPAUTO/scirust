//! Padding, batching and splitting for oracle-labelled [`TrainingSample`]s.
//!
//! The training pipeline ([`crate::generate::build_training_set`]) yields
//! variable-length, token-aligned samples. To feed a model in fixed-shape
//! batches we need three deterministic operations, all defined here:
//!
//! - **Padding** ([`pad_sample`]): extend (or truncate) a single sample's
//!   four parallel vectors to an exact target length, filling new positions
//!   with the neutral pad ids ([`SomVocab::PAD`], [`OWNERSHIP_NA`],
//!   [`BORROW_NA`], invalid `0.0`).
//! - **Batching** ([`batch_samples`]): group samples into fixed-size batches,
//!   each padded to the longest sample *in that batch*, with a per-position
//!   `pad_mask` marking real tokens. A trailing partial batch is kept (with
//!   its true, smaller row count) unless `drop_last` is set.
//! - **Splitting** ([`train_val_split`]): partition samples into train/val
//!   using a seeded Fisher–Yates shuffle. The two halves are disjoint and
//!   together cover every input sample exactly once; identical `seed` and
//!   `val_ratio` always reproduce the same partition.
//!
//! Pad positions use the dedicated "not-applicable" class ids so a loss that
//! is masked by `pad_mask` never learns from them and an unmasked loss treats
//! them as the neutral class rather than a real label.

use crate::generate::TrainingSample;
use scirust_core::nn::rng::PcgEngine;
use scirust_som_symbolic::{BORROW_NA, OWNERSHIP_NA};
use scirust_som_tokenizer::SomVocab;

/// Pad value for the per-token invalid-fault channel.
const INVALID_PAD: f32 = 0.0;

/// Pad or truncate `sample` to exactly `target_len` positions.
///
/// - If the sample is shorter, the four parallel vectors are extended with
///   the neutral pad ids: token [`SomVocab::PAD`], ownership [`OWNERSHIP_NA`],
///   borrow [`BORROW_NA`], invalid [`INVALID_PAD`].
/// - If the sample is longer, every vector is truncated to `target_len`.
/// - The returned `pad_mask` is `1.0` for an original (real) token and `0.0`
///   for a padded one; for a truncated sample it is all `1.0`.
///
/// All five output vectors have length exactly `target_len`.
pub fn pad_sample(sample: &TrainingSample, target_len: usize) -> PaddedSample {
    let real = sample.token_ids.len().min(target_len);

    let mut token_ids = Vec::with_capacity(target_len);
    let mut ownership = Vec::with_capacity(target_len);
    let mut borrow = Vec::with_capacity(target_len);
    let mut invalid = Vec::with_capacity(target_len);
    let mut pad_mask = Vec::with_capacity(target_len);

    token_ids.extend_from_slice(&sample.token_ids[..real]);
    ownership.extend_from_slice(&sample.ownership[..real]);
    borrow.extend_from_slice(&sample.borrow[..real]);
    invalid.extend_from_slice(&sample.invalid[..real]);
    pad_mask.resize(real, 1.0);

    if target_len > real
    {
        let pad = target_len - real;
        token_ids.resize(target_len, SomVocab::PAD);
        ownership.resize(target_len, OWNERSHIP_NA);
        borrow.resize(target_len, BORROW_NA);
        invalid.resize(target_len, INVALID_PAD);
        pad_mask.resize(target_len, 0.0);
        debug_assert_eq!(pad, target_len - real);
    }

    PaddedSample {
        token_ids,
        ownership,
        borrow,
        invalid,
        pad_mask,
    }
}

/// One sample padded/truncated to a fixed length, with a real-token mask.
///
/// All five vectors share the same length ([`PaddedSample::len`]).
#[derive(Debug, Clone, PartialEq)]
pub struct PaddedSample {
    pub token_ids: Vec<usize>,
    pub ownership: Vec<usize>,
    pub borrow: Vec<usize>,
    pub invalid: Vec<f32>,
    /// `1.0` for a real token, `0.0` for a pad position.
    pub pad_mask: Vec<f32>,
}

impl PaddedSample {
    /// Padded sequence length (identical for every channel).
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    /// Number of real (non-pad) tokens.
    pub fn real_len(&self) -> usize {
        self.pad_mask.iter().filter(|&&m| m > 0.5).count()
    }
}

/// A fixed-shape batch: `rows` samples, each padded to `seq_len` tokens.
///
/// The channel buffers are row-major and flat: position `r * seq_len + c`
/// holds row `r`, column `c`. `pad_mask` masks the padded tail of each row.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    /// Number of samples in the batch (the trailing batch may be smaller).
    pub rows: usize,
    /// Padded sequence length shared by every row.
    pub seq_len: usize,
    /// Token ids, row-major (`rows * seq_len`).
    pub token_ids: Vec<usize>,
    /// Ownership class ids, row-major.
    pub ownership: Vec<usize>,
    /// Borrow class ids, row-major.
    pub borrow: Vec<usize>,
    /// Invalid-fault targets, row-major.
    pub invalid: Vec<f32>,
    /// `1.0` real / `0.0` pad, row-major.
    pub pad_mask: Vec<f32>,
}

impl Batch {
    /// Total token slots in the batch (`rows * seq_len`).
    pub fn n_slots(&self) -> usize {
        self.rows * self.seq_len
    }

    /// Build a batch from samples, padding each to `seq_len`.
    ///
    /// `seq_len` must be at least the longest sample's length, otherwise the
    /// over-long samples would be silently truncated; this is checked.
    fn from_samples(samples: &[TrainingSample], seq_len: usize) -> Batch {
        let rows = samples.len();
        let mut token_ids = Vec::with_capacity(rows * seq_len);
        let mut ownership = Vec::with_capacity(rows * seq_len);
        let mut borrow = Vec::with_capacity(rows * seq_len);
        let mut invalid = Vec::with_capacity(rows * seq_len);
        let mut pad_mask = Vec::with_capacity(rows * seq_len);
        for s in samples
        {
            debug_assert!(
                s.token_ids.len() <= seq_len,
                "seq_len must cover the longest sample"
            );
            let p = pad_sample(s, seq_len);
            token_ids.extend_from_slice(&p.token_ids);
            ownership.extend_from_slice(&p.ownership);
            borrow.extend_from_slice(&p.borrow);
            invalid.extend_from_slice(&p.invalid);
            pad_mask.extend_from_slice(&p.pad_mask);
        }
        Batch {
            rows,
            seq_len,
            token_ids,
            ownership,
            borrow,
            invalid,
            pad_mask,
        }
    }

    /// Borrow row `r` of a row-major channel as a `seq_len` slice.
    pub fn row<'a, T>(&self, channel: &'a [T], r: usize) -> &'a [T] {
        let start = r * self.seq_len;
        &channel[start..start + self.seq_len]
    }
}

/// Split `samples` into fixed-size batches.
///
/// Samples are taken in order, `batch_size` at a time. Each batch is padded to
/// the longest sample it contains (so different batches may have different
/// `seq_len`), and carries a `pad_mask`. When the sample count is not a
/// multiple of `batch_size`, the final, smaller batch is **kept** with its true
/// row count unless `drop_last` is `true`, in which case it is dropped.
///
/// No sample is duplicated, and with `drop_last == false` none is lost: the
/// total `rows` across the returned batches equals `samples.len()`.
///
/// # Panics
/// Panics if `batch_size == 0`.
pub fn batch_samples(samples: &[TrainingSample], batch_size: usize, drop_last: bool) -> Vec<Batch> {
    assert!(batch_size > 0, "batch_size must be non-zero");
    let mut batches = Vec::with_capacity(samples.len() / batch_size + 1);
    for chunk in samples.chunks(batch_size)
    {
        if drop_last && chunk.len() < batch_size
        {
            break;
        }
        let seq_len = chunk.iter().map(|s| s.token_ids.len()).max().unwrap_or(0);
        batches.push(Batch::from_samples(chunk, seq_len));
    }
    batches
}

/// A deterministic train/validation partition.
///
/// `train` and `val` are disjoint and together hold every input sample exactly
/// once (`train.len() + val.len() == n`).
#[derive(Debug, Clone)]
pub struct SplitDataset {
    pub train: Vec<TrainingSample>,
    pub val: Vec<TrainingSample>,
}

/// Uniform integer in `[0, bound)` from `rng`, rejection-sampled to avoid the
/// modulo bias of a bare `next_u32() % bound`.
///
/// `bound` must be non-zero.
fn rand_below(rng: &mut PcgEngine, bound: u32) -> u32 {
    debug_assert!(bound > 0);
    // Largest multiple of `bound` that fits in u32; values at or above it are
    // rejected so every residue class is equally likely.
    let zone = u32::MAX - (u32::MAX % bound);
    loop
    {
        let r = rng.next_u32();
        if r < zone
        {
            return r % bound;
        }
    }
}

/// In-place Fisher–Yates shuffle of `items` driven by `rng`.
///
/// Iterates from the last index down to 1, swapping each element with a
/// uniformly chosen earlier-or-equal index. Deterministic in the RNG stream.
fn shuffle_in_place<T>(items: &mut [T], rng: &mut PcgEngine) {
    let n = items.len();
    if n < 2
    {
        return;
    }
    let mut i = n - 1;
    while i >= 1
    {
        let j = rand_below(rng, (i + 1) as u32) as usize;
        items.swap(i, j);
        i -= 1;
    }
}

/// Partition `samples` into train/val with a fixed `val_ratio` and `seed`.
///
/// The validation count is `round(n * val_ratio)`, clamped to `[0, n]`; the
/// remainder is training. Indices `0..n` are shuffled with a seeded
/// Fisher–Yates ([`shuffle_in_place`]); the first `n_train` shuffled indices
/// form the training set and the rest the validation set, so the two are a
/// disjoint cover of the input. Cloning is used so the input slice is left
/// untouched and the operation is referentially transparent in `(seed,
/// val_ratio)`.
///
/// `val_ratio` is clamped to `[0.0, 1.0]`.
pub fn train_val_split(samples: &[TrainingSample], val_ratio: f64, seed: u64) -> SplitDataset {
    let n = samples.len();
    let ratio = val_ratio.clamp(0.0, 1.0);
    let n_val = ((n as f64) * ratio).round() as usize;
    let n_val = n_val.min(n);
    let n_train = n - n_val;

    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = PcgEngine::new(seed);
    shuffle_in_place(&mut order, &mut rng);

    let mut train = Vec::with_capacity(n_train);
    let mut val = Vec::with_capacity(n_val);
    for (rank, &idx) in order.iter().enumerate()
    {
        if rank < n_train
        {
            train.push(samples[idx].clone());
        }
        else
        {
            val.push(samples[idx].clone());
        }
    }
    SplitDataset { train, val }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_som_symbolic::{
        BORROW_MUT, BORROW_NONE, BORROW_SHARED, OWNERSHIP_MOVED, OWNERSHIP_OWNED,
    };

    /// Hand-built sample of a chosen length with distinguishable channels.
    ///
    /// Token ids start at 10 + base so they are clearly non-pad (PAD == 0) and
    /// unique per sample; ownership/borrow cycle through real (non-NA) classes;
    /// invalid marks the last token of the sample as a fault.
    fn sample(base: usize, len: usize) -> TrainingSample {
        let token_ids: Vec<usize> = (0..len).map(|i| 10 + base * 100 + i).collect();
        let own_cycle = [OWNERSHIP_OWNED, OWNERSHIP_MOVED];
        let bor_cycle = [BORROW_NONE, BORROW_SHARED, BORROW_MUT];
        let ownership: Vec<usize> = (0..len).map(|i| own_cycle[i % own_cycle.len()]).collect();
        let borrow: Vec<usize> = (0..len).map(|i| bor_cycle[i % bor_cycle.len()]).collect();
        let invalid: Vec<f32> = (0..len)
            .map(|i| if i + 1 == len { 1.0 } else { 0.0 })
            .collect();
        TrainingSample {
            token_ids,
            ownership,
            borrow,
            invalid,
        }
    }

    #[test]
    fn pad_extends_to_exact_length_with_pad_ids() {
        // A length-3 sample padded to 5: two PAD tail positions.
        let s = sample(0, 3);
        let p = pad_sample(&s, 5);

        assert_eq!(p.len(), 5);
        // Real prefix preserved verbatim.
        assert_eq!(p.token_ids, vec![10, 11, 12, SomVocab::PAD, SomVocab::PAD]);
        assert_eq!(
            p.ownership,
            vec![
                OWNERSHIP_OWNED,
                OWNERSHIP_MOVED,
                OWNERSHIP_OWNED,
                OWNERSHIP_NA,
                OWNERSHIP_NA
            ]
        );
        assert_eq!(
            p.borrow,
            vec![BORROW_NONE, BORROW_SHARED, BORROW_MUT, BORROW_NA, BORROW_NA]
        );
        assert_eq!(p.invalid, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(p.pad_mask, vec![1.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(p.real_len(), 3);
    }

    #[test]
    fn pad_to_equal_length_is_identity_mask_all_ones() {
        let s = sample(1, 4);
        let p = pad_sample(&s, 4);
        assert_eq!(p.len(), 4);
        assert_eq!(p.token_ids, s.token_ids);
        assert_eq!(p.ownership, s.ownership);
        assert_eq!(p.borrow, s.borrow);
        assert_eq!(p.invalid, s.invalid);
        assert_eq!(p.pad_mask, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(p.real_len(), 4);
    }

    #[test]
    fn pad_truncates_when_target_shorter() {
        // Length-5 sample truncated to 2: exactly the first two positions,
        // mask all real (nothing was padded).
        let s = sample(2, 5);
        let p = pad_sample(&s, 2);
        assert_eq!(p.len(), 2);
        assert_eq!(p.token_ids, vec![210, 211]);
        assert_eq!(p.ownership, vec![OWNERSHIP_OWNED, OWNERSHIP_MOVED]);
        assert_eq!(p.borrow, vec![BORROW_NONE, BORROW_SHARED]);
        assert_eq!(p.invalid, vec![0.0, 0.0]);
        assert_eq!(p.pad_mask, vec![1.0, 1.0]);
        assert_eq!(p.real_len(), 2);
    }

    #[test]
    fn batch_shapes_pad_to_max_in_batch_and_keep_partial() {
        // Five samples of lengths 2,3,1,4,2 with batch_size 2, keep partial.
        let samples = vec![
            sample(0, 2),
            sample(1, 3),
            sample(2, 1),
            sample(3, 4),
            sample(4, 2),
        ];
        let batches = batch_samples(&samples, 2, false);
        // 5 samples / 2 => 3 batches: rows 2,2,1.
        assert_eq!(batches.len(), 3);
        assert_eq!((batches[0].rows, batches[0].seq_len), (2, 3)); // max(2,3)
        assert_eq!((batches[1].rows, batches[1].seq_len), (2, 4)); // max(1,4)
        assert_eq!((batches[2].rows, batches[2].seq_len), (1, 2)); // last partial

        // Row-major buffers are sized rows*seq_len for every channel.
        for b in &batches
        {
            assert_eq!(b.token_ids.len(), b.n_slots());
            assert_eq!(b.ownership.len(), b.n_slots());
            assert_eq!(b.borrow.len(), b.n_slots());
            assert_eq!(b.invalid.len(), b.n_slots());
            assert_eq!(b.pad_mask.len(), b.n_slots());
        }

        // Batch 0: row 0 (len 2) padded to 3, row 1 (len 3) full.
        let b0 = &batches[0];
        assert_eq!(b0.row(&b0.token_ids, 0), &[10, 11, SomVocab::PAD]);
        assert_eq!(b0.row(&b0.pad_mask, 0), &[1.0, 1.0, 0.0]);
        assert_eq!(b0.row(&b0.token_ids, 1), &[110, 111, 112]);
        assert_eq!(b0.row(&b0.pad_mask, 1), &[1.0, 1.0, 1.0]);

        // Batch 1: row 0 (len 1) padded to 4, row 1 (len 4) full.
        let b1 = &batches[1];
        assert_eq!(
            b1.row(&b1.token_ids, 0),
            &[210, SomVocab::PAD, SomVocab::PAD, SomVocab::PAD]
        );
        assert_eq!(b1.row(&b1.pad_mask, 0), &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(b1.row(&b1.token_ids, 1), &[310, 311, 312, 313]);
    }

    #[test]
    fn batch_drop_last_discards_partial_only() {
        let samples = vec![
            sample(0, 2),
            sample(1, 3),
            sample(2, 1),
            sample(3, 4),
            sample(4, 2),
        ];
        // drop_last: 5/2 => keep 2 full batches (4 rows), drop the trailing 1.
        let kept = batch_samples(&samples, 2, true);
        assert_eq!(kept.len(), 2);
        let kept_rows: usize = kept.iter().map(|b| b.rows).sum();
        assert_eq!(kept_rows, 4);

        // An exact multiple keeps everything regardless of drop_last.
        let exact = vec![sample(0, 2), sample(1, 2), sample(2, 2), sample(3, 2)];
        let a = batch_samples(&exact, 2, true);
        let b = batch_samples(&exact, 2, false);
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        assert_eq!(a.iter().map(|x| x.rows).sum::<usize>(), 4);
    }

    #[test]
    fn batching_keeps_every_sample_exactly_once() {
        // Distinct first-token ids let us recover original membership.
        let samples: Vec<TrainingSample> = (0..7).map(|i| sample(i, 2 + (i % 3))).collect();
        let batches = batch_samples(&samples, 3, false);
        // 7/3 => 3 batches of rows 3,3,1.
        assert_eq!(batches.iter().map(|b| b.rows).sum::<usize>(), 7);

        // Reconstruct the first real token of every row in order and compare
        // to the inputs' first tokens — no loss, no duplication, order kept.
        let mut seen_first: Vec<usize> = Vec::new();
        for b in &batches
        {
            for r in 0..b.rows
            {
                seen_first.push(b.row(&b.token_ids, r)[0]);
            }
        }
        let expected_first: Vec<usize> = samples.iter().map(|s| s.token_ids[0]).collect();
        assert_eq!(seen_first, expected_first);
    }

    #[test]
    fn split_is_exact_partition_and_seed_deterministic() {
        // 10 samples, 30% val => 3 val, 7 train.
        let samples: Vec<TrainingSample> = (0..10).map(|i| sample(i, 2)).collect();
        let s1 = train_val_split(&samples, 0.3, 123);
        let s2 = train_val_split(&samples, 0.3, 123);

        assert_eq!(s1.train.len(), 7);
        assert_eq!(s1.val.len(), 3);

        // Deterministic in the seed: same membership both times.
        let ids =
            |v: &[TrainingSample]| -> Vec<usize> { v.iter().map(|s| s.token_ids[0]).collect() };
        assert_eq!(ids(&s1.train), ids(&s2.train));
        assert_eq!(ids(&s1.val), ids(&s2.val));

        // Disjoint cover: union of first-token ids == all inputs, no overlap.
        let mut union: Vec<usize> = ids(&s1.train);
        union.extend(ids(&s1.val));
        union.sort_unstable();
        let all: Vec<usize> = (0..10).map(|i| 10 + i * 100).collect();
        assert_eq!(union, all, "every sample appears exactly once");

        let train_set: std::collections::HashSet<usize> = ids(&s1.train).into_iter().collect();
        let val_set: std::collections::HashSet<usize> = ids(&s1.val).into_iter().collect();
        assert!(
            train_set.is_disjoint(&val_set),
            "train and val must not overlap"
        );
    }

    #[test]
    fn split_exact_membership_for_known_seed() {
        // Pin the exact partition for a fixed seed so a change in the shuffle
        // is caught. Derived by running the documented Fisher–Yates with this
        // RNG; asserted against the recovered first-token ids.
        let samples: Vec<TrainingSample> = (0..6).map(|i| sample(i, 2)).collect();
        let split = train_val_split(&samples, 0.5, 99);
        assert_eq!(split.train.len(), 3);
        assert_eq!(split.val.len(), 3);

        // The shuffle below is recomputed independently with the same RNG and
        // algorithm; train = first 3 shuffled indices, val = last 3.
        let mut order: Vec<usize> = (0..6).collect();
        let mut rng = PcgEngine::new(99);
        let mut i = order.len() - 1;
        while i >= 1
        {
            // mirror rand_below: rejection-sampled uniform in [0, i].
            let bound = (i + 1) as u32;
            let zone = u32::MAX - (u32::MAX % bound);
            let j = loop
            {
                let r = rng.next_u32();
                if r < zone
                {
                    break (r % bound) as usize;
                }
            };
            order.swap(i, j);
            i -= 1;
        }
        let expect_train: Vec<usize> = order[..3].iter().map(|&idx| 10 + idx * 100).collect();
        let expect_val: Vec<usize> = order[3..].iter().map(|&idx| 10 + idx * 100).collect();

        let got_train: Vec<usize> = split.train.iter().map(|s| s.token_ids[0]).collect();
        let got_val: Vec<usize> = split.val.iter().map(|s| s.token_ids[0]).collect();
        assert_eq!(got_train, expect_train);
        assert_eq!(got_val, expect_val);
    }

    #[test]
    fn split_edge_ratios() {
        let samples: Vec<TrainingSample> = (0..5).map(|i| sample(i, 2)).collect();
        // ratio 0 => all train.
        let z = train_val_split(&samples, 0.0, 1);
        assert_eq!(z.train.len(), 5);
        assert_eq!(z.val.len(), 0);
        // ratio 1 => all val.
        let o = train_val_split(&samples, 1.0, 1);
        assert_eq!(o.train.len(), 0);
        assert_eq!(o.val.len(), 5);
        // ratio clamps above 1.
        let c = train_val_split(&samples, 2.5, 1);
        assert_eq!(c.val.len(), 5);
    }

    #[test]
    fn shuffle_is_a_permutation() {
        // A seeded shuffle must permute, never drop or duplicate.
        let mut v: Vec<usize> = (0..50).collect();
        let mut rng = PcgEngine::new(2024);
        shuffle_in_place(&mut v, &mut rng);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        // With this seed the order actually changed (not the identity).
        assert_ne!(v, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn real_dataset_batches_lose_no_samples_and_pad_correctly() {
        // End-to-end against the real generator: every produced sample lands
        // in exactly one batch row, and padded rows match their seq_len.
        let samples = crate::generate::build_training_set(42, 13, 32);
        assert!(!samples.is_empty());
        let batch_size = 4;
        let batches = batch_samples(&samples, batch_size, false);
        let total_rows: usize = batches.iter().map(|b| b.rows).sum();
        assert_eq!(total_rows, samples.len(), "no sample lost across batches");

        for b in &batches
        {
            assert!(b.rows <= batch_size);
            assert_eq!(b.pad_mask.len(), b.rows * b.seq_len);
            for r in 0..b.rows
            {
                let mask = b.row(&b.pad_mask, r);
                let toks = b.row(&b.token_ids, r);
                // Real prefix is contiguous then pure padding.
                let real = mask.iter().filter(|&&m| m > 0.5).count();
                assert!(real >= 1 && real <= b.seq_len);
                for c in 0..b.seq_len
                {
                    if c < real
                    {
                        assert!(mask[c] > 0.5);
                    }
                    else
                    {
                        assert!(mask[c] < 0.5);
                        assert_eq!(toks[c], SomVocab::PAD);
                    }
                }
            }
        }
    }
}
