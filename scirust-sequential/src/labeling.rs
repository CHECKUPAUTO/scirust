//! Sequence labeling utilities: BIO tagging, sequence alignment, edit distance.

/// BIO (Begin-Inside-Outside) tagging utilities.
pub mod bio {
    /// A labeled token in BIO format.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct BIOTag {
        pub label: String,
        pub tag_type: TagType,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum TagType {
        Begin,
        Inside,
        Outside,
    }

    impl TagType {
        pub fn as_str(&self) -> &'static str {
            match self
            {
                TagType::Begin => "B",
                TagType::Inside => "I",
                TagType::Outside => "O",
            }
        }
    }

    impl std::fmt::Display for TagType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.as_str())
        }
    }

    impl std::fmt::Display for BIOTag {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            if self.tag_type == TagType::Outside
            {
                write!(f, "O")
            }
            else
            {
                write!(f, "{}-{}", self.tag_type, self.label)
            }
        }
    }

    /// Convert BIO-encoded tag strings into structured BIOTags.
    ///
    /// Tag format: "B-LABEL", "I-LABEL", or "O".
    pub fn parse_bio_tags(tags: &[&str]) -> Vec<BIOTag> {
        tags.iter()
            .map(|tag| {
                if *tag == "O"
                {
                    BIOTag {
                        label: String::new(),
                        tag_type: TagType::Outside,
                    }
                }
                else if let Some(rest) = tag.strip_prefix("B-")
                {
                    BIOTag {
                        label: rest.to_string(),
                        tag_type: TagType::Begin,
                    }
                }
                else if let Some(rest) = tag.strip_prefix("I-")
                {
                    BIOTag {
                        label: rest.to_string(),
                        tag_type: TagType::Inside,
                    }
                }
                else
                {
                    BIOTag {
                        label: tag.to_string(),
                        tag_type: TagType::Outside,
                    }
                }
            })
            .collect()
    }

    /// Convert BIO tags to a list of entity spans: (start, end, label).
    ///
    /// `end` is exclusive.
    pub fn extract_spans(tags: &[BIOTag]) -> Vec<(usize, usize, String)> {
        let mut spans = Vec::new();
        let mut current_start: Option<usize> = None;
        let mut current_label = String::new();

        for (i, tag) in tags.iter().enumerate()
        {
            match tag.tag_type
            {
                TagType::Begin =>
                {
                    if let Some(start) = current_start
                    {
                        spans.push((start, i, current_label.clone()));
                    }
                    current_start = Some(i);
                    current_label = tag.label.clone();
                },
                TagType::Inside =>
                {
                    if current_start.is_none()
                    {
                        current_start = Some(i);
                        current_label = tag.label.clone();
                    }
                },
                TagType::Outside =>
                {
                    if let Some(start) = current_start
                    {
                        spans.push((start, i, current_label.clone()));
                        current_start = None;
                        current_label.clear();
                    }
                },
            }
        }
        if let Some(start) = current_start
        {
            spans.push((start, tags.len(), current_label));
        }
        spans
    }

    /// Validate that a BIO tag sequence is well-formed.
    ///
    /// Returns `Ok(spans)` if valid, `Err(message)` if invalid.
    pub fn validate_bio(tags: &[BIOTag]) -> Result<Vec<(usize, usize, String)>, String> {
        for (i, tag) in tags.iter().enumerate()
        {
            if tag.tag_type == TagType::Inside && i == 0
            {
                return Err("I-tag at position 0 without preceding B-tag".to_string());
            }
            if tag.tag_type == TagType::Inside
            {
                if let Some(prev) = tags.get(i - 1)
                {
                    if prev.tag_type == TagType::Outside
                    {
                        return Err(format!(
                            "I-{} at position {} without preceding B-{}",
                            tag.label, i, tag.label
                        ));
                    }
                    if prev.tag_type == TagType::Begin && prev.label != tag.label
                    {
                        return Err(format!(
                            "I-{} at position {} follows B-{}",
                            tag.label, i, prev.label
                        ));
                    }
                    if prev.tag_type == TagType::Inside && prev.label != tag.label
                    {
                        return Err(format!(
                            "I-{} at position {} follows I-{}",
                            tag.label, i, prev.label
                        ));
                    }
                }
            }
        }
        Ok(extract_spans(tags))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parse_bio() {
            let tags = parse_bio_tags(&["B-PER", "I-PER", "O", "B-LOC"]);
            assert_eq!(
                tags[0],
                BIOTag {
                    label: "PER".into(),
                    tag_type: TagType::Begin
                }
            );
            assert_eq!(
                tags[1],
                BIOTag {
                    label: "PER".into(),
                    tag_type: TagType::Inside
                }
            );
            assert_eq!(
                tags[2],
                BIOTag {
                    label: "".into(),
                    tag_type: TagType::Outside
                }
            );
            assert_eq!(
                tags[3],
                BIOTag {
                    label: "LOC".into(),
                    tag_type: TagType::Begin
                }
            );
        }

        #[test]
        fn extract_spans_works() {
            let tags = parse_bio_tags(&["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]);
            let spans = extract_spans(&tags);
            assert_eq!(spans.len(), 2);
            assert_eq!(spans[0], (0, 2, "PER".into()));
            assert_eq!(spans[1], (3, 5, "LOC".into()));
        }

        #[test]
        fn validate_valid() {
            let tags = parse_bio_tags(&["B-PER", "I-PER", "O"]);
            assert!(validate_bio(&tags).is_ok());
        }

        #[test]
        fn validate_invalid_i_without_b() {
            let tags = parse_bio_tags(&["I-PER", "O"]);
            assert!(validate_bio(&tags).is_err());
        }
    }
}

/// Needleman-Wunsch global sequence alignment.
///
/// Returns `(aligned_seq1, aligned_seq2, score)`.
pub fn needleman_wunsch(
    seq1: &[usize],
    seq2: &[usize],
    match_score: f64,
    mismatch_penalty: f64,
    gap_penalty: f64,
) -> (Vec<Option<usize>>, Vec<Option<usize>>, f64) {
    let n = seq1.len();
    let m = seq2.len();

    let mut dp = vec![vec![0.0f64; m + 1]; n + 1];
    let mut traceback = vec![vec![0u8; m + 1]; n + 1]; // 0=diag, 1=up, 2=left

    for i in 1..=n
    {
        dp[i][0] = dp[i - 1][0] + gap_penalty;
        traceback[i][0] = 1;
    }
    for j in 1..=m
    {
        dp[0][j] = dp[0][j - 1] + gap_penalty;
        traceback[0][j] = 2;
    }

    for i in 1..=n
    {
        for j in 1..=m
        {
            let score_diag = if seq1[i - 1] == seq2[j - 1]
            {
                match_score
            }
            else
            {
                mismatch_penalty
            };
            let from_diag = dp[i - 1][j - 1] + score_diag;
            let from_up = dp[i - 1][j] + gap_penalty;
            let from_left = dp[i][j - 1] + gap_penalty;

            if from_diag >= from_up && from_diag >= from_left
            {
                dp[i][j] = from_diag;
                traceback[i][j] = 0;
            }
            else if from_up >= from_left
            {
                dp[i][j] = from_up;
                traceback[i][j] = 1;
            }
            else
            {
                dp[i][j] = from_left;
                traceback[i][j] = 2;
            }
        }
    }

    let mut aligned1 = Vec::new();
    let mut aligned2 = Vec::new();
    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0
    {
        if i > 0 && j > 0 && traceback[i][j] == 0
        {
            aligned1.push(Some(seq1[i - 1]));
            aligned2.push(Some(seq2[j - 1]));
            i -= 1;
            j -= 1;
        }
        else if i > 0 && traceback[i][j] == 1
        {
            aligned1.push(Some(seq1[i - 1]));
            aligned2.push(None);
            i -= 1;
        }
        else if j > 0
        {
            aligned1.push(None);
            aligned2.push(Some(seq2[j - 1]));
            j -= 1;
        }
        else
        {
            break;
        }
    }

    aligned1.reverse();
    aligned2.reverse();
    (aligned1, aligned2, dp[n][m])
}

/// Levenshtein edit distance between two slices.
///
/// Returns the minimum number of insertions, deletions, and substitutions.
#[allow(clippy::needless_range_loop)]
pub fn edit_distance(seq1: &[usize], seq2: &[usize]) -> usize {
    let n = seq1.len();
    let m = seq2.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];

    for i in 0..=n
    {
        dp[i][0] = i;
    }
    for j in 0..=m
    {
        dp[0][j] = j;
    }

    for i in 1..=n
    {
        for j in 1..=m
        {
            let cost = if seq1[i - 1] == seq2[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j - 1] + cost)
                .min(dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1);
        }
    }

    dp[n][m]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn needleman_wunsch_identical() {
        let seq = vec![1, 2, 3, 4];
        let (a1, a2, score) = needleman_wunsch(&seq, &seq, 1.0, -1.0, -1.0);
        assert_eq!(score, 4.0);
        for (x, y) in a1.iter().zip(a2.iter())
        {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn needleman_wunsch_empty_vs_nonempty() {
        let seq1: Vec<usize> = vec![];
        let seq2 = vec![1, 2, 3];
        let (a1, a2, score) = needleman_wunsch(&seq1, &seq2, 1.0, -1.0, -2.0);
        assert_eq!(a1.len(), 3);
        assert_eq!(a2.len(), 3);
        assert!((score - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn edit_distance_identical() {
        assert_eq!(edit_distance(&[1, 2, 3], &[1, 2, 3]), 0);
    }

    #[test]
    fn edit_distance_empty() {
        assert_eq!(edit_distance(&[], &[1, 2, 3]), 3);
        assert_eq!(edit_distance(&[1, 2, 3], &[]), 3);
    }

    #[test]
    fn edit_distance_classic() {
        assert_eq!(edit_distance(&[1, 2, 3], &[1, 4, 3]), 1);
        assert_eq!(edit_distance(&[], &[]), 0);
    }
}
