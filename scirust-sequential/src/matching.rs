//! Pattern matching in sequences: KMP, Boyer-Moore, LCS, DTW.

/// KMP (Knuth-Morris-Pratt) pattern matching.
pub fn kmp(text: &[usize], pattern: &[usize]) -> Vec<usize> {
    if pattern.is_empty()
    {
        return (0..=text.len()).collect();
    }
    if text.len() < pattern.len()
    {
        return Vec::new();
    }

    let lps = compute_lps(pattern);
    let mut results = Vec::new();
    let mut i = 0usize;
    let mut j = 0usize;

    while i < text.len()
    {
        if text[i] == pattern[j]
        {
            i += 1;
            j += 1;
            if j == pattern.len()
            {
                results.push(i - j);
                j = lps[j - 1];
            }
        }
        else if j > 0
        {
            j = lps[j - 1];
        }
        else
        {
            i += 1;
        }
    }

    results
}

fn compute_lps(pattern: &[usize]) -> Vec<usize> {
    let m = pattern.len();
    let mut lps = vec![0usize; m];
    let mut len = 0usize;
    let mut i = 1usize;

    while i < m
    {
        if pattern[i] == pattern[len]
        {
            len += 1;
            lps[i] = len;
            i += 1;
        }
        else if len > 0
        {
            len = lps[len - 1];
        }
        else
        {
            lps[i] = 0;
            i += 1;
        }
    }

    lps
}

/// Boyer-Moore pattern matching using the bad-character heuristic.
pub fn boyer_moore(text: &[usize], pattern: &[usize]) -> Vec<usize> {
    if pattern.is_empty()
    {
        return (0..=text.len()).collect();
    }
    if text.len() < pattern.len()
    {
        return Vec::new();
    }

    let n = text.len();
    let m = pattern.len();
    let alphabet_size = 256;

    let mut bad_char = vec![-1i64; alphabet_size];
    for i in 0..m
    {
        bad_char[pattern[i] % alphabet_size] = i as i64;
    }

    let mut results = Vec::new();
    let mut s = 0i64;

    while s <= (n as i64 - m as i64)
    {
        let mut j = m as i64 - 1;

        while j >= 0 && pattern[j as usize] == text[(s + j) as usize]
        {
            j -= 1;
        }

        if j < 0
        {
            results.push(s as usize);
            s += 1;
        }
        else
        {
            let bad_idx = text[(s + j) as usize] % alphabet_size;
            let shift = (j - bad_char[bad_idx]).max(1);
            s += shift;
        }
    }

    results
}

/// Longest Common Subsequence length (space-optimized).
pub fn longest_common_subsequence_len(seq1: &[usize], seq2: &[usize]) -> usize {
    let n = seq1.len();
    let m = seq2.len();
    let mut prev = vec![0usize; m + 1];
    let mut curr = vec![0usize; m + 1];

    for i in 1..=n
    {
        for j in 1..=m
        {
            if seq1[i - 1] == seq2[j - 1]
            {
                curr[j] = prev[j - 1] + 1;
            }
            else
            {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev[m]
}

/// Longest Common Subsequence returning the actual subsequence.
pub fn longest_common_subsequence(seq1: &[usize], seq2: &[usize]) -> Vec<usize> {
    let n = seq1.len();
    let m = seq2.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];

    for i in 1..=n
    {
        for j in 1..=m
        {
            if seq1[i - 1] == seq2[j - 1]
            {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            }
            else
            {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    let mut result = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 && j > 0
    {
        if seq1[i - 1] == seq2[j - 1]
        {
            result.push(seq1[i - 1]);
            i -= 1;
            j -= 1;
        }
        else if dp[i - 1][j] > dp[i][j - 1]
        {
            i -= 1;
        }
        else
        {
            j -= 1;
        }
    }
    result.reverse();
    result
}

/// Dynamic Time Warping distance between two 1-D sequences.
pub fn dynamic_time_warping(seq1: &[f64], seq2: &[f64], step_size: Option<usize>) -> f64 {
    let n = seq1.len();
    let m = seq2.len();
    let inf = f64::INFINITY;

    let mut dp = vec![vec![inf; m + 1]; n + 1];
    dp[0][0] = 0.0;

    for i in 1..=n
    {
        let j_start = step_size
            .map(|s| if i > s { i - s } else { 1 })
            .unwrap_or(1);
        let j_end = step_size.map(|s| i.saturating_add(s).min(m)).unwrap_or(m);

        for j in j_start..=j_end
        {
            let cost = (seq1[i - 1] - seq2[j - 1]).powi(2);
            dp[i][j] = cost + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
        }
    }

    dp[n][m].sqrt()
}

/// DTW that also returns the optimal warping path.
pub fn dynamic_time_warping_with_path(seq1: &[f64], seq2: &[f64]) -> (f64, Vec<(usize, usize)>) {
    let n = seq1.len();
    let m = seq2.len();
    let inf = f64::INFINITY;

    let mut dp = vec![vec![inf; m + 1]; n + 1];
    let mut traceback = vec![vec![0u8; m + 1]; n + 1];
    dp[0][0] = 0.0;

    for i in 1..=n
    {
        for j in 1..=m
        {
            let cost = (seq1[i - 1] - seq2[j - 1]).powi(2);
            let candidates = [dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]];
            let mut best = candidates[0];
            let mut best_idx = 0u8;
            for (k, &c) in candidates.iter().enumerate()
            {
                if c < best
                {
                    best = c;
                    best_idx = k as u8;
                }
            }
            dp[i][j] = cost + best;
            traceback[i][j] = best_idx;
        }
    }

    let mut path = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0
    {
        path.push((i - 1, j - 1));
        match traceback[i][j]
        {
            0 => i -= 1,
            1 => j -= 1,
            _ =>
            {
                i -= 1;
                j -= 1;
            },
        }
    }
    path.reverse();
    (dp[n][m].sqrt(), path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmp_basic() {
        let text = vec![0, 1, 2, 0, 1, 2, 3];
        let pattern = vec![0, 1, 2];
        assert_eq!(kmp(&text, &pattern), vec![0, 3]);
    }

    #[test]
    fn kmp_no_match() {
        let text = vec![0, 1, 2, 3];
        assert!(kmp(&text, &[5, 6]).is_empty());
    }

    #[test]
    fn kmp_empty_pattern() {
        let text = vec![1, 2, 3];
        assert_eq!(kmp(&text, &[]), vec![0, 1, 2, 3]);
    }

    #[test]
    fn kmp_pattern_equals_text() {
        let text = vec![1, 2, 3];
        assert_eq!(kmp(&text, &text.clone()), vec![0]);
    }

    #[test]
    fn bm_basic() {
        let text = vec![0, 1, 2, 0, 1, 2, 3];
        let pattern = vec![0, 1, 2];
        assert_eq!(boyer_moore(&text, &pattern), vec![0, 3]);
    }

    #[test]
    fn bm_no_match() {
        let text = vec![0, 1, 2, 3];
        assert!(boyer_moore(&text, &[5, 6]).is_empty());
    }

    #[test]
    fn bm_repeated_pattern() {
        let text = vec![1, 1, 1, 1, 1];
        let pattern = vec![1, 1];
        assert_eq!(boyer_moore(&text, &pattern), vec![0, 1, 2, 3]);
    }

    #[test]
    fn lcs_identical() {
        let seq = vec![1, 2, 3, 4];
        assert_eq!(longest_common_subsequence_len(&seq, &seq), 4);
    }

    #[test]
    fn lcs_empty() {
        assert_eq!(longest_common_subsequence_len(&[], &[]), 0);
        assert_eq!(longest_common_subsequence_len(&[1, 2], &[]), 0);
    }

    #[test]
    fn lcs_classic() {
        let s1 = vec![1, 2, 3, 4, 5];
        let s2 = vec![2, 4, 5, 1, 3];
        assert_eq!(longest_common_subsequence_len(&s1, &s2), 3);
    }

    #[test]
    fn lcs_subsequence_values() {
        let s1 = vec![1, 0, 0, 1, 0, 1, 0, 1];
        let s2 = vec![0, 1, 0, 1, 0, 0, 1, 0];
        let lcs = longest_common_subsequence(&s1, &s2);
        assert_eq!(lcs.len(), longest_common_subsequence_len(&s1, &s2));
    }

    #[test]
    fn dtw_identical() {
        let seq = vec![1.0, 2.0, 3.0, 4.0];
        let dist = dynamic_time_warping(&seq, &seq, None);
        assert!((dist - 0.0).abs() < 1e-10);
    }

    #[test]
    fn dtw_shifted() {
        let s1 = vec![0.0, 1.0, 2.0, 3.0];
        let s2 = vec![1.0, 2.0, 3.0, 4.0];
        let dist = dynamic_time_warping(&s1, &s2, None);
        assert!(dist > 0.0);
        assert!(dist < 5.0);
    }

    #[test]
    fn dtw_with_path_consistency() {
        let s1 = vec![1.0, 2.0, 3.0];
        let s2 = vec![1.0, 3.0, 5.0];
        let (dist, path) = dynamic_time_warping_with_path(&s1, &s2);
        assert!(dist > 0.0);
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(2, 2)));
        assert!(path.len() >= 3);
    }

    #[test]
    fn dtw_banded_matches_full() {
        let s1: Vec<f64> = (0..10).map(|x| x as f64).collect();
        let s2: Vec<f64> = (0..10).map(|x| (x as f64) + 0.5).collect();
        let full = dynamic_time_warping(&s1, &s2, None);
        let banded = dynamic_time_warping(&s1, &s2, Some(10));
        assert!((full - banded).abs() < 1e-10);
    }
}
