//! Two-dimensional detection: 2-D CFAR and detection clustering.
//!
//! The pulse-Doppler / FMCW front end produces a range-Doppler magnitude map
//! ([`super::doppler::range_doppler_map`], [`super::fmcw::range_doppler`]). The
//! detection stage turns that surface into a short list of targets, exactly as
//! the reference pipelines (OpenRadar) do: run **2-D CFAR** to get an adaptive
//! per-cell detection mask, then **cluster** the mask so one physical target —
//! which lights up a little blob of adjacent cells — becomes a single centroid
//! rather than a scatter of detections. Those centroids are what a tracker
//! (next stage) associates across frames.
//!
//! Both steps are dependency-free; the CFAR reuses the closed-form threshold
//! scaling of [`super::cfar::ca_cfar_alpha`].

use super::cfar::ca_cfar_alpha;

/// A clustered detection: one target formed from a connected blob of CFAR
/// detections. `range` and `doppler` are the amplitude-weighted centroid bin
/// coordinates (fractional), `amplitude` the peak cell magnitude in the blob,
/// and `cells` the number of detected cells it spans.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub range: f64,
    pub doppler: f64,
    pub amplitude: f64,
    pub cells: usize,
}

/// Two-dimensional cell-averaging CFAR over a range-Doppler **power** map
/// (`power[range][doppler]`, magnitude-squared).
///
/// For each cell under test the noise level is the mean of the training cells
/// in the square window of half-width `train + guard`, excluding the inner
/// `(2·guard+1)²` guard region (which shields the target's own spread). A
/// detection is flagged when the cell exceeds `α · mean(training)` with `α`
/// from [`ca_cfar_alpha`] over `N = (2(train+guard)+1)² − (2·guard+1)²` training
/// cells — the same closed form as the 1-D detector, so the achieved `P_fa` is
/// exactly `pfa` in homogeneous exponential noise. Cells without a full window
/// (within `train + guard` of an edge) are never flagged. Returns an all-`false`
/// mask of the input shape on a degenerate parameter or too-small / ragged map.
#[allow(clippy::needless_range_loop)] // 2-D windowed sums — indices are the algorithm
pub fn ca_cfar_2d(power: &[Vec<f64>], train: usize, guard: usize, pfa: f64) -> Vec<Vec<bool>> {
    let rows = power.len();
    if rows == 0
    {
        return Vec::new();
    }
    let cols = power[0].len();
    let mut det = vec![vec![false; cols]; rows];
    let w = train + guard;
    if train == 0
        || pfa <= 0.0
        || pfa >= 1.0
        || cols == 0
        || power.iter().any(|r| r.len() != cols)
        || rows < 2 * w + 1
        || cols < 2 * w + 1
    {
        return det;
    }
    let n_ref = (2 * w + 1) * (2 * w + 1) - (2 * guard + 1) * (2 * guard + 1);
    let alpha = ca_cfar_alpha(n_ref, pfa);
    for r in w..rows - w
    {
        for c in w..cols - w
        {
            // Training sum = full window minus the guard region (which holds
            // the CUT); their difference is the pure reference-cell energy.
            let mut full = 0.0;
            for i in r - w..=r + w
            {
                for j in c - w..=c + w
                {
                    full += power[i][j];
                }
            }
            let mut inner = 0.0;
            for i in r - guard..=r + guard
            {
                for j in c - guard..=c + guard
                {
                    inner += power[i][j];
                }
            }
            let noise = (full - inner) / n_ref as f64;
            det[r][c] = power[r][c] > alpha * noise;
        }
    }
    det
}

/// Cluster a 2-D detection `mask` into targets by 8-connected connected-
/// component labelling, weighting each component's centroid by the `map`
/// amplitudes. Returns one [`Detection`] per connected blob, strongest peak
/// first. `map` supplies the amplitudes for the centroid and peak; it must have
/// the same shape as `mask`. Empty if the mask is empty or the shapes disagree.
///
/// A blob whose amplitudes are all zero falls back to the plain geometric
/// centroid of its cells.
#[allow(clippy::needless_range_loop)] // grid scan seeds the flood fill by (r, c)
pub fn cluster_detections(mask: &[Vec<bool>], map: &[Vec<f64>]) -> Vec<Detection> {
    let rows = mask.len();
    if rows == 0 || map.len() != rows
    {
        return Vec::new();
    }
    let cols = mask[0].len();
    if cols == 0 || mask.iter().any(|r| r.len() != cols) || map.iter().any(|r| r.len() != cols)
    {
        return Vec::new();
    }
    let mut visited = vec![vec![false; cols]; rows];
    let mut out: Vec<Detection> = Vec::new();
    for r0 in 0..rows
    {
        for c0 in 0..cols
        {
            if !mask[r0][c0] || visited[r0][c0]
            {
                continue;
            }
            // Flood-fill this 8-connected component.
            visited[r0][c0] = true;
            let mut stack = vec![(r0, c0)];
            let (mut wsum, mut rw, mut cw) = (0.0, 0.0, 0.0);
            let (mut rsum, mut csum) = (0.0, 0.0);
            let mut peak = f64::NEG_INFINITY;
            let mut cells = 0usize;
            while let Some((r, c)) = stack.pop()
            {
                let a = map[r][c];
                let weight = a.max(0.0);
                wsum += weight;
                rw += r as f64 * weight;
                cw += c as f64 * weight;
                rsum += r as f64;
                csum += c as f64;
                if a > peak
                {
                    peak = a;
                }
                cells += 1;
                for dr in -1i64..=1
                {
                    for dc in -1i64..=1
                    {
                        if dr == 0 && dc == 0
                        {
                            continue;
                        }
                        let nr = r as i64 + dr;
                        let nc = c as i64 + dc;
                        if nr < 0 || nc < 0 || nr >= rows as i64 || nc >= cols as i64
                        {
                            continue;
                        }
                        let (nr, nc) = (nr as usize, nc as usize);
                        if mask[nr][nc] && !visited[nr][nc]
                        {
                            visited[nr][nc] = true;
                            stack.push((nr, nc));
                        }
                    }
                }
            }
            let (range, doppler) = if wsum > 0.0
            {
                (rw / wsum, cw / wsum)
            }
            else
            {
                (rsum / cells as f64, csum / cells as f64)
            };
            out.push(Detection {
                range,
                doppler,
                amplitude: peak,
                cells,
            });
        }
    }
    out.sort_by(|a, b| b.amplitude.total_cmp(&a.amplitude));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic LCG producing unit-mean exponential noise (the model
    /// CFAR is designed for), so the statistical false-alarm test is
    /// reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
        }
        fn exponential(&mut self) -> f64 {
            -self.uniform().ln()
        }
    }

    #[test]
    fn ca_cfar_2d_flags_a_point_target_on_a_flat_floor() {
        let mut power = vec![vec![1.0; 21]; 21];
        power[10][10] = 100.0;
        let det = ca_cfar_2d(&power, 3, 1, 0.01);
        assert!(det[10][10], "missed the point target");
        // The flat floor stays below threshold: exactly one detection.
        let total: usize = det.iter().flatten().filter(|&&d| d).count();
        assert_eq!(total, 1);
    }

    #[test]
    fn ca_cfar_2d_holds_the_design_false_alarm_rate() {
        let (train, guard, pfa) = (4usize, 1usize, 0.02);
        let mut rng = Lcg(0x00D0_0D1E);
        let (rows, cols) = (120usize, 120usize);
        let power: Vec<Vec<f64>> = (0..rows)
            .map(|_| (0..cols).map(|_| rng.exponential()).collect())
            .collect();
        let det = ca_cfar_2d(&power, train, guard, pfa);
        let w = train + guard;
        let tested = (rows - 2 * w) * (cols - 2 * w);
        let alarms: usize = det.iter().flatten().filter(|&&d| d).count();
        let empirical = alarms as f64 / tested as f64;
        assert!(
            (empirical - pfa).abs() < 0.01,
            "empirical P_fa {empirical} vs {pfa}"
        );
    }

    #[test]
    fn cluster_finds_two_separated_targets_strongest_first() {
        let (rows, cols) = (10usize, 10usize);
        let mut mask = vec![vec![false; cols]; rows];
        let mut map = vec![vec![0.0; cols]; rows];
        // Blob A around (2.2, 2.6), peak 3.
        for &(r, c, a) in &[(2, 2, 1.0), (2, 3, 3.0), (3, 2, 1.0)]
        {
            mask[r][c] = true;
            map[r][c] = a;
        }
        // Blob B around (7, 7.5), peak 4.
        for &(r, c, a) in &[(7, 7, 4.0), (7, 8, 4.0)]
        {
            mask[r][c] = true;
            map[r][c] = a;
        }
        let dets = cluster_detections(&mask, &map);
        assert_eq!(dets.len(), 2);
        // Strongest peak first: blob B.
        assert_eq!(dets[0].amplitude, 4.0);
        assert!((dets[0].range - 7.0).abs() < 1e-9 && (dets[0].doppler - 7.5).abs() < 1e-9);
        assert_eq!(dets[0].cells, 2);
        // Then blob A: weighted centroid (11/5, 13/5).
        assert_eq!(dets[1].amplitude, 3.0);
        assert!((dets[1].range - 2.2).abs() < 1e-9 && (dets[1].doppler - 2.6).abs() < 1e-9);
        assert_eq!(dets[1].cells, 3);
    }

    #[test]
    fn cluster_merges_diagonally_adjacent_cells() {
        // Two cells touching only at a corner form one 8-connected component.
        let mut mask = vec![vec![false; 4]; 4];
        let map = vec![vec![1.0; 4]; 4];
        mask[1][1] = true;
        mask[2][2] = true;
        let dets = cluster_detections(&mask, &map);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].cells, 2);
        assert!((dets[0].range - 1.5).abs() < 1e-9 && (dets[0].doppler - 1.5).abs() < 1e-9);
    }

    #[test]
    fn ca_cfar_2d_then_cluster_localizes_two_targets() {
        // End-to-end: a flat floor with two well-separated strong cells →
        // 2-D CFAR mask → clustering → two detections at those cells.
        let (rows, cols) = (24usize, 24usize);
        let mut power = vec![vec![1.0; cols]; rows];
        power[6][6] = 80.0;
        power[17][18] = 120.0;
        let mask = ca_cfar_2d(&power, 3, 1, 0.01);
        let dets = cluster_detections(&mask, &power);
        assert_eq!(dets.len(), 2);
        // Strongest first: the (17, 18) target.
        assert!((dets[0].range - 17.0).abs() < 1e-9 && (dets[0].doppler - 18.0).abs() < 1e-9);
        assert!((dets[1].range - 6.0).abs() < 1e-9 && (dets[1].doppler - 6.0).abs() < 1e-9);
    }

    #[test]
    fn detect_guards() {
        // Empty / shape-mismatched clustering → empty.
        assert!(cluster_detections(&[], &[]).is_empty());
        let mask = vec![vec![true; 3]; 2];
        let wrong = vec![vec![0.0; 3]; 3];
        assert!(cluster_detections(&mask, &wrong).is_empty());
        // CFAR guards: too small, and bad P_fa → all false.
        assert!(
            ca_cfar_2d(&vec![vec![1.0; 4]; 4], 3, 1, 0.01)
                .iter()
                .flatten()
                .all(|&d| !d)
        );
        assert!(
            ca_cfar_2d(&vec![vec![1.0; 21]; 21], 3, 1, 0.0)
                .iter()
                .flatten()
                .all(|&d| !d)
        );
        assert!(
            ca_cfar_2d(&vec![vec![1.0; 21]; 21], 0, 1, 0.01)
                .iter()
                .flatten()
                .all(|&d| !d)
        );
    }
}
