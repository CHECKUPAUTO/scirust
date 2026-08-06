//! Deterministic 3-D region labelling, overlap metrics and topology events.
//!
//! Ported from the ITD research simulator's Mission 8 structural machinery
//! (`itd_research.mission8.vortex_regions`), validated against the reference
//! implementation as an oracle (`tests/region3.rs`) — the crate's standing
//! acceptance discipline.
//!
//! ## What it provides
//!
//! 1. **Connected components** ([`label_components`]) of a 3-D boolean mask
//!    under 6-connectivity, found by a raster-scan-seeded, explicit-stack flood
//!    fill — no recursion, no hashing, so labelling order is bit-reproducible
//!    across platforms. `periodic = true` treats every axis as wrapping (a
//!    torus): a structure straddling e.g. `x = 0` / `x = nx − 1` is one
//!    component. That is physically correct for a genuinely periodic simulation
//!    domain and **wrong for an arbitrary sub-window cutout** of a larger
//!    domain, whose edges are not physical boundaries — callers must match the
//!    flag to the provenance of the data (the Mission 8 wraparound lesson).
//! 2. **Region records** ([`Labeling::records`]) — per-component volume and
//!    cell-index centroid, with a minimum-size floor to reject derivative-noise
//!    speckle. The centroid of a component that wraps across a periodic seam is
//!    only approximate (a naive index mean lands mid-domain); volume is exact.
//! 3. **Overlap metrics** ([`iou`], [`dice`], [`max_nearest_distance`],
//!    [`centroid_distance`]) between masks. Where the reference returns `NaN`
//!    for an undefined metric (empty union, empty operand) this port returns a
//!    typed absence (`Ok(None)`) instead — never a silent sentinel.
//! 4. **Topology events** ([`detect_topology_events`]) — persistent changes of
//!    the per-frame component count ([`core_count_series`]). A change counts
//!    only when the *pre*-change count was already stable for `persistence`
//!    frames **and** the post-change count stays stable for `persistence`
//!    frames. Requiring only the post-window lets a one-frame blip that reverts
//!    (`2, 2, 1, 2, 2`) fire a spurious event at the recovery point — a real
//!    bug found and fixed in the reference during Mission 8, pinned here by the
//!    oracle fixture.

use crate::error::{ItdError, Result};
use crate::field3::Field3;

/// How a scalar field is compared against the threshold to build a mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    /// Select cells with `value > threshold` (e.g. Q-criterion `Q > 0`).
    Greater,
    /// Select cells with `value < threshold` (e.g. λ₂ `< 0`).
    Less,
}

/// A dense row-major 3-D boolean mask with the same axis convention as
/// [`Field3`] (`z` outermost, `x` innermost).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mask3 {
    nz: usize,
    ny: usize,
    nx: usize,
    data: Vec<bool>,
}

impl Mask3 {
    /// Builds a mask from row-major data. Fails if `data.len() != nz * ny * nx`
    /// or any dimension is zero.
    ///
    /// # Errors
    ///
    /// [`ItdError::ShapeMismatch`] on a zero dimension or a length that does
    /// not match the declared shape.
    pub fn from_vec(nz: usize, ny: usize, nx: usize, data: Vec<bool>) -> Result<Self> {
        if nz == 0 || ny == 0 || nx == 0
        {
            return Err(ItdError::ShapeMismatch(
                "mask dimensions must be non-zero".into(),
            ));
        }
        let expected = nz * ny * nx;
        if data.len() != expected
        {
            return Err(ItdError::ShapeMismatch(format!(
                "expected {expected} values for {nz}x{ny}x{nx}, got {}",
                data.len()
            )));
        }
        Ok(Self { nz, ny, nx, data })
    }

    /// The `(nz, ny, nx)` shape.
    #[inline]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.nz, self.ny, self.nx)
    }

    /// Value at plane `k`, row `i`, column `j`.
    #[inline]
    pub fn get(&self, k: usize, i: usize, j: usize) -> bool {
        self.data[(k * self.ny + i) * self.nx + j]
    }

    /// The underlying row-major data.
    #[inline]
    pub fn as_slice(&self) -> &[bool] {
        &self.data
    }

    /// Number of selected (`true`) cells.
    pub fn count(&self) -> usize {
        self.data.iter().filter(|&&v| v).count()
    }
}

/// Thresholds a finite scalar field into a mask.
///
/// The reference silently treats non-finite cells as background (`NaN > t` is
/// false); this port rejects non-finite fields up front, matching the
/// validation discipline the rest of the crate applies.
///
/// # Errors
///
/// [`ItdError::NonFinite`] if the field contains a non-finite value, or if
/// `threshold` is non-finite.
pub fn threshold_mask(field: &Field3, threshold: f64, comparison: Comparison) -> Result<Mask3> {
    if !threshold.is_finite()
    {
        return Err(ItdError::NonFinite("threshold must be finite".into()));
    }
    if !field.all_finite()
    {
        return Err(ItdError::NonFinite(
            "field passed to threshold_mask contains a non-finite value".into(),
        ));
    }
    let (nz, ny, nx) = field.shape();
    let data = field
        .as_slice()
        .iter()
        .map(|&v| match comparison
        {
            Comparison::Greater => v > threshold,
            Comparison::Less => v < threshold,
        })
        .collect();
    Mask3::from_vec(nz, ny, nx, data)
}

/// The result of labelling a mask: a dense label field plus component sizes.
///
/// Label `0` is background; components are numbered `1..=component_count()` in
/// raster-scan seeding order (`z` outermost, `x` innermost), which makes the
/// numbering deterministic and identical to the reference implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Labeling {
    nz: usize,
    ny: usize,
    nx: usize,
    labels: Vec<usize>,
    sizes: Vec<usize>,
}

impl Labeling {
    /// The `(nz, ny, nx)` shape.
    #[inline]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.nz, self.ny, self.nx)
    }

    /// The dense row-major label field (`0` = background).
    #[inline]
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Component sizes, indexed by `label - 1`.
    #[inline]
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// Number of connected components (before any size floor).
    #[inline]
    pub fn component_count(&self) -> usize {
        self.sizes.len()
    }

    /// Per-component volume and cell-index centroid, keeping only components
    /// with at least `min_cells` cells.
    ///
    /// The centroid is the arithmetic mean of `(z, y, x)` cell indices. Index
    /// sums are exact in `f64` for any realistic grid, so centroids match the
    /// reference bit-for-bit. For a component wrapping across a periodic seam
    /// the naive mean is only approximate (documented reference behaviour).
    pub fn records(&self, min_cells: usize) -> Vec<RegionRecord> {
        let component_count = self.sizes.len();
        let mut sums = vec![[0.0_f64; 3]; component_count];
        for (offset, &label) in self.labels.iter().enumerate()
        {
            if label == 0
            {
                continue;
            }
            let j = offset % self.nx;
            let i = (offset / self.nx) % self.ny;
            let k = offset / (self.nx * self.ny);
            let sum = &mut sums[label - 1];
            sum[0] += k as f64;
            sum[1] += i as f64;
            sum[2] += j as f64;
        }
        let mut records = Vec::new();
        for (index, &size) in self.sizes.iter().enumerate()
        {
            if size < min_cells
            {
                continue;
            }
            let n = size as f64;
            records.push(RegionRecord {
                label: index + 1,
                volume_cells: size,
                centroid: [sums[index][0] / n, sums[index][1] / n, sums[index][2] / n],
            });
        }
        records
    }
}

/// One connected region: its label, cell count and `(z, y, x)` index centroid.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionRecord {
    /// The component's label in the [`Labeling`] (1-based).
    pub label: usize,
    /// Number of cells in the component.
    pub volume_cells: usize,
    /// Arithmetic mean of the component's `(z, y, x)` cell indices.
    pub centroid: [f64; 3],
}

/// Labels the 6-connected components of a mask.
///
/// A raster scan (`z` outermost, `x` innermost) seeds components in
/// deterministic order; an explicit stack performs the flood fill (no
/// recursion, no hashing), so the result is bit-reproducible. With
/// `periodic = true` every axis wraps (see the module docs for when that is —
/// and is not — physically correct).
pub fn label_components(mask: &Mask3, periodic: bool) -> Labeling {
    let (nz, ny, nx) = mask.shape();
    let mut labels = vec![0_usize; nz * ny * nx];
    let mut sizes: Vec<usize> = Vec::new();
    let mut stack: Vec<(usize, usize, usize)> = Vec::new();
    let offsets: [(isize, isize, isize); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];
    let flat = |k: usize, i: usize, j: usize| (k * ny + i) * nx + j;

    for k0 in 0..nz
    {
        for i0 in 0..ny
        {
            for j0 in 0..nx
            {
                let seed = flat(k0, i0, j0);
                if !mask.as_slice()[seed] || labels[seed] != 0
                {
                    continue;
                }
                let current = sizes.len() + 1;
                let mut size = 0_usize;
                labels[seed] = current;
                stack.push((k0, i0, j0));
                while let Some((k, i, j)) = stack.pop()
                {
                    size += 1;
                    for &(dk, di, dj) in &offsets
                    {
                        let (mut kk, mut ii, mut jj) =
                            (k as isize + dk, i as isize + di, j as isize + dj);
                        if periodic
                        {
                            kk = kk.rem_euclid(nz as isize);
                            ii = ii.rem_euclid(ny as isize);
                            jj = jj.rem_euclid(nx as isize);
                        }
                        else if kk < 0
                            || ii < 0
                            || jj < 0
                            || kk >= nz as isize
                            || ii >= ny as isize
                            || jj >= nx as isize
                        {
                            continue;
                        }
                        let neighbor = flat(kk as usize, ii as usize, jj as usize);
                        if mask.as_slice()[neighbor] && labels[neighbor] == 0
                        {
                            labels[neighbor] = current;
                            stack.push((kk as usize, ii as usize, jj as usize));
                        }
                    }
                }
                sizes.push(size);
            }
        }
    }
    Labeling {
        nz,
        ny,
        nx,
        labels,
        sizes,
    }
}

/// Thresholds a field, labels its components and returns the size-filtered
/// region records alongside the full labelling.
///
/// # Errors
///
/// Propagates [`threshold_mask`]'s [`ItdError::NonFinite`] validation.
pub fn detect_regions(
    field: &Field3,
    threshold: f64,
    comparison: Comparison,
    min_cells: usize,
    periodic: bool,
) -> Result<(Labeling, Vec<RegionRecord>)> {
    let mask = threshold_mask(field, threshold, comparison)?;
    let labeling = label_components(&mask, periodic);
    let records = labeling.records(min_cells);
    Ok((labeling, records))
}

fn require_same_shape(a: &Mask3, b: &Mask3) -> Result<()> {
    if a.shape() != b.shape()
    {
        return Err(ItdError::ShapeMismatch(format!(
            "mask shapes differ: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }
    Ok(())
}

/// Intersection-over-union of two masks; `Ok(None)` when both are empty
/// (the reference returns `NaN` there).
///
/// # Errors
///
/// [`ItdError::ShapeMismatch`] when the masks' shapes differ.
pub fn iou(a: &Mask3, b: &Mask3) -> Result<Option<f64>> {
    require_same_shape(a, b)?;
    let mut intersection = 0_usize;
    let mut union = 0_usize;
    for (&va, &vb) in a.as_slice().iter().zip(b.as_slice())
    {
        if va && vb
        {
            intersection += 1;
        }
        if va || vb
        {
            union += 1;
        }
    }
    if union == 0
    {
        return Ok(None);
    }
    Ok(Some(intersection as f64 / union as f64))
}

/// Dice coefficient of two masks; `Ok(None)` when both are empty (the
/// reference returns `NaN` there).
///
/// # Errors
///
/// [`ItdError::ShapeMismatch`] when the masks' shapes differ.
pub fn dice(a: &Mask3, b: &Mask3) -> Result<Option<f64>> {
    require_same_shape(a, b)?;
    let mut intersection = 0_usize;
    for (&va, &vb) in a.as_slice().iter().zip(b.as_slice())
    {
        if va && vb
        {
            intersection += 1;
        }
    }
    let denominator = a.count() + b.count();
    if denominator == 0
    {
        return Ok(None);
    }
    Ok(Some(2.0 * intersection as f64 / denominator as f64))
}

/// Euclidean distance between two `(z, y, x)` centroids.
pub fn centroid_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dz = a[0] - b[0];
    let dy = a[1] - b[1];
    let dx = a[2] - b[2];
    (dz * dz + dy * dy + dx * dx).sqrt()
}

/// Symmetrised Hausdorff-like maximum nearest-cell distance between two masks;
/// `Ok(None)` when either mask is empty (the reference returns `NaN` there).
///
/// `O(n_a · n_b)` on selected-cell coordinates — intended for the small oracle
/// and diagnostic masks this module targets, never for raw full-resolution
/// volumes (the reference carries the same caveat).
///
/// # Errors
///
/// [`ItdError::ShapeMismatch`] when the masks' shapes differ.
pub fn max_nearest_distance(a: &Mask3, b: &Mask3) -> Result<Option<f64>> {
    require_same_shape(a, b)?;
    let cells = |mask: &Mask3| -> Vec<[f64; 3]> {
        let (_, ny, nx) = mask.shape();
        mask.as_slice()
            .iter()
            .enumerate()
            .filter(|(_, &v)| v)
            .map(|(offset, _)| {
                let j = offset % nx;
                let i = (offset / nx) % ny;
                let k = offset / (nx * ny);
                [k as f64, i as f64, j as f64]
            })
            .collect()
    };
    let pa = cells(a);
    let pb = cells(b);
    if pa.is_empty() || pb.is_empty()
    {
        return Ok(None);
    }
    let directed = |p: &[[f64; 3]], q: &[[f64; 3]]| -> f64 {
        let mut worst = 0.0_f64;
        for cp in p
        {
            let mut nearest = f64::INFINITY;
            for cq in q
            {
                let d = centroid_distance(*cp, *cq);
                if d < nearest
                {
                    nearest = d;
                }
            }
            if nearest > worst
            {
                worst = nearest;
            }
        }
        worst
    };
    Ok(Some(directed(&pa, &pb).max(directed(&pb, &pa))))
}

/// Per-frame component count (after the size floor) of a field sequence — the
/// ITD-independent structural time series topology events are detected on.
///
/// # Errors
///
/// Propagates [`detect_regions`]'s validation for each frame.
pub fn core_count_series(
    fields: &[Field3],
    threshold: f64,
    comparison: Comparison,
    min_cells: usize,
    periodic: bool,
) -> Result<Vec<usize>> {
    let mut counts = Vec::with_capacity(fields.len());
    for field in fields
    {
        let (_, records) = detect_regions(field, threshold, comparison, min_cells, periodic)?;
        counts.push(records.len());
    }
    Ok(counts)
}

/// The direction of a persistent component-count change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyEventKind {
    /// The count decreased (structures merged).
    CoreMerger,
    /// The count increased (a structure split).
    CoreSplit,
}

/// One persistent topology-change event in a component-count series.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyEvent {
    /// Merger or split.
    pub kind: TopologyEventKind,
    /// First frame holding the new count.
    pub frame_index: usize,
    /// The count before the change.
    pub from_count: usize,
    /// The count after the change.
    pub to_count: usize,
}

/// Detects frames where the component count changes and the change persists.
///
/// A change at frame `i` is an event only when the previous count was already
/// stable for `persistence` frames (`counts[i-persistence..i]`, clipped at the
/// start) **and** the new count stays stable for `persistence` frames
/// (`counts[i..i+persistence]`, clipped at the end). The pre-window condition
/// is what rejects the recovery edge of a transient blip — `2, 2, 1, 2, 2`
/// must produce no event at all (oracle-pinned regression case).
pub fn detect_topology_events(counts: &[usize], persistence: usize) -> Vec<TopologyEvent> {
    let n = counts.len();
    let mut events = Vec::new();
    for i in 1..n
    {
        if counts[i] == counts[i - 1]
        {
            continue;
        }
        let post = &counts[i..n.min(i + persistence)];
        let pre = &counts[i.saturating_sub(persistence)..i];
        let post_stable = post.iter().all(|&c| c == counts[i]);
        let pre_stable = pre.iter().all(|&c| c == counts[i - 1]);
        if post_stable && pre_stable
        {
            events.push(TopologyEvent {
                kind: if counts[i] < counts[i - 1]
                {
                    TopologyEventKind::CoreMerger
                }
                else
                {
                    TopologyEventKind::CoreSplit
                },
                frame_index: i,
                from_count: counts[i - 1],
                to_count: counts[i],
            });
        }
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seam_mask() -> Mask3 {
        // A single cell at x = 0 and one at x = nx - 1 on the same (z, y) line:
        // two bounded components, one periodic component.
        let (nz, ny, nx) = (3, 3, 4);
        let mut data = vec![false; nz * ny * nx];
        data[(ny + 1) * nx] = true;
        data[(ny + 1) * nx + (nx - 1)] = true;
        Mask3::from_vec(nz, ny, nx, data).expect("valid mask")
    }

    #[test]
    fn periodic_wraparound_joins_what_bounded_splits() {
        let mask = seam_mask();
        assert_eq!(label_components(&mask, false).component_count(), 2);
        assert_eq!(label_components(&mask, true).component_count(), 1);
    }

    #[test]
    fn labelling_is_deterministic() {
        let mask = seam_mask();
        let first = label_components(&mask, true);
        let second = label_components(&mask, true);
        assert_eq!(first, second);
    }

    #[test]
    fn records_apply_the_min_cells_floor() {
        let mask = seam_mask();
        let labeling = label_components(&mask, false);
        assert_eq!(labeling.records(1).len(), 2);
        assert_eq!(labeling.records(2).len(), 0);
    }

    #[test]
    fn threshold_mask_rejects_non_finite_input() {
        let mut field = Field3::zeros(2, 2, 2);
        *field.get_mut(0, 1, 1) = f64::INFINITY;
        assert!(matches!(
            threshold_mask(&field, 0.5, Comparison::Greater),
            Err(ItdError::NonFinite(_))
        ));
        let ok = Field3::zeros(2, 2, 2);
        assert!(matches!(
            threshold_mask(&ok, f64::NAN, Comparison::Greater),
            Err(ItdError::NonFinite(_))
        ));
    }

    #[test]
    fn overlap_metrics_report_typed_absence_not_a_sentinel() {
        let empty = Mask3::from_vec(2, 2, 2, vec![false; 8]).expect("valid mask");
        assert_eq!(iou(&empty, &empty), Ok(None));
        assert_eq!(dice(&empty, &empty), Ok(None));
        assert_eq!(max_nearest_distance(&empty, &empty), Ok(None));
    }

    #[test]
    fn overlap_metrics_reject_shape_mismatch() {
        let a = Mask3::from_vec(2, 2, 2, vec![true; 8]).expect("valid mask");
        let b = Mask3::from_vec(2, 2, 3, vec![true; 12]).expect("valid mask");
        assert!(matches!(iou(&a, &b), Err(ItdError::ShapeMismatch(_))));
        assert!(matches!(dice(&a, &b), Err(ItdError::ShapeMismatch(_))));
        assert!(matches!(
            max_nearest_distance(&a, &b),
            Err(ItdError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn identical_masks_have_perfect_overlap() {
        let mask = seam_mask();
        assert_eq!(iou(&mask, &mask), Ok(Some(1.0)));
        assert_eq!(dice(&mask, &mask), Ok(Some(1.0)));
        assert_eq!(max_nearest_distance(&mask, &mask), Ok(Some(0.0)));
    }

    #[test]
    fn transient_blip_recovery_fires_no_event() {
        assert!(detect_topology_events(&[2, 2, 1, 2, 2], 2).is_empty());
    }

    #[test]
    fn sustained_change_fires_exactly_one_event() {
        let events = detect_topology_events(&[2, 2, 2, 1, 1, 1], 2);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, TopologyEventKind::CoreMerger);
        assert_eq!(events[0].frame_index, 3);
        assert_eq!(events[0].from_count, 2);
        assert_eq!(events[0].to_count, 1);
    }

    #[test]
    fn split_direction_is_detected_too() {
        let events = detect_topology_events(&[1, 1, 1, 3, 3, 3], 2);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, TopologyEventKind::CoreSplit);
    }
}
