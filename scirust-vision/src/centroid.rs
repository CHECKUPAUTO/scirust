//! Sub-pixel spot and star centroiding for EO/IR pointing and tracking.
//!
//! A tracker turns a bright spot — a star, a beacon, a hot target — into a
//! precise angular measurement by locating the spot's *intensity-weighted*
//! centre, which lands between the pixels and so beats the pixel pitch. The
//! centre of gravity `x̄ = Σ x·I / Σ I` (and likewise `ȳ`) is the workhorse:
//! exact on a single pixel, exact on the midpoint of a symmetric blob, and
//! smoothly interpolating in between (two equal pixels at columns 2 and 4 sit at
//! 3.0; a 1:3 split across columns 2 and 3 sits at 2.75).
//!
//! Two refinements matter in practice. A constant background pedestal drags the
//! centre of gravity toward the frame centre; subtracting a threshold and
//! clamping the negatives to zero removes that bias — [`thresholded_centroid`].
//! And a bright interloper elsewhere in the frame corrupts the estimate;
//! restricting the sum to a window around a seed rejects it — [`windowed_centroid`].
//! All three degrade gracefully: an image with no signal returns its geometric
//! centre. Images are `&[f64]` in row-major order with an explicit `width` and
//! `height`. Dependency-free (`std` only).

/// The geometric centre `((width−1)/2, (height−1)/2)` of the grid, the neutral
/// fallback when there is no intensity to weight. A zero extent maps to `0.0`.
fn geometric_center(width: usize, height: usize) -> (f64, f64) {
    let cx = if width == 0
    {
        0.0
    }
    else
    {
        (width - 1) as f64 / 2.0
    };
    let cy = if height == 0
    {
        0.0
    }
    else
    {
        (height - 1) as f64 / 2.0
    };
    (cx, cy)
}

/// Accumulate `(Σ x·w, Σ y·w, Σ w)` over the half-open window
/// `[x0, x1) × [y0, y1)`, where each pixel weight `w` is its intensity minus
/// `pedestal`, with negative (and zero) weights dropped. Out-of-range indices are
/// skipped, so a malformed length can never panic.
fn accumulate(
    image: &[f64],
    width: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    pedestal: f64,
) -> (f64, f64, f64) {
    let mut sum_xi = 0.0;
    let mut sum_yi = 0.0;
    let mut sum_i = 0.0;
    for y in y0..y1
    {
        for x in x0..x1
        {
            let w = match image.get(y * width + x)
            {
                Some(&value) => value - pedestal,
                None => continue,
            };
            if w <= 0.0
            {
                continue;
            }
            sum_xi += x as f64 * w;
            sum_yi += y as f64 * w;
            sum_i += w;
        }
    }
    (sum_xi, sum_yi, sum_i)
}

/// The intensity-weighted centre of gravity `(Σ x·I / Σ I, Σ y·I / Σ I)` of the
/// whole frame — the sub-pixel spot location. A single bright pixel returns its
/// own `(x, y)`; a symmetric blob returns its midpoint. With no intensity
/// (empty or all-zero frame) it returns the geometric centre.
pub fn center_of_gravity(image: &[f64], width: usize, height: usize) -> (f64, f64) {
    if width == 0 || height == 0
    {
        return geometric_center(width, height);
    }
    let (sum_xi, sum_yi, sum_i) = accumulate(image, width, 0, 0, width, height, 0.0);
    if sum_i <= 0.0
    {
        return geometric_center(width, height);
    }
    (sum_xi / sum_i, sum_yi / sum_i)
}

/// The centre of gravity after subtracting `threshold` from every pixel and
/// clamping the negatives to zero — this removes a uniform background pedestal
/// that would otherwise bias the estimate toward the frame centre. Returns the
/// geometric centre when nothing survives the threshold.
pub fn thresholded_centroid(
    image: &[f64],
    width: usize,
    height: usize,
    threshold: f64,
) -> (f64, f64) {
    if width == 0 || height == 0
    {
        return geometric_center(width, height);
    }
    let (sum_xi, sum_yi, sum_i) = accumulate(image, width, 0, 0, width, height, threshold);
    if sum_i <= 0.0
    {
        return geometric_center(width, height);
    }
    (sum_xi / sum_i, sum_yi / sum_i)
}

/// The centre of gravity restricted to the square window of half-width `radius`
/// around `(seed_x, seed_y)`, clamped to the frame — this rejects a bright
/// interloper outside the window. With no intensity inside the window it returns
/// the window's geometric centre (or the frame's if the window is empty).
pub fn windowed_centroid(
    image: &[f64],
    width: usize,
    height: usize,
    seed_x: usize,
    seed_y: usize,
    radius: usize,
) -> (f64, f64) {
    if width == 0 || height == 0
    {
        return geometric_center(width, height);
    }
    let x0 = seed_x.saturating_sub(radius);
    let y0 = seed_y.saturating_sub(radius);
    let x1 = (seed_x + radius + 1).min(width);
    let y1 = (seed_y + radius + 1).min(height);
    let (sum_xi, sum_yi, sum_i) = accumulate(image, width, x0, y0, x1, y1, 0.0);
    if sum_i <= 0.0
    {
        if x0 < x1 && y0 < y1
        {
            return ((x0 + x1 - 1) as f64 / 2.0, (y0 + y1 - 1) as f64 / 2.0);
        }
        return geometric_center(width, height);
    }
    (sum_xi / sum_i, sum_yi / sum_i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_bright_pixel_marks_its_own_location() {
        // One hot pixel at (3, 1) in a 5×5 frame: the centroid is exactly there.
        let mut image = vec![0.0_f64; 25];
        image[5 + 3] = 7.0;
        let (cx, cy) = center_of_gravity(&image, 5, 5);
        assert!((cx - 3.0).abs() < 1e-12);
        assert!((cy - 1.0).abs() < 1e-12);
    }

    #[test]
    fn symmetric_plateau_sits_at_its_center() {
        // A 3×3 plateau of equal intensity centred at (2, 2) in a 5×5 frame.
        let mut image = vec![0.0_f64; 25];
        for y in 1..=3
        {
            for x in 1..=3
            {
                image[y * 5 + x] = 4.0;
            }
        }
        let (cx, cy) = center_of_gravity(&image, 5, 5);
        assert!((cx - 2.0).abs() < 1e-12);
        assert!((cy - 2.0).abs() < 1e-12);
    }

    #[test]
    fn sub_pixel_splits_interpolate_between_pixels() {
        // Two equal pixels at columns 2 and 4 → midpoint 3.0.
        let equal = vec![0.0_f64, 0.0, 1.0, 0.0, 1.0];
        let (cx, cy) = center_of_gravity(&equal, 5, 1);
        assert!((cx - 3.0).abs() < 1e-12);
        assert!(cy.abs() < 1e-12);

        // A 1:3 split across columns 2 and 3 → (2·1 + 3·3)/4 = 2.75.
        let split = vec![0.0_f64, 0.0, 1.0, 3.0, 0.0];
        let (sx, _sy) = center_of_gravity(&split, 5, 1);
        assert!((sx - 2.75).abs() < 1e-12);
    }

    #[test]
    fn centroid_is_invariant_under_intensity_scaling() {
        // Scaling every pixel by a constant leaves the weighted centre unchanged.
        let image = vec![
            0.0_f64, 1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 5.0, 1.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 1.0,
        ];
        let (bx, by) = center_of_gravity(&image, 4, 4);
        let scaled: Vec<f64> = image.iter().map(|&v| v * 9.0).collect();
        let (sx, sy) = center_of_gravity(&scaled, 4, 4);
        assert!((sx - bx).abs() < 1e-12);
        assert!((sy - by).abs() < 1e-12);
    }

    #[test]
    fn thresholding_removes_a_dc_pedestal() {
        // Blob = two hot pixels at (3, 1) and (1, 3); its centre of gravity is (2, 2).
        let mut blob = vec![0.0_f64; 25];
        blob[5 + 3] = 10.0;
        blob[3 * 5 + 1] = 10.0;
        let (bx, by) = center_of_gravity(&blob, 5, 5);

        // Add a uniform DC pedestal of 2.0 everywhere; thresholding it away must
        // recover exactly the blob-only centroid.
        let pedestal = 2.0_f64;
        let with_dc: Vec<f64> = blob.iter().map(|&v| v + pedestal).collect();
        let (tx, ty) = thresholded_centroid(&with_dc, 5, 5, pedestal);
        assert!((tx - bx).abs() < 1e-12);
        assert!((ty - by).abs() < 1e-12);
        assert!((tx - 2.0).abs() < 1e-12);
        assert!((ty - 2.0).abs() < 1e-12);
    }

    #[test]
    fn windowed_centroid_ignores_a_distant_bright_pixel() {
        // Near blob at columns 2,3 of row 2 plus a bright interloper at (8, 8).
        let mut image = vec![0.0_f64; 100];
        image[2 * 10 + 2] = 5.0;
        image[2 * 10 + 3] = 5.0;
        image[8 * 10 + 8] = 100.0;

        // A window around the seed excludes the interloper: centroid = (2.5, 2.0).
        let (wx, wy) = windowed_centroid(&image, 10, 10, 2, 2, 2);
        assert!((wx - 2.5).abs() < 1e-12);
        assert!((wy - 2.0).abs() < 1e-12);

        // The full-frame centroid is dragged far away by the interloper.
        let (fx, _fy) = center_of_gravity(&image, 10, 10);
        assert!(
            fx > 5.0,
            "full-frame centroid {fx} should be pulled by the interloper"
        );
    }

    #[test]
    fn degenerate_inputs_return_the_geometric_center() {
        // An all-zero 4×4 frame falls back to its geometric centre (1.5, 1.5).
        let zero = vec![0.0_f64; 16];
        let (cx, cy) = center_of_gravity(&zero, 4, 4);
        assert!((cx - 1.5).abs() < 1e-12);
        assert!((cy - 1.5).abs() < 1e-12);

        // An empty frame maps to the origin.
        let (ex, ey) = center_of_gravity(&[], 0, 0);
        assert!(ex.abs() < 1e-12);
        assert!(ey.abs() < 1e-12);

        // A window over dead pixels returns the window centre, not a NaN.
        let (wx, wy) = windowed_centroid(&zero, 4, 4, 1, 1, 1);
        assert!((wx - 1.0).abs() < 1e-12);
        assert!((wy - 1.0).abs() < 1e-12);
    }
}
