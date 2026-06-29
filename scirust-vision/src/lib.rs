//! Computer vision patterns: CNN layers, object detection, image classification, segmentation.

use serde::{Deserialize, Serialize};

// ─── Image Representation ───────────────────────────────────────────────────

/// A 2D grayscale image (height × width).
#[derive(Debug, Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f64>,
}

impl Image {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    pub fn from_vec(width: usize, height: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), width * height);
        Self {
            width,
            height,
            data,
        }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, val: f64) {
        self.data[y * self.width + x] = val;
    }

    pub fn mean(&self) -> f64 {
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    pub fn std(&self) -> f64 {
        let m = self.mean();
        let var = self.data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / self.data.len() as f64;
        var.sqrt()
    }
}

/// A 2D convolution kernel.
#[derive(Debug, Clone)]
pub struct Kernel {
    pub size: usize,
    pub data: Vec<f64>,
}

impl Kernel {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            data: vec![0.0; size * size],
        }
    }

    pub fn from_vec(size: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), size * size);
        Self { size, data }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.size + x]
    }

    /// Predefined edge detection kernel (Sobel X).
    pub fn sobel_x() -> Self {
        Self::from_vec(3, vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0])
    }

    /// Predefined edge detection kernel (Sobel Y).
    pub fn sobel_y() -> Self {
        Self::from_vec(3, vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0])
    }

    /// Gaussian blur kernel.
    pub fn gaussian(sigma: f64) -> Self {
        let size = (6.0 * sigma).ceil() as usize | 1; // ensure odd
        let mut data = vec![0.0; size * size];
        let half = size / 2;
        let mut sum = 0.0;
        for y in 0..size
        {
            for x in 0..size
            {
                let dx = x as f64 - half as f64;
                let dy = y as f64 - half as f64;
                let val = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                data[y * size + x] = val;
                sum += val;
            }
        }
        for v in &mut data
        {
            *v /= sum;
        }
        Self { size, data }
    }

    /// Laplacian kernel for edge detection.
    pub fn laplacian() -> Self {
        Self::from_vec(3, vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0])
    }
}

// ─── Convolution ────────────────────────────────────────────────────────────

/// Apply 2D convolution with a kernel to an image.
pub fn convolve2d(image: &Image, kernel: &Kernel) -> Image {
    let mut out = Image::new(image.width, image.height);
    let half = kernel.size / 2;

    for y in 0..image.height
    {
        for x in 0..image.width
        {
            let mut sum = 0.0;
            for ky in 0..kernel.size
            {
                for kx in 0..kernel.size
                {
                    let ix = x as isize + kx as isize - half as isize;
                    let iy = y as isize + ky as isize - half as isize;
                    if ix >= 0
                        && iy >= 0
                        && (ix as usize) < image.width
                        && (iy as usize) < image.height
                    {
                        sum += image.get(ix as usize, iy as usize) * kernel.get(kx, ky);
                    }
                }
            }
            out.set(x, y, sum);
        }
    }
    out
}

/// Apply max pooling (2×2 by default).
pub fn max_pool2d(image: &Image, pool_size: usize) -> Image {
    let new_w = image.width / pool_size;
    let new_h = image.height / pool_size;
    let mut out = Image::new(new_w, new_h);

    for y in 0..new_h
    {
        for x in 0..new_w
        {
            let mut max_val = f64::NEG_INFINITY;
            for py in 0..pool_size
            {
                for px in 0..pool_size
                {
                    let val = image.get(x * pool_size + px, y * pool_size + py);
                    if val > max_val
                    {
                        max_val = val;
                    }
                }
            }
            out.set(x, y, max_val);
        }
    }
    out
}

/// Apply average pooling.
pub fn avg_pool2d(image: &Image, pool_size: usize) -> Image {
    let new_w = image.width / pool_size;
    let new_h = image.height / pool_size;
    let mut out = Image::new(new_w, new_h);
    let area = (pool_size * pool_size) as f64;

    for y in 0..new_h
    {
        for x in 0..new_w
        {
            let mut sum = 0.0;
            for py in 0..pool_size
            {
                for px in 0..pool_size
                {
                    sum += image.get(x * pool_size + px, y * pool_size + py);
                }
            }
            out.set(x, y, sum / area);
        }
    }
    out
}

// ─── Activation Functions ───────────────────────────────────────────────────

/// ReLU activation on an image.
pub fn relu(image: &Image) -> Image {
    Image::from_vec(
        image.width,
        image.height,
        image.data.iter().map(|&x| x.max(0.0)).collect(),
    )
}

/// Sigmoid activation on an image.
pub fn sigmoid(image: &Image) -> Image {
    Image::from_vec(
        image.width,
        image.height,
        image
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect(),
    )
}

/// Softmax on a 1D feature vector.
pub fn softmax(values: &[f64]) -> Vec<f64> {
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = values.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
}

// ─── Feature Extraction ─────────────────────────────────────────────────────

/// Histogram of Oriented Gradients (HOG) descriptor.
pub fn hog(image: &Image, cell_size: usize, n_bins: usize) -> Vec<f64> {
    let gx = convolve2d(image, &Kernel::sobel_x());
    let gy = convolve2d(image, &Kernel::sobel_y());

    let cells_x = image.width / cell_size;
    let cells_y = image.height / cell_size;
    let mut descriptor = Vec::with_capacity(cells_x * cells_y * n_bins);

    for cy in 0..cells_y
    {
        for cx in 0..cells_x
        {
            let mut histogram = vec![0.0; n_bins];
            for y in 0..cell_size
            {
                for x in 0..cell_size
                {
                    let px = cx * cell_size + x;
                    let py = cy * cell_size + y;
                    let mag = (gx.get(px, py).powi(2) + gy.get(px, py).powi(2)).sqrt();
                    let angle = gy.get(px, py).atan2(gx.get(px, py)) + std::f64::consts::PI;
                    let bin = ((angle / (2.0 * std::f64::consts::PI) * n_bins as f64) as usize)
                        .min(n_bins - 1);
                    histogram[bin] += mag;
                }
            }
            descriptor.extend_from_slice(&histogram);
        }
    }

    // L2 normalize
    let norm: f64 = descriptor.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > f64::EPSILON
    {
        for v in &mut descriptor
        {
            *v /= norm;
        }
    }
    descriptor
}

/// Local Binary Pattern (LBP) feature for a pixel.
pub fn lbp(image: &Image, x: usize, y: usize) -> u8 {
    let center = image.get(x, y);
    let mut pattern: u8 = 0;
    let neighbors = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    for (i, &(dx, dy)) in neighbors.iter().enumerate()
    {
        let nx = x as isize + dx;
        let ny = y as isize + dy;
        if nx >= 0
            && nx < image.width as isize
            && ny >= 0
            && ny < image.height as isize
            && image.get(nx as usize, ny as usize) >= center
        {
            pattern |= 1 << i;
        }
    }
    pattern
}

/// Compute LBP histogram for an image region.
pub fn lbp_histogram(image: &Image, x0: usize, y0: usize, w: usize, h: usize) -> Vec<f64> {
    let mut hist = vec![0.0; 256];
    for y in y0..(y0 + h).min(image.height)
    {
        for x in x0..(x0 + w).min(image.width)
        {
            let code = lbp(image, x, y) as usize;
            hist[code] += 1.0;
        }
    }
    let total: f64 = hist.iter().sum();
    if total > 0.0
    {
        for v in &mut hist
        {
            *v /= total;
        }
    }
    hist
}

/// Haar-like feature (simple rectangle features for face detection).
pub fn haar_feature(
    image: &Image,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    feature_type: HaarFeature,
) -> f64 {
    match feature_type
    {
        HaarFeature::EdgeHorizontal =>
        {
            let top = region_sum(image, x, y, w, h / 2);
            let bottom = region_sum(image, x, y + h / 2, w, h / 2);
            top - bottom
        },
        HaarFeature::EdgeVertical =>
        {
            let left = region_sum(image, x, y, w / 2, h);
            let right = region_sum(image, x + w / 2, y, w / 2, h);
            left - right
        },
        HaarFeature::LineHorizontal =>
        {
            let top = region_sum(image, x, y, w, h / 3);
            let mid = region_sum(image, x, y + h / 3, w, h / 3);
            let bot = region_sum(image, x, y + 2 * h / 3, w, h / 3);
            top + bot - mid
        },
        HaarFeature::LineVertical =>
        {
            let left = region_sum(image, x, y, w / 3, h);
            let mid = region_sum(image, x + w / 3, y, w / 3, h);
            let right = region_sum(image, x + 2 * w / 3, y, w / 3, h);
            left + right - mid
        },
        HaarFeature::FourRectangle =>
        {
            let tl = region_sum(image, x, y, w / 2, h / 2);
            let tr = region_sum(image, x + w / 2, y, w / 2, h / 2);
            let bl = region_sum(image, x, y + h / 2, w / 2, h / 2);
            let br = region_sum(image, x + w / 2, y + h / 2, w / 2, h / 2);
            tl + br - tr - bl
        },
    }
}

#[derive(Debug, Clone, Copy)]
pub enum HaarFeature {
    EdgeHorizontal,
    EdgeVertical,
    LineHorizontal,
    LineVertical,
    FourRectangle,
}

fn region_sum(image: &Image, x: usize, y: usize, w: usize, h: usize) -> f64 {
    let mut sum = 0.0;
    for dy in 0..h
    {
        for dx in 0..w
        {
            let px = x + dx;
            let py = y + dy;
            if px < image.width && py < image.height
            {
                sum += image.get(px, py);
            }
        }
    }
    sum
}

// ─── Object Detection ───────────────────────────────────────────────────────

/// Bounding box for detected objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub confidence: f64,
    pub class_id: usize,
    pub class_name: String,
}

impl BoundingBox {
    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    /// Intersection over Union (IoU) with another bounding box.
    pub fn iou(&self, other: &BoundingBox) -> f64 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        if x2 <= x1 || y2 <= y1
        {
            return 0.0;
        }

        let intersection = (x2 - x1) * (y2 - y1);
        let union = self.area() + other.area() - intersection;
        intersection / union
    }
}

/// Non-Maximum Suppression to remove overlapping detections.
pub fn nms(boxes: &mut Vec<BoundingBox>, iou_threshold: f64) {
    boxes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    let mut keep = vec![true; boxes.len()];

    for i in 0..boxes.len()
    {
        if !keep[i]
        {
            continue;
        }
        for j in (i + 1)..boxes.len()
        {
            if keep[j] && boxes[i].iou(&boxes[j]) > iou_threshold
            {
                keep[j] = false;
            }
        }
    }

    let mut idx = 0;
    boxes.retain(|_| {
        let keep_it = keep[idx];
        idx += 1;
        keep_it
    });
}

/// Template matching using sum of squared differences (SSD).
pub fn template_match(image: &Image, template: &Image) -> Vec<(usize, usize, f64)> {
    let mut results = Vec::new();
    let tw = template.width;
    let th = template.height;

    if tw > image.width || th > image.height
    {
        return results;
    }

    for y in 0..=(image.height - th)
    {
        for x in 0..=(image.width - tw)
        {
            let mut ssd = 0.0;
            for ty in 0..th
            {
                for tx in 0..tw
                {
                    let diff = image.get(x + tx, y + ty) - template.get(tx, ty);
                    ssd += diff * diff;
                }
            }
            results.push((x, y, ssd));
        }
    }

    results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    results
}

/// Match template and return best match location.
pub fn match_template_best(image: &Image, template: &Image) -> Option<(usize, usize, f64)> {
    template_match(image, template).into_iter().next()
}

// ─── Morphological Operations ───────────────────────────────────────────────

/// Apply dilation to a binary image.
pub fn dilate(image: &Image, size: usize) -> Image {
    let mut out = Image::new(image.width, image.height);
    let half = size / 2;

    for y in 0..image.height
    {
        for x in 0..image.width
        {
            let mut max_val = 0.0;
            for ky in 0..size
            {
                for kx in 0..size
                {
                    let ix = x as isize + kx as isize - half as isize;
                    let iy = y as isize + ky as isize - half as isize;
                    if ix >= 0
                        && iy >= 0
                        && (ix as usize) < image.width
                        && (iy as usize) < image.height
                    {
                        max_val = max_val.max(image.get(ix as usize, iy as usize));
                    }
                }
            }
            out.set(x, y, max_val);
        }
    }
    out
}

/// Apply erosion to a binary image.
pub fn erode(image: &Image, size: usize) -> Image {
    let mut out = Image::new(image.width, image.height);
    let half = size / 2;

    for y in 0..image.height
    {
        for x in 0..image.width
        {
            let mut min_val = 1.0;
            for ky in 0..size
            {
                for kx in 0..size
                {
                    let ix = x as isize + kx as isize - half as isize;
                    let iy = y as isize + ky as isize - half as isize;
                    if ix >= 0
                        && iy >= 0
                        && (ix as usize) < image.width
                        && (iy as usize) < image.height
                    {
                        min_val = min_val.min(image.get(ix as usize, iy as usize));
                    }
                    else
                    {
                        min_val = 0.0; // boundary is considered background
                    }
                }
            }
            out.set(x, y, min_val);
        }
    }
    out
}

// ─── Image Segmentation ─────────────────────────────────────────────────────

/// Simple threshold-based segmentation.
pub fn threshold(image: &Image, threshold_val: f64) -> Image {
    Image::from_vec(
        image.width,
        image.height,
        image
            .data
            .iter()
            .map(|&x| if x >= threshold_val { 1.0 } else { 0.0 })
            .collect(),
    )
}

/// Otsu's method for automatic threshold selection.
#[allow(clippy::needless_range_loop)]
pub fn otsu_threshold(image: &Image) -> f64 {
    let mut hist = vec![0u64; 256];
    let min = image.data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = image.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range < f64::EPSILON
    {
        return min;
    }

    for &v in &image.data
    {
        let bin = ((v - min) / range * 255.0) as usize;
        hist[bin.min(255)] += 1;
    }

    let total = image.data.len() as f64;
    let sum_all: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();
    let mut sum_bg = 0.0;
    let mut weight_bg = 0.0;
    let mut max_variance = 0.0;
    let mut best_threshold = 0;

    for t in 0..256
    {
        weight_bg += hist[t] as f64;
        if weight_bg < f64::EPSILON
        {
            continue;
        }
        let weight_fg = total - weight_bg;
        if weight_fg < f64::EPSILON
        {
            break;
        }

        sum_bg += t as f64 * hist[t] as f64;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_all - sum_bg) / weight_fg;
        let variance = weight_bg * weight_fg * (mean_bg - mean_fg).powi(2);

        if variance > max_variance
        {
            max_variance = variance;
            best_threshold = t;
        }
    }

    min + (best_threshold as f64 / 255.0) * range
}

/// Connected component labeling (4-connectivity).
#[allow(clippy::needless_range_loop)]
pub fn connected_components(binary: &Image) -> Vec<Vec<(usize, usize)>> {
    let w = binary.width;
    let h = binary.height;
    let mut labels = vec![vec![0i32; w]; h];
    let mut current_label = 0;

    for y in 0..h
    {
        for x in 0..w
        {
            if binary.get(x, y) > 0.5 && labels[y][x] == 0
            {
                current_label += 1;
                let mut stack = vec![(x, y)];
                while let Some((cx, cy)) = stack.pop()
                {
                    if cx < w && cy < h && binary.get(cx, cy) > 0.5 && labels[cy][cx] == 0
                    {
                        labels[cy][cx] = current_label;
                        if cx > 0
                        {
                            stack.push((cx - 1, cy));
                        }
                        if cx + 1 < w
                        {
                            stack.push((cx + 1, cy));
                        }
                        if cy > 0
                        {
                            stack.push((cx, cy - 1));
                        }
                        if cy + 1 < h
                        {
                            stack.push((cx, cy + 1));
                        }
                    }
                }
            }
        }
    }

    let mut components = vec![Vec::new(); current_label as usize + 1];
    for y in 0..h
    {
        for x in 0..w
        {
            let label = labels[y][x] as usize;
            if label > 0
            {
                components[label].push((x, y));
            }
        }
    }
    components.remove(0); // remove label 0
    components
}

/// Flood fill from a seed point.
pub fn flood_fill(image: &Image, seed_x: usize, seed_y: usize, fill_value: f64) -> Image {
    let mut out = image.clone();
    let target = image.get(seed_x, seed_y);
    let w = image.width;
    let h = image.height;

    if (target - fill_value).abs() < f64::EPSILON
    {
        return out;
    }

    let mut stack = vec![(seed_x, seed_y)];
    let mut visited = vec![false; w * h];

    while let Some((x, y)) = stack.pop()
    {
        let idx = y * w + x;
        if x >= w || y >= h || visited[idx]
        {
            continue;
        }
        if (image.get(x, y) - target).abs() > f64::EPSILON
        {
            continue;
        }

        visited[idx] = true;
        out.set(x, y, fill_value);

        if x > 0
        {
            stack.push((x - 1, y));
        }
        if x + 1 < w
        {
            stack.push((x + 1, y));
        }
        if y > 0
        {
            stack.push((x, y - 1));
        }
        if y + 1 < h
        {
            stack.push((x, y + 1));
        }
    }

    out
}

// ─── Canny Edge Detection ───────────────────────────────────────────────────

/// Simplified Canny edge detection.
pub fn canny(image: &Image, low_thresh: f64, high_thresh: f64) -> Image {
    // 1. Gaussian blur
    let blurred = convolve2d(image, &Kernel::gaussian(1.4));

    // 2. Gradient computation
    let gx = convolve2d(&blurred, &Kernel::sobel_x());
    let gy = convolve2d(&blurred, &Kernel::sobel_y());

    // 3. Magnitude and direction
    let mut magnitude = Vec::with_capacity(image.width * image.height);
    let mut direction = Vec::with_capacity(image.width * image.height);

    for i in 0..(image.width * image.height)
    {
        magnitude.push((gx.data[i].powi(2) + gy.data[i].powi(2)).sqrt());
        direction.push(gy.data[i].atan2(gx.data[i]));
    }

    // 4. Non-maximum suppression
    let w = image.width;
    let h = image.height;
    let mut nms = vec![0.0f64; w * h];

    for y in 1..(h - 1)
    {
        for x in 1..(w - 1)
        {
            let idx = y * w + x;
            let angle = direction[idx] * 180.0 / std::f64::consts::PI;
            let angle = if angle < 0.0 { angle + 180.0 } else { angle };

            let (q, r) = if (angle < 22.5) || (angle >= 157.5)
            {
                (magnitude[y * w + x + 1], magnitude[y * w + x - 1])
            }
            else if angle < 67.5
            {
                (
                    magnitude[(y - 1) * w + x + 1],
                    magnitude[(y + 1) * w + x - 1],
                )
            }
            else if angle < 112.5
            {
                (magnitude[(y - 1) * w + x], magnitude[(y + 1) * w + x])
            }
            else
            {
                (
                    magnitude[(y - 1) * w + x - 1],
                    magnitude[(y + 1) * w + x + 1],
                )
            };

            if magnitude[idx] >= q && magnitude[idx] >= r
            {
                nms[idx] = magnitude[idx];
            }
        }
    }

    // 5. Double threshold
    let mut edges = vec![0.0f64; w * h];
    for i in 0..(w * h)
    {
        if nms[i] >= high_thresh
        {
            edges[i] = 1.0;
        }
        else if nms[i] >= low_thresh
        {
            edges[i] = 0.5; // weak edge
        }
    }

    // 6. Edge tracking by hysteresis (simple version)
    for y in 1..(h - 1)
    {
        for x in 1..(w - 1)
        {
            let idx = y * w + x;
            if edges[idx] == 0.5
            {
                // Check if connected to strong edge
                let mut has_strong = false;
                for dy in -1i32..=1
                {
                    for dx in -1i32..=1
                    {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0
                            && nx < w as i32
                            && ny >= 0
                            && ny < h as i32
                            && edges[ny as usize * w + nx as usize] == 1.0
                        {
                            has_strong = true;
                        }
                    }
                }
                edges[idx] = if has_strong { 1.0 } else { 0.0 };
            }
        }
    }

    Image::from_vec(w, h, edges)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image() -> Image {
        let mut img = Image::new(8, 8);
        for y in 0..8
        {
            for x in 0..8
            {
                if (x + y) % 2 == 0
                {
                    img.set(x, y, 1.0);
                }
            }
        }
        img
    }

    #[test]
    fn test_image_basic() {
        let img = Image::new(4, 4);
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
        assert_eq!(img.data.len(), 16);
        assert_eq!(img.get(0, 0), 0.0);
    }

    #[test]
    fn test_convolve2d() {
        // Convolving an impulse with the Laplacian kernel reproduces the kernel,
        // centered on the impulse location.
        let mut img = Image::new(5, 5);
        img.set(2, 2, 1.0);
        let result = convolve2d(&img, &Kernel::laplacian());
        assert_eq!(result.width, img.width);
        assert_eq!(result.height, img.height);
        assert_eq!(result.get(2, 2), -4.0);
        assert_eq!(result.get(1, 2), 1.0);
        assert_eq!(result.get(3, 2), 1.0);
        assert_eq!(result.get(2, 1), 1.0);
        assert_eq!(result.get(2, 3), 1.0);
        // Everything else must be zero.
        for y in 0..5
        {
            for x in 0..5
            {
                let on_cross = (x == 2 && y == 2)
                    || (x == 1 && y == 2)
                    || (x == 3 && y == 2)
                    || (x == 2 && y == 1)
                    || (x == 2 && y == 3);
                if !on_cross
                {
                    assert_eq!(result.get(x, y), 0.0, "expected 0 at ({}, {})", x, y);
                }
            }
        }
    }

    #[test]
    fn test_max_pool2d() {
        // 4x4 image, 1..=16 row-major, pool 2x2 -> max of each block.
        let data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
        let img = Image::from_vec(4, 4, data);
        let pooled = max_pool2d(&img, 2);
        assert_eq!(pooled.width, 2);
        assert_eq!(pooled.height, 2);
        assert_eq!(pooled.get(0, 0), 6.0); // max(1,2,5,6)
        assert_eq!(pooled.get(1, 0), 8.0); // max(3,4,7,8)
        assert_eq!(pooled.get(0, 1), 14.0); // max(9,10,13,14)
        assert_eq!(pooled.get(1, 1), 16.0); // max(11,12,15,16)
    }

    #[test]
    fn test_avg_pool2d() {
        let data: Vec<f64> = (1..=16).map(|v| v as f64).collect();
        let img = Image::from_vec(4, 4, data);
        let pooled = avg_pool2d(&img, 2);
        assert_eq!(pooled.width, 2);
        assert_eq!(pooled.height, 2);
        assert_eq!(pooled.get(0, 0), 3.5); // mean(1,2,5,6)
        assert_eq!(pooled.get(1, 0), 5.5); // mean(3,4,7,8)
        assert_eq!(pooled.get(0, 1), 11.5); // mean(9,10,13,14)
        assert_eq!(pooled.get(1, 1), 13.5); // mean(11,12,15,16)
    }

    #[test]
    fn test_sigmoid() {
        let img = Image::from_vec(2, 2, vec![0.0, 100.0, -100.0, 0.0]);
        let result = sigmoid(&img);
        assert!((result.get(0, 0) - 0.5).abs() < 1e-12);
        assert!((result.get(1, 1) - 0.5).abs() < 1e-12);
        assert!((result.get(1, 0) - 1.0).abs() < 1e-12); // sigmoid(100) ≈ 1
        assert!(result.get(0, 1).abs() < 1e-12); // sigmoid(-100) ≈ 0
        for &v in &result.data
        {
            assert!(v > 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_relu() {
        let img = Image::from_vec(2, 2, vec![-1.0, 2.0, -3.0, 4.0]);
        let result = relu(&img);
        assert_eq!(result.data, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_softmax() {
        let values = vec![1.0, 2.0, 3.0];
        let result = softmax(&values);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_hog() {
        let img = test_image();
        let descriptor = hog(&img, 4, 9);
        assert!(!descriptor.is_empty());
        let norm: f64 = descriptor.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lbp() {
        // Center (1,1)=0.5; only neighbor index 0 (-1,-1)=(0,0) and index 7 (1,1)=(2,2)
        // are >= center. Bits 1<<0 and 1<<7 => 0b1000_0001 = 129.
        let mut img = Image::new(3, 3);
        img.set(1, 1, 0.5);
        img.set(0, 0, 1.0);
        img.set(2, 2, 1.0);
        let code = lbp(&img, 1, 1);
        assert_eq!(code, 0b1000_0001);
        assert_eq!(code, 129);
    }

    #[test]
    fn test_haar_feature() {
        // Block-diagonal 1s: tl=4, br=4, tr=0, bl=0 => tl+br-tr-bl = 8.0.
        let img = Image::from_vec(
            4,
            4,
            vec![
                1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
            ],
        );
        let f = haar_feature(&img, 0, 0, 4, 4, HaarFeature::FourRectangle);
        assert_eq!(f, 8.0);
    }

    #[test]
    fn test_iou() {
        let b1 = BoundingBox {
            x: 0.0,
            y: 0.0,
            width: 10.0,
            height: 10.0,
            confidence: 0.9,
            class_id: 0,
            class_name: "a".into(),
        };
        let b2 = BoundingBox {
            x: 5.0,
            y: 5.0,
            width: 10.0,
            height: 10.0,
            confidence: 0.8,
            class_id: 0,
            class_name: "a".into(),
        };
        // Intersection 5x5=25; union 100+100-25=175; IoU = 25/175 = 1/7.
        let iou = b1.iou(&b2);
        assert!((iou - 25.0 / 175.0).abs() < 1e-12);
    }

    #[test]
    fn test_nms() {
        let mut boxes = vec![
            BoundingBox {
                x: 0.0,
                y: 0.0,
                width: 10.0,
                height: 10.0,
                confidence: 0.9,
                class_id: 0,
                class_name: "a".into(),
            },
            BoundingBox {
                x: 1.0,
                y: 1.0,
                width: 10.0,
                height: 10.0,
                confidence: 0.8,
                class_id: 0,
                class_name: "a".into(),
            },
            BoundingBox {
                x: 50.0,
                y: 50.0,
                width: 10.0,
                height: 10.0,
                confidence: 0.7,
                class_id: 0,
                class_name: "a".into(),
            },
        ];
        nms(&mut boxes, 0.5);
        assert_eq!(boxes.len(), 2); // overlapping pair + distant box
    }

    #[test]
    fn test_template_match() {
        // Template's only nonzero pixel is its top-left; placing it so the image's
        // (5,5)=1.0 aligns with the template top-left => best match at (5,5) with SSD 0.
        let mut image = Image::new(10, 10);
        image.set(5, 5, 1.0);
        let template = Image::from_vec(2, 2, vec![1.0, 0.0, 0.0, 0.0]);
        let results = template_match(&image, &template);
        assert!(!results.is_empty());
        assert_eq!(results[0], (5, 5, 0.0));
        assert_eq!(match_template_best(&image, &template), Some((5, 5, 0.0)));
    }

    #[test]
    fn test_threshold() {
        let img = Image::from_vec(2, 2, vec![0.2, 0.8, 0.3, 0.9]);
        let result = threshold(&img, 0.5);
        assert_eq!(result.data, vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_otsu() {
        // 100 pixels: 40 at 0.0, 20 at 100.0, 40 at 255.0. min=0, max=255 => bin(v)=v.
        // Between-class variance is maximized at the bin that splits the two dominant
        // clusters {0} and {255}; the middle level 100 falls on the threshold => 100.0.
        let mut data = vec![0.0; 100];
        for v in data.iter_mut().take(40)
        {
            *v = 0.0;
        }
        for v in data.iter_mut().skip(40).take(20)
        {
            *v = 100.0;
        }
        for v in data.iter_mut().skip(60).take(40)
        {
            *v = 255.0;
        }
        let img = Image::from_vec(10, 10, data);
        let t = otsu_threshold(&img);
        assert_eq!(t, 100.0);
    }

    #[test]
    fn test_connected_components() {
        let img = Image::from_vec(
            4,
            4,
            vec![
                1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
            ],
        );
        let components = connected_components(&img);
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_canny() {
        let mut img = Image::new(20, 20);
        // Draw a vertical line
        for y in 0..20
        {
            img.set(10, y, 1.0);
        }
        let edges = canny(&img, 0.1, 0.3);
        assert_eq!(edges.width, 20);
        assert_eq!(edges.height, 20);
        // A strong vertical edge must produce at least one detected edge pixel
        // near the drawn line (column 10). After Gaussian blur the gradient peaks
        // on the two sides of the line, so check the columns around it.
        let mut found = false;
        for y in 1..19
        {
            for x in 8..=12
            {
                if edges.get(x, y) == 1.0
                {
                    found = true;
                }
            }
        }
        assert!(found, "canny should detect an edge near the vertical line");
    }

    #[test]
    fn test_gaussian_kernel() {
        let k = Kernel::gaussian(1.0);
        let sum: f64 = k.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gaussian_center_is_peak() {
        // A 2D Gaussian is maximal and positive at its center, decreasing with radius.
        let k = Kernel::gaussian(1.0);
        let half = k.size / 2;
        let center = k.get(half, half);
        assert!(center > 0.0);
        assert!(center > k.get(0, 0)); // center exceeds corner
        for &v in &k.data
        {
            assert!(v > 0.0); // all weights positive
            assert!(center >= v); // center is the maximum
        }
        let sum: f64 = k.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_flood_fill() {
        let img = Image::from_vec(3, 3, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let filled = flood_fill(&img, 0, 0, 2.0);
        assert_eq!(filled.get(0, 0), 2.0);
        assert_eq!(filled.get(1, 1), 1.0); // not filled
    }

    #[test]
    fn test_lbp_histogram() {
        let img = test_image();
        let hist = lbp_histogram(&img, 0, 0, 4, 4);
        assert_eq!(hist.len(), 256);
        let sum: f64 = hist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_morphological() {
        let mut img = Image::new(5, 5);
        img.set(2, 2, 1.0);
        let dilated = dilate(&img, 3);
        assert_eq!(dilated.get(2, 2), 1.0);
        assert_eq!(dilated.get(1, 1), 1.0);
        assert_eq!(dilated.get(0, 0), 0.0);

        let eroded = erode(&dilated, 3);
        assert_eq!(eroded.get(2, 2), 1.0);
        assert_eq!(eroded.get(1, 1), 0.0);
    }
}
