//! Target tracking — the α–β track filter and a nearest-neighbour multi-target
//! tracker.
//!
//! Detection ([`super::detect`]) gives, per frame, a set of target centroids.
//! Tracking is the temporal layer: associate this frame's detections with
//! existing tracks, filter each track's state so its position and velocity are
//! smoothed and predictable, and manage track birth and death. The workhorse is
//! the **α–β filter** — a fixed-gain steady-state form of the Kalman filter for
//! a constant-velocity target: cheap, stable, and (for a constant-velocity
//! trajectory) unbiased with zero steady-state lag. A [`MultiTracker`] runs one
//! α–β pair per coordinate for every track and does greedy nearest-neighbour
//! association with a distance gate. Dependency-free.

use super::detect::Detection;

/// A scalar α–β track filter for a constant-velocity state `(x, v)`. Each frame
/// advances the state by `dt`; a measurement corrects it with the fixed gains
/// `alpha` (position) and `beta` (velocity).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlphaBeta {
    alpha: f64,
    beta: f64,
    dt: f64,
    x: f64,
    v: f64,
}

impl AlphaBeta {
    /// A filter with gains `alpha`, `beta`, frame interval `dt`, initialised at
    /// position `x0` with zero velocity.
    pub fn new(alpha: f64, beta: f64, dt: f64, x0: f64) -> Self {
        Self {
            alpha,
            beta,
            dt,
            x: x0,
            v: 0.0,
        }
    }

    /// The predicted position at the next frame, `x + v·dt` (does not mutate).
    pub fn predict(&self) -> f64 {
        self.x + self.v * self.dt
    }

    /// Advance one frame and correct with measurement `z`: predict, then nudge
    /// position by `alpha·residual` and velocity by `beta/dt·residual`.
    pub fn update(&mut self, z: f64) {
        let xp = self.x + self.v * self.dt;
        let residual = z - xp;
        self.x = xp + self.alpha * residual;
        self.v += (self.beta / self.dt) * residual;
    }

    /// Advance one frame with no measurement (pure prediction / coasting).
    pub fn coast(&mut self) {
        self.x += self.v * self.dt;
    }

    /// The current filtered position.
    pub fn position(&self) -> f64 {
        self.x
    }

    /// The current filtered velocity.
    pub fn velocity(&self) -> f64 {
        self.v
    }
}

/// The **critically-damped** α–β gains for a discounting factor `theta` in
/// `(0, 1)`: `α = 1 − θ²`, `β = (1 − θ)²`. Smaller `theta` ⇒ heavier gains
/// (faster response, more noise); larger `theta` ⇒ smoother, slower. `theta` is
/// clamped to `(0, 1)`.
pub fn critically_damped_gains(theta: f64) -> (f64, f64) {
    let th = theta.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    (1.0 - th * th, (1.0 - th) * (1.0 - th))
}

/// One target track: an α–β filter per coordinate (range and Doppler bin), plus
/// a hit/miss lifecycle and the last associated amplitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Track {
    /// A stable identifier, assigned at birth by the [`MultiTracker`].
    pub id: usize,
    range: AlphaBeta,
    doppler: AlphaBeta,
    /// Number of frames this track has been updated with a detection.
    pub hits: usize,
    /// Consecutive frames without an association (reset on a hit).
    pub misses: usize,
    /// Amplitude of the most recently associated detection.
    pub amplitude: f64,
}

impl Track {
    fn new(id: usize, d: &Detection, alpha: f64, beta: f64, dt: f64) -> Self {
        Self {
            id,
            range: AlphaBeta::new(alpha, beta, dt, d.range),
            doppler: AlphaBeta::new(alpha, beta, dt, d.doppler),
            hits: 1,
            misses: 0,
            amplitude: d.amplitude,
        }
    }

    /// The predicted `(range, doppler)` centroid at the next frame.
    pub fn predict(&self) -> (f64, f64) {
        (self.range.predict(), self.doppler.predict())
    }

    fn update(&mut self, d: &Detection) {
        self.range.update(d.range);
        self.doppler.update(d.doppler);
        self.hits += 1;
        self.misses = 0;
        self.amplitude = d.amplitude;
    }

    fn coast(&mut self) {
        self.range.coast();
        self.doppler.coast();
        self.misses += 1;
    }

    /// The current filtered `(range, doppler)` position.
    pub fn position(&self) -> (f64, f64) {
        (self.range.position(), self.doppler.position())
    }

    /// The current filtered `(range, doppler)` velocity (bins per frame).
    pub fn velocity(&self) -> (f64, f64) {
        (self.range.velocity(), self.doppler.velocity())
    }
}

/// A nearest-neighbour multi-target tracker over the detections of
/// [`super::detect::cluster_detections`]. Each [`step`](Self::step) predicts
/// every track, greedily associates detections to the nearest predicted track
/// within a distance gate, updates matched tracks, coasts unmatched ones,
/// spawns tracks for unmatched detections, and drops tracks that have coasted
/// past `max_misses`.
#[derive(Debug, Clone)]
pub struct MultiTracker {
    alpha: f64,
    beta: f64,
    dt: f64,
    max_misses: usize,
    next_id: usize,
    tracks: Vec<Track>,
}

impl MultiTracker {
    /// A tracker whose tracks use gains `alpha`/`beta` at frame interval `dt`,
    /// and are dropped after more than `max_misses` consecutive coasted frames.
    pub fn new(alpha: f64, beta: f64, dt: f64, max_misses: usize) -> Self {
        Self {
            alpha,
            beta,
            dt,
            max_misses,
            next_id: 0,
            tracks: Vec::new(),
        }
    }

    /// The current live tracks.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Advance one frame with the frame's `detections`, associating within a
    /// Euclidean `gate` (in range-Doppler bin distance).
    pub fn step(&mut self, detections: &[Detection], gate: f64) {
        let preds: Vec<(f64, f64)> = self.tracks.iter().map(Track::predict).collect();
        // Candidate (squared distance, track index, detection index) pairs
        // inside the gate, greedily assigned nearest-first, each side once.
        let gate2 = gate * gate;
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (di, d) in detections.iter().enumerate()
        {
            for (ti, &(pr, pd)) in preds.iter().enumerate()
            {
                let dist2 = (d.range - pr).powi(2) + (d.doppler - pd).powi(2);
                if dist2 <= gate2
                {
                    pairs.push((dist2, ti, di));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut track_used = vec![false; self.tracks.len()];
        let mut det_used = vec![false; detections.len()];
        let mut assigned: Vec<Option<usize>> = vec![None; self.tracks.len()];
        for (_d2, ti, di) in pairs
        {
            if !track_used[ti] && !det_used[di]
            {
                track_used[ti] = true;
                det_used[di] = true;
                assigned[ti] = Some(di);
            }
        }
        for (ti, track) in self.tracks.iter_mut().enumerate()
        {
            match assigned[ti]
            {
                Some(di) => track.update(&detections[di]),
                None => track.coast(),
            }
        }
        for (di, d) in detections.iter().enumerate()
        {
            if !det_used[di]
            {
                self.tracks
                    .push(Track::new(self.next_id, d, self.alpha, self.beta, self.dt));
                self.next_id += 1;
            }
        }
        let limit = self.max_misses;
        self.tracks.retain(|t| t.misses <= limit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(range: f64, doppler: f64, amplitude: f64) -> Detection {
        Detection {
            range,
            doppler,
            amplitude,
            cells: 1,
        }
    }

    #[test]
    fn critically_damped_gains_match_the_closed_form() {
        let (a, b) = critically_damped_gains(0.5);
        assert!((a - 0.75).abs() < 1e-12 && (b - 0.25).abs() < 1e-12);
        // Clamped into (0, 1); degenerate inputs stay finite.
        let (a0, b0) = critically_damped_gains(0.0);
        assert!(a0 > 0.0 && a0 <= 1.0 && b0 >= 0.0);
        let (a1, _) = critically_damped_gains(1.0);
        assert!(a1 > 0.0);
    }

    #[test]
    fn alpha_beta_tracks_a_constant_velocity_ramp_with_zero_lag() {
        // Noise-free constant-velocity truth x_k = x0 + v·k·dt. An α–β filter
        // tracks a ramp with zero steady-state error.
        let (alpha, beta) = critically_damped_gains(0.6);
        let (dt, x0, v) = (0.5, 3.0, 2.0);
        let mut f = AlphaBeta::new(alpha, beta, dt, x0);
        for k in 1..=80
        {
            f.update(x0 + v * k as f64 * dt);
        }
        // After convergence the position matches truth and velocity matches v.
        let truth = x0 + v * 80.0 * dt;
        assert!(
            (f.position() - truth).abs() < 1e-6,
            "pos {} vs {truth}",
            f.position()
        );
        assert!(
            (f.velocity() - v).abs() < 1e-6,
            "vel {} vs {v}",
            f.velocity()
        );
    }

    #[test]
    fn alpha_beta_coasting_extrapolates_at_constant_velocity() {
        let (alpha, beta) = critically_damped_gains(0.5);
        let (dt, v) = (1.0, 4.0);
        let mut f = AlphaBeta::new(alpha, beta, dt, 0.0);
        for k in 1..=40
        {
            f.update(v * k as f64 * dt);
        }
        let before = f.position();
        let vel = f.velocity();
        f.coast();
        // A coasted frame advances by exactly one velocity step.
        assert!((f.position() - (before + vel * dt)).abs() < 1e-12);
    }

    #[test]
    fn multitracker_follows_a_single_target() {
        let (alpha, beta) = critically_damped_gains(0.5);
        let mut mt = MultiTracker::new(alpha, beta, 1.0, 3);
        // One target moving +1 range bin, +0.5 doppler bin per frame.
        for k in 0..30
        {
            let z = det(5.0 + k as f64, 10.0 + 0.5 * k as f64, 50.0);
            mt.step(&[z], 5.0);
        }
        assert_eq!(mt.tracks().len(), 1);
        let (vr, vd) = mt.tracks()[0].velocity();
        assert!(
            (vr - 1.0).abs() < 1e-3 && (vd - 0.5).abs() < 1e-3,
            "vel ({vr}, {vd})"
        );
        assert_eq!(mt.tracks()[0].id, 0);
    }

    #[test]
    fn multitracker_keeps_two_separated_targets_apart() {
        let (alpha, beta) = critically_damped_gains(0.5);
        let mut mt = MultiTracker::new(alpha, beta, 1.0, 3);
        // Two targets far apart, each constant-velocity; a tight gate cannot
        // cross-associate them.
        for k in 0..20
        {
            let a = det(2.0 + k as f64, 4.0, 40.0);
            let b = det(60.0 - k as f64, 50.0, 70.0);
            mt.step(&[a, b], 3.0);
        }
        assert_eq!(mt.tracks().len(), 2);
        // Distinct, stable ids born on the first frame.
        let ids: Vec<usize> = mt.tracks().iter().map(|t| t.id).collect();
        assert!(ids.contains(&0) && ids.contains(&1));
    }

    #[test]
    fn multitracker_spawns_then_drops_a_lost_track() {
        let (alpha, beta) = critically_damped_gains(0.5);
        let mut mt = MultiTracker::new(alpha, beta, 1.0, 2);
        mt.step(&[det(10.0, 10.0, 30.0)], 3.0);
        assert_eq!(mt.tracks().len(), 1);
        // No detections: coast until it exceeds max_misses (2) and is dropped.
        mt.step(&[], 3.0); // misses = 1
        assert_eq!(mt.tracks().len(), 1);
        mt.step(&[], 3.0); // misses = 2 (== max, kept)
        assert_eq!(mt.tracks().len(), 1);
        mt.step(&[], 3.0); // misses = 3 (> max, dropped)
        assert!(mt.tracks().is_empty());
    }
}
