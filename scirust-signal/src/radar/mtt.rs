//! Multi-target tracking from polar radar measurements with statistical gating.
//!
//! [`super::track::MultiTracker`] runs α–β filters over Cartesian
//! range-Doppler centroids with a Euclidean distance gate. This module raises
//! the fidelity on both counts: each track is a full [`RadarEkf`] fed **polar
//! (range/bearing)** returns, and association uses a **statistical gate** — the
//! normalised innovation squared (NIS, a Mahalanobis distance) tested against a
//! χ² quantile — so the gate automatically tightens or widens with each track's
//! own uncertainty instead of using one fixed radius for all.
//!
//! Each [`step`](RadarMultiTracker::step) predicts every track, gates every
//! (track, measurement) pair by NIS, greedily associates nearest-first, updates
//! matched tracks, coasts unmatched ones, spawns a track for every unmatched
//! measurement, and drops tracks that have coasted past `max_misses`.
//! Dependency-free.

use super::ekf::RadarEkf;

/// One target track: an extended Kalman filter over the target's Cartesian
/// state plus a hit/miss lifecycle.
#[derive(Debug, Clone, PartialEq)]
pub struct RadarTrack {
    /// A stable identifier assigned at birth by the [`RadarMultiTracker`].
    pub id: usize,
    ekf: RadarEkf,
    /// Number of frames this track has been updated with a measurement.
    pub hits: usize,
    /// Consecutive coasted frames without an association (reset on a hit).
    pub misses: usize,
}

impl RadarTrack {
    /// The current filtered Cartesian position `(x, y)`.
    pub fn position(&self) -> (f64, f64) {
        self.ekf.position()
    }

    /// The current filtered Cartesian velocity `(vₓ, v_y)`.
    pub fn velocity(&self) -> (f64, f64) {
        self.ekf.velocity()
    }

    /// The underlying extended Kalman filter.
    pub fn ekf(&self) -> &RadarEkf {
        &self.ekf
    }
}

/// A multi-target tracker over polar `(range, bearing)` measurements, one
/// [`RadarEkf`] per track, associating by a NIS validation gate.
#[derive(Debug, Clone)]
pub struct RadarMultiTracker {
    dt: f64,
    q: f64,
    range_var: f64,
    bearing_var: f64,
    gate_nis: f64,
    max_misses: usize,
    init_var: f64,
    next_id: usize,
    tracks: Vec<RadarTrack>,
}

impl RadarMultiTracker {
    /// A tracker at frame interval `dt` and process-noise intensity `q`, with
    /// per-measurement `range_var`/`bearing_var`, a NIS `gate` (a χ²-with-2-d.o.f.
    /// threshold, e.g. `9.21` for 99 %), tracks dropped after more than
    /// `max_misses` coasted frames, and newly-born tracks initialised with
    /// isotropic covariance `init_var`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dt: f64,
        q: f64,
        range_var: f64,
        bearing_var: f64,
        gate: f64,
        max_misses: usize,
        init_var: f64,
    ) -> Self {
        Self {
            dt,
            q,
            range_var,
            bearing_var,
            gate_nis: gate,
            max_misses,
            init_var,
            next_id: 0,
            tracks: Vec::new(),
        }
    }

    /// The current live tracks.
    pub fn tracks(&self) -> &[RadarTrack] {
        &self.tracks
    }

    /// Advance one frame with the frame's polar `measurements` (`(range,
    /// bearing)` pairs).
    pub fn step(&mut self, measurements: &[(f64, f64)]) {
        // Predict every track once (this is also the coast for unmatched ones).
        for t in &mut self.tracks
        {
            t.ekf.predict();
        }
        // Gated candidate (NIS, track index, measurement index) triples.
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (mi, &(range, bearing)) in measurements.iter().enumerate()
        {
            for (ti, t) in self.tracks.iter().enumerate()
            {
                if let Some(nis) = t.ekf.nis(range, bearing, self.range_var, self.bearing_var)
                {
                    if nis <= self.gate_nis
                    {
                        pairs.push((nis, ti, mi));
                    }
                }
            }
        }
        // Greedy nearest-first assignment, each track and measurement once.
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut track_used = vec![false; self.tracks.len()];
        let mut meas_used = vec![false; measurements.len()];
        let mut assigned: Vec<Option<usize>> = vec![None; self.tracks.len()];
        for (_nis, ti, mi) in pairs
        {
            if !track_used[ti] && !meas_used[mi]
            {
                track_used[ti] = true;
                meas_used[mi] = true;
                assigned[ti] = Some(mi);
            }
        }
        // Update matched tracks (already predicted, so call update directly);
        // coast the rest.
        for (ti, t) in self.tracks.iter_mut().enumerate()
        {
            match assigned[ti]
            {
                Some(mi) =>
                {
                    let (range, bearing) = measurements[mi];
                    t.ekf
                        .update(range, bearing, self.range_var, self.bearing_var);
                    t.hits += 1;
                    t.misses = 0;
                },
                None => t.misses += 1,
            }
        }
        // Spawn a track for every unassociated measurement (polar → Cartesian).
        for (mi, &(range, bearing)) in measurements.iter().enumerate()
        {
            if !meas_used[mi]
            {
                let (x0, y0) = (range * bearing.cos(), range * bearing.sin());
                let ekf = RadarEkf::new(self.dt, self.q, x0, y0, self.init_var);
                self.tracks.push(RadarTrack {
                    id: self.next_id,
                    ekf,
                    hits: 1,
                    misses: 0,
                });
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

    /// The true polar measurement of a Cartesian point.
    fn polar(x: f64, y: f64) -> (f64, f64) {
        (x.hypot(y), y.atan2(x))
    }

    fn tracker() -> RadarMultiTracker {
        // dt=1, modest process noise, tight measurement noise, 99% gate (χ²₂).
        RadarMultiTracker::new(1.0, 1e-3, 1e-3, 1e-6, 9.21, 3, 25.0)
    }

    #[test]
    fn follows_a_single_target() {
        let mut mt = tracker();
        let (x0, y0, vx, vy) = (30.0_f64, 10.0_f64, 1.0_f64, 0.5_f64);
        for k in 0..30
        {
            let (tx, ty) = (x0 + vx * k as f64, y0 + vy * k as f64);
            mt.step(&[polar(tx, ty)]);
        }
        assert_eq!(mt.tracks().len(), 1);
        let (px, py) = mt.tracks()[0].position();
        let (tx, ty) = (x0 + vx * 29.0, y0 + vy * 29.0);
        assert!(
            (px - tx).abs() < 0.5 && (py - ty).abs() < 0.5,
            "pos ({px},{py})"
        );
        assert_eq!(mt.tracks()[0].id, 0);
    }

    #[test]
    fn keeps_two_separated_targets_apart() {
        let mut mt = tracker();
        for k in 0..25
        {
            let a = polar(20.0 + k as f64, 15.0);
            let b = polar(-25.0 - k as f64, -40.0);
            mt.step(&[a, b]);
        }
        assert_eq!(mt.tracks().len(), 2);
        let ids: Vec<usize> = mt.tracks().iter().map(|t| t.id).collect();
        assert!(ids.contains(&0) && ids.contains(&1));
    }

    #[test]
    fn nis_gate_rejects_clutter_and_spawns_a_new_track() {
        let mut mt = tracker();
        // Establish a track.
        for k in 0..8
        {
            mt.step(&[polar(50.0 + k as f64, 0.0)]);
        }
        assert_eq!(mt.tracks().len(), 1);
        let before = mt.tracks()[0].position();
        // One frame whose only measurement is far-off clutter: it fails the NIS
        // gate, so the real track coasts (untainted) and clutter spawns a track.
        mt.step(&[polar(-60.0, 70.0)]);
        assert_eq!(mt.tracks().len(), 2, "clutter should spawn its own track");
        // The established track coasted forward, not toward the clutter.
        let after = mt.tracks().iter().find(|t| t.id == 0).unwrap().position();
        assert!(
            after.0 > before.0 && after.1.abs() < 5.0,
            "track pulled by clutter: {after:?}"
        );
    }

    #[test]
    fn spawns_then_drops_a_lost_track() {
        let mut mt = RadarMultiTracker::new(1.0, 1e-3, 1e-3, 1e-6, 9.21, 2, 25.0);
        mt.step(&[polar(40.0, 40.0)]);
        assert_eq!(mt.tracks().len(), 1);
        mt.step(&[]); // misses = 1
        mt.step(&[]); // misses = 2 (== max, kept)
        assert_eq!(mt.tracks().len(), 1);
        mt.step(&[]); // misses = 3 (> max, dropped)
        assert!(mt.tracks().is_empty());
    }

    #[test]
    fn nis_is_small_on_target_and_large_off_target() {
        // A direct check of the gating statistic behind the tracker.
        let mut ekf = RadarEkf::new(1.0, 1e-3, 20.0, 20.0, 5.0);
        for k in 1..=10
        {
            ekf.step(
                polar(20.0 + k as f64, 20.0).0,
                polar(20.0 + k as f64, 20.0).1,
                1e-3,
                1e-6,
            );
        }
        ekf.predict();
        let (r, b) = ekf.predicted_measurement();
        let on = ekf.nis(r, b, 1e-3, 1e-6).unwrap();
        let off = ekf.nis(r + 20.0, b + 0.4, 1e-3, 1e-6).unwrap();
        assert!(on < 1.0, "on-target NIS {on}");
        assert!(off > on * 100.0, "off-target NIS {off} vs {on}");
    }

    #[test]
    fn empty_frames_on_an_empty_tracker_are_inert() {
        let mut mt = tracker();
        mt.step(&[]);
        assert!(mt.tracks().is_empty());
    }
}
