//! Episodic memory log — an append-only, time-indexed, bounded store over
//! [`crate::Event`].
//!
//! `scirust-events-core` already defines the natural episodic-memory record
//! ([`crate::Event`]: id, timestamp, label, confidence, data snapshot) and
//! [`crate::EventRuntime`](scirust_events_runtime::EventRuntime) produces a
//! `Vec<Event>` — but today those events are returned to the caller and
//! dropped. [`EpisodicEventLog`] turns them into a **store**: append-only,
//! monotonically growing unless a capacity is set, queryable by timestamp
//! range, and (when bounded) evicting the oldest entries so memory stays finite
//! regardless of how long the agent runs.
//!
//! Everything is plain, deterministic bookkeeping in fixed order.

use crate::Event;
use std::collections::BTreeMap;

/// A total-order key for `f64` timestamps so they can index a `BTreeMap`:
/// negative-infinity sorts first, NaN sorts last, and every other value has a
/// unique, monotone position. (We never expect NaN in a timestamp, but
/// `to_bits` alone is not a total order for floats, so we fold the sign.)
fn ord_key(t: f64) -> u64 {
    let bits = t.to_bits();
    if t.is_nan()
    {
        return u64::MAX;
    }
    if t < 0.0
    {
        // Negative: flip all bits so -inf (0xff_f..0) becomes 0x00_0..0 (smallest).
        !bits
    }
    else
    {
        // Non-negative: flip only the sign bit so +0 sits just above the negatives.
        bits ^ (1u64 << 63)
    }
}

/// An append-only, time-indexed episodic store over [`Event`]s.
pub struct EpisodicEventLog {
    events: Vec<Event>,
    /// `timestamp-ord-key -> indices` into `events`, for range queries.
    by_time: BTreeMap<u64, Vec<usize>>,
    next_id: u64,
    capacity: Option<usize>,
}

impl EpisodicEventLog {
    /// Unbounded log (grows forever; use for short-lived runs or tests).
    pub fn new() -> Self {
        Self::with_capacity(None)
    }

    /// Bounded log: when `capacity` is `Some(n)` at most `n` events are kept;
    /// recording beyond it evicts the oldest (and re-indexes the time map).
    pub fn with_capacity(capacity: Option<usize>) -> Self {
        Self {
            events: Vec::new(),
            by_time: BTreeMap::new(),
            next_id: 1,
            capacity,
        }
    }

    /// Number of stored events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Append an event. If its `id` is `0`, a fresh monotone id is assigned; the
    /// assigned id is returned. Enforces the capacity by evicting the oldest
    /// event when exceeded (the recorded event itself may be evicted if it is
    /// the oldest, but its assigned id is still returned).
    pub fn record(&mut self, mut event: Event) -> u64 {
        if event.id == 0
        {
            event.id = self.next_id;
            self.next_id += 1;
        }
        else if event.id >= self.next_id
        {
            self.next_id = event.id + 1;
        }
        let id = event.id;
        let key = ord_key(event.timestamp);
        let idx = self.events.len();
        self.by_time.entry(key).or_default().push(idx);
        self.events.push(event);
        if let Some(cap) = self.capacity
        {
            while self.events.len() > cap
            {
                self.evict_oldest();
            }
        }
        id
    }

    /// Convenience: record from raw fields.
    pub fn record_event(
        &mut self,
        timestamp: f64,
        label_en: &str,
        label_fr: &str,
        confidence: f32,
        data_snapshot: Option<Vec<f32>>,
    ) -> u64 {
        self.record(Event {
            id: 0,
            timestamp,
            label_en: label_en.to_string(),
            label_fr: label_fr.to_string(),
            confidence,
            data_snapshot,
        })
    }

    /// Drop the oldest event (smallest timestamp, ties broken by lowest index)
    /// and rebuild the time index. No-op if empty.
    pub fn evict_oldest(&mut self) -> Option<Event> {
        if self.events.is_empty()
        {
            return None;
        }
        // Find the smallest key with an entry; remove the first index it holds.
        let first_key = *self.by_time.keys().next().unwrap();
        let removed_idx = self.by_time.get_mut(&first_key).unwrap().remove(0);
        if self
            .by_time
            .get(&first_key)
            .map(|v| v.is_empty())
            .unwrap_or(true)
        {
            self.by_time.remove(&first_key);
        }
        let event = self.events.remove(removed_idx);
        // Rebuild the index (indices shifted by the removal).
        self.reindex();
        Some(event)
    }

    fn reindex(&mut self) {
        self.by_time.clear();
        for (i, e) in self.events.iter().enumerate()
        {
            self.by_time
                .entry(ord_key(e.timestamp))
                .or_default()
                .push(i);
        }
    }

    /// All events whose timestamp lies in `[t0, t1]` (inclusive), in
    /// chronological then insertion order. Cheap via the time index.
    pub fn iter_range(&self, t0: f64, t1: f64) -> Vec<&Event> {
        let lo = ord_key(t0);
        let hi = ord_key(t1);
        let mut out = Vec::new();
        for idxs in self.by_time.range(lo..=hi).map(|(_, v)| v)
        {
            for &i in idxs
            {
                out.push(&self.events[i]);
            }
        }
        out
    }

    /// Iterate over all stored events in chronological then insertion order.
    pub fn iter_chrono(&self) -> Vec<&Event> {
        let mut out = Vec::with_capacity(self.events.len());
        for idxs in self.by_time.values()
        {
            for &i in idxs
            {
                out.push(&self.events[i]);
            }
        }
        out
    }

    /// Borrow the raw storage (insertion order).
    pub fn events(&self) -> &[Event] {
        &self.events
    }
}

impl Default for EpisodicEventLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assigns_monotone_ids_and_indexes_by_time() {
        let mut log = EpisodicEventLog::new();
        let a = log.record_event(3.0, "a", "a", 1.0, None);
        let b = log.record_event(1.0, "b", "b", 0.5, None);
        let c = log.record_event(2.0, "c", "c", 0.9, None);
        assert_eq!([a, b, c], [1, 2, 3]);
        // Chronological order is 1.0, 2.0, 3.0.
        let chrono: Vec<&str> = log
            .iter_chrono()
            .iter()
            .map(|e| e.label_en.as_str())
            .collect();
        assert_eq!(chrono, vec!["b", "c", "a"]);
    }

    #[test]
    fn range_query_is_inclusive_and_cheap() {
        let mut log = EpisodicEventLog::new();
        for t in [10.0, 20.0, 20.0, 30.0, -5.0]
        {
            log.record_event(t, "x", "x", 0.1, None);
        }
        let in_range: Vec<&str> = log
            .iter_range(20.0, 30.0)
            .iter()
            .map(|e| e.label_en.as_str())
            .collect();
        assert_eq!(in_range, vec!["x", "x", "x"]); // 20, 20, 30
        assert_eq!(log.iter_range(100.0, 200.0).len(), 0);
    }

    #[test]
    fn capacity_evicts_oldest() {
        let mut log = EpisodicEventLog::with_capacity(Some(2));
        log.record_event(1.0, "first", "premier", 1.0, None);
        log.record_event(2.0, "second", "second", 1.0, None);
        log.record_event(3.0, "third", "troisieme", 1.0, None);
        assert_eq!(log.len(), 2);
        // The oldest (t=1.0) must have been evicted.
        let labels: Vec<&str> = log
            .iter_chrono()
            .iter()
            .map(|e| e.label_en.as_str())
            .collect();
        assert_eq!(labels, vec!["second", "third"]);
    }

    #[test]
    fn preserves_an_explicit_id_and_advances_next() {
        let mut log = EpisodicEventLog::new();
        let id0 = log.record(Event {
            id: 42,
            timestamp: 0.0,
            label_en: "given".into(),
            label_fr: "donne".into(),
            confidence: 0.2,
            data_snapshot: None,
        });
        assert_eq!(id0, 42);
        let id1 = log.record_event(0.0, "auto", "auto", 0.1, None);
        assert_eq!(id1, 43);
    }
}
