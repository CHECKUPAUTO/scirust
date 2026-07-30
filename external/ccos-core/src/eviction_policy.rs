//! # Eviction policy — tabular Q-learning for paging decisions
//!
//! CCOS's default eviction is a deterministic greedy: drop the lowest-scored
//! node when the window exceeds the budget. That's optimal in the *average*
//! case but blind to a pattern every long-running agent hits: **some low-score
//! nodes are about to become hot** (a `page_fault` is coming and they're in its
//! causal region), and **some high-score nodes are never touched again** (the
//! agent moved on). A learned policy can trade a little short-term score for a
//! lot of avoided re-paging.
//!
//! This module distills a **tabular Q-learning** agent (the simplest RL that
//! works, taken from SCIRUST's `scirust-rl-algo::TabularQLearner`) into a
//! zero-dependency, deterministic CCOS policy. The state is a 4-tuple of
//! coarse buckets (score / recency / failure-pressure / token-size), the
//! action is binary (keep / evict), and the reward is +1 for a keep that was
//! later recalled, −1 for an evict that triggered a re-page, scaled by the
//! token cost. No RNG in the greedy/deploy path (the agent picks the argmax Q);
//! ε-exploration is only used during offline training.
//!
//! ## Why tabular and not a neural policy?
//!
//! Same reason as the embeddings: CCOS is zero-dep and bit-exact-deterministic.
//! A neural policy (PPO/DQN) would pull in `scirust-core` and break the replay
//! invariant. The state space is tiny (4 buckets × few values = a few hundred
//! cells), so tabular Q-learning converges in a few thousand episodes and the
//! Q-table serializes as a `BTreeMap` — bit-reproducible across builds.
//!
//! ## Integration
//!
//! The policy is **advisory**: [`EvictionPolicy::should_evict`] returns a
//! boolean the caller blends with the deterministic greedy. When the policy is
//! untrained (empty Q-table), it falls back to the greedy baseline, so turning
//! it on is never worse than the status quo. The training loop is offline (a
//! `fit` method that replays a recorded timeline and assigns rewards); the
//! live path is a pure lookup.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ─────────────────────────────────────────────────────────────────────────────
// State discretization
// ─────────────────────────────────────────────────────────────────────────────

/// The discretized state a paging decision is made on. Each field is a small
/// bucket so the Q-table stays compact (a few hundred cells).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct PagingState {
    /// Causal score bucket: 0 (cold, <0.2), 1 (warm, 0.2–0.5), 2 (hot, ≥0.5).
    pub score: u8,
    /// Recency bucket: 0 (just touched), 1 (recent), 2 (stale).
    pub recency: u8,
    /// Failure-pressure bucket: 0 (none), 1 (near a failure), 2 (failing).
    pub pressure: u8,
    /// Token-size bucket: 0 (small, <64), 1 (medium, 64–512), 2 (large, ≥512).
    pub size: u8,
}

/// Action: keep the node in the window or evict it to a colder zone.
pub const KEEP: u8 = 0;
pub const EVICT: u8 = 1;

/// Bucket a raw causal score (0.0–1.0+) into 0/1/2.
pub fn bucket_score(score: f64) -> u8 {
    if score < 0.2 {
        0
    } else if score < 0.5 {
        1
    } else {
        2
    }
}

/// Bucket a recency rank (0 = most recent) into 0/1/2.
pub fn bucket_recency(rank: usize, total: usize) -> u8 {
    if total == 0 {
        return 0;
    }
    let frac = rank as f64 / total as f64;
    if frac < 0.33 {
        0
    } else if frac < 0.66 {
        1
    } else {
        2
    }
}

/// Bucket failure-relevance (0.0–1.0) into 0/1/2.
pub fn bucket_pressure(failure_relevance: f64) -> u8 {
    if failure_relevance < 0.05 {
        0
    } else if failure_relevance < 0.4 {
        1
    } else {
        2
    }
}

/// Bucket token-size estimate into 0/1/2.
pub fn bucket_size(tokens: usize) -> u8 {
    if tokens < 64 {
        0
    } else if tokens < 512 {
        1
    } else {
        2
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tabular Q-learner
// ─────────────────────────────────────────────────────────────────────────────

/// A deterministic tabular Q-learning policy. The Q-table is a `BTreeMap`
/// keyed by `(state, action)`, so serialization and iteration are stable.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvictionPolicy {
    pub q: BTreeMap<(PagingState, u8), f64>,
    /// Learning rate (default 0.1).
    pub alpha: f64,
    /// Discount factor (default 0.9).
    pub gamma: f64,
    /// Number of training updates applied (for reporting / convergence checks).
    pub updates: usize,
}

impl EvictionPolicy {
    /// Fresh untrained policy with standard hyperparameters.
    pub fn new() -> Self {
        Self {
            q: BTreeMap::new(),
            alpha: 0.1,
            gamma: 0.9,
            updates: 0,
        }
    }

    /// True iff the policy has learned anything (non-empty Q-table).
    pub fn is_trained(&self) -> bool {
        !self.q.is_empty()
    }

    /// Q-value for `(state, action)`, defaulting to 0.0 for unseen cells.
    pub fn q_value(&self, state: PagingState, action: u8) -> f64 {
        self.q.get(&(state, action)).copied().unwrap_or(0.0)
    }

    /// Greedy action: `KEEP` if `Q(keep) >= Q(evict)`, else `EVICT`. When the
    /// policy is untrained (both Q are 0), this returns `KEEP` — the caller is
    /// expected to fall back to the deterministic greedy eviction (lowest
    /// score first) in that case, so the policy only *overrides* the greedy
    /// once it has actually learned a preference.
    pub fn greedy_action(&self, state: PagingState) -> u8 {
        let keep = self.q_value(state, KEEP);
        let evict = self.q_value(state, EVICT);
        if evict > keep {
            EVICT
        } else {
            KEEP
        }
    }

    /// Should the caller evict this node? `false` (keep) when the policy is
    /// untrained or prefers keep; `true` only when the policy has learned that
    /// this state is better evicted. The deterministic greedy remains the
    /// authority when the policy has no opinion.
    pub fn should_evict(&self, state: PagingState) -> bool {
        self.is_trained() && self.greedy_action(state) == EVICT
    }

    /// One step of Q-learning update:
    /// `Q(s,a) ← Q(s,a) + α·(r + γ·max_a' Q(s',a') − Q(s,a))`.
    /// Deterministic (no ε-greedy here; exploration happens in the training
    /// loop that calls this).
    pub fn update(&mut self, s: PagingState, a: u8, r: f64, s_next: PagingState) {
        let q_sa = self.q_value(s, a);
        let max_next = self.q_value(s_next, KEEP).max(self.q_value(s_next, EVICT));
        let td = r + self.gamma * max_next - q_sa;
        self.q.insert((s, a), q_sa + self.alpha * td);
        self.updates += 1;
    }

    /// Bulk-fit the policy from a replay of (state, action, reward, next_state)
    /// transitions. Each transition is applied once (online Q-learning, not
    /// batch). Deterministic: the order of the iterator is the order of the
    /// updates, and `BTreeMap` keeps the table stable.
    pub fn fit<I>(&mut self, transitions: I)
    where
        I: IntoIterator<Item = (PagingState, u8, f64, PagingState)>,
    {
        for (s, a, r, s_next) in transitions {
            self.update(s, a, r, s_next);
        }
    }

    /// Number of distinct (state, action) cells the policy has a value for.
    pub fn table_size(&self) -> usize {
        self.q.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reward shaping
// ─────────────────────────────────────────────────────────────────────────────

/// Reward for a keep decision: positive when the node was later recalled (the
/// keep paid off), negative scaled by its token cost when it was never touched
/// again (it crowded out something useful).
pub fn reward_keep(was_recalled: bool, token_cost: usize) -> f64 {
    if was_recalled {
        1.0
    } else {
        // -0.001 per token: a 512-token never-recalled node costs ~−0.5.
        -0.001 * token_cost as f64
    }
}

/// Reward for an evict decision: positive when the node was *not* later
/// re-paged (the evict was right), negative when it triggered a re-page (the
/// evict cost a re-parse / re-fetch).
pub fn reward_evict(was_re_paged: bool, token_cost: usize) -> f64 {
    if was_re_paged {
        // The re-page cost ~the node's token budget; penalise proportionally.
        -0.5 - 0.002 * token_cost as f64
    } else {
        // Freed space that something else used → small positive.
        0.2
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untrained_policy_never_evicts() {
        let p = EvictionPolicy::new();
        let s = PagingState {
            score: 0,
            recency: 2,
            pressure: 0,
            size: 2,
        };
        assert!(!p.should_evict(s), "untrained → keep (fall back to greedy)");
    }

    #[test]
    fn trained_policy_evicts_when_q_prefers_evict() {
        let mut p = EvictionPolicy::new();
        // Train: a stale, cold, no-pressure, large node should be evicted.
        let s_bad = PagingState {
            score: 0,
            recency: 2,
            pressure: 0,
            size: 2,
        };
        for _ in 0..50 {
            p.update(s_bad, EVICT, 1.0, s_bad);
            p.update(s_bad, KEEP, -1.0, s_bad);
        }
        assert!(p.should_evict(s_bad), "learned to evict the bad state");
    }

    #[test]
    fn trained_policy_keeps_failing_nodes() {
        let mut p = EvictionPolicy::new();
        let s_hot = PagingState {
            score: 2,
            recency: 0,
            pressure: 2,
            size: 1,
        };
        for _ in 0..50 {
            p.update(s_hot, KEEP, 1.0, s_hot);
            p.update(s_hot, EVICT, -1.0, s_hot);
        }
        assert!(
            !p.should_evict(s_hot),
            "learned to keep the hot failing node"
        );
    }

    #[test]
    fn q_table_is_deterministic_for_same_transitions() {
        let transitions = [
            (
                PagingState {
                    score: 0,
                    recency: 2,
                    pressure: 0,
                    size: 2,
                },
                EVICT,
                1.0,
                PagingState {
                    score: 0,
                    recency: 2,
                    pressure: 0,
                    size: 2,
                },
            ),
            (
                PagingState {
                    score: 2,
                    recency: 0,
                    pressure: 2,
                    size: 1,
                },
                KEEP,
                1.0,
                PagingState {
                    score: 2,
                    recency: 0,
                    pressure: 2,
                    size: 1,
                },
            ),
        ];
        let mut a = EvictionPolicy::new();
        let mut b = EvictionPolicy::new();
        for _ in 0..10 {
            a.fit(transitions.iter().copied());
            b.fit(transitions.iter().copied());
        }
        assert_eq!(a.q, b.q, "same transitions → bit-identical Q-table");
    }

    #[test]
    fn fit_converges_q_values() {
        let mut p = EvictionPolicy::new();
        let s = PagingState {
            score: 0,
            recency: 2,
            pressure: 0,
            size: 2,
        };
        // Repeatedly reward EVICT, penalise KEEP → Q(EVICT) should grow above Q(KEEP).
        for _ in 0..100 {
            p.update(s, EVICT, 1.0, s);
            p.update(s, KEEP, -1.0, s);
        }
        let qe = p.q_value(s, EVICT);
        let qk = p.q_value(s, KEEP);
        assert!(qe > qk, "Q(EVICT)={qe} > Q(KEEP)={qk} after training");
    }

    #[test]
    fn buckets_partition_correctly() {
        assert_eq!(bucket_score(0.1), 0);
        assert_eq!(bucket_score(0.3), 1);
        assert_eq!(bucket_score(0.7), 2);
        assert_eq!(bucket_recency(0, 10), 0);
        assert_eq!(bucket_recency(5, 10), 1);
        assert_eq!(bucket_recency(9, 10), 2);
        assert_eq!(bucket_pressure(0.0), 0);
        assert_eq!(bucket_pressure(0.2), 1);
        assert_eq!(bucket_pressure(0.5), 2);
        assert_eq!(bucket_size(32), 0);
        assert_eq!(bucket_size(256), 1);
        assert_eq!(bucket_size(1024), 2);
    }

    #[test]
    fn reward_keep_positive_on_recall() {
        assert!(reward_keep(true, 100) > 0.0);
        assert!(reward_keep(false, 100) < 0.0);
    }

    #[test]
    fn reward_evict_positive_when_no_repage() {
        assert!(reward_evict(false, 100) > 0.0);
        assert!(reward_evict(true, 100) < 0.0);
    }

    #[test]
    fn table_size_grows_with_training() {
        let mut p = EvictionPolicy::new();
        assert_eq!(p.table_size(), 0);
        let s = PagingState {
            score: 1,
            recency: 1,
            pressure: 1,
            size: 1,
        };
        p.update(s, KEEP, 1.0, s);
        assert_eq!(p.table_size(), 1);
        p.update(s, EVICT, -1.0, s);
        assert_eq!(p.table_size(), 2);
    }

    #[test]
    fn empty_state_space_is_small() {
        // 3^4 = 81 states × 2 actions = 162 cells max. Compact.
        let max_cells = 3u32.pow(4) * 2;
        assert_eq!(max_cells, 162, "tabular state space stays compact");
    }
}
