//! Group-conditional drift monitoring (continual program, phase 6.2).
//!
//! # The blind spot this exists to close
//!
//! Phase 6.1's [`DriftMonitor`] compares a rolling **median** against a frozen
//! reference. The median is the robust choice, and phase 6.1 defended it on
//! exactly that ground: a minority of wild scores can neither fake nor mask
//! the signal.
//!
//! That property cuts both ways, and the second edge is sharper than it looks.
//! A minority of wild scores cannot move the pooled median — *including when
//! those scores are a real subpopulation whose model has genuinely collapsed*.
//! Feed one monitor a stream that is 85% healthy and 15% catastrophic and the
//! pooled median sits calmly inside its threshold while one sixth of the
//! traffic is served predictions nobody is watching. The breakdown point that
//! protects against contamination is indistinguishable, from inside the
//! monitor, from a breakdown point that hides a stratum.
//!
//! This is the drift-dimension twin of the argument phase 4E.6 made for
//! coverage: marginal coverage can look perfect while a group is starved, so
//! Mondrian conformal prediction conditions on the group. [`GroupDriftMonitor`]
//! conditions the same way — one independent [`DriftMonitor`] per declared
//! group, each with **its own** calibrated reference scale, so a group whose
//! residuals are naturally larger is not permanently alarmed and a group whose
//! residuals are naturally small cannot have its collapse averaged away.
//!
//! # What conditioning buys, and what it does not
//!
//! It buys **localization in the declared partition**: the monitor reports
//! *which* groups are alarmed, in canonical order. It does **not** buy cause
//! attribution — phase 6.1's rule stands unchanged, and a group label is not
//! an explanation. Above all it does not buy safety outside the partition the
//! caller declared: a grouping that does not separate the failure mode leaves
//! the monitor exactly as blind as the pooled one, and nothing here can detect
//! that the grouping was wrong.
//!
//! # Silence is not health
//!
//! Conditioning multiplies the warm-up cost: every group needs its own
//! `minimum_samples` before it says anything, and a rare group reaches that
//! slowly — or never. A group with no traffic is not a quiet group, it is an
//! **unmonitored** one, and the two must never render the same. So the
//! aggregate [`GroupDriftState::Quiet`] is deliberately not the whole story,
//! and [`GroupDriftMonitor::warming_groups`] exists to make the residual
//! blindness reportable rather than implicit.
//!
//! For the same reason an observation carrying an undeclared group is a typed
//! error, never quietly pooled into a default bucket: silent pooling is
//! precisely the failure this module exists to prevent, and re-introducing it
//! on the ingest path would be self-defeating.
//!
//! All deterministic; no RNG, no clock.

use core::fmt;

use crate::drift_monitor::{DriftMonitor, DriftMonitorConfig, DriftMonitorError, DriftState};

/// Configuration for a [`GroupDriftMonitor`].
#[derive(Clone, Debug, PartialEq)]
pub struct GroupDriftConfig {
    /// One `(group name, configuration)` pair per declared group. Each group
    /// carries its own calibrated `reference_scale`; sharing one scale across
    /// groups with different natural residual magnitudes is the mistake this
    /// signature exists to make awkward.
    pub groups: Vec<(String, DriftMonitorConfig)>,
    /// How many groups must be simultaneously alarmed before the aggregate
    /// state is [`GroupDriftState::Alarmed`]. Must be in `1..=groups.len()`.
    /// `1` — any single group — is the safe default: a stratum failing alone
    /// is still a failure.
    pub groups_to_alarm: usize,
}

impl GroupDriftConfig {
    /// Builds a configuration giving every group the same drift parameters
    /// except for its own reference scale, alarming on any single group.
    ///
    /// # Errors
    ///
    /// [`GroupDriftError`] under the same conditions as
    /// [`GroupDriftMonitor::new`].
    pub fn uniform(
        template: DriftMonitorConfig,
        scales: &[(&str, f64)],
    ) -> Result<Self, GroupDriftError> {
        if scales.is_empty()
        {
            return Err(GroupDriftError::NoGroups);
        }
        let groups = scales
            .iter()
            .map(|(name, scale)| {
                (
                    (*name).to_string(),
                    DriftMonitorConfig {
                        reference_scale: *scale,
                        ..template
                    },
                )
            })
            .collect();
        Ok(Self {
            groups,
            groups_to_alarm: 1,
        })
    }
}

/// The aggregate state across all declared groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupDriftState {
    /// No group has met its warm-up, so nothing is actionable yet.
    Warming,
    /// At least one group is actionable and fewer than `groups_to_alarm` are
    /// alarmed. Read it together with [`GroupDriftMonitor::warming_groups`] —
    /// groups still warming are unmonitored, not healthy.
    Quiet,
    /// At least `groups_to_alarm` groups are alarmed.
    Alarmed,
}

/// Typed group-drift errors.
#[derive(Clone, Debug, PartialEq)]
pub enum GroupDriftError {
    /// No groups were declared.
    NoGroups,
    /// A group name was empty or blank.
    EmptyGroupName,
    /// The same group name was declared twice.
    DuplicateGroup {
        /// The repeated name.
        name: String,
    },
    /// The alarm quorum was zero or exceeded the number of groups.
    InvalidQuorum {
        /// The requested quorum.
        groups_to_alarm: usize,
        /// The number of declared groups.
        groups: usize,
    },
    /// An observation arrived for a group that was never declared. It is
    /// refused rather than pooled into a default bucket: silent pooling is the
    /// blindness this module exists to remove.
    UnknownGroup {
        /// The undeclared group.
        name: String,
    },
    /// One group's configuration or observed score was invalid.
    Group {
        /// The group whose monitor rejected the input.
        name: String,
        /// The underlying drift-monitor error.
        source: DriftMonitorError,
    },
}

impl fmt::Display for GroupDriftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NoGroups => formatter.write_str("at least one group must be declared"),
            Self::EmptyGroupName => formatter.write_str("group names must not be empty"),
            Self::DuplicateGroup { name } =>
            {
                write!(formatter, "group '{name}' was declared more than once")
            },
            Self::InvalidQuorum {
                groups_to_alarm,
                groups,
            } => write!(
                formatter,
                "alarm quorum {groups_to_alarm} must be in 1..={groups}"
            ),
            Self::UnknownGroup { name } => write!(
                formatter,
                "group '{name}' was never declared; an undeclared stratum is refused rather than \
                 pooled"
            ),
            Self::Group { name, source } => write!(formatter, "group '{name}': {source}"),
        }
    }
}

impl std::error::Error for GroupDriftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self
        {
            Self::Group { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// One group's externally visible drift state.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupReport {
    /// The group name.
    pub name: String,
    /// That group's drift state.
    pub state: DriftState,
    /// That group's window median as a multiple of its own reference scale.
    pub ratio: f64,
    /// How many observations that group has received in total.
    pub observations: usize,
}

/// A drift monitor conditioned on a declared partition of the stream. See the
/// module documentation for what conditioning does and does not buy.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupDriftMonitor {
    groups: Vec<String>,
    monitors: Vec<DriftMonitor>,
    counts: Vec<usize>,
    groups_to_alarm: usize,
}

impl GroupDriftMonitor {
    /// Builds a monitor over the declared groups.
    ///
    /// Group order is canonicalized by name (`str` ordering), so every
    /// accessor returns groups in a deterministic order independent of
    /// declaration order.
    ///
    /// # Errors
    ///
    /// [`GroupDriftError`] when no groups are declared, a name is empty or
    /// repeated, the quorum is out of range, or a group's drift configuration
    /// is invalid.
    pub fn new(config: GroupDriftConfig) -> Result<Self, GroupDriftError> {
        if config.groups.is_empty()
        {
            return Err(GroupDriftError::NoGroups);
        }
        if config.groups_to_alarm == 0 || config.groups_to_alarm > config.groups.len()
        {
            return Err(GroupDriftError::InvalidQuorum {
                groups_to_alarm: config.groups_to_alarm,
                groups: config.groups.len(),
            });
        }
        let mut declared = config.groups;
        declared.sort_by(|left, right| left.0.cmp(&right.0));
        let mut groups = Vec::with_capacity(declared.len());
        let mut monitors = Vec::with_capacity(declared.len());
        for (name, drift) in declared
        {
            if name.trim().is_empty()
            {
                return Err(GroupDriftError::EmptyGroupName);
            }
            if groups.last().is_some_and(|previous| *previous == name)
            {
                return Err(GroupDriftError::DuplicateGroup { name });
            }
            let monitor = DriftMonitor::new(drift).map_err(|source| GroupDriftError::Group {
                name: name.clone(),
                source,
            })?;
            groups.push(name);
            monitors.push(monitor);
        }
        let counts = vec![0; groups.len()];
        Ok(Self {
            groups,
            monitors,
            counts,
            groups_to_alarm: config.groups_to_alarm,
        })
    }

    /// Records one residual score against its group and returns the aggregate
    /// state.
    ///
    /// # Errors
    ///
    /// [`GroupDriftError::UnknownGroup`] for an undeclared group and
    /// [`GroupDriftError::Group`] for an invalid score; in both cases every
    /// group's state is left untouched.
    pub fn observe(&mut self, group: &str, score: f64) -> Result<GroupDriftState, GroupDriftError> {
        let index = self
            .index_of(group)
            .ok_or_else(|| GroupDriftError::UnknownGroup {
                name: group.to_string(),
            })?;
        self.monitors[index]
            .observe(score)
            .map_err(|source| GroupDriftError::Group {
                name: self.groups[index].clone(),
                source,
            })?;
        self.counts[index] += 1;
        Ok(self.state())
    }

    /// The aggregate state without observing anything.
    pub fn state(&self) -> GroupDriftState {
        let alarmed = self
            .monitors
            .iter()
            .filter(|monitor| monitor.state() == DriftState::Alarmed)
            .count();
        if alarmed >= self.groups_to_alarm
        {
            return GroupDriftState::Alarmed;
        }
        if self
            .monitors
            .iter()
            .any(|monitor| monitor.state() != DriftState::Warming)
        {
            GroupDriftState::Quiet
        }
        else
        {
            GroupDriftState::Warming
        }
    }

    /// The alarmed groups, in canonical order.
    pub fn alarmed_groups(&self) -> Vec<&str> {
        self.groups_in_state(DriftState::Alarmed)
    }

    /// The groups that have not met their warm-up, in canonical order. These
    /// are **unmonitored**, not healthy — see the module documentation.
    pub fn warming_groups(&self) -> Vec<&str> {
        self.groups_in_state(DriftState::Warming)
    }

    /// Every group's state, in canonical order.
    pub fn reports(&self) -> Vec<GroupReport> {
        self.groups
            .iter()
            .enumerate()
            .map(|(index, name)| GroupReport {
                name: name.clone(),
                state: self.monitors[index].state(),
                ratio: self.monitors[index].ratio(),
                observations: self.counts[index],
            })
            .collect()
    }

    /// One group's state, or `None` when it was never declared.
    pub fn group_state(&self, group: &str) -> Option<DriftState> {
        self.index_of(group)
            .map(|index| self.monitors[index].state())
    }

    /// One group's window median as a multiple of its own reference scale, or
    /// `None` when it was never declared.
    pub fn group_ratio(&self, group: &str) -> Option<f64> {
        self.index_of(group)
            .map(|index| self.monitors[index].ratio())
    }

    /// How many observations one group has received, or `None` when it was
    /// never declared.
    pub fn observations(&self, group: &str) -> Option<usize> {
        self.index_of(group).map(|index| self.counts[index])
    }

    /// The declared groups, in canonical order.
    pub fn groups(&self) -> Vec<&str> {
        self.groups.iter().map(String::as_str).collect()
    }

    fn groups_in_state(&self, state: DriftState) -> Vec<&str> {
        self.groups
            .iter()
            .enumerate()
            .filter(|(index, _)| self.monitors[*index].state() == state)
            .map(|(_, name)| name.as_str())
            .collect()
    }

    fn index_of(&self, group: &str) -> Option<usize> {
        self.groups
            .binary_search_by(|candidate| candidate.as_str().cmp(group))
            .ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template() -> DriftMonitorConfig {
        DriftMonitorConfig {
            reference_scale: 1.0,
            alarm_ratio: 3.0,
            window: 8,
            minimum_samples: 4,
            consecutive_breaches_to_alarm: 3,
            consecutive_normals_to_clear: 5,
        }
    }

    fn four_groups() -> GroupDriftConfig {
        GroupDriftConfig::uniform(
            template(),
            &[("east", 1.0), ("north", 1.0), ("south", 1.0), ("west", 1.0)],
        )
        .unwrap()
    }

    /// The phase's thesis, as an executable check: a pooled monitor fed the
    /// *same* mixed stream stays Quiet while the conditioned monitor alarms on
    /// the collapsed stratum. The minority that cannot move a median is
    /// exactly the minority a median cannot protect.
    #[test]
    fn the_pooled_median_is_blind_to_a_minority_group_collapse() {
        let mut pooled = DriftMonitor::new(template()).unwrap();
        let mut conditioned = GroupDriftMonitor::new(four_groups()).unwrap();

        // Six healthy observations per wild one: the collapsed stratum is a
        // clear minority of the pooled stream.
        let healthy = ["east", "north", "south"];
        for round in 0..40
        {
            for (offset, group) in healthy.iter().enumerate()
            {
                let score = 0.9 + 0.05 * ((round + offset) % 3) as f64;
                pooled.observe(score).unwrap();
                conditioned.observe(group, score).unwrap();
            }
            pooled.observe(0.9).unwrap();
            pooled.observe(0.9).unwrap();
            let collapsed = 12.0;
            pooled.observe(collapsed).unwrap();
            conditioned.observe("west", collapsed).unwrap();
        }

        assert_eq!(
            pooled.state(),
            DriftState::Quiet,
            "the pooled median stayed inside its threshold at ratio {}",
            pooled.ratio()
        );
        assert!(pooled.ratio() < 3.0);
        assert_eq!(conditioned.state(), GroupDriftState::Alarmed);
        assert_eq!(conditioned.alarmed_groups(), vec!["west"]);
        assert_eq!(conditioned.group_state("east"), Some(DriftState::Quiet));
    }

    #[test]
    fn a_uniformly_healthy_stream_alarms_nowhere() {
        let mut monitor = GroupDriftMonitor::new(four_groups()).unwrap();
        for round in 0..30
        {
            for group in ["east", "north", "south", "west"]
            {
                monitor
                    .observe(group, 0.8 + 0.1 * (round % 3) as f64)
                    .unwrap();
            }
        }
        assert_eq!(monitor.state(), GroupDriftState::Quiet);
        assert!(monitor.alarmed_groups().is_empty());
        assert!(monitor.warming_groups().is_empty());
    }

    #[test]
    fn each_group_is_judged_against_its_own_reference_scale() {
        // `west` naturally runs an order of magnitude hotter. With its own
        // scale it is quiet; judged against `east`'s it would alarm at once.
        let config =
            GroupDriftConfig::uniform(template(), &[("east", 1.0), ("west", 10.0)]).unwrap();
        let mut monitor = GroupDriftMonitor::new(config).unwrap();
        for _ in 0..20
        {
            monitor.observe("east", 1.0).unwrap();
            monitor.observe("west", 10.0).unwrap();
        }
        assert_eq!(monitor.state(), GroupDriftState::Quiet);
        assert_eq!(monitor.group_ratio("west"), Some(1.0));
        assert_eq!(monitor.group_ratio("east"), Some(1.0));
    }

    #[test]
    fn a_group_with_no_traffic_is_reported_as_unmonitored_not_healthy() {
        let mut monitor = GroupDriftMonitor::new(four_groups()).unwrap();
        for _ in 0..20
        {
            monitor.observe("east", 0.9).unwrap();
        }
        // The aggregate reads Quiet, which is true and insufficient: three of
        // four strata have never been looked at.
        assert_eq!(monitor.state(), GroupDriftState::Quiet);
        assert_eq!(monitor.warming_groups(), vec!["north", "south", "west"]);
        assert_eq!(monitor.observations("north"), Some(0));
    }

    #[test]
    fn the_quorum_can_demand_more_than_one_alarmed_group() {
        let mut config = four_groups();
        config.groups_to_alarm = 2;
        let mut monitor = GroupDriftMonitor::new(config).unwrap();
        for _ in 0..20
        {
            for group in ["east", "north", "south"]
            {
                monitor.observe(group, 0.9).unwrap();
            }
            monitor.observe("west", 12.0).unwrap();
        }
        assert_eq!(monitor.alarmed_groups(), vec!["west"]);
        assert_eq!(
            monitor.state(),
            GroupDriftState::Quiet,
            "one alarmed group does not meet a quorum of two"
        );
        for _ in 0..20
        {
            monitor.observe("south", 12.0).unwrap();
        }
        assert_eq!(monitor.alarmed_groups(), vec!["south", "west"]);
        assert_eq!(monitor.state(), GroupDriftState::Alarmed);
    }

    #[test]
    fn an_undeclared_group_is_refused_rather_than_pooled() {
        let mut monitor = GroupDriftMonitor::new(four_groups()).unwrap();
        let before = monitor.clone();
        assert_eq!(
            monitor.observe("northeast", 0.9),
            Err(GroupDriftError::UnknownGroup {
                name: "northeast".to_string()
            })
        );
        assert_eq!(monitor, before, "a refused observation changes nothing");
    }

    #[test]
    fn an_invalid_score_names_its_group_and_changes_nothing() {
        let mut monitor = GroupDriftMonitor::new(four_groups()).unwrap();
        monitor.observe("east", 0.9).unwrap();
        let before = monitor.clone();
        assert!(matches!(
            monitor.observe("east", f64::NAN),
            Err(GroupDriftError::Group { ref name, source: DriftMonitorError::InvalidScore { .. } })
                if name == "east"
        ));
        assert_eq!(monitor, before);
        assert_eq!(monitor.observations("east"), Some(1));
    }

    #[test]
    fn groups_are_canonically_ordered_regardless_of_declaration_order() {
        let forward =
            GroupDriftConfig::uniform(template(), &[("east", 1.0), ("north", 1.0), ("west", 1.0)])
                .unwrap();
        let reversed =
            GroupDriftConfig::uniform(template(), &[("west", 1.0), ("north", 1.0), ("east", 1.0)])
                .unwrap();
        let forward = GroupDriftMonitor::new(forward).unwrap();
        let reversed = GroupDriftMonitor::new(reversed).unwrap();
        assert_eq!(forward.groups(), vec!["east", "north", "west"]);
        assert_eq!(forward, reversed);
    }

    #[test]
    fn invalid_configurations_are_typed_errors() {
        assert_eq!(
            GroupDriftConfig::uniform(template(), &[]).unwrap_err(),
            GroupDriftError::NoGroups
        );
        let mut bad_quorum = four_groups();
        bad_quorum.groups_to_alarm = 5;
        assert_eq!(
            GroupDriftMonitor::new(bad_quorum).unwrap_err(),
            GroupDriftError::InvalidQuorum {
                groups_to_alarm: 5,
                groups: 4
            }
        );
        let mut zero_quorum = four_groups();
        zero_quorum.groups_to_alarm = 0;
        assert!(matches!(
            GroupDriftMonitor::new(zero_quorum),
            Err(GroupDriftError::InvalidQuorum { .. })
        ));
        let duplicated =
            GroupDriftConfig::uniform(template(), &[("east", 1.0), ("east", 1.0)]).unwrap();
        assert_eq!(
            GroupDriftMonitor::new(duplicated).unwrap_err(),
            GroupDriftError::DuplicateGroup {
                name: "east".to_string()
            }
        );
        let blank = GroupDriftConfig::uniform(template(), &[("  ", 1.0)]).unwrap();
        assert_eq!(
            GroupDriftMonitor::new(blank).unwrap_err(),
            GroupDriftError::EmptyGroupName
        );
        let bad_scale = GroupDriftConfig::uniform(template(), &[("east", 0.0)]).unwrap();
        assert!(matches!(
            GroupDriftMonitor::new(bad_scale),
            Err(GroupDriftError::Group {
                source: DriftMonitorError::InvalidReferenceScale { .. },
                ..
            })
        ));
    }

    #[test]
    fn a_cleared_group_clears_the_aggregate() {
        let mut monitor = GroupDriftMonitor::new(four_groups()).unwrap();
        for _ in 0..20
        {
            monitor.observe("west", 12.0).unwrap();
        }
        assert_eq!(monitor.state(), GroupDriftState::Alarmed);
        for _ in 0..20
        {
            monitor.observe("west", 0.9).unwrap();
        }
        assert_eq!(monitor.state(), GroupDriftState::Quiet);
        assert!(monitor.alarmed_groups().is_empty());
    }

    #[test]
    fn identical_event_sequences_produce_identical_monitors() {
        let run = || {
            let mut monitor = GroupDriftMonitor::new(four_groups()).unwrap();
            for round in 0..50
            {
                for (offset, group) in ["east", "north", "south", "west"].iter().enumerate()
                {
                    monitor
                        .observe(group, 0.5 + 0.4 * ((round + offset) % 7) as f64)
                        .unwrap();
                }
            }
            monitor
        };
        assert_eq!(run(), run());
        assert_eq!(run().reports(), run().reports());
    }
}
