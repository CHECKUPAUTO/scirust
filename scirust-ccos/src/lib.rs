#![forbid(unsafe_code)]
//! Cognitive Context Orchestration System (CCOS) interfacing and SoulLink Multi-Agent Mesh Network.

use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};
use scirust_elliptic_discovery::{SearchPlan, execute_campaign};

/// Domain separator for CCOS campaign artifacts.
pub const DOMAIN: &[u8] = b"SCIRUST-ELLIPTIC-DISCOVERY/CAMPAIGN/V1";

/// Validation state for campaign runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationState {
    Pending,
    Certified,
    Refuted,
}

impl ValidationState {
    pub const fn tag(self) -> u8 {
        match self {
            Self::Pending => 0,
            Self::Certified => 1,
            Self::Refuted => 2,
        }
    }
}

/// An immutable, verifiable artifact in the CCOS semantic memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CcosCampaignArtifact {
    pub sequence: u64,
    pub plan_fingerprint: [u8; 32],
    pub campaign_fingerprint: [u8; 32],
    pub state: ValidationState,
    pub timestamp_ns: u128,
    pub prev_hash: [u8; 32],
    pub chain_hash: [u8; 32],
}

impl CcosCampaignArtifact {
    /// Computes the canonical chain hash of this artifact.
    pub fn compute_chain_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(DOMAIN);
        hasher.update(self.sequence.to_be_bytes());
        hasher.update(self.plan_fingerprint);
        hasher.update(self.campaign_fingerprint);
        hasher.update([self.state.tag()]);
        hasher.update(self.timestamp_ns.to_be_bytes());
        hasher.update(self.prev_hash);
        hasher.finalize().into()
    }

    /// Verifies if the artifact's chain hash matches the payload.
    pub fn verify(&self, expected_prev: [u8; 32]) -> bool {
        self.prev_hash == expected_prev && self.chain_hash == self.compute_chain_hash()
    }
}

/// Temporal knowledge graph/semantic memory log of CCOS.
#[derive(Clone, Debug, Default)]
pub struct CcosSemanticMemory {
    artifacts: Vec<CcosCampaignArtifact>,
}

impl CcosSemanticMemory {
    pub fn new() -> Self {
        Self {
            artifacts: Vec::new(),
        }
    }

    /// Appends a new immutable campaign artifact to the semantic memory.
    pub fn append(
        &mut self,
        plan_fingerprint: [u8; 32],
        campaign_fingerprint: [u8; 32],
        state: ValidationState,
    ) -> &CcosCampaignArtifact {
        let sequence = self.artifacts.len() as u64;
        let prev_hash = self
            .artifacts
            .last()
            .map(|art| art.chain_hash)
            .unwrap_or([0u8; 32]);
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        let mut artifact = CcosCampaignArtifact {
            sequence,
            plan_fingerprint,
            campaign_fingerprint,
            state,
            timestamp_ns,
            prev_hash,
            chain_hash: [0u8; 32],
        };
        artifact.chain_hash = artifact.compute_chain_hash();
        self.artifacts.push(artifact);
        self.artifacts.last().unwrap()
    }

    pub fn artifacts(&self) -> &[CcosCampaignArtifact] {
        &self.artifacts
    }

    /// Verifies the cryptographic integrity of the entire memory chain.
    pub fn verify_integrity(&self) -> bool {
        let mut expected_prev = [0u8; 32];
        for artifact in &self.artifacts {
            if !artifact.verify(expected_prev) {
                return false;
            }
            expected_prev = artifact.chain_hash;
        }
        true
    }
}

/// Orchestrator for the Mesh Network SoulLink multi-agent network.
pub struct SoulLinkMesh {
    agents: Vec<String>,
    pub memory: CcosSemanticMemory,
}

impl SoulLinkMesh {
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            memory: CcosSemanticMemory::new(),
        }
    }

    /// Registers a new sub-agent.
    pub fn register_agent(&mut self, agent_id: &str) {
        if !self.agents.iter().any(|a| a == agent_id) {
            self.agents.push(agent_id.to_string());
        }
    }

    /// Submits a dynamic SearchPlan from an agent, validating it via the canonical controls.
    pub fn submit_plan(&mut self, agent_id: &str, plan: SearchPlan) -> Result<&CcosCampaignArtifact, String> {
        self.register_agent(agent_id);

        let plan_fingerprint = plan.fingerprint();

        // 1. Log the Pending state to CCOS semantic memory
        self.memory.append(plan_fingerprint, [0u8; 32], ValidationState::Pending);

        // 2. Execute the campaign deterministically (reproducing 6 strict canonical controls)
        let campaign_run = execute_campaign(plan);
        let campaign_fingerprint = campaign_run.fingerprint();

        // 3. Complete the orchestration with Certified or Refuted status
        let final_state = if campaign_run.controls_valid() {
            ValidationState::Certified
        } else {
            ValidationState::Refuted
        };

        let final_artifact = self.memory.append(plan_fingerprint, campaign_fingerprint, final_state);
        Ok(final_artifact)
    }

    pub fn registered_agents(&self) -> &[String] {
        &self.agents
    }
}

impl Default for SoulLinkMesh {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_memory_chaining() {
        let mut mem = CcosSemanticMemory::new();
        assert!(mem.verify_integrity());

        mem.append([1; 32], [2; 32], ValidationState::Pending);
        assert!(mem.verify_integrity());
        assert_eq!(mem.artifacts().len(), 1);

        mem.append([3; 32], [4; 32], ValidationState::Certified);
        assert!(mem.verify_integrity());
        assert_eq!(mem.artifacts().len(), 2);
    }

    #[test]
    fn test_soul_link_mesh_orchestration() {
        let mut mesh = SoulLinkMesh::new();
        let plan = SearchPlan::new(17, 1, 2, 3, 1, 1).expect("valid plan");
        let result = mesh.submit_plan("agent-alpha", plan);
        assert!(result.is_ok());
        let artifact = result.unwrap();
        assert_eq!(artifact.state, ValidationState::Certified);
        assert!(mesh.memory.verify_integrity());
        assert_eq!(mesh.registered_agents(), &["agent-alpha".to_string()]);
    }
}
