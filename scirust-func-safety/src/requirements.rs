use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Status of a safety requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequirementStatus {
    Draft,
    Verified,
    Implemented,
    Tested,
    Certified,
    Deprecated,
}

/// A safety requirement with traceability links.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Requirement {
    pub id: String,
    pub description: String,
    pub asil_level: super::asil::AsilLevel,
    pub status: RequirementStatus,
    /// Linked source code locations (file:line)
    pub code_links: Vec<String>,
    /// Linked test cases
    pub test_links: Vec<String>,
    /// Parent requirement ID (for hierarchical decomposition)
    pub parent: Option<String>,
}

impl Requirement {
    pub fn new(id: &str, description: &str, asil: super::asil::AsilLevel) -> Self {
        Self {
            id: id.to_string(),
            description: description.to_string(),
            asil_level: asil,
            status: RequirementStatus::Draft,
            code_links: Vec::new(),
            test_links: Vec::new(),
            parent: None,
        }
    }

    pub fn add_code_link(&mut self, file: &str, line: u32) {
        self.code_links.push(format!("{}:{}", file, line));
    }

    pub fn add_test_link(&mut self, test_id: &str) {
        self.test_links.push(test_id.to_string());
    }

    pub fn is_fully_traced(&self) -> bool {
        !self.code_links.is_empty() && !self.test_links.is_empty()
    }
}

/// Traceability matrix mapping requirements to code and tests.
///
/// Required for ISO 26262 Part 6 (Product Software) compliance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceabilityMatrix {
    pub requirements: HashMap<String, Requirement>,
    /// Code locations covered by at least one requirement
    pub covered_code: HashMap<String, Vec<String>>, // code_loc -> req_ids
    /// Tests linked to requirements
    pub covered_tests: HashMap<String, Vec<String>>, // test_id -> req_ids
}

impl TraceabilityMatrix {
    pub fn new() -> Self {
        Self {
            requirements: HashMap::new(),
            covered_code: HashMap::new(),
            covered_tests: HashMap::new(),
        }
    }

    pub fn add_requirement(&mut self, req: Requirement) {
        for code_link in &req.code_links
        {
            self.covered_code
                .entry(code_link.clone())
                .or_default()
                .push(req.id.clone());
        }
        for test_link in &req.test_links
        {
            self.covered_tests
                .entry(test_link.clone())
                .or_default()
                .push(req.id.clone());
        }
        self.requirements.insert(req.id.clone(), req);
    }

    /// Check coverage: all requirements must have code and test links.
    pub fn check_coverage(&self) -> Vec<String> {
        self.requirements
            .values()
            .filter(|r| !r.is_fully_traced())
            .map(|r| r.id.clone())
            .collect()
    }

    /// Coverage percentage.
    pub fn coverage_percent(&self) -> f32 {
        if self.requirements.is_empty()
        {
            return 0.0;
        }
        let covered = self
            .requirements
            .values()
            .filter(|r| r.is_fully_traced())
            .count();
        (covered as f32 / self.requirements.len() as f32) * 100.0
    }

    /// Export as JSON for documentation/dossiers.
    pub fn export_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    /// Generate a text report for certification dossiers.
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# Traceability Matrix Report\n\n");
        report.push_str(&format!(
            "Total requirements: {}\n",
            self.requirements.len()
        ));
        report.push_str(&format!("Coverage: {:.1}%\n\n", self.coverage_percent()));
        report.push_str("## Requirements\n\n");
        let mut reqs: Vec<&Requirement> = self.requirements.values().collect();
        reqs.sort_by(|a, b| a.id.cmp(&b.id));
        for r in reqs
        {
            report.push_str(&format!(
                "- **{}** [{}]: {} ({})\n",
                r.id,
                r.asil_level.label(),
                r.description,
                r.status_label()
            ));
            if !r.code_links.is_empty()
            {
                report.push_str(&format!("  - Code: {}\n", r.code_links.join(", ")));
            }
            if !r.test_links.is_empty()
            {
                report.push_str(&format!("  - Tests: {}\n", r.test_links.join(", ")));
            }
            if !r.is_fully_traced()
            {
                report.push_str("  - **WARNING: Incomplete traceability**\n");
            }
        }
        report
    }
}

impl Requirement {
    pub fn status_label(&self) -> &'static str {
        match self.status
        {
            RequirementStatus::Draft => "Draft",
            RequirementStatus::Verified => "Verified",
            RequirementStatus::Implemented => "Implemented",
            RequirementStatus::Tested => "Tested",
            RequirementStatus::Certified => "Certified",
            RequirementStatus::Deprecated => "Deprecated",
        }
    }
}

impl Default for TraceabilityMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::asil::AsilLevel;
    use super::*;

    #[test]
    fn test_requirement_creation() {
        let req = Requirement::new("REQ-001", "Brake detection within 5ms", AsilLevel::D);
        assert_eq!(req.id, "REQ-001");
        assert_eq!(req.asil_level, AsilLevel::D);
        assert!(!req.is_fully_traced());
    }

    #[test]
    fn test_requirement_with_links() {
        let mut req = Requirement::new("REQ-001", "Test", AsilLevel::B);
        req.add_code_link("src/main.rs", 42);
        req.add_test_link("test_brake_001");
        assert!(req.is_fully_traced());
    }

    #[test]
    fn test_traceability_matrix_coverage() {
        let mut tm = TraceabilityMatrix::new();
        let mut req1 = Requirement::new("REQ-001", "Complete", AsilLevel::B);
        req1.add_code_link("src/lib.rs", 10);
        req1.add_test_link("test_1");
        tm.add_requirement(req1);
        let req2 = Requirement::new("REQ-002", "Incomplete", AsilLevel::A);
        tm.add_requirement(req2);
        let uncovered = tm.check_coverage();
        assert_eq!(uncovered, vec!["REQ-002"]);
        assert!((tm.coverage_percent() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_export_json() {
        let mut tm = TraceabilityMatrix::new();
        let req = Requirement::new("REQ-001", "Test", AsilLevel::A);
        tm.add_requirement(req);
        let json = tm.export_json().unwrap();
        assert!(json.contains("REQ-001"));
    }

    #[test]
    fn test_generate_report() {
        let mut tm = TraceabilityMatrix::new();
        let req = Requirement::new("REQ-001", "Brake control", AsilLevel::D);
        tm.add_requirement(req);
        let report = tm.generate_report();
        assert!(report.contains("REQ-001"));
        assert!(report.contains("ASIL-D"));
        assert!(report.contains("Incomplete"));
    }

    #[test]
    fn test_empty_matrix_coverage() {
        let tm = TraceabilityMatrix::new();
        assert_eq!(tm.coverage_percent(), 0.0);
        assert!(tm.check_coverage().is_empty());
    }
}
