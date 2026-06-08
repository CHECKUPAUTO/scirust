use crate::core::{Result, Reasoner};

pub struct Rule {
    pub head: String,
    pub body: Vec<String>,
}

pub struct RuleEngine {
    pub rules: Vec<Rule>,
}

impl RuleEngine {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    pub fn add_rule(&mut self, head: &str, body: Vec<&str>) {
        self.rules.push(Rule {
            head: head.to_string(),
            body: body.into_iter().map(|s| s.to_string()).collect(),
        });
    }

    pub fn forward_chain(&self) -> Result<Vec<String>> {
        // Forward chaining implementation
        Ok(vec![])
    }
}

impl Reasoner for RuleEngine {
    fn name(&self) -> &str {
        "RuleEngine"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_engine_name() {
        let re = RuleEngine::new();
        assert_eq!(re.name(), "RuleEngine");
    }
}
