use crate::core::Reasoner;
use crate::graph::kg::KnowledgeGraph;

pub struct GraphReasoning {
    pub kg: KnowledgeGraph,
}

impl GraphReasoning {
    pub fn new(kg: KnowledgeGraph) -> Self {
        Self { kg }
    }

    pub fn find_path(&self, _start: &str, _end: &str) -> Vec<String> {
        // Path finding implementation over KG
        vec![]
    }
}

impl Reasoner for GraphReasoning {
    fn name(&self) -> &str {
        "GraphReasoning"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::kg::KnowledgeGraph;

    #[test]
    fn test_graph_reasoning_name() {
        let kg = KnowledgeGraph::new();
        let gr = GraphReasoning::new(kg);
        assert_eq!(gr.name(), "GraphReasoning");
    }
}
