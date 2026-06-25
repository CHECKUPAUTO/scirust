use crate::core::Reasoner;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Entity(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Relation(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub subject: Entity,
    pub relation: Relation,
    pub object: Entity,
}

#[derive(Default)]
pub struct KnowledgeGraph {
    pub triples: HashSet<Triple>,
    pub entities: HashSet<Entity>,
    pub relations: HashSet<Relation>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_triple(&mut self, s: &str, r: &str, o: &str) {
        let subject = Entity(s.to_string());
        let relation = Relation(r.to_string());
        let object = Entity(o.to_string());

        self.entities.insert(subject.clone());
        self.entities.insert(object.clone());
        self.relations.insert(relation.clone());

        self.triples.insert(Triple {
            subject,
            relation,
            object,
        });
    }

    pub fn get_objects(&self, s: &str, r: &str) -> Vec<Entity> {
        let subject = Entity(s.to_string());
        let relation = Relation(r.to_string());

        self.triples
            .iter()
            .filter(|t| t.subject == subject && t.relation == relation)
            .map(|t| t.object.clone())
            .collect()
    }
}

impl Reasoner for KnowledgeGraph {
    fn name(&self) -> &str {
        "KnowledgeGraph"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kg_basic() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("Paris", "is_capital_of", "France");
        let objects = kg.get_objects("Paris", "is_capital_of");
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].0, "France");
    }

    #[test]
    fn kg_get_objects_returns_all_matches() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("A", "likes", "B");
        kg.add_triple("A", "likes", "C");
        kg.add_triple("A", "hates", "D");
        let objs = kg.get_objects("A", "likes");
        assert_eq!(objs.len(), 2);
        // triples is a HashSet, so sort before comparing.
        let mut names: Vec<String> = objs.iter().map(|e| e.0.clone()).collect();
        names.sort();
        assert_eq!(names, vec!["B".to_string(), "C".to_string()]);
    }
}
