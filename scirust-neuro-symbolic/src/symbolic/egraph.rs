use scirust_symbolic::Expr;
use std::collections::{HashMap, HashSet};

/// E-Graph supporting equality saturation primitives: insertion, `union`
/// (merging equivalence classes via union-find) and `rebuild` (restoring
/// congruence closure so that structurally-identical nodes over merged children
/// become equivalent).
#[derive(Default)]
pub struct EGraph {
    /// Mapping from canonical E-class ID to its contents.
    pub classes: HashMap<usize, EClass>,
    /// Memoization table to find E-class by (canonical) node.
    pub memo: HashMap<ENode, usize>,
    /// Union-find parent array (indexed by E-class ID).
    pub uf: Vec<usize>,
    /// Next available ID for a new E-class.
    pub next_id: usize,
}

/// A node in the E-Graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ENode {
    Const(String),
    Var(String),
    Op(String, Vec<usize>),
}

/// An E-class containing equivalent nodes.
#[derive(Debug, Default)]
pub struct EClass {
    pub id: usize,
    pub nodes: HashSet<ENode>,
    pub parents: HashSet<(ENode, usize)>,
}

impl EGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an expression to the E-Graph and return its E-class ID.
    pub fn add_expr(&mut self, expr: &Expr) -> usize {
        let node = match expr
        {
            Expr::Const(c) => ENode::Const(c.to_string()),
            Expr::Var(v) => ENode::Var(v.clone()),
            Expr::Add(a, b) => self.binop("+", a, b),
            Expr::Sub(a, b) => self.binop("-", a, b),
            Expr::Mul(a, b) => self.binop("*", a, b),
            Expr::Div(a, b) => self.binop("/", a, b),
            Expr::Pow(a, b) => self.binop("^", a, b),
            Expr::Neg(a) => self.unop("neg", a),
            Expr::Sin(a) => self.unop("sin", a),
            Expr::Cos(a) => self.unop("cos", a),
            Expr::Exp(a) => self.unop("exp", a),
            Expr::Ln(a) => self.unop("ln", a),
            Expr::Sqrt(a) => self.unop("sqrt", a),
            Expr::Abs(a) => self.unop("abs", a),
        };
        self.add_node(node)
    }

    fn binop(&mut self, op: &str, a: &Expr, b: &Expr) -> ENode {
        let id1 = self.add_expr(a);
        let id2 = self.add_expr(b);
        ENode::Op(op.to_string(), vec![id1, id2])
    }

    fn unop(&mut self, op: &str, a: &Expr) -> ENode {
        let id = self.add_expr(a);
        ENode::Op(op.to_string(), vec![id])
    }

    fn add_node(&mut self, node: ENode) -> usize {
        let canon = self.canonicalize(&node);
        if let Some(&id) = self.memo.get(&canon)
        {
            return self.find(id);
        }
        let id = self.next_id;
        self.next_id += 1;
        debug_assert_eq!(id, self.uf.len());
        self.uf.push(id);
        let mut class = EClass {
            id,
            ..Default::default()
        };
        class.nodes.insert(canon.clone());
        self.classes.insert(id, class);
        self.memo.insert(canon, id);
        id
    }

    /// Find the canonical (root) class ID with path compression.
    pub fn find(&mut self, id: usize) -> usize {
        let mut root = id;
        while self.uf[root] != root
        {
            root = self.uf[root];
        }
        let mut cur = id;
        while self.uf[cur] != root
        {
            let next = self.uf[cur];
            self.uf[cur] = root;
            cur = next;
        }
        root
    }

    /// Merge the two E-classes containing `id1` and `id2`.
    pub fn union(&mut self, id1: usize, id2: usize) {
        let a = self.find(id1);
        let b = self.find(id2);
        if a == b
        {
            return;
        }
        self.uf[a] = b;
        if let Some(class_a) = self.classes.remove(&a)
        {
            let class_b = self.classes.get_mut(&b).expect("root class must exist");
            for n in class_a.nodes
            {
                class_b.nodes.insert(n);
            }
            for p in class_a.parents
            {
                class_b.parents.insert(p);
            }
        }
    }

    /// True iff the two IDs are in the same equivalence class.
    pub fn equivalent(&mut self, id1: usize, id2: usize) -> bool {
        self.find(id1) == self.find(id2)
    }

    fn canonicalize(&mut self, node: &ENode) -> ENode {
        match node
        {
            ENode::Op(op, children) =>
            {
                ENode::Op(op.clone(), children.iter().map(|&c| self.find(c)).collect())
            },
            other => other.clone(),
        }
    }

    /// Restore congruence closure: re-canonicalise every node and union any two
    /// classes that contain the same canonical node, iterating to a fixpoint.
    pub fn rebuild(&mut self) {
        loop
        {
            let mut seen: HashMap<ENode, usize> = HashMap::new();
            let mut to_union: Vec<(usize, usize)> = Vec::new();
            let roots: Vec<usize> = self.classes.keys().copied().collect();
            for cid in roots
            {
                let nodes: Vec<ENode> = self.classes[&cid].nodes.iter().cloned().collect();
                for node in nodes
                {
                    let cnode = self.canonicalize(&node);
                    match seen.get(&cnode)
                    {
                        Some(&other) if self.find(other) != self.find(cid) =>
                        {
                            to_union.push((other, cid));
                        },
                        Some(_) =>
                        {},
                        None =>
                        {
                            seen.insert(cnode, cid);
                        },
                    }
                }
            }
            if to_union.is_empty()
            {
                break;
            }
            for (a, b) in to_union
            {
                self.union(a, b);
            }
        }

        // Rebuild the memo table with canonical nodes / roots.
        self.memo.clear();
        let roots: Vec<usize> = self.classes.keys().copied().collect();
        for cid in roots
        {
            let nodes: Vec<ENode> = self.classes[&cid].nodes.iter().cloned().collect();
            let root = self.find(cid);
            for node in nodes
            {
                let c = self.canonicalize(&node);
                self.memo.insert(c, root);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_egraph_add() {
        let mut eg = EGraph::new();
        let expr = Expr::Add(Box::new(Expr::Const(1.0)), Box::new(Expr::Const(2.0)));
        let id = eg.add_expr(&expr);
        assert!(eg.classes.contains_key(&id));
    }

    #[test]
    fn union_makes_classes_equivalent() {
        let mut eg = EGraph::new();
        let x = eg.add_expr(&Expr::Var("x".into()));
        let y = eg.add_expr(&Expr::Var("y".into()));
        assert!(!eg.equivalent(x, y));
        eg.union(x, y);
        assert!(eg.equivalent(x, y));
    }

    #[test]
    fn rebuild_propagates_congruence() {
        // f(x) and f(y): once x≡y, congruence ⇒ f(x)≡f(y) after rebuild.
        let mut eg = EGraph::new();
        let fx = eg.add_expr(&Expr::Sin(Box::new(Expr::Var("x".into()))));
        let fy = eg.add_expr(&Expr::Sin(Box::new(Expr::Var("y".into()))));
        assert!(!eg.equivalent(fx, fy));
        let x = eg.add_expr(&Expr::Var("x".into()));
        let y = eg.add_expr(&Expr::Var("y".into()));
        eg.union(x, y);
        eg.rebuild();
        assert!(eg.equivalent(fx, fy), "congruence: x≡y ⇒ sin(x)≡sin(y)");
    }
}
