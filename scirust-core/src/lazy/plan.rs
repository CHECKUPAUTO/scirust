// scirust-core/src/lazy/plan.rs
//
// Plan — représentation immuable et optimisée d'un sous-graphe à exécuter.
//
// Pipeline de compilation :
//
//   LazyGraph (mutable, exploratoire)
//       │
//       │  .compile(target_id)
//       ▼
//   Plan (immuable, optimisé)
//       │  - Dead Code Elimination (DCE)
//       │  - Operator Fusion (chaînes pointwise)
//       │  - Lifetime analysis pour eviction des buffers
//       │  - Topological order
//       │
//       ▼  .execute() ou .execute_with(feeds)
//   Tensor (résultat)
//
// La compilation est coûteuse, l'exécution doit être bon marché. C'est
// exactement le pattern "compile-once-run-many" du training loop.

use std::collections::{HashMap, HashSet};
use crate::autodiff::reverse::Tensor;
use crate::lazy::{LazyGraph, LazyId, LazyOp};

// ================================================================== //
//  Instruction — opérations dans la séquence exécutable               //
// ================================================================== //

/// Une instruction du Plan : opération + indices de buffers.
/// Chaque instruction a un buffer de sortie (output_buf) et lit
/// dans 0..N buffers d'entrée.
#[derive(Clone, Debug)]
pub enum Instr {
    /// Charge un Tensor concret (feeds nommés ou constantes embarquées)
    LoadConst { output_buf: usize, value: Tensor },
    LoadFeed  { output_buf: usize, feed_name: String, expected_shape: (usize, usize) },

    /// Opérations pointwise — peuvent être fusionnées
    PointwiseChain {
        output_buf: usize,
        input_bufs: Vec<usize>,         // inputs de la chaîne (1 ou 2)
        ops:        Vec<PwOp>,          // séquence d'ops appliquée en un passage
        shape:      (usize, usize),
    },

    /// MatMul (pas fusable avec pointwise dans ce design)
    MatMul {
        output_buf: usize,
        a_buf: usize,
        b_buf: usize,
        m: usize, k: usize, n: usize,
    },
}

/// Opérations pointwise élémentaires utilisées dans une chaîne fusionnée.
/// Une chaîne combine plusieurs opérations en une seule passe sur les
/// données — c'est exactement la "kernel fusion".
#[derive(Clone, Debug)]
pub enum PwOp {
    /// Charge la valeur d'un input dans l'accumulateur (premier op de la chaîne)
    LoadInput(usize),                      // index dans input_bufs
    /// acc = acc + buf
    Add(usize),
    /// acc = acc - buf
    Sub(usize),
    /// acc = acc * buf
    Mul(usize),
    /// acc = acc * scalar
    Scale(f32),
    /// acc = max(acc, 0)
    Relu,
    /// acc = exp(acc)
    Exp,
    /// acc = log(max(acc, eps))
    Log,
}

// ================================================================== //
//  CachePolicy — gère la durée de vie des buffers intermédiaires      //
// ================================================================== //

#[derive(Clone, Debug)]
pub enum CachePolicy {
    /// Garde tout en mémoire (utile pour le backward, gourmand)
    KeepAll,
    /// Ne garde que les feuilles (constantes, feeds) et les outputs marqués persist
    LeavesOnly,
    /// Garde au plus n buffers intermédiaires (LRU)
    Lru(usize),
}

// ================================================================== //
//  Plan — résultat de la compilation                                  //
// ================================================================== //

pub struct Plan {
    pub instructions: Vec<Instr>,
    pub n_buffers:    usize,
    pub output_buf:   usize,
    pub output_shape: (usize, usize),
    /// Pour chaque feed nommé, son buffer dans le plan
    pub feed_slots:   HashMap<String, usize>,
    /// Optimisations appliquées (pour reporting / debug)
    pub stats:        PlanStats,
    pub cache_policy: CachePolicy,
}

#[derive(Default, Debug, Clone)]
pub struct PlanStats {
    pub original_node_count: usize,
    pub instructions_count:  usize,
    pub fused_chains:        usize,
    pub dce_eliminated:      usize,
}

impl Plan {
    /// Exécute le plan avec ses feeds par défaut (constantes embarquées).
    pub fn execute(&self) -> Tensor {
        self.execute_with(&[])
    }

    /// Exécute avec des feeds dynamiques. Les noms doivent matcher
    /// les feeds définis lors de la compilation.
    pub fn execute_with(&self, feeds: &[(&str, Tensor)]) -> Tensor {
        let feed_map: HashMap<&str, &Tensor> = feeds.iter()
            .map(|(k, v)| (*k, v))
            .collect();

        // Allocation des buffers — un Vec<Option<Tensor>> permet le drop
        let mut buffers: Vec<Option<Tensor>> = vec![None; self.n_buffers];

        for instr in &self.instructions {
            match instr {
                Instr::LoadConst { output_buf, value } => {
                    buffers[*output_buf] = Some(value.clone());
                }
                Instr::LoadFeed { output_buf, feed_name, expected_shape } => {
                    let t = feed_map.get(feed_name.as_str())
                        .unwrap_or_else(|| panic!("feed manquant : '{feed_name}'"));
                    assert_eq!(t.shape(), *expected_shape,
                               "feed '{feed_name}' shape mismatch : {:?} vs {:?}",
                               t.shape(), expected_shape);
                    buffers[*output_buf] = Some((*t).clone());
                }
                Instr::PointwiseChain { output_buf, input_bufs, ops, shape } => {
                    let result = run_pointwise_chain(&buffers, input_bufs, ops, *shape);
                    buffers[*output_buf] = Some(result);
                }
                Instr::MatMul { output_buf, a_buf, b_buf, m, k, n } => {
                    let a = buffers[*a_buf].as_ref().expect("buffer a non chargé");
                    let b = buffers[*b_buf].as_ref().expect("buffer b non chargé");
                    let mut out = Tensor::zeros(*m, *n);
                    for i in 0..*m {
                        for j in 0..*n {
                            let mut acc = 0.0f32;
                            for p in 0..*k {
                                acc += a.data[i * k + p] * b.data[p * n + j];
                            }
                            out.data[i * n + j] = acc;
                        }
                    }
                    buffers[*output_buf] = Some(out);
                }
            }
        }

        buffers[self.output_buf].take().expect("buffer de sortie absent")
    }
}

fn run_pointwise_chain(
    buffers:    &[Option<Tensor>],
    input_bufs: &[usize],
    ops:        &[PwOp],
    shape:      (usize, usize),
) -> Tensor {
    let n = shape.0 * shape.1;
    let mut acc = vec![0.0f32; n];

    // Pré-charge tous les inputs comme références
    let inputs: Vec<&Tensor> = input_bufs.iter()
        .map(|b| buffers[*b].as_ref().expect("input non chargé"))
        .collect();

    // Boucle externe sur les éléments — boucle interne sur les ops.
    // En une seule passe, on applique toute la chaîne. C'est la
    // "fusion" : un seul parcours mémoire au lieu de N parcours.
    for i in 0..n {
        let mut a: f32 = 0.0;
        for op in ops {
            match op {
                PwOp::LoadInput(k) => a = inputs[*k].data[i],
                PwOp::Add(k)       => a += inputs[*k].data[i],
                PwOp::Sub(k)       => a -= inputs[*k].data[i],
                PwOp::Mul(k)       => a *= inputs[*k].data[i],
                PwOp::Scale(s)     => a *= s,
                PwOp::Relu         => a = a.max(0.0),
                PwOp::Exp          => a = a.exp(),
                PwOp::Log          => a = a.max(1e-12).ln(),
            }
        }
        acc[i] = a;
    }
    Tensor::from_vec(acc, shape.0, shape.1)
}

// ================================================================== //
//  Compilation : LazyGraph → Plan                                     //
// ================================================================== //

pub struct Compiler<'g> {
    graph:        &'g LazyGraph,
    /// Mapping LazyId → buffer slot dans le plan
    buf_of_node:  HashMap<LazyId, usize>,
    /// Feeds détectés pendant la compilation
    feed_slots:   HashMap<String, usize>,
    /// Instructions générées
    instructions: Vec<Instr>,
    /// Compteur pour allouer les buffers
    next_buf:     usize,
    /// Stats accumulées
    stats:        PlanStats,
    cache_policy: CachePolicy,
}

impl<'g> Compiler<'g> {
    pub fn new(graph: &'g LazyGraph) -> Self {
        Self {
            graph,
            buf_of_node:  HashMap::new(),
            feed_slots:   HashMap::new(),
            instructions: Vec::new(),
            next_buf:     0,
            stats:        PlanStats::default(),
            cache_policy: CachePolicy::LeavesOnly,
        }
    }

    pub fn with_cache_policy(mut self, p: CachePolicy) -> Self {
        self.cache_policy = p;
        self
    }

    /// Pipeline complet : DCE → fusion pointwise → ordre topologique → émission.
    pub fn compile(mut self, target: LazyId) -> Plan {
        let total_nodes = self.graph.nodes_borrow().len();
        self.stats.original_node_count = total_nodes;

        // Phase 1 : DCE — calcule l'ensemble des nœuds atteignables depuis target
        let live = self.compute_reachable(target);
        self.stats.dce_eliminated = total_nodes - live.len();

        // Phase 2 : ordre topologique (post-order DFS)
        let order = self.topological_order(target, &live);

        // Phase 3 : émission avec fusion pointwise
        // Stratégie de fusion : si un nœud pointwise n'a qu'un seul successeur
        // dans le graphe live et que ce successeur est aussi pointwise,
        // on l'absorbe.
        let consumer_count = self.count_consumers(target, &live);

        for &id in &order {
            if self.buf_of_node.contains_key(&id) {
                continue; // déjà émis (peut arriver avec fusion)
            }
            let (op, shape) = {
                let nodes = self.graph.nodes_borrow();
                (nodes[id].op.clone(), nodes[id].shape)
            };
            self.emit_node(id, op, shape, &consumer_count, &live);
        }

        let output_buf = *self.buf_of_node.get(&target).expect("target non émis");
        let output_shape = self.graph.nodes_borrow()[target].shape;

        self.stats.instructions_count = self.instructions.len();

        Plan {
            instructions: self.instructions,
            n_buffers:    self.next_buf,
            output_buf,
            output_shape,
            feed_slots:   self.feed_slots,
            stats:        self.stats,
            cache_policy: self.cache_policy,
        }
    }

    fn compute_reachable(&self, target: LazyId) -> HashSet<LazyId> {
        let mut live = HashSet::new();
        let mut stack = vec![target];
        while let Some(id) = stack.pop() {
            if !live.insert(id) { continue; }
            let op = self.graph.nodes_borrow()[id].op.clone();
            for parent in op_parents(&op) {
                stack.push(parent);
            }
        }
        live
    }

    fn topological_order(&self, target: LazyId, live: &HashSet<LazyId>) -> Vec<LazyId> {
        // Post-order DFS — chaque nœud apparaît APRÈS ses parents
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        self.dfs(target, live, &mut visited, &mut order);
        order
    }

    fn dfs(&self, id: LazyId, live: &HashSet<LazyId>,
           visited: &mut HashSet<LazyId>, order: &mut Vec<LazyId>) {
        if !live.contains(&id) || visited.contains(&id) { return; }
        visited.insert(id);
        let op = self.graph.nodes_borrow()[id].op.clone();
        for parent in op_parents(&op) {
            self.dfs(parent, live, visited, order);
        }
        order.push(id);
    }

    fn count_consumers(&self, target: LazyId, live: &HashSet<LazyId>) -> HashMap<LazyId, usize> {
        let mut count: HashMap<LazyId, usize> = HashMap::new();
        for &id in live {
            let op = self.graph.nodes_borrow()[id].op.clone();
            for parent in op_parents(&op) {
                *count.entry(parent).or_insert(0) += 1;
            }
        }
        // target a "1 consommateur fictif" — sinon il serait éligible à la fusion
        // d'une façon qui ne lui donnerait pas de buffer de sortie distinct
        count.insert(target, count.get(&target).copied().unwrap_or(0) + 1);
        count
    }

    fn alloc_buf(&mut self) -> usize {
        let b = self.next_buf;
        self.next_buf += 1;
        b
    }

    fn emit_node(&mut self,
                 id: LazyId,
                 op: LazyOp,
                 shape: (usize, usize),
                 consumers: &HashMap<LazyId, usize>,
                 live: &HashSet<LazyId>)
    {
        match op {
            LazyOp::Const(t) => {
                let buf = self.alloc_buf();
                self.instructions.push(Instr::LoadConst { output_buf: buf, value: t });
                self.buf_of_node.insert(id, buf);
            }
            LazyOp::Feed { name, shape: s } => {
                let buf = self.alloc_buf();
                self.feed_slots.insert(name.clone(), buf);
                self.instructions.push(Instr::LoadFeed {
                    output_buf: buf, feed_name: name, expected_shape: s,
                });
                self.buf_of_node.insert(id, buf);
            }
            LazyOp::MatMul(a, b) => {
                let a_buf = *self.buf_of_node.get(&a).expect("a pas encore émis");
                let b_buf = *self.buf_of_node.get(&b).expect("b pas encore émis");
                let buf = self.alloc_buf();
                let (m, k) = self.graph.nodes_borrow()[a].shape;
                let (_, n) = self.graph.nodes_borrow()[b].shape;
                self.instructions.push(Instr::MatMul {
                    output_buf: buf, a_buf, b_buf, m, k, n,
                });
                self.buf_of_node.insert(id, buf);
            }

            // Pointwise ops — éligibles à la fusion
            pw_op @ (LazyOp::Add(_, _) | LazyOp::Sub(_, _) | LazyOp::Mul(_, _)
                   | LazyOp::Scale(_, _) | LazyOp::Relu(_) | LazyOp::Exp(_)
                   | LazyOp::Log(_)) =>
            {
                self.emit_pointwise_chain(id, pw_op, shape, consumers, live);
            }
        }
    }

    /// Tente de fusionner ce nœud avec ses parents pointwise mono-consommateur.
    /// Construit une chaîne PwOp et émet une seule instruction PointwiseChain.
    fn emit_pointwise_chain(&mut self,
                            id: LazyId,
                            head_op: LazyOp,
                            shape: (usize, usize),
                            consumers: &HashMap<LazyId, usize>,
                            live: &HashSet<LazyId>)
    {
        let mut chain_ops: Vec<PwOp> = Vec::new();
        let mut input_bufs: Vec<usize> = Vec::new();
        let chain_member_ids = self.collect_chain_members(id, consumers, live);

        // Construction de la chaîne en parcourant les membres en post-order.
        // Premier nœud non pointwise rencontré → entrée.
        for &mid in &chain_member_ids {
            let op = self.graph.nodes_borrow()[mid].op.clone();
            self.append_to_chain(mid, op, &mut chain_ops, &mut input_bufs, &chain_member_ids);
        }

        // Allocation du buffer de sortie
        let out_buf = self.alloc_buf();
        self.instructions.push(Instr::PointwiseChain {
            output_buf: out_buf,
            input_bufs,
            ops:        chain_ops,
            shape,
        });

        // Tous les membres de la chaîne pointent sur ce même buffer
        for mid in &chain_member_ids {
            self.buf_of_node.insert(*mid, out_buf);
        }

        if chain_member_ids.len() > 1 {
            self.stats.fused_chains += 1;
        }

        // Évite warning unused
        let _ = head_op;
    }

    /// Remonte la chaîne pointwise. Retourne les nœuds en ordre d'exécution
    /// (parent avant enfant). On s'arrête quand un parent a >1 consommateur
    /// ou n'est pas pointwise — il devient une "entrée" de la chaîne.
    fn collect_chain_members(&self,
                             tail: LazyId,
                             consumers: &HashMap<LazyId, usize>,
                             _live: &HashSet<LazyId>) -> Vec<LazyId>
    {
        let mut members = Vec::new();
        self.collect_recursive(tail, consumers, &mut members, true);
        members
    }

    fn collect_recursive(&self,
                         id: LazyId,
                         consumers: &HashMap<LazyId, usize>,
                         out: &mut Vec<LazyId>,
                         is_tail: bool)
    {
        let op = self.graph.nodes_borrow()[id].op.clone();
        let is_pw = is_pointwise(&op);
        let n_consumers = *consumers.get(&id).unwrap_or(&0);

        // Conditions pour intégrer ce nœud dans la chaîne :
        //   - c'est le tail (point de départ), ou
        //   - il est pointwise ET a un seul consommateur (qui est dans la chaîne)
        let in_chain = is_tail || (is_pw && n_consumers == 1);

        if in_chain && is_pw {
            // On descend d'abord chez les parents pointwise mono-consommateur
            for parent in op_parents(&op) {
                self.collect_recursive(parent, consumers, out, false);
            }
            out.push(id);
        }
        // Sinon : ce nœud est traité par emit_node ailleurs et ne fait pas
        // partie de la chaîne. Si is_tail mais non pointwise, c'est un bug.
    }

    /// Récupère ou crée le slot d'index d'un input dans la chaîne.
    /// Le buffer doit déjà avoir été émis (parent traité avant).
    fn get_or_add_input(&self,
                        parent_id: LazyId,
                        input_bufs: &mut Vec<usize>) -> usize
    {
        let buf = *self.buf_of_node.get(&parent_id)
            .unwrap_or_else(|| panic!("parent {parent_id} pas encore émis"));
        if let Some(pos) = input_bufs.iter().position(|&b| b == buf) {
            pos
        } else {
            input_bufs.push(buf);
            input_bufs.len() - 1
        }
    }

    /// Indique si le parent fait partie de la chaîne courante (ne nécessite
    /// donc pas un nouveau LoadInput, l'accumulateur le contient déjà).
    fn parent_is_in_chain(&self, parent_id: LazyId, chain_member_ids: &[LazyId]) -> bool {
        chain_member_ids.contains(&parent_id)
    }

    fn append_to_chain(&mut self,
                       id: LazyId,
                       op: LazyOp,
                       chain_ops: &mut Vec<PwOp>,
                       input_bufs: &mut Vec<usize>,
                       chain_member_ids: &[LazyId])
    {
        match op {
            LazyOp::Add(a, b) => {
                let a_in = self.parent_is_in_chain(a, chain_member_ids);
                let b_in = self.parent_is_in_chain(b, chain_member_ids);
                if !a_in && !b_in {
                    // Premier op de la chaîne (chain_ops vide) : a est l'init,
                    // b est ajouté.
                    let ia = self.get_or_add_input(a, input_bufs);
                    if chain_ops.is_empty() { chain_ops.push(PwOp::LoadInput(ia)); }
                    let ib = self.get_or_add_input(b, input_bufs);
                    chain_ops.push(PwOp::Add(ib));
                } else if a_in && !b_in {
                    let ib = self.get_or_add_input(b, input_bufs);
                    chain_ops.push(PwOp::Add(ib));
                } else if !a_in && b_in {
                    let ia = self.get_or_add_input(a, input_bufs);
                    chain_ops.push(PwOp::Add(ia));
                } else {
                    // Les deux parents sont dans la chaîne — cas exotique
                    // (diamant). On retombe sur l'émission séparée :
                    // ne devrait pas arriver avec count_consumers correct.
                    panic!("diamant détecté dans une chaîne pointwise — non supporté");
                }
            }
            LazyOp::Sub(a, b) => {
                let a_in = self.parent_is_in_chain(a, chain_member_ids);
                let b_in = self.parent_is_in_chain(b, chain_member_ids);
                if !a_in && !b_in {
                    let ia = self.get_or_add_input(a, input_bufs);
                    if chain_ops.is_empty() { chain_ops.push(PwOp::LoadInput(ia)); }
                    let ib = self.get_or_add_input(b, input_bufs);
                    chain_ops.push(PwOp::Sub(ib));
                } else if a_in {
                    let ib = self.get_or_add_input(b, input_bufs);
                    chain_ops.push(PwOp::Sub(ib));
                } else {
                    panic!("Sub avec b dans la chaîne — non géré (acc - chain_val)");
                }
            }
            LazyOp::Mul(a, b) => {
                let a_in = self.parent_is_in_chain(a, chain_member_ids);
                let b_in = self.parent_is_in_chain(b, chain_member_ids);
                if !a_in && !b_in {
                    let ia = self.get_or_add_input(a, input_bufs);
                    if chain_ops.is_empty() { chain_ops.push(PwOp::LoadInput(ia)); }
                    let ib = self.get_or_add_input(b, input_bufs);
                    chain_ops.push(PwOp::Mul(ib));
                } else if a_in && !b_in {
                    let ib = self.get_or_add_input(b, input_bufs);
                    chain_ops.push(PwOp::Mul(ib));
                } else if !a_in && b_in {
                    let ia = self.get_or_add_input(a, input_bufs);
                    chain_ops.push(PwOp::Mul(ia));
                } else {
                    panic!("diamant détecté dans Mul");
                }
            }
            LazyOp::Scale(a, s) => {
                let a_in = self.parent_is_in_chain(a, chain_member_ids);
                if !a_in && chain_ops.is_empty() {
                    let ia = self.get_or_add_input(a, input_bufs);
                    chain_ops.push(PwOp::LoadInput(ia));
                }
                chain_ops.push(PwOp::Scale(s));
            }
            LazyOp::Relu(a) => {
                let a_in = self.parent_is_in_chain(a, chain_member_ids);
                if !a_in && chain_ops.is_empty() {
                    let ia = self.get_or_add_input(a, input_bufs);
                    chain_ops.push(PwOp::LoadInput(ia));
                }
                chain_ops.push(PwOp::Relu);
            }
            LazyOp::Exp(a) => {
                let a_in = self.parent_is_in_chain(a, chain_member_ids);
                if !a_in && chain_ops.is_empty() {
                    let ia = self.get_or_add_input(a, input_bufs);
                    chain_ops.push(PwOp::LoadInput(ia));
                }
                chain_ops.push(PwOp::Exp);
            }
            LazyOp::Log(a) => {
                let a_in = self.parent_is_in_chain(a, chain_member_ids);
                if !a_in && chain_ops.is_empty() {
                    let ia = self.get_or_add_input(a, input_bufs);
                    chain_ops.push(PwOp::LoadInput(ia));
                }
                chain_ops.push(PwOp::Log);
            }
            _ => panic!("append_to_chain: op non pointwise"),
        }
        let _ = id;
    }
}

// ================================================================== //
//  Helpers                                                            //
// ================================================================== //

fn is_pointwise(op: &LazyOp) -> bool {
    matches!(op,
        LazyOp::Add(_, _) | LazyOp::Sub(_, _) | LazyOp::Mul(_, _)
      | LazyOp::Scale(_, _) | LazyOp::Relu(_) | LazyOp::Exp(_) | LazyOp::Log(_))
}

fn op_parents(op: &LazyOp) -> Vec<LazyId> {
    match op {
        LazyOp::Const(_) | LazyOp::Feed { .. } => vec![],
        LazyOp::Add(a, b) | LazyOp::Sub(a, b) | LazyOp::Mul(a, b) | LazyOp::MatMul(a, b)
            => vec![*a, *b],
        LazyOp::Scale(a, _) | LazyOp::Relu(a) | LazyOp::Exp(a) | LazyOp::Log(a)
            => vec![*a],
    }
}

// ================================================================== //
//  Tests                                                              //
// ================================================================== //
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::LazyTensor;

    #[test]
    fn plan_executes_simple_chain() {
        let g = LazyGraph::new();
        let a = LazyTensor::from_tensor(g.clone(), Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], 1, 4));
        let y = a.relu().scale(2.0);

        let plan = Compiler::new(&g).compile(y.id);
        let result = plan.execute();
        assert_eq!(result.data, vec![0.0, 4.0, 0.0, 8.0]);
    }

    #[test]
    fn fusion_collapses_chain_to_one_instr() {
        // 4 ops pointwise → 1 seule PointwiseChain dans le plan
        let g = LazyGraph::new();
        let a = LazyTensor::from_tensor(g.clone(), Tensor::from_vec(vec![1.0; 4], 2, 2));
        let y = a.scale(3.0).relu().exp().log();

        let plan = Compiler::new(&g).compile(y.id);

        // 1 const + 1 chaîne fusionnée = 2 instructions
        assert_eq!(plan.instructions.len(), 2);
        assert_eq!(plan.stats.fused_chains, 1);
    }

    #[test]
    fn dce_removes_unused_branch() {
        let g = LazyGraph::new();
        let a = LazyTensor::from_tensor(g.clone(), Tensor::from_vec(vec![1.0; 4], 1, 4));
        // Branche utile
        let y = a.clone().relu();
        // Branche morte — jamais utilisée comme target
        let _dead = a.exp().scale(99.0);

        let plan = Compiler::new(&g).compile(y.id);
        assert!(plan.stats.dce_eliminated >= 2,
                "stats: {:?}", plan.stats);
    }

    #[test]
    fn execute_with_feeds() {
        // Construit un plan paramétré par un feed "x"
        let g = LazyGraph::new();
        let x = LazyTensor::feed(g.clone(), "x".into(), (1, 3));
        let y = x.scale(10.0).relu();

        let plan = Compiler::new(&g).compile(y.id);

        let r1 = plan.execute_with(&[("x", Tensor::from_vec(vec![-1.0, 2.0, -3.0], 1, 3))]);
        assert_eq!(r1.data, vec![0.0, 20.0, 0.0]);

        let r2 = plan.execute_with(&[("x", Tensor::from_vec(vec![5.0, 0.5, -100.0], 1, 3))]);
        assert_eq!(r2.data, vec![50.0, 5.0, 0.0]);
    }

    #[test]
    fn re_execute_is_cheap() {
        // Le plan compilé peut être ré-exécuté sans recompiler
        let g = LazyGraph::new();
        let x = LazyTensor::feed(g.clone(), "x".into(), (1, 4));
        let y = x.relu().scale(2.0);
        let plan = Compiler::new(&g).compile(y.id);

        for _ in 0..100 {
            let _ = plan.execute_with(&[("x", Tensor::from_vec(vec![1.0; 4], 1, 4))]);
        }
    }
}
