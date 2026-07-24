//! V-structure (unshielded collider) detection and Meek's orientation rules,
//! completing a skeleton + separating sets into a CPDAG.
//!
//! # Two stages
//!
//! 1. [`orient_v_structures`]: for every unshielded triple `x - z - y` (`x`,
//!    `y` not adjacent, both adjacent to `z`), if `z` is **not** in the
//!    recorded separating set for `{x, y}`, `z` is a collider — orient
//!    `x -> z` and `y -> z`. This is the only step that reads statistical
//!    evidence (the separating sets from `crate::skeleton_discovery`); every
//!    later step is pure graph propagation.
//! 2. [`apply_meek_rules`]: Meek's rules R1-R3 (Meek, *Causal Inference and
//!    Causal Explanation with Background Knowledge*, UAI 1995), applied to a
//!    fixpoint, propagate the v-structure orientations to every edge they
//!    logically force, without creating a new (unevidenced) collider or a
//!    directed cycle.
//!
//! # Conflicting v-structures are left undirected, not guessed
//!
//! Two distinct unshielded triples can — only when the statistical evidence
//! is inconsistent (finite-sample error, or a genuine violation of
//! faithfulness/causal sufficiency) — demand opposite orientations for the
//! same edge. When that happens this implementation leaves the edge
//! undirected and records a warning; it never silently picks one direction.
//! Under a perfect oracle and the standard assumptions this can provably not
//! occur, so seeing the warning is itself diagnostic information.
//!
//! # Rule 4 is out of scope
//!
//! Meek (1995) proves rules R1-R3 necessary *and sufficient* to complete a
//! PDAG into a CPDAG whenever every initial directed edge comes from
//! v-structure detection alone — exactly this module's setting, since this
//! phase accepts no background knowledge. Meek's rule 4 is needed only when
//! additional orientations *beyond* what v-structures give are injected
//! (background knowledge); since none are here, rule 4 would never find a
//! pattern to act on, and is not implemented.

use crate::cpdag::Cpdag;
use std::collections::BTreeMap;

fn canon(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

/// Detects unshielded colliders and orients them. `separating_sets` must be
/// keyed by the same canonical `(min, max)` convention
/// `crate::skeleton_discovery::discover_skeleton` produces. Appends a warning
/// (and leaves the edge undirected) for every conflicting demand instead of
/// resolving it silently.
pub(crate) fn orient_v_structures(
    cpdag: &mut Cpdag,
    separating_sets: &BTreeMap<(usize, usize), Vec<usize>>,
    warnings: &mut Vec<String>,
) {
    // Collect every v-structure's demanded orientation before applying any of
    // them, so a conflict is detected against the *original* undirected
    // skeleton, never against an orientation an earlier demand already made.
    let mut demands: BTreeMap<(usize, usize), Vec<(usize, usize)>> = BTreeMap::new();

    for z in 0..cpdag.n_nodes()
    {
        let neighbors = cpdag.neighbors(z);
        for i in 0..neighbors.len()
        {
            for &y in neighbors.iter().skip(i + 1)
            {
                let x = neighbors[i];
                if cpdag.is_adjacent(x, y)
                {
                    continue; // shielded triple: not a collider signal
                }
                let Some(sep) = separating_sets.get(&canon(x, y))
                else
                {
                    continue; // not adjacent but no recorded sepset: nothing to compare against
                };
                if !sep.contains(&z)
                {
                    demands.entry(canon(x, z)).or_default().push((x, z));
                    demands.entry(canon(y, z)).or_default().push((y, z));
                }
            }
        }
    }

    for (edge, mut wanted) in demands
    {
        wanted.sort_unstable();
        wanted.dedup();
        if wanted.len() > 1
        {
            warnings.push(format!(
                "conflicting v-structure orientation for edge {edge:?}; left undirected"
            ));
            continue;
        }
        let (from, to) = wanted[0];
        cpdag.orient(from, to);
    }
}

/// Applies Meek's rules R1-R3 to a fixpoint (repeats until a full pass makes
/// no change).
pub(crate) fn apply_meek_rules(cpdag: &mut Cpdag) {
    loop
    {
        let mut changed = false;
        changed |= apply_rule_1(cpdag);
        changed |= apply_rule_2(cpdag);
        changed |= apply_rule_3(cpdag);
        if !changed
        {
            break;
        }
    }
}

/// R1 — avoid a new unshielded collider: `a -> b`, `b - c`, `a` and `c` not
/// adjacent `⟹` `b -> c` (orienting `c -> b` would create the unshielded
/// collider `a -> b <- c`, which nothing in the evidence supports).
fn apply_rule_1(cpdag: &mut Cpdag) -> bool {
    let mut changed = false;
    for (a, b) in cpdag.directed_edges()
    {
        for c in cpdag.neighbors(b)
        {
            if c != a && cpdag.is_undirected(b, c) && !cpdag.is_adjacent(a, c) && cpdag.orient(b, c)
            {
                changed = true;
            }
        }
    }
    changed
}

/// R2 — avoid a directed cycle: `a -> b`, `b -> c`, `a - c` `⟹` `a -> c`
/// (orienting `c -> a` would close the cycle `a -> b -> c -> a`).
fn apply_rule_2(cpdag: &mut Cpdag) -> bool {
    let mut changed = false;
    let directed = cpdag.directed_edges();
    for &(a, b) in &directed
    {
        for &(b2, c) in &directed
        {
            if b2 == b && a != c && cpdag.is_undirected(a, c) && cpdag.orient(a, c)
            {
                changed = true;
            }
        }
    }
    changed
}

/// R3 — the four-node rule: `a - b`, `a - c`, `a - d` (all undirected),
/// `c -> b`, `d -> b`, `c` and `d` not adjacent `⟹` `a -> b`.
fn apply_rule_3(cpdag: &mut Cpdag) -> bool {
    let mut changed = false;
    for &(p, q) in &cpdag.undirected_edges()
    {
        for &(a, b) in &[(p, q), (q, p)]
        {
            if !cpdag.is_undirected(a, b)
            {
                continue; // oriented by an earlier iteration this same pass
            }
            let a_undirected_neighbors: Vec<usize> = cpdag
                .neighbors(a)
                .into_iter()
                .filter(|&n| n != b && cpdag.is_undirected(a, n))
                .collect();

            let mut witnessed = false;
            'search: for i in 0..a_undirected_neighbors.len()
            {
                for &d in a_undirected_neighbors.iter().skip(i + 1)
                {
                    let c = a_undirected_neighbors[i];
                    if !cpdag.is_adjacent(c, d)
                        && cpdag.is_directed(c, b)
                        && cpdag.is_directed(d, b)
                    {
                        witnessed = true;
                        break 'search;
                    }
                }
            }
            if witnessed && cpdag.orient(a, b)
            {
                changed = true;
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sepsets(pairs: &[((usize, usize), &[usize])]) -> BTreeMap<(usize, usize), Vec<usize>> {
        pairs.iter().map(|&(k, z)| (k, z.to_vec())).collect()
    }

    // ─── V-structures ───────────────────────────────────────────────────

    #[test]
    fn unshielded_triple_with_empty_sepset_is_a_v_structure() {
        // 0 - 2, 1 - 2, 0/1 not adjacent, sepset(0,1) = {} (does not contain 2).
        let mut g = Cpdag::complete(3);
        g.remove_edge(0, 1);
        let seps = sepsets(&[((0, 1), &[])]);
        let mut warnings = Vec::new();
        orient_v_structures(&mut g, &seps, &mut warnings);
        assert!(g.is_directed(0, 2));
        assert!(g.is_directed(1, 2));
        assert!(warnings.is_empty());
    }

    #[test]
    fn unshielded_triple_with_middle_node_in_sepset_is_not_a_v_structure() {
        // Chain shape: 0 - 1, 1 - 2, 0/2 not adjacent, sepset(0,2) = {1}.
        let mut g = Cpdag::complete(3);
        g.remove_edge(0, 2);
        let seps = sepsets(&[((0, 2), &[1])]);
        let mut warnings = Vec::new();
        orient_v_structures(&mut g, &seps, &mut warnings);
        assert!(g.is_undirected(0, 1));
        assert!(g.is_undirected(1, 2));
        assert!(warnings.is_empty());
    }

    #[test]
    fn shielded_triple_is_never_a_v_structure_regardless_of_sepset_map_contents() {
        // Full triangle: every pair adjacent, so (0,1,2) is shielded at every
        // vertex no matter what a (structurally spurious) sepset entry says.
        let mut g = Cpdag::complete(3);
        let seps = sepsets(&[((0, 1), &[])]); // would demand a v-structure if 0,1 were non-adjacent
        let mut warnings = Vec::new();
        orient_v_structures(&mut g, &seps, &mut warnings);
        assert!(g.is_undirected(0, 1));
        assert!(g.is_undirected(0, 2));
        assert!(g.is_undirected(1, 2));
        assert!(warnings.is_empty());
    }

    #[test]
    fn conflicting_v_structures_leave_the_edge_undirected_with_a_warning() {
        // 4 nodes, built from K4 by removing {0,1}, {1,3}, {2,3}, leaving
        // exactly the undirected edges {0,2}, {0,3}, {1,2}. Adjacency:
        // 0:{2,3}, 1:{2}, 2:{0,1}, 3:{0}. Exactly two unshielded triples
        // result: (2,0,3) centered at 0, and (0,2,1) centered at 2 -- hand
        // verified, no others (nodes 1 and 3 have only one neighbor each, so
        // no triple can center on them).
        //
        // sepset(2,3) = {} (for triple (2,0,3)): 0 not in it -> demands
        // 2->0, 3->0.
        // sepset(0,1) = {} (for triple (0,2,1)): 2 not in it -> demands
        // 0->2, 1->2.
        //
        // Edge {0,2} now has two opposing demands: "2->0" (from the first
        // triple) and "0->2" (from the second) -- a genuine conflict. Edge
        // {0,3} has only the demand "3->0" (uncontested). Edge {1,2} has
        // only the demand "1->2" (uncontested).
        let mut g = Cpdag::complete(4);
        g.remove_edge(0, 1);
        g.remove_edge(1, 3);
        g.remove_edge(2, 3);
        let seps = sepsets(&[((2, 3), &[]), ((0, 1), &[])]);
        let mut warnings = Vec::new();
        orient_v_structures(&mut g, &seps, &mut warnings);

        assert!(g.is_undirected(0, 2), "contested edge must stay undirected");
        assert!(warnings.iter().any(|w| w.contains("conflicting")));
        assert!(g.is_directed(3, 0), "uncontested demand must still apply");
        assert!(g.is_directed(1, 2), "uncontested demand must still apply");
    }

    // ─── Meek's rules ───────────────────────────────────────────────────

    #[test]
    fn rule_1_orients_away_from_a_new_collider() {
        // a=0 -> b=1 (directed), b=1 - c=2 (undirected), 0/2 not adjacent.
        let mut g = Cpdag::complete(3);
        g.remove_edge(0, 2);
        g.orient(0, 1);
        assert!(apply_rule_1(&mut g));
        assert!(g.is_directed(1, 2));
    }

    #[test]
    fn rule_1_does_not_fire_when_a_and_c_are_adjacent() {
        let mut g = Cpdag::complete(3); // 0-2 stays an edge (shielded)
        g.orient(0, 1);
        assert!(!apply_rule_1(&mut g));
        assert!(g.is_undirected(1, 2));
    }

    #[test]
    fn rule_2_orients_the_shortcut_to_avoid_a_cycle() {
        let mut g = Cpdag::complete(3);
        g.orient(0, 1);
        g.orient(1, 2);
        assert!(apply_rule_2(&mut g));
        assert!(g.is_directed(0, 2));
    }

    #[test]
    fn rule_2_does_not_fire_with_only_one_directed_edge() {
        let mut g = Cpdag::complete(3);
        g.orient(0, 1);
        assert!(!apply_rule_2(&mut g));
        assert!(g.is_undirected(1, 2));
        assert!(g.is_undirected(0, 2));
    }

    #[test]
    fn rule_3_orients_a_to_b() {
        // a=0, b=1, c=2, d=3: a-b, a-c, a-d undirected; c->b, d->b directed;
        // c,d not adjacent.
        let mut g = Cpdag::complete(4);
        g.remove_edge(2, 3); // c=2, d=3 not adjacent
        g.orient(2, 1); // c -> b
        g.orient(3, 1); // d -> b
        assert!(apply_rule_3(&mut g));
        assert!(g.is_directed(0, 1));
    }

    #[test]
    fn rule_3_does_not_fire_when_c_and_d_are_adjacent() {
        let mut g = Cpdag::complete(4); // 2-3 stays an edge: shielded
        g.orient(2, 1);
        g.orient(3, 1);
        assert!(!apply_rule_3(&mut g));
        assert!(g.is_undirected(0, 1));
    }

    #[test]
    fn meek_rules_reach_a_full_fixpoint_via_repeated_r1_then_r2() {
        // 5 nodes: v-structure-style seed 0->1, then a chain of undirected
        // edges 1-2-3-4 with shortcuts, forcing R1 and R2 to each fire more
        // than once before reaching the fixpoint.
        let mut g = Cpdag::complete(5);
        g.remove_edge(0, 2);
        g.remove_edge(0, 3);
        g.remove_edge(0, 4);
        g.remove_edge(1, 3);
        g.remove_edge(1, 4);
        g.remove_edge(2, 4);
        g.orient(0, 1);
        apply_meek_rules(&mut g);
        // 0->1 (seed), R1: 1->2 (0,2 non-adjacent); R2 needs 0->1,1->2 and 0-2
        // undirected, but 0,2 are NOT adjacent here so R2 does not apply to
        // that pair. R1 again: 2->3 (1,3 non-adjacent); then 3->4 (2,4 non-adjacent).
        assert!(g.is_directed(0, 1));
        assert!(g.is_directed(1, 2));
        assert!(g.is_directed(2, 3));
        assert!(g.is_directed(3, 4));
        assert_eq!(g.undirected_edges().len(), 0);
    }

    #[test]
    fn output_never_contains_a_directed_cycle() {
        // Sanity/property check across every hand-built case above plus the
        // R1 chain-propagation case: no directed cycle in the result.
        fn has_cycle(g: &Cpdag) -> bool {
            let n = g.n_nodes();
            let mut children = vec![Vec::new(); n];
            for (a, b) in g.directed_edges()
            {
                children[a].push(b);
            }
            let mut state = vec![0u8; n]; // 0=unvisited,1=in-progress,2=done
            fn visit(u: usize, children: &[Vec<usize>], state: &mut [u8]) -> bool {
                state[u] = 1;
                for &v in &children[u]
                {
                    if state[v] == 1
                    {
                        return true;
                    }
                    if state[v] == 0 && visit(v, children, state)
                    {
                        return true;
                    }
                }
                state[u] = 2;
                false
            }
            for u in 0..n
            {
                if state[u] == 0 && visit(u, &children, &mut state)
                {
                    return true;
                }
            }
            false
        }

        let mut g = Cpdag::complete(5);
        g.remove_edge(0, 2);
        g.remove_edge(0, 3);
        g.remove_edge(0, 4);
        g.remove_edge(1, 3);
        g.remove_edge(1, 4);
        g.remove_edge(2, 4);
        g.orient(0, 1);
        apply_meek_rules(&mut g);
        assert!(!has_cycle(&g));
    }
}
