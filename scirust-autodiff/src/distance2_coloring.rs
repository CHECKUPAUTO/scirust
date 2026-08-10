//! Deterministic Distance-2 coloring for symmetric sparse Hessian graphs.
//!
//! Vertices at graph distance one or two receive different colors. This is the
//! seed-planning primitive required by compressed Hessian techniques. Planning is
//! CSR-based and uses generation-marked dense scratch arrays rather than hash
//! tables in the coloring loop.

/// Validation/planning error for a symmetric CSR graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Distance2ColoringError {
    InvalidRowOffsets,
    VertexOutOfBounds { vertex: usize, vertices: usize },
    DuplicateNeighbor { vertex: usize, neighbor: usize },
    NonCanonicalRow { vertex: usize },
    AsymmetricEdge { from: usize, to: usize },
    TooManyColors { required: usize, available: usize },
}

impl core::fmt::Display for Distance2ColoringError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::InvalidRowOffsets => write!(f, "invalid symmetric-graph CSR row offsets"),
            Self::VertexOutOfBounds { vertex, vertices } => {
                write!(f, "neighbor vertex {vertex} is outside 0..{vertices}")
            },
            Self::DuplicateNeighbor { vertex, neighbor } => {
                write!(f, "vertex {vertex} contains duplicate neighbor {neighbor}")
            },
            Self::NonCanonicalRow { vertex } => {
                write!(f, "vertex {vertex} adjacency row is not sorted")
            },
            Self::AsymmetricEdge { from, to } => {
                write!(f, "edge {from}->{to} has no matching {to}->{from} edge")
            },
            Self::TooManyColors { required, available } => write!(
                f,
                "distance-2 coloring requires {required} directions, but only {available} are available"
            ),
        }
    }
}

impl std::error::Error for Distance2ColoringError {}

/// Validated symmetric adjacency graph stored as CSR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymmetricSparsityGraph {
    vertices: usize,
    row_offsets: Vec<usize>,
    neighbors: Vec<usize>,
}

impl SymmetricSparsityGraph {
    pub fn new(
        vertices: usize,
        row_offsets: Vec<usize>,
        neighbors: Vec<usize>,
    ) -> Result<Self, Distance2ColoringError> {
        if row_offsets.len() != vertices + 1
            || row_offsets.first().copied() != Some(0)
            || row_offsets.last().copied() != Some(neighbors.len())
            || row_offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(Distance2ColoringError::InvalidRowOffsets);
        }

        for vertex in 0..vertices
        {
            let row = &neighbors[row_offsets[vertex]..row_offsets[vertex + 1]];
            for &neighbor in row
            {
                if neighbor >= vertices
                {
                    return Err(Distance2ColoringError::VertexOutOfBounds {
                        vertex: neighbor,
                        vertices,
                    });
                }
            }
            for pair in row.windows(2)
            {
                if pair[0] == pair[1]
                {
                    return Err(Distance2ColoringError::DuplicateNeighbor {
                        vertex,
                        neighbor: pair[0],
                    });
                }
                if pair[0] > pair[1]
                {
                    return Err(Distance2ColoringError::NonCanonicalRow { vertex });
                }
            }
        }

        let graph = Self {
            vertices,
            row_offsets,
            neighbors,
        };
        for from in 0..vertices
        {
            for &to in graph.neighbors(from).expect("validated vertex")
            {
                if from == to
                {
                    continue;
                }
                let reverse = graph.neighbors(to).expect("validated vertex");
                if reverse.binary_search(&from).is_err()
                {
                    return Err(Distance2ColoringError::AsymmetricEdge { from, to });
                }
            }
        }
        Ok(graph)
    }

    pub fn vertices(&self) -> usize {
        self.vertices
    }

    pub fn edge_entries(&self) -> usize {
        self.neighbors.len()
    }

    pub fn neighbors(&self, vertex: usize) -> Option<&[usize]> {
        let start = *self.row_offsets.get(vertex)?;
        let end = *self.row_offsets.get(vertex + 1)?;
        self.neighbors.get(start..end)
    }

    /// Greedy deterministic Distance-2 coloring in ascending vertex order.
    pub fn color_distance2(&self) -> Distance2Coloring {
        if self.vertices == 0
        {
            return Distance2Coloring {
                colors: Vec::new(),
                color_count: 0,
            };
        }

        let mut colors = vec![usize::MAX; self.vertices];
        let mut forbidden_generation = vec![0usize; self.vertices + 1];
        let mut generation = 1usize;
        let mut color_count = 0usize;

        for vertex in 0..self.vertices
        {
            if generation == usize::MAX
            {
                forbidden_generation.fill(0);
                generation = 1;
            }

            for &neighbor in self.neighbors(vertex).expect("validated vertex")
            {
                mark_earlier_color(
                    neighbor,
                    vertex,
                    &colors,
                    &mut forbidden_generation,
                    generation,
                );
                for &distance2 in self.neighbors(neighbor).expect("validated neighbor")
                {
                    if distance2 != vertex
                    {
                        mark_earlier_color(
                            distance2,
                            vertex,
                            &colors,
                            &mut forbidden_generation,
                            generation,
                        );
                    }
                }
            }

            let mut color = 0usize;
            while forbidden_generation[color] == generation
            {
                color += 1;
            }
            colors[vertex] = color;
            color_count = color_count.max(color + 1);
            generation += 1;
        }

        Distance2Coloring {
            colors,
            color_count,
        }
    }

    /// Validate the Distance-2 invariant for an externally supplied coloring.
    pub fn validate_coloring(&self, coloring: &Distance2Coloring) -> bool {
        if coloring.colors.len() != self.vertices
        {
            return false;
        }
        for vertex in 0..self.vertices
        {
            let color = coloring.colors[vertex];
            for &neighbor in self.neighbors(vertex).expect("validated vertex")
            {
                if neighbor != vertex && coloring.colors[neighbor] == color
                {
                    return false;
                }
                for &distance2 in self.neighbors(neighbor).expect("validated neighbor")
                {
                    if distance2 != vertex && coloring.colors[distance2] == color
                    {
                        return false;
                    }
                }
            }
        }
        true
    }
}

fn mark_earlier_color(
    other: usize,
    current: usize,
    colors: &[usize],
    forbidden_generation: &mut [usize],
    generation: usize,
) {
    if other >= current
    {
        return;
    }
    let color = colors[other];
    if color != usize::MAX
    {
        forbidden_generation[color] = generation;
    }
}

/// Seed-plan result for compressed sparse Hessian calculations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Distance2Coloring {
    colors: Vec<usize>,
    color_count: usize,
}

impl Distance2Coloring {
    pub fn colors(&self) -> &[usize] {
        &self.colors
    }

    pub fn color_count(&self) -> usize {
        self.color_count
    }

    pub fn color_of(&self, vertex: usize) -> Option<usize> {
        self.colors.get(vertex).copied()
    }

    pub fn require_width(&self, width: usize) -> Result<(), Distance2ColoringError> {
        if self.color_count > width
        {
            return Err(Distance2ColoringError::TooManyColors {
                required: self.color_count,
                available: width,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph(vertices: usize) -> SymmetricSparsityGraph {
        let mut offsets = Vec::with_capacity(vertices + 1);
        let mut neighbors = Vec::new();
        offsets.push(0);
        for vertex in 0..vertices
        {
            if vertex > 0
            {
                neighbors.push(vertex - 1);
            }
            if vertex + 1 < vertices
            {
                neighbors.push(vertex + 1);
            }
            offsets.push(neighbors.len());
        }
        SymmetricSparsityGraph::new(vertices, offsets, neighbors).unwrap()
    }

    fn square_grid(side: usize) -> SymmetricSparsityGraph {
        let vertices = side * side;
        let mut offsets = Vec::with_capacity(vertices + 1);
        let mut neighbors = Vec::new();
        offsets.push(0);
        for row in 0..side
        {
            for col in 0..side
            {
                let mut local = [usize::MAX; 4];
                let mut len = 0usize;
                if row > 0
                {
                    local[len] = (row - 1) * side + col;
                    len += 1;
                }
                if col > 0
                {
                    local[len] = row * side + col - 1;
                    len += 1;
                }
                if col + 1 < side
                {
                    local[len] = row * side + col + 1;
                    len += 1;
                }
                if row + 1 < side
                {
                    local[len] = (row + 1) * side + col;
                    len += 1;
                }
                local[..len].sort_unstable();
                neighbors.extend_from_slice(&local[..len]);
                offsets.push(neighbors.len());
            }
        }
        SymmetricSparsityGraph::new(vertices, offsets, neighbors).unwrap()
    }

    #[test]
    fn path_distance2_coloring_uses_three_colors() {
        let graph = path_graph(32);
        let coloring = graph.color_distance2();
        assert_eq!(coloring.color_count(), 3);
        assert!(graph.validate_coloring(&coloring));
    }

    #[test]
    fn grid_coloring_satisfies_distance2_without_assuming_chromatic_optimum() {
        let graph = square_grid(8);
        let coloring = graph.color_distance2();
        assert!(coloring.color_count() >= 5);
        assert!(graph.validate_coloring(&coloring));
    }

    #[test]
    fn rejects_asymmetric_graph() {
        let error = SymmetricSparsityGraph::new(2, vec![0, 1, 1], vec![1]).unwrap_err();
        assert_eq!(error, Distance2ColoringError::AsymmetricEdge { from: 0, to: 1 });
    }

    #[test]
    fn width_gate_is_explicit() {
        let coloring = path_graph(8).color_distance2();
        assert_eq!(
            coloring.require_width(2),
            Err(Distance2ColoringError::TooManyColors {
                required: 3,
                available: 2,
            })
        );
        assert!(coloring.require_width(3).is_ok());
    }
}
