//! # scirust-rl-algo — RL-Based Algorithm Discovery
//!
//! This crate fuses reinforcement learning with symbolic algorithm search.
//! It provides:
//!
//! - **Algorithm representation**: instruction-level encoding with mutation/crossover
//! - **RL environments**: search over algorithm space with correctness/efficiency/simplicity rewards
//! - **Policy gradient methods**: REINFORCE with baseline, simplified Actor-Critic
//! - **Tabular Q-learning**: discretized state, epsilon-greedy, experience replay
//! - **Meta-learning**: algorithm templates, transfer learning, feature extraction
//! - **Search heuristics**: simulated annealing, beam search, MCTS, progressive widening
//! - **Verification**: test-suite generation, invariant inference, CEGAR

use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Default seed — each constructible with an explicit seed for reproducibility
// ---------------------------------------------------------------------------
#[allow(dead_code)]
const RL_ALGO_DEFAULT_SEED: u64 = 0x414C_474F;

// ===========================================================================
// 1. ALGORITHM REPRESENTATION
// ===========================================================================

/// A single instruction in our low-level algorithm language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Instruction {
    /// Load immediate value into register
    Load(usize, i64),
    /// Store register value into another register
    Store(usize, usize),
    /// Add: r[dst] = r[src1] + r[src2]
    Add(usize, usize, usize),
    /// Sub: r[dst] = r[src1] - r[src2]
    Sub(usize, usize, usize),
    /// Mul: r[dst] = r[src1] * r[src2]
    Mul(usize, usize, usize),
    /// Div: r[dst] = r[src1] / r[src2] (integer division, zero-safe)
    Div(usize, usize, usize),
    /// Compare: sets flags for JumpIf
    Cmp(usize, usize),
    /// Unconditional jump to instruction index
    Jump(isize),
    /// Conditional jump: jumps if last cmp result matches condition
    JumpIf(isize, CmpCondition),
    /// Swap register values
    Swap(usize, usize),
    /// Push register value onto stack
    Push(usize),
    /// Pop stack into register
    Pop(usize),
    /// Return / halt
    Return,
}

/// Conditions used by JumpIf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CmpCondition {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

/// The result of a comparison operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CmpResult {
    Equal,
    Less,
    Greater,
}

/// A complete algorithm: a list of instructions operating over some registers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Algorithm {
    pub instructions: Vec<Instruction>,
    pub num_registers: usize,
    pub stack_capacity: usize,
}

impl Algorithm {
    pub fn new(num_registers: usize, stack_capacity: usize) -> Self {
        Self {
            instructions: Vec::new(),
            num_registers,
            stack_capacity,
        }
    }

    /// Total length in instructions.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Whether the algorithm is empty.
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Execute the algorithm on the given input registers.
    /// Returns `Ok(final_registers)` or `Err(error_message)`.
    pub fn execute(&self, input_registers: &[i64]) -> Result<Vec<i64>, String> {
        let n = self.num_registers;
        let mut regs = vec![0i64; n];
        let copy_len = input_registers.len().min(n);
        regs[..copy_len].copy_from_slice(&input_registers[..copy_len]);

        let mut stack: Vec<i64> = Vec::with_capacity(self.stack_capacity);
        let mut pc: isize = 0;
        let mut cmp_result = CmpResult::Equal;
        let max_steps = 10_000;
        let mut steps = 0;

        while pc >= 0 && (pc as usize) < self.instructions.len()
        {
            if steps >= max_steps
            {
                return Err("Execution exceeded max steps".to_string());
            }
            steps += 1;

            let instr = self.instructions[pc as usize];
            match instr
            {
                Instruction::Load(r, val) =>
                {
                    if r < n
                    {
                        regs[r] = val;
                    }
                    pc += 1;
                },
                Instruction::Store(src, dst) =>
                {
                    if src < n && dst < n
                    {
                        regs[dst] = regs[src];
                    }
                    pc += 1;
                },
                Instruction::Add(dst, a, b) =>
                {
                    if dst < n && a < n && b < n
                    {
                        regs[dst] = regs[a].wrapping_add(regs[b]);
                    }
                    pc += 1;
                },
                Instruction::Sub(dst, a, b) =>
                {
                    if dst < n && a < n && b < n
                    {
                        regs[dst] = regs[a].wrapping_sub(regs[b]);
                    }
                    pc += 1;
                },
                Instruction::Mul(dst, a, b) =>
                {
                    if dst < n && a < n && b < n
                    {
                        regs[dst] = regs[a].wrapping_mul(regs[b]);
                    }
                    pc += 1;
                },
                Instruction::Div(dst, a, b) =>
                {
                    if dst < n && a < n && b < n
                    {
                        if regs[b] == 0
                        {
                            regs[dst] = 0;
                        }
                        else
                        {
                            regs[dst] = regs[a] / regs[b];
                        }
                    }
                    pc += 1;
                },
                Instruction::Cmp(a, b) =>
                {
                    if a < n && b < n
                    {
                        cmp_result = match regs[a].cmp(&regs[b])
                        {
                            std::cmp::Ordering::Less => CmpResult::Less,
                            std::cmp::Ordering::Equal => CmpResult::Equal,
                            std::cmp::Ordering::Greater => CmpResult::Greater,
                        };
                    }
                    pc += 1;
                },
                Instruction::Jump(offset) =>
                {
                    pc += offset;
                },
                Instruction::JumpIf(offset, cond) =>
                {
                    let take = match cond
                    {
                        CmpCondition::Equal => cmp_result == CmpResult::Equal,
                        CmpCondition::NotEqual => cmp_result != CmpResult::Equal,
                        CmpCondition::Less => cmp_result == CmpResult::Less,
                        CmpCondition::LessEqual =>
                        {
                            cmp_result == CmpResult::Less || cmp_result == CmpResult::Equal
                        },
                        CmpCondition::Greater => cmp_result == CmpResult::Greater,
                        CmpCondition::GreaterEqual =>
                        {
                            cmp_result == CmpResult::Greater || cmp_result == CmpResult::Equal
                        },
                    };
                    if take
                    {
                        pc += offset;
                    }
                    else
                    {
                        pc += 1;
                    }
                },
                Instruction::Swap(a, b) =>
                {
                    if a < n && b < n
                    {
                        regs.swap(a, b);
                    }
                    pc += 1;
                },
                Instruction::Push(r) =>
                {
                    if r < n && stack.len() < self.stack_capacity
                    {
                        stack.push(regs[r]);
                    }
                    pc += 1;
                },
                Instruction::Pop(r) =>
                {
                    if r < n
                    {
                        regs[r] = stack.pop().unwrap_or(0);
                    }
                    pc += 1;
                },
                Instruction::Return =>
                {
                    break;
                },
            }
        }

        Ok(regs)
    }

    /// Compute a simple cost: shorter = simpler.
    pub fn simplicity_cost(&self) -> f64 {
        self.instructions.len() as f64
    }

    /// Estimate time complexity: count loop depth via Jumps.
    /// This is a heuristic based on the number of backward jumps.
    pub fn estimated_complexity(&self) -> f64 {
        let mut backward_jumps = 0u64;
        let mut max_offset = 0i64;
        for instr in &self.instructions
        {
            match instr
            {
                Instruction::Jump(offset) | Instruction::JumpIf(offset, _)
                    if *offset < 0
                    => {
                        backward_jumps += 1;
                        max_offset = max_offset.max(offset.abs() as i64);
                    },
                _ =>
                {},
            }
        }
        // Heuristic: polynomial degree based on nesting
        if backward_jumps == 0
        {
            1.0
        }
        else if backward_jumps <= 2
        {
            2.0_f64.powf(backward_jumps as f64)
        }
        else
        {
            3.0_f64.powf(backward_jumps as f64)
        }
    }
}

// ===========================================================================
// 2. RL ENVIRONMENT FOR ALGORITHM SEARCH
// ===========================================================================

/// The state of an algorithm search episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgoSearchState {
    pub algorithm: Algorithm,
    pub tests_passed: usize,
    pub total_tests: usize,
}

/// Actions that can be taken in the algorithm search space.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlgoAction {
    /// Add an instruction at position idx
    AddInstruction(usize, Instruction),
    /// Remove instruction at position idx
    RemoveInstruction(usize),
    /// Modify instruction at position idx
    ModifyInstruction(usize, Instruction),
    /// Swap two instructions
    SwapInstructions(usize, usize),
    /// NOP (no operation — useful for variable-length episodes)
    Noop,
}

/// Specification of a problem to solve: input examples and expected outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemSpec {
    pub name: String,
    pub num_registers: usize,
    pub stack_capacity: usize,
    pub test_cases: Vec<(Vec<i64>, Vec<i64>)>,
    pub max_instructions: usize,
}

impl ProblemSpec {
    pub fn new(name: &str, num_registers: usize, max_instructions: usize) -> Self {
        Self {
            name: name.to_string(),
            num_registers,
            stack_capacity: 16,
            test_cases: Vec::new(),
            max_instructions,
        }
    }

    pub fn with_test(mut self, input: Vec<i64>, expected: Vec<i64>) -> Self {
        self.test_cases.push((input, expected));
        self
    }

    pub fn evaluate_correctness(&self, algo: &Algorithm) -> f64 {
        if self.test_cases.is_empty()
        {
            return 0.0;
        }
        let mut passed = 0;
        for (input, expected) in &self.test_cases
        {
            if let Ok(output) = algo.execute(input) {
                let ok = expected.iter().zip(&output).all(|(e, o)| e == o)
                    && expected.len() == output.len();
                if ok
                {
                    passed += 1;
                }
            }
        }
        passed as f64 / self.test_cases.len() as f64
    }

    pub fn all_passed(&self, algo: &Algorithm) -> bool {
        (self.evaluate_correctness(algo) - 1.0).abs() < f64::EPSILON
    }
}

/// The main RL environment trait for algorithm search.
pub trait AlgoEnv {
    fn reset(&mut self) -> AlgoSearchState;
    fn step(&mut self, action: &AlgoAction) -> (AlgoSearchState, f64, bool);
    fn observe(&self) -> AlgoSearchState;
    fn reward(&self, state: &AlgoSearchState) -> f64;
    fn available_actions(&self, state: &AlgoSearchState) -> Vec<AlgoAction>;
    fn is_terminal(&self, state: &AlgoSearchState) -> bool;
}

/// Concrete RL environment that learns algorithms for a given ProblemSpec.
pub struct AlgoSearchEnv {
    pub problem: ProblemSpec,
    pub state: AlgoSearchState,
    pub rng: RefCell<StdRng>,
    pub instruction_set: Vec<Instruction>,
    pub reward_correctness_weight: f64,
    pub reward_efficiency_weight: f64,
    pub reward_simplicity_weight: f64,
    step_count: usize,
    max_steps: usize,
}

impl AlgoSearchEnv {
    pub fn new(problem: ProblemSpec, seed: u64) -> Self {
        let state = AlgoSearchState {
            algorithm: Algorithm::new(problem.num_registers, problem.stack_capacity),
            tests_passed: 0,
            total_tests: problem.test_cases.len(),
        };
        Self {
            problem,
            state,
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
            instruction_set: Self::default_instruction_set(),
            reward_correctness_weight: 10.0,
            reward_efficiency_weight: 1.0,
            reward_simplicity_weight: 0.5,
            step_count: 0,
            max_steps: 200,
        }
    }

    fn default_instruction_set() -> Vec<Instruction> {
        vec![
            Instruction::Load(0, 0),
            Instruction::Store(0, 1),
            Instruction::Add(0, 0, 1),
            Instruction::Sub(0, 0, 1),
            Instruction::Mul(0, 0, 1),
            Instruction::Div(0, 0, 1),
            Instruction::Cmp(0, 1),
            Instruction::Jump(1),
            Instruction::Jump(-1),
            Instruction::JumpIf(1, CmpCondition::Equal),
            Instruction::JumpIf(-1, CmpCondition::Less),
            Instruction::Swap(0, 1),
            Instruction::Push(0),
            Instruction::Pop(0),
            Instruction::Return,
        ]
    }

    pub fn with_reward_weights(
        mut self,
        correctness: f64,
        efficiency: f64,
        simplicity: f64,
    ) -> Self {
        self.reward_correctness_weight = correctness;
        self.reward_efficiency_weight = efficiency;
        self.reward_simplicity_weight = simplicity;
        self
    }

    pub fn rand_reg(&self, rng: &mut StdRng) -> usize {
        let n = self.problem.num_registers;
        if n == 0 { 0 } else { rng.gen_range(0..n) }
    }

    pub fn generate_instruction(&self, rng: &mut StdRng) -> Instruction {
        let n = self.problem.num_registers.max(1);
        let r0 = rng.gen_range(0..n);
        let r1 = rng.gen_range(0..n);
        let r2 = rng.gen_range(0..n);
        let val = rng.gen_range(-100..101_i64);
        match rng.gen_range(0..13_u32)
        {
            0 => Instruction::Load(r0, val),
            1 => Instruction::Store(r0, r1),
            2 => Instruction::Add(r0, r1, r2),
            3 => Instruction::Sub(r0, r1, r2),
            4 => Instruction::Mul(r0, r1, r2),
            5 => Instruction::Div(r0, r1, r2),
            6 => Instruction::Cmp(r0, r1),
            7 => Instruction::Jump(rng.gen_range(-5..6)),
            8 => Instruction::JumpIf(
                rng.gen_range(-5..6),
                [
                    CmpCondition::Equal,
                    CmpCondition::NotEqual,
                    CmpCondition::Less,
                    CmpCondition::Greater,
                    CmpCondition::LessEqual,
                    CmpCondition::GreaterEqual,
                ][rng.gen_range(0..6_usize)],
            ),
            9 => Instruction::Swap(r0, r1),
            10 => Instruction::Push(r0),
            11 => Instruction::Pop(r0),
            _ => Instruction::Return,
        }
    }

    pub fn generate_instruction_for_problem(num_registers: usize, rng: &mut StdRng) -> Instruction {
        let n = num_registers.max(1);
        let r0 = rng.gen_range(0..n);
        let r1 = rng.gen_range(0..n);
        let r2 = rng.gen_range(0..n);
        let val = rng.gen_range(-100..101_i64);
        match rng.gen_range(0..13_u32)
        {
            0 => Instruction::Load(r0, val),
            1 => Instruction::Store(r0, r1),
            2 => Instruction::Add(r0, r1, r2),
            3 => Instruction::Sub(r0, r1, r2),
            4 => Instruction::Mul(r0, r1, r2),
            5 => Instruction::Div(r0, r1, r2),
            6 => Instruction::Cmp(r0, r1),
            7 => Instruction::Jump(rng.gen_range(-5..6)),
            8 => Instruction::JumpIf(
                rng.gen_range(-5..6),
                [
                    CmpCondition::Equal,
                    CmpCondition::NotEqual,
                    CmpCondition::Less,
                    CmpCondition::Greater,
                    CmpCondition::LessEqual,
                    CmpCondition::GreaterEqual,
                ][rng.gen_range(0..6_usize)],
            ),
            9 => Instruction::Swap(r0, r1),
            10 => Instruction::Push(r0),
            11 => Instruction::Pop(r0),
            _ => Instruction::Return,
        }
    }
}

impl AlgoEnv for AlgoSearchEnv {
    fn reset(&mut self) -> AlgoSearchState {
        self.step_count = 0;
        self.state = AlgoSearchState {
            algorithm: Algorithm::new(self.problem.num_registers, self.problem.stack_capacity),
            tests_passed: 0,
            total_tests: self.problem.test_cases.len(),
        };
        self.state.clone()
    }

    fn step(&mut self, action: &AlgoAction) -> (AlgoSearchState, f64, bool) {
        self.step_count += 1;
        let algo = &mut self.state.algorithm;
        let mut rng = self.rng.borrow_mut();

        match action
        {
            AlgoAction::AddInstruction(idx, instr) =>
            {
                let pos = (*idx).min(algo.instructions.len());
                algo.instructions.insert(pos, *instr);
            },
            AlgoAction::RemoveInstruction(idx) =>
            {
                if !algo.instructions.is_empty()
                {
                    let pos = *idx % algo.instructions.len();
                    algo.instructions.remove(pos);
                }
            },
            AlgoAction::ModifyInstruction(idx, instr) =>
            {
                if !algo.instructions.is_empty()
                {
                    let pos = *idx % algo.instructions.len();
                    algo.instructions[pos] = *instr;
                }
            },
            AlgoAction::SwapInstructions(a, b) =>
            {
                let len = algo.instructions.len();
                if len >= 2
                {
                    let i = *a % len;
                    let j = *b % len;
                    if i != j
                    {
                        algo.instructions.swap(i, j);
                    }
                }
            },
            AlgoAction::Noop =>
            {},
        }

        // Truncate if too long
        if algo.instructions.len() > self.problem.max_instructions
        {
            algo.instructions.truncate(self.problem.max_instructions);
        }

        // Random insertion if empty
        if algo.instructions.is_empty()
        {
            let n = self.problem.num_registers;
            let instr = Self::generate_instruction_for_problem(n, &mut rng);
            algo.instructions.push(instr);
        }

        // Evaluate
        let correctness = self.problem.evaluate_correctness(algo);
        self.state.tests_passed = (correctness * self.state.total_tests as f64) as usize;

        let r = self.reward(&self.state.clone());
        let done = self.is_terminal(&self.state);
        (self.state.clone(), r, done)
    }

    fn observe(&self) -> AlgoSearchState {
        self.state.clone()
    }

    fn reward(&self, state: &AlgoSearchState) -> f64 {
        let correctness = self.problem.evaluate_correctness(&state.algorithm);
        let simplicity = 1.0 / (1.0 + state.algorithm.simplicity_cost());
        let efficiency = 1.0 / state.algorithm.estimated_complexity();
        self.reward_correctness_weight * correctness
            + self.reward_simplicity_weight * simplicity
            + self.reward_efficiency_weight * efficiency
    }

    fn available_actions(&self, state: &AlgoSearchState) -> Vec<AlgoAction> {
        let mut actions = vec![AlgoAction::Noop];
        let len = state.algorithm.instructions.len();
        let n = self.problem.num_registers;

        for pos in 0..=len
        {
            let instr = {
                let mut rng = self.rng.borrow_mut();
                Self::generate_instruction_for_problem(n, &mut rng)
            };
            actions.push(AlgoAction::AddInstruction(pos, instr));
        }

        if len > 0
        {
            actions.push(AlgoAction::RemoveInstruction(0));
            let instr = {
                let mut rng = self.rng.borrow_mut();
                Self::generate_instruction_for_problem(n, &mut rng)
            };
            actions.push(AlgoAction::ModifyInstruction(0, instr));
        }

        if len >= 2
        {
            actions.push(AlgoAction::SwapInstructions(0, 1));
        }

        actions
    }

    fn is_terminal(&self, state: &AlgoSearchState) -> bool {
        self.step_count >= self.max_steps || self.problem.all_passed(&state.algorithm)
    }
}

// ===========================================================================
// 3. ALGORITHM MUTATION AND CROSSOVER
// ===========================================================================

/// Mutate an algorithm in-place.
pub fn mutate_algorithm(algo: &mut Algorithm, rng: &mut StdRng, mutation_rate: f64) {
    random_instruction_for_algo(algo, rng, mutation_rate);
}

fn random_instruction_for_algo(algo: &mut Algorithm, rng: &mut StdRng, mutation_rate: f64) {
    let n = algo.num_registers.max(1);

    let rand_reg = |rng: &mut StdRng| rng.gen_range(0..n);

    // Possibly add an instruction
    if rng.gen::<f64>() < mutation_rate
    {
        let r0 = rand_reg(rng);
        let r1 = rand_reg(rng);
        let r2 = rand_reg(rng);
        let instr = match rng.gen_range(0..8_u32)
        {
            0 => Instruction::Load(r0, rng.gen_range(-50..51)),
            1 => Instruction::Add(r0, r1, r2),
            2 => Instruction::Sub(r0, r1, r2),
            3 => Instruction::Mul(r0, r1, r2),
            4 => Instruction::Cmp(r0, r1),
            5 => Instruction::Jump(rng.gen_range(-3..4)),
            6 => Instruction::Swap(r0, r1),
            _ => Instruction::Return,
        };
        let pos = if algo.instructions.is_empty()
        {
            0
        }
        else
        {
            rng.gen_range(0..algo.instructions.len())
        };
        algo.instructions.insert(pos, instr);
    }

    // Possibly remove an instruction
    if algo.instructions.len() > 1 && rng.gen::<f64>() < mutation_rate * 0.5
    {
        let pos = rng.gen_range(0..algo.instructions.len());
        algo.instructions.remove(pos);
    }

    // Possibly modify an instruction
    if !algo.instructions.is_empty() && rng.gen::<f64>() < mutation_rate
    {
        let pos = rng.gen_range(0..algo.instructions.len());
        let r0 = rand_reg(rng);
        let r1 = rand_reg(rng);
        let r2 = rand_reg(rng);
        let instr = match rng.gen_range(0..8_u32)
        {
            0 => Instruction::Load(r0, rng.gen_range(-50..51)),
            1 => Instruction::Add(r0, r1, r2),
            2 => Instruction::Sub(r0, r1, r2),
            3 => Instruction::Mul(r0, r1, r2),
            4 => Instruction::Cmp(r0, r1),
            5 => Instruction::Jump(rng.gen_range(-3..4)),
            6 => Instruction::Swap(r0, r1),
            _ => Instruction::Return,
        };
        algo.instructions[pos] = instr;
    }

    // Possibly swap two instructions
    if algo.instructions.len() >= 2 && rng.gen::<f64>() < mutation_rate * 0.3
    {
        let a = rng.gen_range(0..algo.instructions.len());
        let b = rng.gen_range(0..algo.instructions.len());
        algo.instructions.swap(a, b);
    }
}

/// Crossover between two algorithms, producing one child.
pub fn crossover_algorithms(
    parent_a: &Algorithm,
    parent_b: &Algorithm,
    rng: &mut StdRng,
) -> Algorithm {
    let max_len = parent_a.instructions.len().max(parent_b.instructions.len());
    let split = if max_len == 0
    {
        0
    }
    else
    {
        rng.gen_range(0..max_len)
    };

    let mut child = Algorithm::new(parent_a.num_registers, parent_a.stack_capacity);
    for i in 0..split
    {
        if i < parent_a.instructions.len()
        {
            child.instructions.push(parent_a.instructions[i]);
        }
    }
    for i in split..max_len
    {
        if i < parent_b.instructions.len()
        {
            child.instructions.push(parent_b.instructions[i]);
        }
    }
    child
}

// ===========================================================================
// 4. POLICY GRADIENT METHODS
// ===========================================================================

/// A simple feed-forward neural network for policy/value approximation.
#[derive(Debug, Clone)]
pub struct FeedForwardNet {
    pub w1: Vec<Vec<f64>>,
    pub b1: Vec<f64>,
    pub w2: Vec<Vec<f64>>,
    pub b2: Vec<f64>,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

impl FeedForwardNet {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, rng: &mut StdRng) -> Self {
        let scale = (2.0 / input_dim as f64).sqrt();
        let mut w1 = vec![vec![0.0; hidden_dim]; input_dim];
        let mut b1 = vec![0.0; hidden_dim];
        let mut w2 = vec![vec![0.0; output_dim]; hidden_dim];
        let mut b2 = vec![0.0; output_dim];

        for row in w1.iter_mut()
        {
            for v in row.iter_mut()
            {
                *v = rng.gen_range(-scale..scale);
            }
        }
        for row in w2.iter_mut()
        {
            for v in row.iter_mut()
            {
                *v = rng.gen_range(-scale..scale);
            }
        }
        // Initialize biases to small values
        for v in b1.iter_mut()
        {
            *v = rng.gen_range(-0.01..0.01);
        }
        for v in b2.iter_mut()
        {
            *v = rng.gen_range(-0.01..0.01);
        }

        Self {
            w1,
            b1,
            w2,
            b2,
            input_dim,
            hidden_dim,
            output_dim,
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut hidden = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim
        {
            let sum: f64 = input.iter().zip(&self.w1).map(|(x, row)| x * row[j]).sum();
            hidden[j] = relu(sum + self.b1[j]);
        }
        let mut output = vec![0.0; self.output_dim];
        for j in 0..self.output_dim
        {
            let sum: f64 = hidden
                .iter()
                .enumerate()
                .map(|(i, h)| h * self.w2[i][j])
                .sum();
            output[j] = sum + self.b2[j];
        }
        output
    }

    /// Softmax over outputs for policy network.
    pub fn forward_softmax(&self, input: &[f64]) -> Vec<f64> {
        let logits = self.forward(input);
        softmax(&logits)
    }

    /// Train with SGD on a single sample.
    pub fn train_sgd(&mut self, input: &[f64], target: &[f64], lr: f64) {
        // Forward
        let mut hidden = vec![0.0; self.hidden_dim];
        let mut pre_hidden = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim
        {
            let sum: f64 = input.iter().zip(&self.w1).map(|(x, row)| x * row[j]).sum();
            pre_hidden[j] = sum + self.b1[j];
            hidden[j] = relu(pre_hidden[j]);
        }
        let mut output = vec![0.0; self.output_dim];
        let mut pre_output = vec![0.0; self.output_dim];
        for j in 0..self.output_dim
        {
            let sum: f64 = hidden
                .iter()
                .enumerate()
                .map(|(i, h)| h * self.w2[i][j])
                .sum();
            pre_output[j] = sum + self.b2[j];
            output[j] = pre_output[j];
        }

        // Output layer gradient (MSE derivative)
        let mut out_grad = vec![0.0; self.output_dim];
        for j in 0..self.output_dim
        {
            out_grad[j] = 2.0 * (output[j] - target[j]) / self.output_dim as f64;
        }

        // Gradients for w2, b2
        for i in 0..self.hidden_dim
        {
            for j in 0..self.output_dim
            {
                self.w2[i][j] -= lr * out_grad[j] * hidden[i];
            }
        }
        for j in 0..self.output_dim
        {
            self.b2[j] -= lr * out_grad[j];
        }

        // Hidden layer gradient
        let mut hidden_grad = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim
        {
            let sum: f64 = (0..self.output_dim)
                .map(|j| out_grad[j] * self.w2[i][j])
                .sum();
            hidden_grad[i] = sum * relu_deriv(pre_hidden[i]);
        }

        // Gradients for w1, b1
        for (k, &x) in input.iter().enumerate()
        {
            for i in 0..self.hidden_dim
            {
                self.w1[k][i] -= lr * hidden_grad[i] * x;
            }
        }
        for i in 0..self.hidden_dim
        {
            self.b1[i] -= lr * hidden_grad[i];
        }
    }
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_deriv(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn softmax(x: &[f64]) -> Vec<f64> {
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|v| (v - max_x).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0
    {
        vec![1.0 / x.len() as f64; x.len()]
    }
    else
    {
        exps.iter().map(|v| v / sum).collect()
    }
}

/// Convert algorithm state to a fixed-size feature vector.
pub fn algo_to_features(algo: &Algorithm, max_len: usize) -> Vec<f64> {
    let mut feats = Vec::with_capacity(max_len * 3);
    for i in 0..max_len
    {
        if i < algo.instructions.len()
        {
            match algo.instructions[i]
            {
                Instruction::Load(r, v) =>
                {
                    feats.push(0.0);
                    feats.push(r as f64 / algo.num_registers.max(1) as f64);
                    feats.push(v as f64 / 100.0);
                },
                Instruction::Store(a, b) =>
                {
                    feats.push(1.0);
                    feats.push(a as f64 / algo.num_registers.max(1) as f64);
                    feats.push(b as f64 / algo.num_registers.max(1) as f64);
                },
                Instruction::Add(d, a, b) =>
                {
                    feats.push(2.0);
                    feats.push(d as f64 / algo.num_registers.max(1) as f64);
                    feats.push(a as f64 / algo.num_registers.max(1) as f64 + b as f64 * 0.01);
                },
                Instruction::Sub(_, a, b) =>
                {
                    feats.push(3.0);
                    feats.push(a as f64 / algo.num_registers.max(1) as f64);
                    feats.push(b as f64 / algo.num_registers.max(1) as f64);
                },
                Instruction::Mul(_, a, b) =>
                {
                    feats.push(4.0);
                    feats.push(a as f64 / algo.num_registers.max(1) as f64);
                    feats.push(b as f64 / algo.num_registers.max(1) as f64);
                },
                Instruction::Div(_, a, b) =>
                {
                    feats.push(5.0);
                    feats.push(a as f64 / algo.num_registers.max(1) as f64);
                    feats.push(b as f64 / algo.num_registers.max(1) as f64);
                },
                Instruction::Cmp(a, b) =>
                {
                    feats.push(6.0);
                    feats.push(a as f64 / algo.num_registers.max(1) as f64);
                    feats.push(b as f64 / algo.num_registers.max(1) as f64);
                },
                Instruction::Jump(off) =>
                {
                    feats.push(7.0);
                    feats.push(off as f64 / 10.0);
                    feats.push(0.0);
                },
                Instruction::JumpIf(off, cond) =>
                {
                    feats.push(8.0);
                    feats.push(off as f64 / 10.0);
                    feats.push(cond as u8 as f64 / 6.0);
                },
                Instruction::Swap(a, b) =>
                {
                    feats.push(9.0);
                    feats.push(a as f64 / algo.num_registers.max(1) as f64);
                    feats.push(b as f64 / algo.num_registers.max(1) as f64);
                },
                Instruction::Push(r) =>
                {
                    feats.push(10.0);
                    feats.push(r as f64 / algo.num_registers.max(1) as f64);
                    feats.push(0.0);
                },
                Instruction::Pop(r) =>
                {
                    feats.push(11.0);
                    feats.push(r as f64 / algo.num_registers.max(1) as f64);
                    feats.push(0.0);
                },
                Instruction::Return =>
                {
                    feats.push(12.0);
                    feats.push(0.0);
                    feats.push(0.0);
                },
            }
        }
        else
        {
            feats.push(-1.0);
            feats.push(0.0);
            feats.push(0.0);
        }
    }
    feats.push(algo.num_registers as f64);
    feats.push(algo.instructions.len() as f64 / max_len.max(1) as f64);
    feats
}

/// REINFORCE with baseline agent.
pub struct ReinforceAgent {
    pub policy_net: FeedForwardNet,
    pub value_net: FeedForwardNet,
    pub lr_policy: f64,
    pub lr_value: f64,
    pub gamma: f64,
    pub feat_dim: usize,
    pub action_dim: usize,
    pub rng: RefCell<StdRng>,
}

impl ReinforceAgent {
    pub fn new(
        feat_dim: usize,
        hidden_dim: usize,
        action_dim: usize,
        lr_policy: f64,
        lr_value: f64,
        gamma: f64,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let policy_net = FeedForwardNet::new(feat_dim, hidden_dim, action_dim, &mut rng);
        let mut rng2 = StdRng::seed_from_u64(seed.wrapping_add(1));
        let value_net = FeedForwardNet::new(feat_dim, hidden_dim, 1, &mut rng2);
        Self {
            policy_net,
            value_net,
            lr_policy,
            lr_value,
            gamma,
            feat_dim,
            action_dim,
            rng: RefCell::new(StdRng::seed_from_u64(seed.wrapping_add(2))),
        }
    }

    /// Sample an action from the policy distribution.
    pub fn act(&self, features: &[f64]) -> usize {
        let probs = self.policy_net.forward_softmax(features);
        let mut rng = self.rng.borrow_mut();
        let sample: f64 = rng.gen();
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate()
        {
            cum += p;
            if sample <= cum
            {
                return i;
            }
        }
        probs.len() - 1
    }

    /// Compute value estimate.
    pub fn value(&self, features: &[f64]) -> f64 {
        self.value_net.forward(features)[0]
    }

    /// Store an episode trajectory for later training.
    pub fn train_episode(&mut self, trajectory: &EpisodeTrajectory) {
        let t = trajectory;
        // Compute discounted returns
        let mut returns = vec![0.0; t.rewards.len()];

        // Baseline values
        let values: Vec<f64> = t.features.iter().map(|f| self.value(f)).collect();

        // Compute returns (backward)
        let mut g = 0.0;
        for i in (0..t.rewards.len()).rev()
        {
            g = t.rewards[i] + self.gamma * g;
            returns[i] = g;
        }

        // Advantages
        let advantages: Vec<f64> = returns.iter().zip(&values).map(|(r, v)| r - v).collect();

        // Update policy (REINFORCE)
        for i in 0..t.features.len()
        {
            let feat = &t.features[i];
            let probs = self.policy_net.forward_softmax(feat);
            let mut policy_target = vec![0.0; self.action_dim];
            for j in 0..self.action_dim
            {
                policy_target[j] = probs[j];
            }
            // REINFORCE gradient: increase log-prob of taken action scaled by advantage
            let log_prob = (probs[t.actions[i]].max(1e-10)).ln();
            let scale = (advantages[i] * log_prob).clamp(-10.0, 10.0);
            policy_target[t.actions[i]] += self.lr_policy * scale;
            // Normalize
            let sum: f64 = policy_target.iter().sum();
            if sum > 0.0
            {
                for v in &mut policy_target
                {
                    *v /= sum;
                }
            }
            self.policy_net
                .train_sgd(feat, &policy_target, self.lr_policy);
        }

        // Update value network
        for i in 0..t.features.len()
        {
            let v_target = vec![returns[i]];
            self.value_net
                .train_sgd(&t.features[i], &v_target, self.lr_value);
        }
    }

    /// Run a full training loop.
    pub fn train_loop(
        &mut self,
        env: &mut impl AlgoEnv,
        episodes: usize,
        max_steps_per_episode: usize,
        _action_set_size: usize,
    ) -> Vec<f64> {
        let mut episode_rewards = Vec::new();

        for _ep in 0..episodes
        {
            let state = env.reset();
            let mut trajectory = EpisodeTrajectory::new();

            for _step in 0..max_steps_per_episode
            {
                let features = algo_to_features(&state.algorithm, 10);
                let action_idx = self.act(&features);

                let actions = env.available_actions(&state);
                let action = if actions.is_empty()
                {
                    AlgoAction::Noop
                }
                else
                {
                    actions[action_idx % actions.len()].clone()
                };

                let (_next_state, reward, done) = env.step(&action);

                trajectory.push(&features, action_idx, reward);

                if done
                {
                    break;
                }
                if _step == max_steps_per_episode - 1
                {
                    break;
                }
            }

            let total_reward: f64 = trajectory.rewards.iter().sum();
            episode_rewards.push(total_reward);

            if !trajectory.features.is_empty()
            {
                self.train_episode(&trajectory);
            }
        }

        episode_rewards
    }
}

/// Actor-Critic agent (simplified).
pub struct ActorCriticAgent {
    pub policy_net: FeedForwardNet,
    pub value_net: FeedForwardNet,
    pub lr_policy: f64,
    pub lr_value: f64,
    pub gamma: f64,
    pub rng: RefCell<StdRng>,
}

impl ActorCriticAgent {
    pub fn new(
        feat_dim: usize,
        hidden_dim: usize,
        action_dim: usize,
        lr_policy: f64,
        lr_value: f64,
        gamma: f64,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let policy_net = FeedForwardNet::new(feat_dim, hidden_dim, action_dim, &mut rng);
        let mut rng2 = StdRng::seed_from_u64(seed.wrapping_add(1));
        let value_net = FeedForwardNet::new(feat_dim, hidden_dim, 1, &mut rng2);
        Self {
            policy_net,
            value_net,
            lr_policy,
            lr_value,
            gamma,
            rng: RefCell::new(StdRng::seed_from_u64(seed.wrapping_add(2))),
        }
    }

    pub fn act(&self, features: &[f64]) -> usize {
        let probs = self.policy_net.forward_softmax(features);
        let mut rng = self.rng.borrow_mut();
        let sample: f64 = rng.gen();
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate()
        {
            cum += p;
            if sample <= cum
            {
                return i;
            }
        }
        probs.len() - 1
    }

    /// Single-step actor-critic update (TD(0)).
    pub fn update(
        &mut self,
        state_feats: &[f64],
        action_idx: usize,
        reward: f64,
        next_state_feats: &[f64],
        done: bool,
        _action_dim: usize,
    ) {
        let v = self.value_net.forward(state_feats)[0];
        let v_next = if done
        {
            0.0
        }
        else
        {
            self.value_net.forward(next_state_feats)[0]
        };
        let td_error = reward + self.gamma * v_next - v;

        // Update value
        let v_target = vec![reward + self.gamma * v_next];
        self.value_net
            .train_sgd(state_feats, &v_target, self.lr_value);

        // Update policy
        let probs = self.policy_net.forward_softmax(state_feats);
        let mut policy_target = probs.clone();
        let log_prob = (probs[action_idx].max(1e-10)).ln();
        let grad_scale = td_error * log_prob;
        policy_target[action_idx] += self.lr_policy * grad_scale;
        let sum: f64 = policy_target.iter().sum();
        if sum > 0.0
        {
            for v in &mut policy_target
            {
                *v /= sum;
            }
        }
        self.policy_net
            .train_sgd(state_feats, &policy_target, self.lr_policy);
    }
}

/// Trajectory accumulated during an episode.
#[derive(Debug, Clone, Default)]
pub struct EpisodeTrajectory {
    pub features: Vec<Vec<f64>>,
    pub actions: Vec<usize>,
    pub rewards: Vec<f64>,
}

impl EpisodeTrajectory {
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
        }
    }

    pub fn push(&mut self, features: &[f64], action: usize, reward: f64) {
        self.features.push(features.to_vec());
        self.actions.push(action);
        self.rewards.push(reward);
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

// ===========================================================================
// 5. Q-LEARNING FOR ALGORITHM SEARCH
// ===========================================================================

/// Discretized state representation for tabular Q-learning.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DiscretizedState {
    /// Number of instructions (bucketed)
    pub len_bucket: usize,
    /// Most common instruction type (0-12)
    pub dominant_op: Option<u8>,
    /// Whether the algorithm has a backward jump
    pub has_loop: bool,
    /// Correctness bucket (0.0, 0.25, 0.5, 0.75, 1.0 -> 0..4)
    pub correctness_bucket: usize,
}

/// Tabular Q-learning agent for algorithm search.
pub struct TabularQLearner {
    pub q_table: HashMap<(DiscretizedState, usize), f64>,
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub epsilon_decay: f64,
    pub min_epsilon: f64,
    pub rng: RefCell<StdRng>,
}

impl TabularQLearner {
    pub fn new(alpha: f64, gamma: f64, epsilon: f64, epsilon_decay: f64, seed: u64) -> Self {
        Self {
            q_table: HashMap::new(),
            alpha,
            gamma,
            epsilon,
            epsilon_decay,
            min_epsilon: 0.01,
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }

    pub fn discretize_state(&self, state: &AlgoSearchState) -> DiscretizedState {
        let algo = &state.algorithm;
        let len_bucket = match algo.instructions.len()
        {
            0 => 0,
            1..=3 => 1,
            4..=7 => 2,
            8..=15 => 3,
            _ => 4,
        };

        let dominant_op = most_common_instruction(algo);

        let has_loop = algo
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Jump(o) if *o < 0));

        let correctness_bucket = if state.total_tests == 0
        {
            0
        }
        else
        {
            let ratio = state.tests_passed as f64 / state.total_tests as f64;
            (ratio * 4.0).min(3.0) as usize
        };

        DiscretizedState {
            len_bucket,
            dominant_op,
            has_loop,
            correctness_bucket,
        }
    }

    pub fn get_q(&self, state: &DiscretizedState, action_idx: usize) -> f64 {
        *self
            .q_table
            .get(&(state.clone(), action_idx))
            .unwrap_or(&0.0)
    }

    pub fn act(&self, state: &DiscretizedState, num_actions: usize) -> usize {
        let mut rng = self.rng.borrow_mut();
        if rng.gen::<f64>() < self.epsilon
        {
            rng.gen_range(0..num_actions.max(1))
        }
        else
        {
            (0..num_actions)
                .max_by(|a, b| {
                    let qa = self.get_q(state, *a);
                    let qb = self.get_q(state, *b);
                    qa.partial_cmp(&qb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0)
        }
    }

    pub fn update(
        &mut self,
        state: &DiscretizedState,
        action_idx: usize,
        reward: f64,
        next_state: &DiscretizedState,
        num_actions: usize,
        done: bool,
    ) {
        let max_q_next = if done
        {
            0.0
        }
        else
        {
            (0..num_actions)
                .map(|a| self.get_q(next_state, a))
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let old_q = self.get_q(state, action_idx);
        let new_q = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q);
        self.q_table.insert((state.clone(), action_idx), new_q);
    }

    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.min_epsilon);
    }

    pub fn train_loop(
        &mut self,
        env: &mut impl AlgoEnv,
        episodes: usize,
        max_steps_per_episode: usize,
    ) -> Vec<f64> {
        let mut episode_rewards = Vec::new();

        for _ep in 0..episodes
        {
            let state = env.reset();
            let disc_state = self.discretize_state(&state);
            let actions = env.available_actions(&state);
            let mut total_reward = 0.0;

            for _step in 0..max_steps_per_episode
            {
                let action_idx = self.act(&disc_state, actions.len());
                let action = if actions.is_empty()
                {
                    AlgoAction::Noop
                }
                else
                {
                    actions[action_idx % actions.len()].clone()
                };

                let (next_state, reward, done) = env.step(&action);
                total_reward += reward;

                let next_disc = self.discretize_state(&next_state);
                let next_actions = env.available_actions(&next_state);

                self.update(
                    &disc_state,
                    action_idx,
                    reward,
                    &next_disc,
                    next_actions.len(),
                    done,
                );

                if done
                {
                    break;
                }
            }

            episode_rewards.push(total_reward);
            self.decay_epsilon();
        }

        episode_rewards
    }
}

/// Experience replay buffer.
pub struct ExperienceReplay {
    pub buffer: VecDeque<Experience>,
    pub capacity: usize,
}

#[derive(Debug, Clone)]
pub struct Experience {
    pub state: DiscretizedState,
    pub action_idx: usize,
    pub reward: f64,
    pub next_state: DiscretizedState,
    pub done: bool,
}

impl ExperienceReplay {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, exp: Experience) {
        if self.buffer.len() >= self.capacity
        {
            self.buffer.pop_front();
        }
        self.buffer.push_back(exp);
    }

    pub fn sample_batch(&self, batch_size: usize, rng: &mut StdRng) -> Vec<&Experience> {
        let n = self.buffer.len();
        if n == 0 || batch_size == 0
        {
            return Vec::new();
        }
        let size = batch_size.min(n);
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev()
        {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        indices[..size].iter().map(|&i| &self.buffer[i]).collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

fn most_common_instruction(algo: &Algorithm) -> Option<u8> {
    let mut counts = [0u32; 13];
    for instr in &algo.instructions
    {
        let idx = match instr
        {
            Instruction::Load(..) => 0,
            Instruction::Store(..) => 1,
            Instruction::Add(..) => 2,
            Instruction::Sub(..) => 3,
            Instruction::Mul(..) => 4,
            Instruction::Div(..) => 5,
            Instruction::Cmp(..) => 6,
            Instruction::Jump(..) => 7,
            Instruction::JumpIf(..) => 8,
            Instruction::Swap(..) => 9,
            Instruction::Push(..) => 10,
            Instruction::Pop(..) => 11,
            Instruction::Return => 12,
        };
        counts[idx] += 1;
    }
    counts
        .iter()
        .enumerate()
        .max_by_key(|(_, &c)| c)
        .filter(|(_, &c)| c > 0)
        .map(|(i, _)| i as u8)
}

// ===========================================================================
// 6. META-LEARNING
// ===========================================================================

/// An algorithm template: a pattern extracted from successful solutions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmTemplate {
    pub problem_name: String,
    pub instructions: Vec<Instruction>,
    pub num_registers: usize,
    pub performance_score: f64,
    pub usage_count: u64,
}

/// Meta-learner that stores and retrieves algorithm templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearner {
    pub templates: Vec<AlgorithmTemplate>,
    pub feature_extractor: ProblemFeatureExtractor,
}

impl Default for MetaLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaLearner {
    pub fn new() -> Self {
        Self {
            templates: Vec::new(),
            feature_extractor: ProblemFeatureExtractor::new(),
        }
    }

    /// Store a successful algorithm as a template.
    pub fn store_template(&mut self, problem: &ProblemSpec, algo: &Algorithm, performance: f64) {
        // Update existing template if one exists
        for t in &mut self.templates
        {
            if t.problem_name == problem.name
            {
                if performance > t.performance_score
                {
                    t.instructions = algo.instructions.clone();
                    t.performance_score = performance;
                }
                t.usage_count += 1;
                return;
            }
        }
        self.templates.push(AlgorithmTemplate {
            problem_name: problem.name.clone(),
            instructions: algo.instructions.clone(),
            num_registers: algo.num_registers,
            performance_score: performance,
            usage_count: 1,
        });
    }

    /// Retrieve the best template for a problem.
    pub fn get_best_template(&self, problem_name: &str) -> Option<&AlgorithmTemplate> {
        self.templates
            .iter()
            .filter(|t| t.problem_name == problem_name)
            .max_by(|a, b| {
                a.performance_score
                    .partial_cmp(&b.performance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Transfer learning: find similar problems and adapt templates.
    pub fn transfer_learn(&self, target_problem: &ProblemSpec) -> Vec<Algorithm> {
        let target_feats = self.feature_extractor.extract(target_problem);
        let mut scored: Vec<(f64, &AlgorithmTemplate)> = self
            .templates
            .iter()
            .map(|t| {
                let src_feats = self.feature_extractor.extract_features(
                    t.num_registers,
                    t.instructions.len(),
                    t.instructions
                        .iter()
                        .filter(|i| matches!(i, Instruction::Jump(_) | Instruction::JumpIf(..)))
                        .count(),
                    t.instructions
                        .iter()
                        .filter(|i| {
                            matches!(
                                i,
                                Instruction::Add(..)
                                    | Instruction::Sub(..)
                                    | Instruction::Mul(..)
                                    | Instruction::Div(..)
                            )
                        })
                        .count(),
                );
                let sim = cosine_similarity(&target_feats, &src_feats);
                (sim, t)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(3)
            .map(|(_, t)| Algorithm {
                instructions: t.instructions.clone(),
                num_registers: t.num_registers,
                stack_capacity: 16,
            })
            .collect()
    }

    /// Select which search strategy to use based on problem features.
    pub fn select_strategy(&self, problem: &ProblemSpec) -> SearchStrategy {
        let feats = self.feature_extractor.extract(problem);
        let complexity = feats[0] + feats[1]; // heuristic

        if complexity < 2.0
        {
            SearchStrategy::TabularQLearning
        }
        else if complexity < 5.0
        {
            SearchStrategy::Reinforce
        }
        else
        {
            SearchStrategy::MCTS
        }
    }
}

/// Search strategies the meta-learner can select from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    TabularQLearning,
    Reinforce,
    ActorCritic,
    SimulatedAnnealing,
    BeamSearch,
    MCTS,
}

/// Feature extractor for problem descriptions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemFeatureExtractor;

impl Default for ProblemFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl ProblemFeatureExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Extract features from a ProblemSpec.
    pub fn extract(&self, problem: &ProblemSpec) -> Vec<f64> {
        self.extract_features(
            problem.num_registers,
            problem.max_instructions,
            problem.test_cases.len(),
            problem
                .test_cases
                .iter()
                .map(|(a, b)| a.len().max(b.len()))
                .max()
                .unwrap_or(0),
        )
    }

    /// Extract features from raw problem characteristics.
    pub fn extract_features(
        &self,
        num_regs: usize,
        max_instr: usize,
        num_tests: usize,
        max_io_len: usize,
    ) -> Vec<f64> {
        vec![
            num_regs as f64,
            max_instr as f64,
            num_tests as f64,
            max_io_len as f64,
        ]
    }
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0
    {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b).take(len).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().take(len).map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().take(len).map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0
    {
        0.0
    }
    else
    {
        dot / (norm_a * norm_b)
    }
}

// ===========================================================================
// 7. SEARCH HEURISTICS
// ===========================================================================

/// Simulated Annealing for algorithm search.
pub struct SimulatedAnnealing {
    pub initial_temp: f64,
    pub cooling_rate: f64,
    pub min_temp: f64,
    pub rng: RefCell<StdRng>,
}

impl SimulatedAnnealing {
    pub fn new(initial_temp: f64, cooling_rate: f64, seed: u64) -> Self {
        Self {
            initial_temp,
            cooling_rate,
            min_temp: 0.001,
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }

    /// Run simulated annealing on the given problem, returning the best algorithm found.
    pub fn search(
        &self,
        problem: &ProblemSpec,
        initial_algo: &Algorithm,
        max_iterations: usize,
    ) -> (Algorithm, f64) {
        let mut current = initial_algo.clone();
        let mut current_score = evaluate_fitness(problem, &current);
        let mut best = current.clone();
        let mut best_score = current_score;
        let mut temp = self.initial_temp;
        let mut rng = self.rng.borrow_mut();

        for _ in 0..max_iterations
        {
            let mut candidate = current.clone();
            mutate_algorithm(&mut candidate, &mut rng, 0.3);
            // Ensure not empty
            if candidate.instructions.is_empty()
            {
                candidate.instructions.push(Instruction::Return);
            }
            let candidate_score = evaluate_fitness(problem, &candidate);

            let delta = candidate_score - current_score;
            if delta > 0.0 || rng.gen::<f64>() < (delta / temp.max(1e-10)).exp()
            {
                current = candidate;
                current_score = candidate_score;
                if current_score > best_score
                {
                    best = current.clone();
                    best_score = current_score;
                }
            }

            temp *= self.cooling_rate;
            if temp < self.min_temp
            {
                break;
            }
        }

        (best, best_score)
    }
}

/// Beam Search over algorithm space.
pub struct BeamSearch {
    pub beam_width: usize,
    pub max_depth: usize,
    pub rng: RefCell<StdRng>,
}

impl BeamSearch {
    pub fn new(beam_width: usize, max_depth: usize, seed: u64) -> Self {
        Self {
            beam_width,
            max_depth,
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }

    /// Run beam search to find the best algorithm.
    pub fn search(&self, problem: &ProblemSpec) -> Option<(Algorithm, f64)> {
        let mut rng = self.rng.borrow_mut();
        let mut beam: Vec<(Algorithm, f64)> = vec![(
            Algorithm::new(problem.num_registers, problem.stack_capacity),
            evaluate_fitness(
                problem,
                &Algorithm::new(problem.num_registers, problem.stack_capacity),
            ),
        )];

        for _depth in 0..self.max_depth
        {
            let mut candidates: Vec<(Algorithm, f64)> = Vec::new();

            for (algo, _) in &beam
            {
                // Generate mutations
                for _ in 0..5
                {
                    let mut mutated = algo.clone();
                    mutate_algorithm(&mut mutated, &mut rng, 0.5);
                    if mutated.instructions.is_empty()
                    {
                        continue;
                    }
                    let score = evaluate_fitness(problem, &mutated);
                    candidates.push((mutated, score));
                }
                // Generate crossover candidates from beam
                if beam.len() >= 2
                {
                    for _ in 0..3
                    {
                        let other_idx = rng.gen_range(0..beam.len());
                        let child = crossover_algorithms(algo, &beam[other_idx].0, &mut rng);
                        if child.instructions.is_empty()
                        {
                            continue;
                        }
                        let score = evaluate_fitness(problem, &child);
                        candidates.push((child, score));
                    }
                }
            }

            // Keep top-k by fitness
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            beam = candidates.into_iter().take(self.beam_width).collect();

            // Check for perfect solution
            if let Some((algo, score)) = beam.first()
            {
                if problem.all_passed(algo)
                {
                    return Some((algo.clone(), *score));
                }
            }
        }

        beam.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
}

/// MCTS (Monte Carlo Tree Search) node for algorithm discovery.
#[derive(Debug, Clone)]
pub struct MctsNode {
    pub algorithm: Algorithm,
    pub visits: u64,
    pub total_value: f64,
    pub children: Vec<(AlgoAction, MctsNode)>,
    pub is_expanded: bool,
}

impl MctsNode {
    pub fn new(algorithm: Algorithm) -> Self {
        Self {
            algorithm,
            visits: 0,
            total_value: 0.0,
            children: Vec::new(),
            is_expanded: false,
        }
    }

    pub fn ucb1(&self, parent_visits: u64, exploration: f64) -> f64 {
        if self.visits == 0
        {
            return f64::INFINITY;
        }
        self.total_value / self.visits as f64
            + exploration * (2.0 * (parent_visits as f64).ln() / self.visits as f64).sqrt()
    }

    pub fn best_child(&self, exploration: f64) -> Option<usize> {
        let pv = self.visits;
        self.children
            .iter()
            .enumerate()
            .max_by(|(_, (_, a)), (_, (_, b))| {
                a.ucb1(pv, exploration)
                    .partial_cmp(&b.ucb1(pv, exploration))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }
}

/// MCTS search engine for algorithm discovery with progressive widening.
pub struct MctsEngine {
    pub exploration_constant: f64,
    pub max_iterations: usize,
    pub progressive_widening_constant: f64,
    pub rng: RefCell<StdRng>,
}

impl MctsEngine {
    pub fn new(exploration_constant: f64, max_iterations: usize, seed: u64) -> Self {
        Self {
            exploration_constant,
            max_iterations,
            progressive_widening_constant: 0.5,
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }

    pub fn search(&self, problem: &ProblemSpec, root_algo: &Algorithm) -> (Algorithm, f64) {
        let mut root = MctsNode::new(root_algo.clone());
        let mut rng = self.rng.borrow_mut();

        for _ in 0..self.max_iterations
        {
            // Selection
            let mut path: Vec<usize> = Vec::new();
            let current = &mut root;

            while current.is_expanded && !current.children.is_empty()
            {
                if let Some(idx) = current.best_child(self.exploration_constant)
                {
                    path.push(idx);
                    // SAFETY: workaround for borrow checker — we re-traverse from root
                    break;
                }
                else
                {
                    break;
                }
            }

            // Re-traverse from root safely
            let mut leaf = &mut root;
            leaf.visits += 1;
            for &idx in &path
            {
                leaf = &mut leaf.children[idx].1;
                leaf.visits += 1;
            }

            // Expansion
            if !leaf.is_expanded
            {
                // Progressive widening: limit children based on visits
                let max_children =
                    (self.progressive_widening_constant * (leaf.visits as f64).sqrt()) as usize;
                let max_children = max_children.max(2).min(10);

                if leaf.children.len() < max_children
                {
                    let actions =
                        generate_expansion_actions(&leaf.algorithm, &mut rng, max_children);
                    for action in actions
                    {
                        let mut child_algo = leaf.algorithm.clone();
                        apply_action(&mut child_algo, &action);
                        if child_algo.instructions.is_empty()
                        {
                            child_algo.instructions.push(Instruction::Return);
                        }
                        leaf.children.push((action, MctsNode::new(child_algo)));
                    }
                }
                leaf.is_expanded = true;

                // Rollout
                let rollout_value = self.rollout(problem, &leaf.algorithm, &mut rng);
                leaf.total_value += rollout_value;

                // Backpropagate
                let mut back = &mut root;
                back.total_value += rollout_value;
                for &idx in &path
                {
                    back = &mut back.children[idx].1;
                    back.total_value += rollout_value;
                }
            }
            else if !leaf.children.is_empty()
            {
                // Select child and rollout
                let idx = rng.gen_range(0..leaf.children.len());
                let child = &mut leaf.children[idx].1;
                let rollout_value = self.rollout(problem, &child.algorithm, &mut rng);
                child.total_value += rollout_value;
                child.visits += 1;

                // Backprop
                let mut back = &mut root;
                back.total_value += rollout_value;
                for &idx in &path
                {
                    back = &mut back.children[idx].1;
                    back.total_value += rollout_value;
                }
            }
            else
            {
                // Leaf with no children — just rollout
                let rollout_value = self.rollout(problem, &leaf.algorithm, &mut rng);
                leaf.total_value += rollout_value;
            }
        }

        // Return best child of root
        let best = root.children.iter().max_by(|(_, a), (_, b)| {
            a.total_value
                .partial_cmp(&b.total_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        match best
        {
            Some((_, node)) =>
            {
                let score = evaluate_fitness(problem, &node.algorithm);
                (node.algorithm.clone(), score)
            },
            None =>
            {
                let score = evaluate_fitness(problem, &root.algorithm);
                (root.algorithm.clone(), score)
            },
        }
    }

    fn rollout(&self, problem: &ProblemSpec, algo: &Algorithm, rng: &mut StdRng) -> f64 {
        let mut current = algo.clone();
        let mut best_score = evaluate_fitness(problem, &current);

        for _ in 0..20
        {
            let mut candidate = current.clone();
            mutate_algorithm(&mut candidate, rng, 0.4);
            if candidate.instructions.is_empty()
            {
                continue;
            }
            let score = evaluate_fitness(problem, &candidate);
            if score > best_score
            {
                best_score = score;
                current = candidate;
            }
            if problem.all_passed(&current)
            {
                break;
            }
        }
        best_score
    }
}

/// Generate expansion actions for MCTS.
fn generate_expansion_actions(
    algo: &Algorithm,
    rng: &mut StdRng,
    max_count: usize,
) -> Vec<AlgoAction> {
    let mut actions = Vec::new();
    let n = algo.num_registers.max(1);

    for _ in 0..max_count
    {
        let r0 = rng.gen_range(0..n);
        let r1 = rng.gen_range(0..n);
        let r2 = rng.gen_range(0..n);
        let action = match rng.gen_range(0..5_u32)
        {
            0 =>
            {
                let instr = Instruction::Add(r0, r1, r2);
                let pos = if algo.instructions.is_empty()
                {
                    0
                }
                else
                {
                    rng.gen_range(0..=algo.instructions.len())
                };
                AlgoAction::AddInstruction(pos, instr)
            },
            1 if !algo.instructions.is_empty() =>
            {
                let pos = rng.gen_range(0..algo.instructions.len());
                let instr = Instruction::Load(r0, rng.gen_range(-10..11));
                AlgoAction::ModifyInstruction(pos, instr)
            },
            2 if algo.instructions.len() >= 2 =>
            {
                let a = rng.gen_range(0..algo.instructions.len());
                let b = rng.gen_range(0..algo.instructions.len());
                AlgoAction::SwapInstructions(a, b)
            },
            3 if !algo.instructions.is_empty() =>
            {
                AlgoAction::RemoveInstruction(rng.gen_range(0..algo.instructions.len()))
            },
            _ =>
            {
                let instr = Instruction::Cmp(r0, r1);
                let pos = if algo.instructions.is_empty()
                {
                    0
                }
                else
                {
                    rng.gen_range(0..=algo.instructions.len())
                };
                AlgoAction::AddInstruction(pos, instr)
            },
        };
        actions.push(action);
    }
    actions
}

/// Apply an action to an algorithm in-place.
fn apply_action(algo: &mut Algorithm, action: &AlgoAction) {
    match action
    {
        AlgoAction::AddInstruction(idx, instr) =>
        {
            let pos = (*idx).min(algo.instructions.len());
            algo.instructions.insert(pos, *instr);
        },
        AlgoAction::RemoveInstruction(idx) =>
        {
            if !algo.instructions.is_empty()
            {
                let pos = *idx % algo.instructions.len();
                algo.instructions.remove(pos);
            }
        },
        AlgoAction::ModifyInstruction(idx, instr) =>
        {
            if !algo.instructions.is_empty()
            {
                let pos = *idx % algo.instructions.len();
                algo.instructions[pos] = *instr;
            }
        },
        AlgoAction::SwapInstructions(a, b) =>
        {
            let len = algo.instructions.len();
            if len >= 2
            {
                let i = *a % len;
                let j = *b % len;
                algo.instructions.swap(i, j);
            }
        },
        AlgoAction::Noop =>
        {},
    }
}

/// Evaluate the fitness of an algorithm for a problem.
/// Combines correctness, simplicity, and efficiency.
pub fn evaluate_fitness(problem: &ProblemSpec, algo: &Algorithm) -> f64 {
    let correctness = problem.evaluate_correctness(algo);
    if correctness < 0.01
    {
        // For completely wrong, still offer some grain based on having instructions
        return algo.instructions.len().min(1) as f64 * 0.1;
    }
    let simplicity = 1.0 / (1.0 + algo.simplicity_cost());
    let efficiency = 1.0 / algo.estimated_complexity().max(1.0);
    correctness * 10.0 + simplicity + efficiency
}

// ===========================================================================
// 8. ALGORITHM VERIFICATION
// ===========================================================================

/// Test suite generator: produces boundary cases and random inputs.
#[derive(Debug, Clone)]
pub struct TestSuiteGenerator {
    pub rng: RefCell<StdRng>,
}

impl TestSuiteGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }

    /// Generate a test suite for a problem specification.
    pub fn generate(
        &self,
        base_inputs: &[i64],
        num_registers: usize,
        num_random: usize,
    ) -> Vec<Vec<i64>> {
        let mut rng = self.rng.borrow_mut();
        let mut tests: Vec<Vec<i64>> = Vec::new();

        // Boundary cases
        tests.push(vec![0; num_registers]); // all zeros
        tests.push(vec![1; num_registers]); // all ones
        tests.push(vec![-1; num_registers]); // all negative one

        // Use base inputs
        if !base_inputs.is_empty()
        {
            let mut pad = base_inputs.to_vec();
            pad.resize(num_registers, 0);
            tests.push(pad);
        }

        // Random inputs
        for _ in 0..num_random
        {
            let test: Vec<i64> = (0..num_registers)
                .map(|_| rng.gen_range(-1000..1001_i64))
                .collect();
            tests.push(test);
        }

        // Edge values
        let mut edges = vec![0i64; num_registers];
        edges[0] = i64::MAX;
        tests.push(edges.clone());
        edges[0] = i64::MIN;
        tests.push(edges.clone());
        edges[0] = 0;
        edges[1] = i64::MAX - 1;
        tests.push(edges);

        tests
    }
}

/// Invariant inference: discovers loop invariants and pre/post conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invariant {
    pub description: String,
    pub register: usize,
    pub invariant_type: InvariantType,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvariantType {
    Constant,
    MonotonicIncreasing,
    MonotonicDecreasing,
    Bounded,
    Parity,
    Divides,
}

/// Invariant inference engine.
pub struct InvariantInferrer;

impl Default for InvariantInferrer {
    fn default() -> Self {
        Self::new()
    }
}

impl InvariantInferrer {
    pub fn new() -> Self {
        Self
    }

    /// Infer invariants by running the algorithm on multiple inputs.
    pub fn infer(
        &self,
        algo: &Algorithm,
        inputs: &[Vec<i64>],
        max_invariants: usize,
    ) -> Vec<Invariant> {
        let mut invariants = Vec::new();
        if inputs.is_empty() || algo.instructions.is_empty()
        {
            return invariants;
        }

        let n = algo.num_registers;

        for reg in 0..n
        {
            let mut all_outputs: Vec<i64> = Vec::new();
            let mut all_inputs: Vec<i64> = Vec::new();

            for input in inputs
            {
                if let Ok(output) = algo.execute(input) {
                    if reg < output.len()
                    {
                        all_outputs.push(output[reg]);
                    }
                    if reg < input.len()
                    {
                        all_inputs.push(input[reg]);
                    }
                }
            }

            if invariants.len() >= max_invariants
            {
                break;
            }

            // Check if output[reg] is constant
            if all_outputs.len() >= 2
            {
                let first = all_outputs[0];
                if all_outputs.iter().all(|&v| v == first)
                {
                    invariants.push(Invariant {
                        description: format!("Register {} is constant (= {})", reg, first),
                        register: reg,
                        invariant_type: InvariantType::Constant,
                        confidence: 1.0,
                    });
                }
            }

            if invariants.len() >= max_invariants
            {
                break;
            }

            // Check monotonicity
            if all_inputs.len() >= 2 && all_outputs.len() >= 2
            {
                let increasing = all_outputs.windows(2).all(|w| w[0] <= w[1]);
                let decreasing = all_outputs.windows(2).all(|w| w[0] >= w[1]);
                if increasing && !decreasing
                {
                    invariants.push(Invariant {
                        description: format!(
                            "Register {} is monotonically increasing w.r.t. input",
                            reg
                        ),
                        register: reg,
                        invariant_type: InvariantType::MonotonicIncreasing,
                        confidence: 0.7,
                    });
                }
                else if decreasing && !increasing
                {
                    invariants.push(Invariant {
                        description: format!(
                            "Register {} is monotonically decreasing w.r.t. input",
                            reg
                        ),
                        register: reg,
                        invariant_type: InvariantType::MonotonicDecreasing,
                        confidence: 0.7,
                    });
                }
            }

            if invariants.len() >= max_invariants
            {
                break;
            }

            // Parity check
            if all_outputs.len() >= 3
            {
                let all_same_parity = all_outputs.iter().all(|&v| v % 2 == all_outputs[0] % 2);
                if all_same_parity
                {
                    invariants.push(Invariant {
                        description: format!(
                            "Register {} has constant parity ({})",
                            reg,
                            if all_outputs[0] % 2 == 0
                            {
                                "even"
                            }
                            else
                            {
                                "odd"
                            }
                        ),
                        register: reg,
                        invariant_type: InvariantType::Parity,
                        confidence: 0.6,
                    });
                }
            }
        }

        invariants
    }
}

/// Counterexample-Guided Abstraction Refinement (CEGAR).
pub struct CegarVerifier {
    pub max_iterations: usize,
    pub rng: RefCell<StdRng>,
}

impl CegarVerifier {
    pub fn new(max_iterations: usize, seed: u64) -> Self {
        Self {
            max_iterations,
            rng: RefCell::new(StdRng::seed_from_u64(seed)),
        }
    }

    /// CEGAR loop: iteratively find counterexamples and refine the algorithm.
    pub fn refine(
        &self,
        problem: &ProblemSpec,
        initial_algo: &Algorithm,
    ) -> Result<Algorithm, Vec<String>> {
        let mut algo = initial_algo.clone();
        let mut rng = self.rng.borrow_mut();
        let mut failures: Vec<String> = Vec::new();

        for iteration in 0..self.max_iterations
        {
            // Find failing test case
            let mut counterexample: Option<Vec<i64>> = None;
            for (input, expected) in &problem.test_cases
            {
                match algo.execute(input)
                {
                    Ok(output) =>
                    {
                        let ok = expected.iter().zip(&output).all(|(e, o)| e == o)
                            && expected.len() == output.len();
                        if !ok
                        {
                            counterexample = Some(input.clone());
                            break;
                        }
                    },
                    Err(e) =>
                    {
                        failures.push(e);
                        counterexample = Some(input.clone());
                        break;
                    },
                }
            }

            match counterexample
            {
                Some(_input) =>
                {
                    // Try to refine: mutate the algorithm
                    let mut candidate = algo.clone();
                    mutate_algorithm(&mut candidate, &mut rng, 0.5);
                    // Keep if better or accept with probability
                    let old_score = evaluate_fitness(problem, &algo);
                    let new_score = evaluate_fitness(problem, &candidate);
                    if new_score > old_score
                    {
                        algo = candidate;
                    }
                },
                None =>
                {
                    // All tests pass
                    return Ok(algo);
                },
            }

            if iteration == self.max_iterations - 1
            {
                failures.push(format!(
                    "CEGAR did not converge after {} iterations",
                    self.max_iterations
                ));
            }
        }

        Err(failures)
    }
}

// ===========================================================================
// 9. TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    fn make_add_problem() -> ProblemSpec {
        ProblemSpec::new("add_two", 3, 20)
            .with_test(vec![1, 2, 0], vec![3, 0, 0])
            .with_test(vec![10, 5, 0], vec![15, 0, 0])
            .with_test(vec![0, 0, 0], vec![0, 0, 0])
    }

    fn make_add_algo() -> Algorithm {
        // Computes R0 + R1 -> R0, zeroes R1
        // Test cases: [1,2,0]->[3,0,0], [10,5,0]->[15,0,0], [0,0,0]->[0,0,0]
        Algorithm {
            instructions: vec![
                Instruction::Add(0, 0, 1),
                Instruction::Load(1, 0),
                Instruction::Return,
            ],
            num_registers: 3,
            stack_capacity: 16,
        }
    }

    // --- Algorithm Representation Tests ---

    #[test]
    fn test_algorithm_creation() {
        let algo = Algorithm::new(4, 8);
        assert_eq!(algo.num_registers, 4);
        assert_eq!(algo.stack_capacity, 8);
        assert!(algo.is_empty());
        assert_eq!(algo.len(), 0);
    }

    #[test]
    fn test_execute_simple_add() {
        let mut algo = Algorithm::new(3, 16);
        algo.instructions = vec![Instruction::Add(2, 0, 1), Instruction::Return];
        let result = algo.execute(&[5, 3, 0]).unwrap();
        assert_eq!(result[0], 5);
        assert_eq!(result[1], 3);
        assert_eq!(result[2], 8);
    }

    #[test]
    fn test_execute_load_and_sub() {
        let mut algo = Algorithm::new(3, 8);
        algo.instructions = vec![
            Instruction::Load(0, 10),
            Instruction::Load(1, 3),
            Instruction::Sub(2, 0, 1),
            Instruction::Return,
        ];
        let result = algo.execute(&[0, 0, 0]).unwrap();
        assert_eq!(result[2], 7);
    }

    #[test]
    fn test_execute_mul() {
        let mut algo = Algorithm::new(3, 8);
        algo.instructions = vec![Instruction::Mul(2, 0, 1), Instruction::Return];
        let result = algo.execute(&[4, 5, 0]).unwrap();
        assert_eq!(result[2], 20);
    }

    #[test]
    fn test_execute_div_by_zero_safe() {
        let mut algo = Algorithm::new(3, 8);
        algo.instructions = vec![Instruction::Div(2, 0, 1), Instruction::Return];
        let result = algo.execute(&[10, 0, 0]).unwrap();
        assert_eq!(result[2], 0); // safe division
    }

    #[test]
    fn test_execute_jump() {
        let mut algo = Algorithm::new(3, 8);
        algo.instructions = vec![
            Instruction::Load(0, 1),
            Instruction::Jump(2),     // skip next
            Instruction::Load(0, 99), // skipped
            Instruction::Return,
        ];
        let result = algo.execute(&[0, 0, 0]).unwrap();
        assert_eq!(result[0], 1);
    }

    #[test]
    fn test_execute_conditional_jump() {
        // Simple test: if R0 == 0, set R1=1; else set R1=2
        let mut algo = Algorithm::new(3, 8);
        algo.instructions = vec![
            Instruction::Load(2, 0),                     // R2 = 0
            Instruction::Cmp(0, 2),                      // compare R0 with 0
            Instruction::JumpIf(3, CmpCondition::Equal), // if equal, skip to Load(1,1) at +3
            Instruction::Load(1, 2),                     // R1 = 2 (not equal case)
            Instruction::Jump(2),                        // skip to return
            Instruction::Load(1, 1),                     // R1 = 1 (equal case)
            Instruction::Return,
        ];
        // R0=0 -> equal -> R1 should be 1
        let result = algo.execute(&[0, 0, 0]).unwrap();
        assert_eq!(result[1], 1);

        // R0=5 -> not equal -> R1 should be 2
        let result2 = algo.execute(&[5, 0, 0]).unwrap();
        assert_eq!(result2[1], 2);
    }

    #[test]
    fn test_execute_swap() {
        let mut algo = Algorithm::new(3, 8);
        algo.instructions = vec![Instruction::Swap(0, 1), Instruction::Return];
        let result = algo.execute(&[1, 2, 0]).unwrap();
        assert_eq!(result[0], 2);
        assert_eq!(result[1], 1);
    }

    #[test]
    fn test_execute_push_pop() {
        let mut algo = Algorithm::new(3, 16);
        algo.instructions = vec![
            Instruction::Push(0),
            Instruction::Load(0, 99),
            Instruction::Pop(0),
            Instruction::Return,
        ];
        let result = algo.execute(&[42, 0, 0]).unwrap();
        assert_eq!(result[0], 42); // restored from stack
    }

    #[test]
    fn test_fitness_correct_algo() {
        let problem = make_add_problem();
        let algo = make_add_algo();
        let fitness = evaluate_fitness(&problem, &algo);
        assert!(
            fitness > 5.0,
            "Perfect algo should have high fitness: {}",
            fitness
        );
    }

    #[test]
    fn test_problem_all_passed() {
        let problem = make_add_problem();
        let algo = make_add_algo();
        assert!(problem.all_passed(&algo));
    }

    #[test]
    fn test_simplicity_cost() {
        let short = Algorithm {
            instructions: vec![Instruction::Return],
            num_registers: 3,
            stack_capacity: 8,
        };
        let long_algo = Algorithm {
            instructions: vec![Instruction::Return; 10],
            num_registers: 3,
            stack_capacity: 8,
        };
        assert!(short.simplicity_cost() < long_algo.simplicity_cost());
    }

    // --- Mutation and Crossover Tests ---

    #[test]
    fn test_mutate_algorithm() {
        let mut algo = make_add_algo();
        let _original_len = algo.len();
        let mut rng = seeded_rng(42);
        mutate_algorithm(&mut algo, &mut rng, 0.8);
        // After mutation with high rate, something should change
        // (either length or content)
        assert!(!algo.instructions.is_empty());
    }

    #[test]
    fn test_crossover_algorithms() {
        let a = Algorithm {
            instructions: vec![
                Instruction::Return,
                Instruction::Load(0, 1),
                Instruction::Add(2, 0, 1),
            ],
            num_registers: 3,
            stack_capacity: 8,
        };
        let b = Algorithm {
            instructions: vec![
                Instruction::Load(0, 10),
                Instruction::Mul(2, 0, 1),
                Instruction::Return,
            ],
            num_registers: 3,
            stack_capacity: 8,
        };
        let mut rng = seeded_rng(99);
        let child = crossover_algorithms(&a, &b, &mut rng);
        assert!(!child.instructions.is_empty());
        assert_eq!(child.num_registers, 3);
    }

    // --- AlgoEnv Tests ---

    #[test]
    fn test_algo_env_reset() {
        let problem = make_add_problem();
        let mut env = AlgoSearchEnv::new(problem, 42);
        let state = env.reset();
        assert!(state.algorithm.is_empty());
        assert_eq!(state.total_tests, 3);
    }

    #[test]
    fn test_algo_env_step() {
        let problem = make_add_problem();
        let mut env = AlgoSearchEnv::new(problem, 42);
        env.reset();
        let (_state, reward, _done) = env.step(&AlgoAction::Noop);
        // Noop should reward the empty algorithm
        assert!(reward.is_finite());
    }

    #[test]
    fn test_algo_env_terminal_on_perfect() {
        let problem = make_add_problem();
        let mut env = AlgoSearchEnv::new(problem, 42);
        env.reset();
        let algo = make_add_algo();
        let state = AlgoSearchState {
            algorithm: algo,
            tests_passed: 3,
            total_tests: 3,
        };
        assert!(env.is_terminal(&state));
    }

    // --- RL Agent Tests ---

    #[test]
    fn test_tabular_q_learning() {
        let problem = make_add_problem();
        let mut env = AlgoSearchEnv::new(problem, 999);
        let mut agent = TabularQLearner::new(0.1, 0.9, 0.3, 0.995, 123);
        let rewards = agent.train_loop(&mut env, 50, 30);
        assert!(!rewards.is_empty());
        // Should see some non-negative rewards
        let has_positive = rewards.iter().any(|&r| r > 0.0);
        assert!(has_positive || rewards.iter().all(|&r| r >= 0.0));
    }

    #[test]
    fn test_experience_replay() {
        let mut buffer = ExperienceReplay::new(10);
        let s = DiscretizedState {
            len_bucket: 0,
            dominant_op: None,
            has_loop: false,
            correctness_bucket: 0,
        };
        for i in 0..15
        {
            buffer.push(Experience {
                state: s.clone(),
                action_idx: i % 3,
                reward: i as f64,
                next_state: s.clone(),
                done: i >= 14,
            });
        }
        assert_eq!(buffer.len(), 10); // capacity capped

        let mut rng = seeded_rng(42);
        let batch = buffer.sample_batch(5, &mut rng);
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn test_reinforce_agent() {
        let problem = make_add_problem();
        let mut env = AlgoSearchEnv::new(problem, 777);
        let mut agent = ReinforceAgent::new(32, 16, 5, 0.01, 0.01, 0.95, 456);
        let rewards = agent.train_loop(&mut env, 20, 20, 5);
        assert!(!rewards.is_empty());
        assert!(rewards.iter().all(|&r| r.is_finite()));
    }

    #[test]
    fn test_actor_critic_single_update() {
        let mut agent = ActorCriticAgent::new(32, 8, 5, 0.01, 0.01, 0.95, 789);
        let feats = vec![0.1; 32];
        let next_feats = vec![0.2; 32];
        // Should not panic
        agent.update(&feats, 2, 1.0, &next_feats, false, 5);
    }

    // --- Heuristic Search Tests ---

    #[test]
    fn test_simulated_annealing() {
        let problem = make_add_problem();
        let initial = Algorithm {
            instructions: vec![Instruction::Add(0, 0, 1), Instruction::Return],
            num_registers: 3,
            stack_capacity: 16,
        };
        let sa = SimulatedAnnealing::new(10.0, 0.95, 101);
        let (best, score) = sa.search(&problem, &initial, 300);
        assert!(!best.instructions.is_empty());
        assert!(score.is_finite());
    }

    #[test]
    fn test_beam_search() {
        let problem = make_add_problem();
        let bs = BeamSearch::new(5, 10, 202);
        let result = bs.search(&problem);
        assert!(result.is_some());
        let (algo, score) = result.unwrap();
        assert!(!algo.instructions.is_empty());
        assert!(score > 0.0);
    }

    #[test]
    fn test_mcts_search() {
        let problem = make_add_problem();
        let root = Algorithm {
            instructions: vec![Instruction::Return],
            num_registers: 3,
            stack_capacity: 16,
        };
        let mcts = MctsEngine::new(1.4, 100, 303);
        let (algo, score) = mcts.search(&problem, &root);
        assert!(!algo.instructions.is_empty());
        assert!(score.is_finite());
    }

    // --- Verification Tests ---

    #[test]
    fn test_test_suite_generator() {
        let gen = TestSuiteGenerator::new(404);
        let tests = gen.generate(&[1, 2, 3], 4, 5);
        assert!(tests.len() >= 5); // 3 boundary + base + random
        for t in &tests
        {
            assert_eq!(t.len(), 4);
        }
    }

    #[test]
    fn test_invariant_inference_constant() {
        let algo = make_add_algo();
        let inferrer = InvariantInferrer::new();
        let inputs: Vec<Vec<i64>> = vec![vec![1, 2, 0], vec![5, 3, 0], vec![10, 20, 0]];
        let invariants = inferrer.infer(&algo, &inputs, 10);
        assert!(!invariants.is_empty());
    }

    #[test]
    fn test_cegar_refinement() {
        let problem = ProblemSpec::new("cegar_test", 3, 10).with_test(vec![2, 3, 0], vec![5, 0, 0]);
        let initial = Algorithm {
            instructions: vec![
                Instruction::Add(0, 0, 1),
                Instruction::Load(1, 0),
                Instruction::Return,
            ],
            num_registers: 3,
            stack_capacity: 16,
        };
        let verifier = CegarVerifier::new(200, 505);
        let result = verifier.refine(&problem, &initial);
        assert!(result.is_ok());
    }

    // --- Meta-Learning Tests ---

    #[test]
    fn test_meta_learner_store_and_retrieve() {
        let mut ml = MetaLearner::new();
        let problem = make_add_problem();
        let algo = make_add_algo();
        ml.store_template(&problem, &algo, 10.0);

        let template = ml.get_best_template("add_two");
        assert!(template.is_some());
        assert_eq!(template.unwrap().instructions.len(), 3);
    }

    #[test]
    fn test_transfer_learning() {
        let mut ml = MetaLearner::new();

        // Store a template for a similar problem
        let prob_a = ProblemSpec::new("add_small", 3, 20).with_test(vec![1, 2, 0], vec![3, 0, 0]);
        let algo = make_add_algo();
        ml.store_template(&prob_a, &algo, 10.0);

        // Transfer to a new but similar problem
        let prob_b =
            ProblemSpec::new("add_large", 3, 20).with_test(vec![100, 200, 0], vec![300, 0, 0]);

        let transferred = ml.transfer_learn(&prob_b);
        assert!(!transferred.is_empty());
    }

    #[test]
    fn test_strategy_selection() {
        let ml = MetaLearner::new();
        let tiny = ProblemSpec::new("tiny", 1, 0).with_test(vec![1], vec![2]);
        let strategy = ml.select_strategy(&tiny);
        assert_eq!(strategy, SearchStrategy::TabularQLearning);

        let medium = ProblemSpec::new("medium", 2, 2).with_test(vec![1, 2], vec![3, 0]);
        let strategy2 = ml.select_strategy(&medium);
        assert!(matches!(strategy2, SearchStrategy::Reinforce));

        let complex = ProblemSpec::new("complex", 8, 50)
            .with_test(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let strategy3 = ml.select_strategy(&complex);
        assert!(matches!(strategy3, SearchStrategy::MCTS));
    }

    // --- Feature Extraction Tests ---

    #[test]
    fn test_algo_to_features() {
        let algo = make_add_algo();
        let features = algo_to_features(&algo, 10);
        assert_eq!(features.len(), 10 * 3 + 2); // 10 instruction slots * 3 + 2 extra
    }

    #[test]
    fn test_problem_feature_extraction() {
        let problem = make_add_problem();
        let extractor = ProblemFeatureExtractor::new();
        let feats = extractor.extract(&problem);
        assert_eq!(feats.len(), 4);
        assert_eq!(feats[0], 3.0); // num_registers
    }

    // --- Serialization Tests ---

    #[test]
    fn test_instruction_serialization() {
        let instr = Instruction::Add(0, 1, 2);
        let json = serde_json::to_string(&instr).unwrap();
        let deser: Instruction = serde_json::from_str(&json).unwrap();
        assert_eq!(instr, deser);
    }

    #[test]
    fn test_algorithm_serialization() {
        let algo = make_add_algo();
        let json = serde_json::to_string(&algo).unwrap();
        let deser: Algorithm = serde_json::from_str(&json).unwrap();
        assert_eq!(algo.instructions, deser.instructions);
        assert_eq!(algo.num_registers, deser.num_registers);
    }

    #[test]
    fn test_discretized_state_serialization() {
        let s = DiscretizedState {
            len_bucket: 2,
            dominant_op: Some(2),
            has_loop: true,
            correctness_bucket: 3,
        };
        let json = serde_json::to_string(&s).unwrap();
        let deser: DiscretizedState = serde_json::from_str(&json).unwrap();
        assert_eq!(s, deser);
    }

    #[test]
    fn test_meta_learner_serialization() {
        let mut ml = MetaLearner::new();
        let problem = ProblemSpec::new("ser_test", 3, 10).with_test(vec![1, 2, 3], vec![6, 0, 0]);
        let algo = Algorithm {
            instructions: vec![Instruction::Add(0, 1, 2), Instruction::Return],
            num_registers: 3,
            stack_capacity: 16,
        };
        ml.store_template(&problem, &algo, 5.0);

        let json = serde_json::to_string(&ml).unwrap();
        let deser: MetaLearner = serde_json::from_str(&json).unwrap();
        assert_eq!(ml.templates.len(), deser.templates.len());
    }

    // --- Edge Cases ---

    #[test]
    fn test_execute_empty_algorithm() {
        let algo = Algorithm::new(3, 8);
        let result = algo.execute(&[1, 2, 3]).unwrap();
        assert_eq!(result, vec![1, 2, 3]); // unchanged
    }

    #[test]
    fn test_execute_infinite_loop_protection() {
        // Jump(0) at pc=0 jumps to self, creating an infinite loop.
        let algo = Algorithm {
            instructions: vec![Instruction::Jump(0)],
            num_registers: 3,
            stack_capacity: 8,
        };
        let result = algo.execute(&[0, 0, 0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("max steps"));
    }

    #[test]
    fn test_evaluate_fitness_completely_wrong() {
        let problem =
            ProblemSpec::new("wrong_test", 3, 10).with_test(vec![1, 2, 0], vec![100, 0, 0]);
        let algo = Algorithm {
            instructions: vec![Instruction::Load(0, 1), Instruction::Return],
            num_registers: 3,
            stack_capacity: 8,
        };
        let fitness = evaluate_fitness(&problem, &algo);
        assert!(
            fitness < 1.0,
            "Wrong algo should have low fitness: {}",
            fitness
        );
    }

    #[test]
    fn test_algogen_action_empty_algo() {
        let mut algo = Algorithm::new(3, 8);
        apply_action(
            &mut algo,
            &AlgoAction::AddInstruction(0, Instruction::Return),
        );
        assert_eq!(algo.len(), 1);
        apply_action(&mut algo, &AlgoAction::RemoveInstruction(0));
        assert_eq!(algo.len(), 0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-10);

        let d = vec![1.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &d);
        assert!(sim > 0.5 && sim < 1.0); // should be around 0.707
    }

    #[test]
    fn test_most_common_instruction() {
        let algo = Algorithm {
            instructions: vec![
                Instruction::Add(0, 1, 2),
                Instruction::Add(1, 2, 3),
                Instruction::Sub(2, 3, 0),
                Instruction::Return,
            ],
            num_registers: 4,
            stack_capacity: 8,
        };
        let dom = most_common_instruction(&algo);
        assert_eq!(dom, Some(2)); // Add is index 2
    }
}
