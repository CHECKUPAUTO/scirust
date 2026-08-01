//! Closed, local-only experiment scope.

use core::fmt;

/// A built-in, locally generated corpus. No variant represents external data.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CorpusKind
{
    /// Every nonsingular curve for primes 5, 7, 11 and 13.
    ExhaustiveSmall,
    /// A seeded holdout sampled from primes between 17 and 97.
    IndependentHoldout,
    /// A seeded sample on the explicit scale ladder.
    ScaleLadder,
}

impl CorpusKind
{
    pub(crate) const fn tag(self) -> u8
    {
        match self
        {
            Self::ExhaustiveSmall => 0,
            Self::IndependentHoldout => 1,
            Self::ScaleLadder => 2,
        }
    }

    /// Stable descriptive name used in reports.
    pub const fn name(self) -> &'static str
    {
        match self
        {
            Self::ExhaustiveSmall => "ExhaustiveSmall",
            Self::IndependentHoldout => "IndependentHoldout",
            Self::ScaleLadder => "ScaleLadder",
        }
    }
}

/// Authorization and deterministic limits for one local research run.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LocalResearchCase
{
    seed: u64,
    corpus: CorpusKind,
    curves_per_prime: u32,
    tuple_budget: u64,
}

impl LocalResearchCase
{
    /// Creates a local-only case. Limits must be nonzero.
    pub fn new(
        seed: u64,
        corpus: CorpusKind,
        curves_per_prime: u32,
        tuple_budget: u64,
    ) -> Result<Self, ScopeError>
    {
        if curves_per_prime == 0
        {
            return Err(ScopeError::ZeroCurveBudget);
        }
        if tuple_budget == 0
        {
            return Err(ScopeError::ZeroTupleBudget);
        }
        Ok(Self {
            seed,
            corpus,
            curves_per_prime,
            tuple_budget,
        })
    }

    /// Explicit deterministic seed.
    pub const fn seed(self) -> u64
    {
        self.seed
    }

    /// Built-in corpus selector.
    pub const fn corpus(self) -> CorpusKind
    {
        self.corpus
    }

    /// Maximum selected curves per prime for sampled corpora.
    pub const fn curves_per_prime(self) -> u32
    {
        self.curves_per_prime
    }

    /// Maximum point tuples evaluated by a later search stage.
    pub const fn tuple_budget(self) -> u64
    {
        self.tuple_budget
    }
}

/// Invalid local experiment limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScopeError
{
    ZeroCurveBudget,
    ZeroTupleBudget,
}

impl fmt::Display for ScopeError
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        match self
        {
            Self::ZeroCurveBudget => write!(formatter, "curve budget must be nonzero"),
            Self::ZeroTupleBudget => write!(formatter, "tuple budget must be nonzero"),
        }
    }
}

impl std::error::Error for ScopeError {}
