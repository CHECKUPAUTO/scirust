//! scirust-algogen — Algorithm Generation Framework
//!
//! Automatic algorithm selection, composition, and complexity analysis
//! for sorting, searching, graph, dynamic-programming, and
//! divide-and-conquer algorithms.
//!
//! # Modules (single-file crate)
//! 1. Sorting – search space, auto-select, hybrid generation, benchmarks
//! 2. Searching – linear / binary / jump / exponential / interpolation /
//!    BST / AVL / hash, data-distribution adaptation
//! 3. Graph – path finding, spanning trees, flow, coloring, property-based
//!    strategy selection
//! 4. Dynamic Programming – suitability detection, memoization / tabulation
//! 5. Divide-and-Conquer – templates, base-case selection, master theorem
//! 6. Complexity – Big-O fitting, empirical measurement, adaptive detection
//! 7. Comparison – head-to-head benchmarks, statistical reporting

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::Instant;

// -----------------------------------------------------------------
// Re-exports (so downstream consumers get a single import surface)
// -----------------------------------------------------------------
pub use scirust_core::error::{Result, SciRustError};
pub use scirust_evo::{GeneticAlgorithm, Individual};

// ================================================================
// 0. Utilities & data-characterisation helpers
// ================================================================

/// Deterministic fast pseudo-random generator (LCG).  Keeps the crate
/// free of an explicit `rand` dependency for library code while still
/// allowing reproducible data generation in benchmarks/tests.
fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    *state
}

/// Characteristic summary of a dataset used for auto-selection decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    pub size: usize,
    /// Fraction of adjacent pairs that are already in order (0=random, 1=fully sorted).
    pub sortedness: f64,
    /// Approximate value span  (min, max).
    pub range: (i64, i64),
    /// Fraction of distinct values (0=all identical, 1=all distinct).
    pub uniqueness: f64,
}

/// Fraction of adjacent ordered pairs (ascending).
pub fn sortedness(data: &[i64]) -> f64 {
    let pairs = data.len().saturating_sub(1);
    if pairs == 0
    {
        return 1.0;
    }
    let ordered = data.windows(2).filter(|w| w[0] <= w[1]).count();
    ordered as f64 / pairs as f64
}

/// Fraction of distinct values.
pub fn uniqueness(data: &[i64]) -> f64 {
    if data.is_empty()
    {
        return 1.0;
    }
    let mut v = data.to_vec();
    v.sort_unstable();
    v.dedup();
    v.len() as f64 / data.len() as f64
}

/// Full characterisation of a slice.
pub fn characterize(data: &[i64]) -> DataCharacteristics {
    if data.is_empty()
    {
        return DataCharacteristics {
            size: 0,
            sortedness: 1.0,
            range: (0, 0),
            uniqueness: 1.0,
        };
    }
    let min = *data.iter().min().unwrap();
    let max = *data.iter().max().unwrap();
    DataCharacteristics {
        size: data.len(),
        sortedness: sortedness(data),
        range: (min, max),
        uniqueness: uniqueness(data),
    }
}

// ================================================================
// 1.  S O R T I N G
// ================================================================

/// Enumerates the search space of sorting strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SortStrategy {
    Bubble,
    Insertion,
    Selection,
    Merge,
    Quick,
    Heap,
    Counting,
    Radix,
    TimsortLike,
    /// Introselect = quicksort + heapsort fallback + insertion for tiny sub-arrays.
    Introselect,
}

impl fmt::Display for SortStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self
        {
            SortStrategy::Bubble => "Bubble",
            SortStrategy::Insertion => "Insertion",
            SortStrategy::Selection => "Selection",
            SortStrategy::Merge => "Merge",
            SortStrategy::Quick => "Quick",
            SortStrategy::Heap => "Heap",
            SortStrategy::Counting => "Counting",
            SortStrategy::Radix => "Radix",
            SortStrategy::TimsortLike => "TimsortLike",
            SortStrategy::Introselect => "Introselect",
        };
        write!(f, "{s}")
    }
}

impl SortStrategy {
    pub fn all() -> Vec<SortStrategy> {
        vec![
            SortStrategy::Bubble,
            SortStrategy::Insertion,
            SortStrategy::Selection,
            SortStrategy::Merge,
            SortStrategy::Quick,
            SortStrategy::Heap,
            SortStrategy::Counting,
            SortStrategy::Radix,
            SortStrategy::TimsortLike,
            SortStrategy::Introselect,
        ]
    }
}

// ---- elementary sorts ----

fn insertion_range(data: &mut [i64], lo: usize, hi: usize) {
    for i in lo..=hi
    {
        let key = data[i];
        let mut j = i;
        while j > lo && data[j - 1] > key
        {
            data[j] = data[j - 1];
            j -= 1;
        }
        data[j] = key;
    }
}

fn bubble_sort(data: &mut [i64]) {
    let n = data.len();
    for i in 0..n
    {
        for j in 0..n - i - 1
        {
            if data[j] > data[j + 1]
            {
                data.swap(j, j + 1);
            }
        }
    }
}

fn insertion_sort(data: &mut [i64]) {
    let n = data.len();
    if n > 0
    {
        insertion_range(data, 0, n - 1);
    }
}

fn selection_sort(data: &mut [i64]) {
    let n = data.len();
    for i in 0..n
    {
        let mut min_idx = i;
        for j in i + 1..n
        {
            if data[j] < data[min_idx]
            {
                min_idx = j;
            }
        }
        data.swap(i, min_idx);
    }
}

// ---- O(n log n) sorts ----

fn merge_sort(data: &mut [i64]) {
    let n = data.len();
    if n <= 1
    {
        return;
    }
    let mid = n / 2;
    merge_sort(&mut data[..mid]);
    merge_sort(&mut data[mid..]);
    let mut merged = data.to_vec();
    let (mut i, mut j, mut k) = (0, mid, 0);
    while i < mid && j < n
    {
        if data[i] <= data[j]
        {
            merged[k] = data[i];
            i += 1;
        }
        else
        {
            merged[k] = data[j];
            j += 1;
        }
        k += 1;
    }
    while i < mid
    {
        merged[k] = data[i];
        i += 1;
        k += 1;
    }
    while j < n
    {
        merged[k] = data[j];
        j += 1;
        k += 1;
    }
    data.copy_from_slice(&merged);
}

fn quicksort(data: &mut [i64]) {
    if data.len() <= 1
    {
        return;
    }
    let pivot = data.len() - 1;
    let mut i = 0;
    for j in 0..pivot
    {
        if data[j] <= data[pivot]
        {
            data.swap(i, j);
            i += 1;
        }
    }
    data.swap(i, pivot);
    quicksort(&mut data[..i]);
    quicksort(&mut data[i + 1..]);
}

fn sift_down(data: &mut [i64], mut i: usize, end: usize) {
    loop
    {
        let child = 2 * i + 1;
        if child >= end
        {
            break;
        }
        let max_child = if child + 1 < end && data[child] < data[child + 1]
        {
            child + 1
        }
        else
        {
            child
        };
        if data[i] >= data[max_child]
        {
            break;
        }
        data.swap(i, max_child);
        i = max_child;
    }
}

fn heapsort(data: &mut [i64]) {
    let n = data.len();
    for i in (0..n / 2).rev()
    {
        sift_down(data, i, n);
    }
    for end in (1..n).rev()
    {
        data.swap(0, end);
        sift_down(data, 0, end);
    }
}

// ---- integer-key sorts ----

fn countingsort(data: &mut [i64]) {
    if data.is_empty()
    {
        return;
    }
    let min = *data.iter().min().unwrap();
    let max = *data.iter().max().unwrap();
    // Compute the value span without overflowing (max - min can exceed i64::MAX)
    // and fall back to a comparison sort when it would require an unreasonable
    // allocation. Counting sort is only sensible for a narrow key range.
    const MAX_RANGE: u128 = 1 << 24;
    let span = (max as i128 - min as i128) as u128;
    if span >= MAX_RANGE
    {
        heapsort(data);
        return;
    }
    let range = span as usize + 1;
    let mut counts = vec![0usize; range];
    for &v in data.iter()
    {
        counts[(v - min) as usize] += 1;
    }
    let mut idx = 0;
    for (vi, &c) in counts.iter().enumerate()
    {
        for _ in 0..c
        {
            data[idx] = vi as i64 + min;
            idx += 1;
        }
    }
}

fn radixsort(data: &mut [i64]) {
    if data.is_empty()
    {
        return;
    }
    // Map i64 -> u64 by flipping the sign bit. This is order-preserving
    // (i64::MIN -> 0, i64::MAX -> u64::MAX) and, unlike subtracting `min`,
    // never overflows for any input range.
    let mut keys: Vec<u64> = data.iter().map(|&v| (v as u64) ^ (1u64 << 63)).collect();
    let max = keys.iter().copied().max().unwrap();
    let mut buf = vec![0u64; keys.len()];
    let mut exp: u64 = 1;
    loop
    {
        let mut bucket = [0usize; 256];
        for &v in keys.iter()
        {
            let digit = (v / exp) % 256;
            bucket[digit as usize] += 1;
        }
        let mut pos = 0;
        for b in bucket.iter_mut()
        {
            let cnt = *b;
            *b = pos;
            pos += cnt;
        }
        for &v in keys.iter()
        {
            let digit = (v / exp) % 256;
            buf[bucket[digit as usize]] = v;
            bucket[digit as usize] += 1;
        }
        keys.copy_from_slice(&buf);
        // Stop once `exp` has covered the most significant digit of `max`.
        // Guard against overflow of `exp *= 256` at the top of the u64 range.
        if max / exp < 256
        {
            break;
        }
        exp *= 256;
    }
    for (d, &k) in data.iter_mut().zip(keys.iter())
    {
        *d = (k ^ (1u64 << 63)) as i64;
    }
}

// ---- hybrid/composite sorts ----

fn timsort_like(data: &mut [i64]) {
    const MIN_RUN: usize = 32;
    let n = data.len();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n
    {
        let mut j = i + 1;
        while j < n && data[j - 1] <= data[j]
        {
            j += 1;
        }
        let run_end = n.min(j.max(i + (MIN_RUN.min(n - i))));
        insertion_range(data, i, run_end.saturating_sub(1));
        runs.push((i, run_end));
        i = run_end;
    }
    while runs.len() > 1
    {
        let mut merged = Vec::new();
        let mut k = 0;
        while k + 1 < runs.len()
        {
            let (l1, r1) = runs[k];
            let (l2, r2) = runs[k + 1];
            let mut tmp = vec![0i64; r2 - l1];
            let (mut a, mut b) = (l1, l2);
            let mut out = 0;
            while a < r1 && b < r2
            {
                if data[a] <= data[b]
                {
                    tmp[out] = data[a];
                    a += 1;
                }
                else
                {
                    tmp[out] = data[b];
                    b += 1;
                }
                out += 1;
            }
            while a < r1
            {
                tmp[out] = data[a];
                a += 1;
                out += 1;
            }
            while b < r2
            {
                tmp[out] = data[b];
                b += 1;
                out += 1;
            }
            data[l1..r2].copy_from_slice(&tmp);
            merged.push((l1, r2));
            k += 2;
        }
        if k < runs.len()
        {
            merged.push(runs[k]);
        }
        runs = merged;
    }
}

fn introselect(data: &mut [i64]) {
    const INSERT_THRESH: usize = 16;
    let max_depth = (2.0 * (data.len() as f64).log2()).ceil() as usize;
    introselect_impl(data, max_depth, INSERT_THRESH);
}

fn introselect_impl(data: &mut [i64], depth: usize, thresh: usize) {
    let n = data.len();
    if n <= thresh
    {
        if n > 1
        {
            insertion_sort(data);
        }
        return;
    }
    if depth == 0
    {
        heapsort(data);
        return;
    }
    let pivot = n - 1;
    let mut i = 0;
    for j in 0..pivot
    {
        if data[j] <= data[pivot]
        {
            data.swap(i, j);
            i += 1;
        }
    }
    data.swap(i, pivot);
    introselect_impl(&mut data[..i], depth.saturating_sub(1), thresh);
    introselect_impl(&mut data[i + 1..], depth.saturating_sub(1), thresh);
}

/// Sort `data` in-place with the chosen strategy.
pub fn sort(data: &mut [i64], strategy: SortStrategy) {
    match strategy
    {
        SortStrategy::Bubble => bubble_sort(data),
        SortStrategy::Insertion => insertion_sort(data),
        SortStrategy::Selection => selection_sort(data),
        SortStrategy::Merge => merge_sort(data),
        SortStrategy::Quick => quicksort(data),
        SortStrategy::Heap => heapsort(data),
        SortStrategy::Counting => countingsort(data),
        SortStrategy::Radix => radixsort(data),
        SortStrategy::TimsortLike => timsort_like(data),
        SortStrategy::Introselect => introselect(data),
    }
}

/// Time a sort strategy on a *copy* of `data`.  Returns nanoseconds.
pub fn benchmark_sort(data: &[i64], strategy: SortStrategy) -> u128 {
    let mut copy = data.to_vec();
    let start = Instant::now();
    sort(&mut copy, strategy);
    start.elapsed().as_nanos()
}

/// Heuristic fitness for `strategy` given `c` (higher = better).
fn fitness_sort(s: &SortStrategy, c: &DataCharacteristics) -> f64 {
    let n = c.size as f64;
    match s
    {
        SortStrategy::Bubble | SortStrategy::Selection => -n * n,
        SortStrategy::Insertion =>
        {
            if c.sortedness > 0.9
            {
                -n
            }
            else
            {
                -n * n
            }
        },
        SortStrategy::Merge | SortStrategy::Heap => -n * n.log2().max(1.0),
        SortStrategy::Quick =>
        {
            let penalty = if c.sortedness > 0.9 { n * n } else { 0.0 };
            -n * n.log2().max(1.0) - penalty
        },
        SortStrategy::Counting | SortStrategy::Radix =>
        {
            let range = (c.range.1 - c.range.0) as f64;
            if range / n.max(1.0) < 100.0
            {
                -n - range * 0.1
            }
            else
            {
                -n * n.log2().max(1.0) - 1000.0
            }
        },
        SortStrategy::TimsortLike =>
        {
            if c.sortedness > 0.3
            {
                -n
            }
            else
            {
                -n * n.log2().max(1.0)
            }
        },
        SortStrategy::Introselect => -n * n.log2().max(1.0),
    }
}

/// Score every sort strategy for the given characteristics (descending fitness).
pub fn select_sort(c: &DataCharacteristics) -> Vec<(SortStrategy, f64)> {
    let mut scores: Vec<(SortStrategy, f64)> = SortStrategy::all()
        .into_iter()
        .map(|s| (s, fitness_sort(&s, c)))
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scores
}

/// Best single sort strategy for the given characteristics.
pub fn best_sort(c: &DataCharacteristics) -> SortStrategy {
    select_sort(c)
        .first()
        .map(|(s, _)| *s)
        .unwrap_or(SortStrategy::Quick)
}

// ================================================================
// 2.  S E A R C H I N G
// ================================================================

/// Search-strategy names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SearchStrategy {
    Linear,
    Binary,
    Jump,
    Exponential,
    Interpolation,
    BST,
    AVL,
    HashMap,
}

impl fmt::Display for SearchStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self
        {
            SearchStrategy::Linear => "Linear",
            SearchStrategy::Binary => "Binary",
            SearchStrategy::Jump => "Jump",
            SearchStrategy::Exponential => "Exponential",
            SearchStrategy::Interpolation => "Interpolation",
            SearchStrategy::BST => "BST",
            SearchStrategy::AVL => "AVL",
            SearchStrategy::HashMap => "HashMap",
        };
        write!(f, "{s}")
    }
}

/// Simplified data-distribution descriptor for search-adaptation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataDistribution {
    Uniform,
    Skewed { factor: f64 },
    Clustered { clusters: usize },
    Sorted,
    ReverseSorted,
}

// -- Basic search algorithms --

pub fn linear_search(data: &[i64], target: i64) -> Option<usize> {
    data.iter().position(|&x| x == target)
}

pub fn binary_search(data: &[i64], target: i64) -> Option<usize> {
    let mut lo = 0isize;
    let mut hi = data.len() as isize - 1;
    while lo <= hi
    {
        let mid = ((lo + hi) / 2) as usize;
        match data[mid].cmp(&target)
        {
            Ordering::Less => lo = mid as isize + 1,
            Ordering::Greater => hi = mid as isize - 1,
            Ordering::Equal => return Some(mid),
        }
    }
    None
}

pub fn jump_search(data: &[i64], target: i64) -> Option<usize> {
    let n = data.len();
    if n == 0
    {
        return None;
    }
    let step = (n as f64).sqrt() as usize;
    let mut prev = 0usize;
    while prev < n && data[prev.min(n - 1)] < target
    {
        prev += step;
    }
    let lo = prev.saturating_sub(step);
    let hi = n.min(prev + step);
    (lo..hi).find(|&i| data[i] == target)
}

pub fn exponential_search(data: &[i64], target: i64) -> Option<usize> {
    let n = data.len();
    if n == 0
    {
        return None;
    }
    if data[0] == target
    {
        return Some(0);
    }
    let mut i = 1;
    while i < n && data[i] <= target
    {
        i *= 2;
    }
    let lo = i / 2;
    let hi = n.min(i + 1);
    (lo..hi).find(|&j| data[j] == target)
}

pub fn interpolation_search(data: &[i64], target: i64) -> Option<usize> {
    let n = data.len();
    if n == 0
    {
        return None;
    }
    let mut lo = 0isize;
    let mut hi = n as isize - 1;
    while lo <= hi && target >= data[lo as usize] && target <= data[hi as usize]
    {
        if lo == hi
        {
            return if data[lo as usize] == target
            {
                Some(lo as usize)
            }
            else
            {
                None
            };
        }
        let denom = data[hi as usize] - data[lo as usize];
        let pos = if denom == 0
        {
            lo
        }
        else
        {
            lo + ((target - data[lo as usize]) as f64 / denom as f64 * (hi - lo) as f64) as isize
        };
        let pos = pos.clamp(lo, hi) as usize;
        match data[pos].cmp(&target)
        {
            Ordering::Equal => return Some(pos),
            Ordering::Less => lo = pos as isize + 1,
            Ordering::Greater => hi = pos as isize - 1,
        }
    }
    None
}

// -- BST --

struct BstNode {
    key: i64,
    index: usize,
    left: Option<Box<BstNode>>,
    right: Option<Box<BstNode>>,
}

impl BstNode {
    fn insert(&mut self, key: i64, index: usize) {
        if key < self.key
        {
            if let Some(ref mut left) = self.left
            {
                left.insert(key, index);
            }
            else
            {
                self.left = Some(Box::new(BstNode {
                    key,
                    index,
                    left: None,
                    right: None,
                }));
            }
        }
        else if let Some(ref mut right) = self.right
        {
            right.insert(key, index);
        }
        else
        {
            self.right = Some(Box::new(BstNode {
                key,
                index,
                left: None,
                right: None,
            }));
        }
    }

    fn search(&self, key: i64) -> Option<usize> {
        match key.cmp(&self.key)
        {
            Ordering::Less => self.left.as_ref().and_then(|n| n.search(key)),
            Ordering::Greater => self.right.as_ref().and_then(|n| n.search(key)),
            Ordering::Equal => Some(self.index),
        }
    }
}

pub fn bst_search(data: &[i64], target: i64) -> Option<usize> {
    if data.is_empty()
    {
        return None;
    }
    let mut root = BstNode {
        key: data[0],
        index: 0,
        left: None,
        right: None,
    };
    for (i, &v) in data.iter().enumerate().skip(1)
    {
        root.insert(v, i);
    }
    root.search(target)
}

// -- AVL --

struct AvlNode {
    key: i64,
    index: usize,
    height: usize,
    left: Option<Box<AvlNode>>,
    right: Option<Box<AvlNode>>,
}

impl AvlNode {
    fn h(n: &Option<Box<AvlNode>>) -> usize {
        n.as_ref().map_or(0, |x| x.height)
    }
    fn bf(&self) -> isize {
        Self::h(&self.right) as isize - Self::h(&self.left) as isize
    }
    fn update(&mut self) {
        self.height = 1 + Self::h(&self.left).max(Self::h(&self.right));
    }
    fn rotate_right(mut self: Box<Self>) -> Box<Self> {
        let mut x = self.left.take().unwrap();
        self.left = x.right.take();
        self.update();
        x.right = Some(self);
        x.update();
        x
    }
    fn rotate_left(mut self: Box<Self>) -> Box<Self> {
        let mut y = self.right.take().unwrap();
        self.right = y.left.take();
        self.update();
        y.left = Some(self);
        y.update();
        y
    }
    fn balance(mut self: Box<Self>) -> Box<Self> {
        self.update();
        let bf = self.bf();
        if bf > 1
        {
            if self.right.as_ref().map_or(0, |r| r.bf()) < 0
            {
                self.right = Some(self.right.unwrap().rotate_right());
            }
            return self.rotate_left();
        }
        if bf < -1
        {
            if self.left.as_ref().map_or(0, |l| l.bf()) > 0
            {
                self.left = Some(self.left.unwrap().rotate_left());
            }
            return self.rotate_right();
        }
        self
    }
    fn insert(mut self: Box<Self>, key: i64, index: usize) -> Box<Self> {
        match key.cmp(&self.key)
        {
            Ordering::Less =>
            {
                self.left = if let Some(left) = self.left.take()
                {
                    Some(left.insert(key, index))
                }
                else
                {
                    Some(Box::new(AvlNode {
                        key,
                        index,
                        height: 1,
                        left: None,
                        right: None,
                    }))
                };
            },
            Ordering::Greater =>
            {
                self.right = if let Some(right) = self.right.take()
                {
                    Some(right.insert(key, index))
                }
                else
                {
                    Some(Box::new(AvlNode {
                        key,
                        index,
                        height: 1,
                        left: None,
                        right: None,
                    }))
                };
            },
            Ordering::Equal => return self,
        }
        self.balance()
    }
    fn search(&self, key: i64) -> Option<usize> {
        match key.cmp(&self.key)
        {
            Ordering::Less => self.left.as_ref().and_then(|n| n.search(key)),
            Ordering::Greater => self.right.as_ref().and_then(|n| n.search(key)),
            Ordering::Equal => Some(self.index),
        }
    }
}

pub fn avl_search(data: &[i64], target: i64) -> Option<usize> {
    if data.is_empty()
    {
        return None;
    }
    let mut root = Box::new(AvlNode {
        key: data[0],
        index: 0,
        height: 1,
        left: None,
        right: None,
    });
    for (i, &v) in data.iter().enumerate().skip(1)
    {
        root = root.insert(v, i);
    }
    root.search(target)
}

// -- Hash --

pub fn hash_search(data: &[i64], target: i64) -> Option<usize> {
    let map: HashMap<i64, usize> = data.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    map.get(&target).copied()
}

/// Dispatched search.
pub fn search(data: &[i64], target: i64, strategy: SearchStrategy) -> Option<usize> {
    match strategy
    {
        SearchStrategy::Linear => linear_search(data, target),
        SearchStrategy::Binary => binary_search(data, target),
        SearchStrategy::Jump => jump_search(data, target),
        SearchStrategy::Exponential => exponential_search(data, target),
        SearchStrategy::Interpolation => interpolation_search(data, target),
        SearchStrategy::BST => bst_search(data, target),
        SearchStrategy::AVL => avl_search(data, target),
        SearchStrategy::HashMap => hash_search(data, target),
    }
}

/// Adaptive search-strategy selection based on data distribution.
pub fn select_search(data: &[i64], target: i64, dist: &DataDistribution) -> SearchStrategy {
    let _ = target;
    let n = data.len();
    if n == 0
    {
        return SearchStrategy::Linear;
    }
    match dist
    {
        DataDistribution::Sorted => SearchStrategy::Binary,
        DataDistribution::ReverseSorted => SearchStrategy::Exponential,
        DataDistribution::Uniform =>
        {
            if n < 100
            {
                SearchStrategy::Linear
            }
            else if n < 10_000
            {
                SearchStrategy::Binary
            }
            else
            {
                SearchStrategy::Interpolation
            }
        },
        DataDistribution::Skewed { .. } | DataDistribution::Clustered { .. } =>
        {
            SearchStrategy::HashMap
        },
    }
}

// ================================================================
// 3.  G R A P H
// ================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub vertices: usize,
    pub edges: Vec<(usize, usize, Option<f64>)>,
    pub directed: bool,
}

impl Graph {
    pub fn new(vertices: usize, directed: bool) -> Self {
        Self {
            vertices,
            edges: Vec::new(),
            directed,
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize, weight: Option<f64>) {
        self.edges.push((u, v, weight));
    }

    /// Adjacency list: for each vertex, list of (neighbor, weight).
    pub fn adjacency(&self) -> Vec<Vec<(usize, f64)>> {
        let mut adj = vec![Vec::new(); self.vertices];
        for &(u, v, w) in &self.edges
        {
            adj[u].push((v, w.unwrap_or(1.0)));
            if !self.directed
            {
                adj[v].push((u, w.unwrap_or(1.0)));
            }
        }
        adj
    }

    pub fn is_sparse(&self) -> bool {
        let max_edges = if self.directed
        {
            self.vertices * (self.vertices - 1)
        }
        else
        {
            self.vertices * (self.vertices - 1) / 2
        };
        if max_edges == 0
        {
            return true;
        }
        self.edges.len() < max_edges / 4
    }

    pub fn has_negative_edges(&self) -> bool {
        self.edges
            .iter()
            .any(|(_, _, w)| w.is_some_and(|x| x < 0.0))
    }
}

// -- Path finding --

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathStrategy {
    BFS,
    DFS,
    Dijkstra,
    AStar,
    BellmanFord,
    FloydWarshall,
}

impl PathStrategy {
    pub fn all() -> Vec<PathStrategy> {
        vec![
            PathStrategy::BFS,
            PathStrategy::DFS,
            PathStrategy::Dijkstra,
            PathStrategy::AStar,
            PathStrategy::BellmanFord,
            PathStrategy::FloydWarshall,
        ]
    }
}

/// Shortest-path distances from `source`.
pub fn shortest_path(graph: &Graph, source: usize, strategy: PathStrategy) -> Vec<Option<f64>> {
    match strategy
    {
        PathStrategy::BFS => bfs_path(graph, source),
        PathStrategy::DFS => dfs_path(graph, source),
        PathStrategy::Dijkstra => dijkstra(graph, source),
        PathStrategy::AStar => astar(graph, source, graph.vertices.saturating_sub(1)),
        PathStrategy::BellmanFord => bellman_ford(graph, source),
        PathStrategy::FloydWarshall =>
        {
            let ap = floyd_warshall(graph);
            if source < ap.len()
            {
                ap[source].clone()
            }
            else
            {
                vec![None; graph.vertices]
            }
        },
    }
}

fn bfs_path(graph: &Graph, source: usize) -> Vec<Option<f64>> {
    let adj = graph.adjacency();
    let mut dist = vec![None; graph.vertices];
    let mut queue = VecDeque::new();
    dist[source] = Some(0.0);
    queue.push_back(source);
    while let Some(u) = queue.pop_front()
    {
        let d = dist[u].unwrap();
        for &(v, _w) in &adj[u]
        {
            if dist[v].is_none()
            {
                dist[v] = Some(d + 1.0);
                queue.push_back(v);
            }
        }
    }
    dist
}

fn dfs_path(graph: &Graph, source: usize) -> Vec<Option<f64>> {
    let adj = graph.adjacency();
    let mut dist = vec![None; graph.vertices];
    let mut stack = vec![(source, 0.0)];
    while let Some((u, d)) = stack.pop()
    {
        if dist[u].is_some()
        {
            continue;
        }
        dist[u] = Some(d);
        for &(v, _w) in adj[u].iter().rev()
        {
            if dist[v].is_none()
            {
                stack.push((v, d + 1.0));
            }
        }
    }
    dist
}

fn dijkstra(graph: &Graph, source: usize) -> Vec<Option<f64>> {
    let adj = graph.adjacency();
    let mut dist = vec![f64::INFINITY; graph.vertices];
    let mut visited = vec![false; graph.vertices];
    dist[source] = 0.0;
    for _ in 0..graph.vertices
    {
        let u = (0..graph.vertices)
            .filter(|i| !visited[*i])
            .min_by(|a, b| dist[*a].partial_cmp(&dist[*b]).unwrap_or(Ordering::Equal));
        match u
        {
            None => break,
            Some(u) =>
            {
                visited[u] = true;
                for &(v, w) in &adj[u]
                {
                    let nd = dist[u] + w;
                    if nd < dist[v]
                    {
                        dist[v] = nd;
                    }
                }
            },
        }
    }
    dist.into_iter()
        .map(|d| if d == f64::INFINITY { None } else { Some(d) })
        .collect()
}

fn astar(graph: &Graph, source: usize, target: usize) -> Vec<Option<f64>> {
    let _ = target;
    dijkstra(graph, source)
}

fn bellman_ford(graph: &Graph, source: usize) -> Vec<Option<f64>> {
    let mut dist = vec![f64::INFINITY; graph.vertices];
    dist[source] = 0.0;
    for _ in 0..graph.vertices
    {
        for &(u, v, w) in &graph.edges
        {
            let w = w.unwrap_or(1.0);
            if dist[u] + w < dist[v]
            {
                dist[v] = dist[u] + w;
            }
            if !graph.directed && dist[v] + w < dist[u]
            {
                dist[u] = dist[v] + w;
            }
        }
    }
    dist.into_iter()
        .map(|d| if d == f64::INFINITY { None } else { Some(d) })
        .collect()
}

#[allow(clippy::needless_range_loop)]
fn floyd_warshall(graph: &Graph) -> Vec<Vec<Option<f64>>> {
    let n = graph.vertices;
    let mut dist = vec![vec![f64::INFINITY; n]; n];
    for i in 0..n
    {
        dist[i][i] = 0.0;
    }
    for &(u, v, w) in &graph.edges
    {
        let w = w.unwrap_or(1.0);
        dist[u][v] = dist[u][v].min(w);
        if !graph.directed
        {
            dist[v][u] = dist[v][u].min(w);
        }
    }
    for k in 0..n
    {
        for i in 0..n
        {
            for j in 0..n
            {
                let through = dist[i][k] + dist[k][j];
                if through < dist[i][j]
                {
                    dist[i][j] = through;
                }
            }
        }
    }
    dist.iter()
        .map(|row| {
            row.iter()
                .map(|&d| if d == f64::INFINITY { None } else { Some(d) })
                .collect()
        })
        .collect()
}

/// Auto-select pathfinding strategy based on graph properties.
pub fn select_path(graph: &Graph) -> PathStrategy {
    if graph.vertices < 2
    {
        return PathStrategy::BFS;
    }
    let weighted = graph
        .edges
        .iter()
        .any(|(_, _, w)| w.is_some() && w.unwrap() != 1.0);
    let negative = graph.has_negative_edges();
    let sparse = graph.is_sparse();
    if negative
    {
        if sparse && graph.vertices < 500
        {
            PathStrategy::BellmanFord
        }
        else
        {
            PathStrategy::FloydWarshall
        }
    }
    else if weighted
    {
        if sparse
        {
            PathStrategy::Dijkstra
        }
        else
        {
            PathStrategy::AStar
        }
    }
    else if sparse
    {
        PathStrategy::BFS
    }
    else
    {
        PathStrategy::FloydWarshall
    }
}

// -- Spanning tree --

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TreeStrategy {
    Prim,
    Kruskal,
    Boruvka,
}

/// Minimum spanning tree: returns list of (u, v, weight) edges.
pub fn spanning_tree(graph: &Graph, strategy: TreeStrategy) -> Vec<(usize, usize, f64)> {
    match strategy
    {
        TreeStrategy::Prim => prim(graph),
        TreeStrategy::Kruskal => kruskal(graph),
        TreeStrategy::Boruvka => boruvka(graph),
    }
}

fn prim(graph: &Graph) -> Vec<(usize, usize, f64)> {
    let adj = graph.adjacency();
    let mut in_tree = vec![false; graph.vertices];
    let mut best = vec![(f64::INFINITY, 0usize); graph.vertices];
    let mut mst = Vec::new();
    best[0] = (0.0, 0);
    for _ in 0..graph.vertices
    {
        let u = (0..graph.vertices)
            .filter(|i| !in_tree[*i] && best[*i].0 < f64::INFINITY)
            .min_by(|a, b| {
                best[*a]
                    .0
                    .partial_cmp(&best[*b].0)
                    .unwrap_or(Ordering::Equal)
            });
        match u
        {
            None => break,
            Some(u) =>
            {
                in_tree[u] = true;
                if best[u].1 != u
                {
                    mst.push((best[u].1, u, best[u].0));
                }
                for &(v, w) in &adj[u]
                {
                    if !in_tree[v] && w < best[v].0
                    {
                        best[v] = (w, u);
                    }
                }
            },
        }
    }
    mst
}

fn kruskal(graph: &Graph) -> Vec<(usize, usize, f64)> {
    let mut edges: Vec<(usize, usize, f64)> = graph
        .edges
        .iter()
        .map(|&(u, v, w)| (u, v, w.unwrap_or(1.0)))
        .collect();
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));
    let mut uf = UnionFind::new(graph.vertices);
    let mut mst = Vec::new();
    for (u, v, w) in edges
    {
        if uf.find(u) != uf.find(v)
        {
            uf.union(u, v);
            mst.push((u, v, w));
        }
    }
    mst
}

#[allow(clippy::needless_range_loop)]
fn boruvka(graph: &Graph) -> Vec<(usize, usize, f64)> {
    let mut mst = Vec::new();
    let mut uf = UnionFind::new(graph.vertices);
    let mut num_components = graph.vertices;
    while num_components > 1
    {
        let mut cheapest = vec![(usize::MAX, usize::MAX, f64::INFINITY); graph.vertices];
        for &(u, v, w) in &graph.edges
        {
            let w = w.unwrap_or(1.0);
            let ru = uf.find(u);
            let rv = uf.find(v);
            if ru != rv
            {
                if w < cheapest[ru].2
                {
                    cheapest[ru] = (u, v, w);
                }
                if w < cheapest[rv].2
                {
                    cheapest[rv] = (u, v, w);
                }
            }
        }
        let mut added = false;
        for i in 0..graph.vertices
        {
            let (u, v, w) = cheapest[i];
            if u == usize::MAX
            {
                continue;
            }
            let ru = uf.find(u);
            let rv = uf.find(v);
            if ru != rv
            {
                uf.union(ru, rv);
                mst.push((u, v, w));
                num_components -= 1;
                added = true;
            }
        }
        if !added
        {
            break;
        }
    }
    mst
}

// -- Union-Find --

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x
        {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry
        {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry])
        {
            Ordering::Less => self.parent[rx] = ry,
            Ordering::Greater => self.parent[ry] = rx,
            Ordering::Equal =>
            {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            },
        }
    }
}

// -- Flow --

/// Return type for max-flow: (total_flow, edge_flow_list).
pub type FlowResult = (f64, Vec<((usize, usize), f64)>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowStrategy {
    FordFulkerson,
    EdmondsKarp,
    DinicLike,
}

/// Compute max flow from s to t. Returns (total_flow, edge_flow_list).
pub fn max_flow(graph: &Graph, source: usize, sink: usize, strategy: FlowStrategy) -> FlowResult {
    match strategy
    {
        FlowStrategy::FordFulkerson => ford_fulkerson(graph, source, sink),
        FlowStrategy::EdmondsKarp => edmonds_karp(graph, source, sink),
        FlowStrategy::DinicLike => dinic_like(graph, source, sink),
    }
}

fn edmonds_karp(graph: &Graph, source: usize, sink: usize) -> FlowResult {
    let n = graph.vertices;
    let mut cap = vec![vec![0.0; n]; n];
    let mut adj = vec![Vec::new(); n];
    for &(u, v, w) in &graph.edges
    {
        let c = w.unwrap_or(1.0);
        cap[u][v] += c;
        adj[u].push(v);
        adj[v].push(u);
    }
    let mut flow = vec![vec![0.0; n]; n];
    let mut total = 0.0;
    loop
    {
        let mut parent = vec![None; n];
        let mut queue = VecDeque::new();
        queue.push_back(source);
        parent[source] = Some((source, f64::INFINITY));
        while let Some(u) = queue.pop_front()
        {
            let flow_u = parent[u].unwrap().1;
            for &v in &adj[u]
            {
                if parent[v].is_none() && cap[u][v] - flow[u][v] > 0.0
                {
                    let df = flow_u.min(cap[u][v] - flow[u][v]);
                    parent[v] = Some((u, df));
                    if v == sink
                    {
                        break;
                    }
                    queue.push_back(v);
                }
            }
            if parent[sink].is_some()
            {
                break;
            }
        }
        match parent[sink]
        {
            None => break,
            Some((_, df)) =>
            {
                total += df;
                let mut v = sink;
                while v != source
                {
                    let u = parent[v].unwrap().0;
                    flow[u][v] += df;
                    flow[v][u] -= df;
                    v = u;
                }
            },
        }
    }
    let mut edge_flows = Vec::new();
    for &(u, v, _) in &graph.edges
    {
        edge_flows.push(((u, v), flow[u][v]));
    }
    (total, edge_flows)
}

fn ford_fulkerson(graph: &Graph, source: usize, sink: usize) -> FlowResult {
    let n = graph.vertices;
    let mut cap = vec![vec![0.0; n]; n];
    let mut adj = vec![Vec::new(); n];
    for &(u, v, w) in &graph.edges
    {
        let c = w.unwrap_or(1.0);
        cap[u][v] += c;
        adj[u].push(v);
        adj[v].push(u);
    }
    let mut flow = vec![vec![0.0; n]; n];
    let mut total = 0.0;

    fn dfs(
        u: usize,
        sink: usize,
        min: f64,
        visited: &mut [bool],
        cap: &[Vec<f64>],
        flow: &mut [Vec<f64>],
        adj: &[Vec<usize>],
    ) -> f64 {
        if u == sink
        {
            return min;
        }
        visited[u] = true;
        for &v in &adj[u]
        {
            if !visited[v] && cap[u][v] - flow[u][v] > 0.0
            {
                let df = dfs(
                    v,
                    sink,
                    min.min(cap[u][v] - flow[u][v]),
                    visited,
                    cap,
                    flow,
                    adj,
                );
                if df > 0.0
                {
                    flow[u][v] += df;
                    flow[v][u] -= df;
                    return df;
                }
            }
        }
        0.0
    }

    loop
    {
        let mut visited = vec![false; n];
        let df = dfs(
            source,
            sink,
            f64::INFINITY,
            &mut visited,
            &cap,
            &mut flow,
            &adj,
        );
        if df == 0.0
        {
            break;
        }
        total += df;
    }

    let mut edge_flows = Vec::new();
    for &(u, v, _) in &graph.edges
    {
        edge_flows.push(((u, v), flow[u][v]));
    }
    (total, edge_flows)
}

fn dinic_like(graph: &Graph, source: usize, sink: usize) -> FlowResult {
    edmonds_karp(graph, source, sink)
}

// -- Graph coloring --

/// Greedy Welsh-Powell coloring. Returns color index per vertex.
pub fn graph_coloring(graph: &Graph) -> Vec<usize> {
    let n = graph.vertices;
    let adj = graph.adjacency();
    let mut degree: Vec<(usize, usize)> = (0..n).map(|i| (adj[i].len(), i)).collect();
    degree.sort_by_key(|b| std::cmp::Reverse(b.0));
    let mut colors = vec![usize::MAX; n];
    for &(_deg, u) in &degree
    {
        let mut used = vec![false; n];
        for &(v, _) in &adj[u]
        {
            if colors[v] != usize::MAX
            {
                used[colors[v]] = true;
            }
        }
        let c = used.iter().position(|&u| !u).unwrap_or(0);
        colors[u] = c;
    }
    colors
}

// ================================================================
// 4.  D Y N A M I C   P R O G R A M M I N G
// ================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DpImplementation {
    Memoization,
    Tabulation,
}

/// Assess DP-suitability.
pub fn is_dp_suitable(optimal_substructure: bool, overlapping: bool) -> bool {
    optimal_substructure && overlapping
}

/// Fibonacci via memoization.
pub fn fib_memo(n: usize) -> u64 {
    let mut memo = vec![0u64; n + 2];
    if n >= 1
    {
        memo[1] = 1;
    }
    fn inner(n: usize, memo: &mut [u64]) -> u64 {
        if n == 0 || memo[n] != 0
        {
            return memo[n];
        }
        let val = inner(n - 1, memo) + inner(n - 2, memo);
        memo[n] = val;
        val
    }
    inner(n, &mut memo)
}

/// Fibonacci via tabulation.
pub fn fib_tab(n: usize) -> u64 {
    if n == 0
    {
        return 0;
    }
    let mut dp = vec![0u64; n + 1];
    dp[1] = 1;
    for i in 2..=n
    {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    dp[n]
}

/// 0/1 Knapsack — tabulation.
pub fn knapsack(weights: &[usize], values: &[u64], capacity: usize) -> u64 {
    let n = weights.len();
    let mut dp = vec![0u64; capacity + 1];
    for i in 0..n
    {
        for w in (weights[i]..=capacity).rev()
        {
            dp[w] = dp[w].max(dp[w - weights[i]] + values[i]);
        }
    }
    dp[capacity]
}

/// Longest Common Subsequence length.
pub fn lcs(a: &[i64], b: &[i64]) -> usize {
    let (m, n) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m
    {
        for j in 1..=n
        {
            if a[i - 1] == b[j - 1]
            {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            }
            else
            {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[m][n]
}

/// Edit distance (Levenshtein).
pub fn edit_distance(a: &str, b: &str) -> usize {
    // Operate on Unicode scalar values, not raw bytes, so multi-byte
    // characters count as a single edit unit.
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    let (m, n) = (ac.len(), bc.len());
    let mut prev = (0..=n).collect::<Vec<usize>>();
    let mut cur = vec![0usize; n + 1];
    for i in 1..=m
    {
        cur[0] = i;
        for j in 1..=n
        {
            let cost = if ac[i - 1] == bc[j - 1]
            {
                0
            }
            else
            {
                1
            };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[n]
}

// ================================================================
// 5.  D I V I D E - A N D - C O N Q U E R
// ================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DaqStrategy {
    /// Merge-sort-like: split in half, recurse, merge.
    BinarySplit,
    /// Quick-sort-like: partition around pivot.
    PivotBased,
    /// Strassen-like: divide into K subproblems.
    MultiWay(usize),
}

/// Merge sort via DaC pattern (returns sorted copy).
pub fn dac_merge_sort(data: &[i64]) -> Vec<i64> {
    if data.len() <= 1
    {
        return data.to_vec();
    }
    let mid = data.len() / 2;
    let left = dac_merge_sort(&data[..mid]);
    let right = dac_merge_sort(&data[mid..]);
    let mut result = Vec::with_capacity(data.len());
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len()
    {
        if left[i] <= right[j]
        {
            result.push(left[i]);
            i += 1;
        }
        else
        {
            result.push(right[j]);
            j += 1;
        }
    }
    result.extend_from_slice(&left[i..]);
    result.extend_from_slice(&right[j..]);
    result
}

/// Maximum subarray sum via divide-and-conquer (crossing Kadane).
#[allow(clippy::needless_range_loop)]
pub fn max_subarray(data: &[i64]) -> i64 {
    if data.is_empty()
    {
        return 0;
    }

    fn dac(a: &[i64], lo: usize, hi: usize) -> i64 {
        if lo == hi
        {
            return a[lo];
        }
        let mid = (lo + hi) / 2;
        let left = dac(a, lo, mid);
        let right = dac(a, mid + 1, hi);

        let mut lsum = i64::MIN;
        let mut s = 0i64;
        for i in (lo..=mid).rev()
        {
            s += a[i];
            lsum = lsum.max(s);
        }

        let mut rsum = i64::MIN;
        s = 0;
        for i in mid + 1..=hi
        {
            s += a[i];
            rsum = rsum.max(s);
        }
        let cross = lsum + rsum;
        left.max(right).max(cross)
    }

    dac(data, 0, data.len() - 1)
}

/// Fast exponentiation x^n (binary exponentiation — divide-and-conquer).
pub fn fast_pow(x: f64, n: u32) -> f64 {
    match n
    {
        0 => 1.0,
        n if n % 2 == 0 =>
        {
            let half = fast_pow(x, n / 2);
            half * half
        },
        _ =>
        {
            let half = fast_pow(x, n / 2);
            half * half * x
        },
    }
}

/// Closest pair in 1D (demonstration template).
pub fn closest_pair_1d(points: &[i64]) -> Option<(i64, i64, i64)> {
    if points.len() < 2
    {
        return None;
    }
    let mut pts = points.to_vec();
    pts.sort_unstable();
    let mut min_dist = i64::MAX;
    let mut pair = (0i64, 0i64);
    for w in pts.windows(2)
    {
        let d = w[1] - w[0];
        if d < min_dist
        {
            min_dist = d;
            pair = (w[0], w[1]);
        }
    }
    Some((pair.0, pair.1, min_dist))
}

/// Master theorem: T(n) = a*T(n/b) + f(n) where f(n) = n^e.
pub fn master_theorem(a: f64, b: f64, f_exponent: f64) -> String {
    let log_ba = if b > 1.0 && a > 0.0
    {
        (a.ln() / b.ln()).abs()
    }
    else
    {
        1.0
    };
    let epsilon = 1e-6;
    if (f_exponent - log_ba).abs() < epsilon
    {
        format!("Θ(n^{:.1} log n)", log_ba)
    }
    else if f_exponent > log_ba
    {
        format!("Θ(n^{:.1})", f_exponent)
    }
    else
    {
        format!("Θ(n^{:.1})", log_ba)
    }
}

// ================================================================
// 6.  C O M P L E X I T Y   A N A L Y S I S
// ================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexityClass {
    Constant,
    Logarithmic,
    Linear,
    Linearithmic,
    Quadratic,
    Cubic,
    Exponential,
    Factorial,
    Unknown,
}

impl ComplexityClass {
    pub fn name(&self) -> &'static str {
        match self
        {
            ComplexityClass::Constant => "O(1)",
            ComplexityClass::Logarithmic => "O(log n)",
            ComplexityClass::Linear => "O(n)",
            ComplexityClass::Linearithmic => "O(n log n)",
            ComplexityClass::Quadratic => "O(n²)",
            ComplexityClass::Cubic => "O(n³)",
            ComplexityClass::Exponential => "O(2ⁿ)",
            ComplexityClass::Factorial => "O(n!)",
            ComplexityClass::Unknown => "Unknown",
        }
    }
}

/// Fit the empirical (size, time) measurements to one of the standard complexity classes.
pub fn fit_complexity(sizes: &[usize], times: &[f64]) -> ComplexityClass {
    if sizes.len() < 3
    {
        return ComplexityClass::Unknown;
    }
    #[allow(clippy::type_complexity)]
    let candidates: [(ComplexityClass, fn(f64) -> f64); 7] = [
        (ComplexityClass::Constant, |_n| 1.0),
        (ComplexityClass::Logarithmic, |n: f64| n.ln().max(1.0)),
        (ComplexityClass::Linear, |n: f64| n),
        (ComplexityClass::Linearithmic, |n: f64| n * n.ln().max(1.0)),
        (ComplexityClass::Quadratic, |n: f64| n * n),
        (ComplexityClass::Cubic, |n: f64| n * n * n),
        (ComplexityClass::Exponential, |n: f64| {
            2.0f64.powf(n.min(50.0))
        }),
    ];
    let mut best = (ComplexityClass::Unknown, f64::INFINITY);
    for (cls, model) in &candidates
    {
        // Scale factor: pick scale to minimise relative error on largest input.
        let last_n = sizes.last().copied().unwrap_or(1) as f64;
        let last_t = times.last().copied().unwrap_or(1.0);
        let predicted_last = model(last_n);
        if predicted_last <= 0.0
        {
            continue;
        }
        let scale = last_t / predicted_last;
        let error: f64 = sizes
            .iter()
            .zip(times.iter())
            .map(|(&s, &t)| {
                let p = model(s as f64) * scale;
                ((t - p) / t.max(1.0)).abs()
            })
            .sum();
        if error < best.1
        {
            best = (*cls, error);
        }
    }
    best.0
}

/// Measure empirical complexity by timing `f` on increasing sizes.
pub fn measure_complexity<F>(f: F, max_size: usize, steps: usize) -> (Vec<usize>, Vec<f64>)
where
    F: Fn(&[i64]) -> i64,
{
    let mut sizes = Vec::new();
    let mut times = Vec::new();
    let mut state: u64 = 42;
    for step in 0..steps
    {
        let size = max_size * (step + 1) / steps;
        let data: Vec<i64> = (0..size)
            .map(|_| {
                lcg(&mut state);
                (state % 1000) as i64
            })
            .collect();
        let start = Instant::now();
        f(&data);
        let elapsed = start.elapsed().as_nanos() as f64;
        sizes.push(size);
        times.push(elapsed);
    }
    (sizes, times)
}

// ================================================================
// 7.  A L G O R I T H M   C O M P A R I S O N
// ================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub strategy: String,
    pub time_ns: u128,
    pub correct: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub algorithm: String,
    pub data_size: usize,
    pub results: Vec<BenchmarkResult>,
    pub winner: String,
}

/// Compare sort strategies head-to-head on the same data.
pub fn compare_sorts(data: &[i64], strategies: &[SortStrategy]) -> ComparisonReport {
    let expected = {
        let mut s = data.to_vec();
        s.sort_unstable();
        s
    };
    let mut results = Vec::new();
    let mut best = (String::new(), u128::MAX);
    for &s in strategies
    {
        let mut copy = data.to_vec();
        let start = Instant::now();
        sort(&mut copy, s);
        let time_ns = start.elapsed().as_nanos();
        let correct = copy == expected;
        results.push(BenchmarkResult {
            strategy: s.to_string(),
            time_ns,
            correct,
        });
        if correct && time_ns < best.1
        {
            best = (s.to_string(), time_ns);
        }
    }
    ComparisonReport {
        algorithm: "sort".into(),
        data_size: data.len(),
        results,
        winner: best.0,
    }
}

/// Compare search strategies on the same target.
pub fn compare_searches(
    data: &[i64],
    target: i64,
    strategies: &[SearchStrategy],
) -> ComparisonReport {
    let mut sorted = data.to_vec();
    sorted.sort_unstable();
    let expected = sorted.iter().position(|&x| x == target);
    let mut results = Vec::new();
    let mut best = (String::new(), u128::MAX);
    for &s in strategies
    {
        let start = Instant::now();
        let result = match s
        {
            SearchStrategy::Linear => linear_search(data, target),
            SearchStrategy::Binary => binary_search(&sorted, target),
            SearchStrategy::Jump => jump_search(&sorted, target),
            SearchStrategy::Exponential => exponential_search(&sorted, target),
            SearchStrategy::Interpolation => interpolation_search(&sorted, target),
            SearchStrategy::BST => bst_search(data, target),
            SearchStrategy::AVL => avl_search(data, target),
            SearchStrategy::HashMap => hash_search(data, target),
        };
        let time_ns = start.elapsed().as_nanos();
        let correct = result == expected;
        results.push(BenchmarkResult {
            strategy: s.to_string(),
            time_ns,
            correct,
        });
        if correct && time_ns < best.1
        {
            best = (s.to_string(), time_ns);
        }
    }
    ComparisonReport {
        algorithm: "search".into(),
        data_size: data.len(),
        results,
        winner: best.0,
    }
}

/// Repeated-measures statistical comparison: mean, stddev per strategy.
pub fn statistical_compare_sorts(
    data: &[i64],
    strategies: &[SortStrategy],
    trials: usize,
) -> Vec<(String, f64, f64)> {
    let mut stats: Vec<(String, Vec<u128>)> = strategies
        .iter()
        .map(|&s| (s.to_string(), Vec::with_capacity(trials)))
        .collect();
    for (name, times) in &mut stats
    {
        let strat = strategies
            .iter()
            .find(|&&s| s.to_string() == *name)
            .copied()
            .unwrap();
        for _ in 0..trials
        {
            let mut copy = data.to_vec();
            let start = Instant::now();
            sort(&mut copy, strat);
            times.push(start.elapsed().as_nanos());
        }
    }
    stats
        .into_iter()
        .map(|(name, times)| {
            let n = times.len() as f64;
            let mean = times.iter().sum::<u128>() as f64 / n;
            let var = times
                .iter()
                .map(|t| (*t as f64 - mean).powi(2))
                .sum::<f64>()
                / n;
            (name, mean, var.sqrt())
        })
        .collect()
}

// ================================================================
// 8.  E V O L U T I O N A R Y   I N T E G R A T I O N
// ================================================================

/// Evolve the best sort strategy for the given data using a GA.
/// Genome encodes strategy-index + tuning parameters.
pub fn evolve_sort(data: &[i64], generations: usize, pop_size: usize) -> (SortStrategy, f64) {
    let mut ga = GeneticAlgorithm::seeded(0x4160_5C12);
    ga.pop_size = pop_size;
    ga.mutation_rate = 0.15;
    ga.crossover_rate = 0.8;
    ga.elitism = 2;
    ga.bounds = (0.0, 10.0);
    let mut pop = ga.init_pop(3);
    for _ in 0..generations
    {
        let data_clone = data.to_vec();
        ga.evolve(&mut pop, |inds| {
            inds.iter()
                .map(|ind| {
                    let g = &ind.genome;
                    let strategy_idx = (g[0] as usize) % SortStrategy::all().len();
                    let strategy = SortStrategy::all()[strategy_idx];
                    let mut copy = data_clone.clone();
                    let start = Instant::now();
                    sort(&mut copy, strategy);
                    let time_ns = start.elapsed().as_nanos() as f64;
                    let correct = copy.windows(2).all(|w| w[0] <= w[1]);
                    if correct { -time_ns + 1e9 } else { -1e12 }
                })
                .collect()
        });
    }
    pop.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(Ordering::Equal));
    let best = &pop[0];
    let strategy_idx = (best.genome[0] as usize) % SortStrategy::all().len();
    (SortStrategy::all()[strategy_idx], best.fitness)
}

// ================================================================
// 9.  T E S T S   (30 tests)
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic test data — avoids external `rand` dependency.
    fn test_vec(size: usize, seed: u64) -> Vec<i64> {
        let mut state = seed;
        (0..size)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state % 2000) as i64 - 1000
            })
            .collect()
    }

    fn check_sorted(data: &[i64]) -> bool {
        data.windows(2).all(|w| w[0] <= w[1])
    }

    // ---- 1–12: sorting ----

    #[test]
    fn test_bubble_sort() {
        let mut v = test_vec(80, 1);
        sort(&mut v, SortStrategy::Bubble);
        assert!(check_sorted(&v));
    }

    #[test]
    fn test_insertion_sort() {
        let mut v = test_vec(80, 2);
        sort(&mut v, SortStrategy::Insertion);
        assert!(check_sorted(&v));
    }

    #[test]
    fn test_selection_sort() {
        let mut v = test_vec(80, 3);
        sort(&mut v, SortStrategy::Selection);
        assert!(check_sorted(&v));
    }

    #[test]
    fn test_merge_sort() {
        let mut v = test_vec(200, 4);
        sort(&mut v, SortStrategy::Merge);
        assert!(check_sorted(&v));
    }

    #[test]
    fn test_quick_sort() {
        let mut v = test_vec(200, 5);
        sort(&mut v, SortStrategy::Quick);
        assert!(check_sorted(&v));
    }

    #[test]
    fn test_heap_sort() {
        let mut v = test_vec(200, 6);
        sort(&mut v, SortStrategy::Heap);
        assert!(check_sorted(&v));
    }

    #[test]
    fn test_counting_sort() {
        let mut v = test_vec(120, 7);
        sort(&mut v, SortStrategy::Counting);
        assert!(check_sorted(&v));
    }

    #[test]
    fn test_radix_sort() {
        let mut v = test_vec(120, 8);
        sort(&mut v, SortStrategy::Radix);
        assert!(check_sorted(&v));
    }

    // Regression: counting sort must not overflow `max - min` nor attempt a
    // gigantic allocation when the value span exceeds i64::MAX. It should
    // degrade gracefully and still sort correctly.
    #[test]
    fn test_counting_sort_wide_range() {
        let mut v = vec![i64::MAX, i64::MIN, 0, -1, 1, i64::MAX - 1, i64::MIN + 1];
        let mut expected = v.clone();
        expected.sort_unstable();
        sort(&mut v, SortStrategy::Counting);
        assert_eq!(v, expected);
    }

    // Regression: radix sort must not overflow when the range is wide enough
    // that `v - min` or `exp *= 256` would wrap. The sign-bit mapping keeps it
    // correct across the full i64 domain.
    #[test]
    fn test_radix_sort_wide_range() {
        let mut v = vec![i64::MAX, i64::MIN, 0, -1, 1, i64::MAX - 3, i64::MIN + 7, 42, -42];
        let mut expected = v.clone();
        expected.sort_unstable();
        sort(&mut v, SortStrategy::Radix);
        assert_eq!(v, expected);
    }

    #[test]
    fn test_timsort_like() {
        let mut v = test_vec(120, 9);
        sort(&mut v, SortStrategy::TimsortLike);
        assert!(check_sorted(&v));
    }

    #[test]
    fn test_introselect() {
        let mut v = test_vec(200, 10);
        sort(&mut v, SortStrategy::Introselect);
        assert!(check_sorted(&v));
    }

    #[test]
    fn test_sort_edge_cases() {
        // empty
        let mut empty: Vec<i64> = vec![];
        sort(&mut empty, SortStrategy::Quick);
        assert!(empty.is_empty());
        // single element
        let mut single = vec![42];
        for &s in &SortStrategy::all()
        {
            sort(&mut single, s);
            assert_eq!(single, vec![42]);
        }
    }

    #[test]
    fn test_sort_nearly_sorted() {
        let mut v: Vec<i64> = (0..100).collect();
        v[5] = 99;
        v[6] = 98;
        for &s in &SortStrategy::all()
        {
            let mut copy = v.clone();
            sort(&mut copy, s);
            assert!(check_sorted(&copy));
        }
    }

    #[test]
    fn test_auto_select_sort() {
        let c = DataCharacteristics {
            size: 100,
            sortedness: 0.1,
            range: (0, 100),
            uniqueness: 1.0,
        };
        let scores = select_sort(&c);
        assert!(!scores.is_empty());
        let best = best_sort(&c);
        assert!(SortStrategy::all().contains(&best));
    }

    #[test]
    fn test_select_sort_nearly_sorted() {
        let c = DataCharacteristics {
            size: 1000,
            sortedness: 0.95,
            range: (0, 100),
            uniqueness: 1.0,
        };
        let best = best_sort(&c);
        assert!(
            best == SortStrategy::Insertion || best == SortStrategy::TimsortLike,
            "expected Insertion or TimsortLike for nearly-sorted data, got {best}"
        );
    }

    // ---- 13–18: searching ----

    #[test]
    fn test_linear_search() {
        let data = vec![5, 3, 8, 1, 9];
        assert_eq!(linear_search(&data, 8), Some(2));
        assert_eq!(linear_search(&data, 0), None);
    }

    #[test]
    fn test_binary_search() {
        let data: Vec<i64> = (0..100).step_by(2).collect();
        assert_eq!(binary_search(&data, 20), Some(10));
        assert_eq!(binary_search(&data, 21), None);
    }

    #[test]
    fn test_jump_and_exponential_search() {
        let data: Vec<i64> = (0..100).collect();
        assert_eq!(jump_search(&data, 77), Some(77));
        assert_eq!(jump_search(&data, 1000), None);
        assert_eq!(exponential_search(&data, 0), Some(0));
        assert_eq!(exponential_search(&data, 99), Some(99));
        assert_eq!(exponential_search(&data, 100), None);
    }

    #[test]
    fn test_interpolation_search() {
        let data: Vec<i64> = (0..200).step_by(2).collect();
        assert_eq!(interpolation_search(&data, 50), Some(25));
        assert_eq!(interpolation_search(&data, 51), None);
    }

    #[test]
    fn test_bst_and_avl_search() {
        let data = vec![5, 3, 8, 1, 9, 2, 7];
        assert_eq!(bst_search(&data, 3), Some(1));
        assert_eq!(bst_search(&data, 10), None);
        assert_eq!(avl_search(&data, 3), Some(1));
        assert_eq!(avl_search(&data, 10), None);
    }

    #[test]
    fn test_hash_search() {
        let data = vec![5, 3, 8, 1, 9];
        assert_eq!(hash_search(&data, 8), Some(2));
        assert_eq!(hash_search(&data, 0), None);
    }

    // ---- 19–24: graph ----

    #[test]
    fn test_bfs_path() {
        let mut g = Graph::new(5, false);
        g.add_edge(0, 1, None);
        g.add_edge(0, 2, None);
        g.add_edge(1, 3, None);
        g.add_edge(2, 4, None);
        let dist = shortest_path(&g, 0, PathStrategy::BFS);
        assert_eq!(dist[0], Some(0.0));
        assert_eq!(dist[1], Some(1.0));
        assert_eq!(dist[3], Some(2.0));
        assert_eq!(dist[4], Some(2.0));
    }

    #[test]
    fn test_dijkstra_path() {
        let mut g = Graph::new(4, false);
        g.add_edge(0, 1, Some(2.0));
        g.add_edge(0, 2, Some(5.0));
        g.add_edge(1, 2, Some(1.0));
        g.add_edge(1, 3, Some(7.0));
        g.add_edge(2, 3, Some(3.0));
        let dist = shortest_path(&g, 0, PathStrategy::Dijkstra);
        assert_eq!(dist[0], Some(0.0));
        assert_eq!(dist[1], Some(2.0));
        assert_eq!(dist[2], Some(3.0));
        assert_eq!(dist[3], Some(6.0));
    }

    #[test]
    fn test_bellman_ford() {
        let mut g = Graph::new(4, true);
        g.add_edge(0, 1, Some(1.0));
        g.add_edge(1, 2, Some(-2.0));
        g.add_edge(2, 3, Some(3.0));
        g.add_edge(0, 3, Some(10.0));
        let dist = shortest_path(&g, 0, PathStrategy::BellmanFord);
        assert_eq!(dist[0], Some(0.0));
        assert_eq!(dist[1], Some(1.0));
        assert_eq!(dist[2], Some(-1.0));
        assert_eq!(dist[3], Some(2.0));
    }

    #[test]
    fn test_mst_all_strategies() {
        let mut g = Graph::new(4, false);
        g.add_edge(0, 1, Some(1.0));
        g.add_edge(0, 2, Some(3.0));
        g.add_edge(1, 2, Some(2.0));
        g.add_edge(1, 3, Some(4.0));
        g.add_edge(2, 3, Some(5.0));
        for &strat in &[
            TreeStrategy::Prim,
            TreeStrategy::Kruskal,
            TreeStrategy::Boruvka,
        ]
        {
            let mst = spanning_tree(&g, strat);
            assert_eq!(mst.len(), 3, "MST {strat:?} should have 3 edges");
            let total: f64 = mst.iter().map(|(_, _, w)| w).sum();
            assert!((total - 7.0).abs() < 0.01, "{strat:?} total {total} != 7");
        }
    }

    #[test]
    fn test_max_flow_ek() {
        let mut g = Graph::new(6, true);
        g.add_edge(0, 1, Some(10.0));
        g.add_edge(0, 2, Some(10.0));
        g.add_edge(1, 2, Some(2.0));
        g.add_edge(1, 3, Some(4.0));
        g.add_edge(1, 4, Some(8.0));
        g.add_edge(2, 4, Some(9.0));
        g.add_edge(3, 5, Some(10.0));
        g.add_edge(4, 3, Some(6.0));
        g.add_edge(4, 5, Some(10.0));
        let (total, _) = max_flow(&g, 0, 5, FlowStrategy::EdmondsKarp);
        assert!((total - 19.0).abs() < 0.01);
    }

    #[test]
    fn test_graph_coloring() {
        let mut g = Graph::new(4, false);
        g.add_edge(0, 1, None);
        g.add_edge(1, 2, None);
        g.add_edge(2, 3, None);
        g.add_edge(3, 0, None);
        let colors = graph_coloring(&g);
        let max_color = *colors.iter().max().unwrap();
        assert!(max_color <= 2, "4-cycle needs at most 2 colors");
        for &(u, v, _) in &g.edges
        {
            assert_ne!(colors[u], colors[v]);
        }
    }

    #[test]
    fn test_select_path() {
        let mut g = Graph::new(10, false);
        g.add_edge(0, 1, None);
        assert_eq!(select_path(&g), PathStrategy::BFS);

        let mut g2 = Graph::new(10, false);
        g2.add_edge(0, 1, Some(5.0));
        g2.add_edge(1, 2, Some(3.0));
        assert_eq!(select_path(&g2), PathStrategy::Dijkstra);
    }

    // ---- 25–28: DP ----

    #[test]
    fn test_fibonacci() {
        assert_eq!(fib_memo(0), 0);
        assert_eq!(fib_memo(1), 1);
        assert_eq!(fib_memo(10), 55);
        assert_eq!(fib_memo(20), 6765);
        assert_eq!(fib_tab(0), 0);
        assert_eq!(fib_tab(1), 1);
        assert_eq!(fib_tab(10), 55);
        assert_eq!(fib_tab(20), 6765);
        for n in 0..30
        {
            assert_eq!(fib_memo(n), fib_tab(n));
        }
    }

    #[test]
    fn test_knapsack() {
        let weights = vec![2, 3, 4, 5];
        let values = vec![3, 4, 5, 6];
        assert_eq!(knapsack(&weights, &values, 5), 7);
    }

    #[test]
    fn test_lcs() {
        assert_eq!(lcs(&[1, 2, 3, 4, 5], &[2, 4, 5, 6]), 3);
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
        assert!(is_dp_suitable(true, true));
        assert!(!is_dp_suitable(false, true));
    }

    // Regression: distance is measured in characters, not bytes. Multi-byte
    // UTF-8 characters must count as a single edit unit.
    #[test]
    fn test_edit_distance_unicode() {
        // Each string is a single character; substituting one for the other
        // is exactly one edit, regardless of their byte lengths.
        assert_eq!(edit_distance("é", "e"), 1); // "é" is 2 bytes, "e" is 1
        assert_eq!(edit_distance("café", "cafe"), 1);
        assert_eq!(edit_distance("naïve", "naive"), 1);
        // Identical multi-byte strings have zero distance.
        assert_eq!(edit_distance("héllo", "héllo"), 0);
        // Emoji (4-byte) counts as one unit.
        assert_eq!(edit_distance("a😀b", "ab"), 1);
    }

    // ---- 29–31: DaC ----

    #[test]
    fn test_dac_merge_sort() {
        let v = test_vec(100, 20);
        let result = dac_merge_sort(&v);
        assert!(check_sorted(&result));
        assert_eq!(result.len(), v.len());
    }

    #[test]
    fn test_max_subarray() {
        assert_eq!(max_subarray(&[-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6);
    }

    #[test]
    fn test_fast_pow_and_master_theorem() {
        assert!((fast_pow(2.0, 0) - 1.0).abs() < 1e-10);
        assert!((fast_pow(2.0, 10) - 1024.0).abs() < 1e-10);
        assert!((fast_pow(3.0, 5) - 243.0).abs() < 1e-10);
        assert_eq!(master_theorem(2.0, 2.0, 1.0), "Θ(n^1.0 log n)");
        assert_eq!(master_theorem(1.0, 2.0, 0.0), "Θ(n^0.0 log n)");
    }

    // ---- 32–33: complexity fitting ----

    #[test]
    fn test_fit_complexity_linear() {
        let sizes = vec![100, 200, 400, 800, 1600];
        let times: Vec<f64> = sizes.iter().map(|&s| s as f64 * 1.0).collect();
        assert_eq!(fit_complexity(&sizes, &times), ComplexityClass::Linear);
    }

    #[test]
    fn test_fit_complexity_quadratic() {
        let sizes = vec![10, 20, 40, 80];
        let times: Vec<f64> = sizes.iter().map(|&s| (s * s) as f64).collect();
        assert_eq!(fit_complexity(&sizes, &times), ComplexityClass::Quadratic);
    }

    #[test]
    fn fit_complexity_classifies_exact_constant() {
        // Flat timings: the Constant model (a fixed cost) fits with zero
        // residual, while every growing model overshoots the small inputs.
        let sizes = vec![10, 20, 40, 80, 160];
        let times = vec![5.0; 5];
        assert_eq!(fit_complexity(&sizes, &times), ComplexityClass::Constant);
    }

    #[test]
    fn fit_complexity_classifies_exact_logarithmic() {
        // times grow exactly as ln(n) — the Logarithmic model's own basis
        // function (and for n ≥ 10 the model's .max(1.0) clamp is inert), so it
        // is a perfect fit; Constant, the only earlier candidate, cannot match
        // a non-flat curve.
        let sizes = vec![10, 20, 40, 80, 160];
        let times: Vec<f64> = sizes.iter().map(|&s| (s as f64).ln()).collect();
        assert_eq!(fit_complexity(&sizes, &times), ComplexityClass::Logarithmic);
    }

    #[test]
    fn fit_complexity_classifies_exact_linearithmic() {
        // times grow exactly as n·ln(n): the Linearithmic model fits with zero
        // residual, while the plain Linear model is off by the ln(n) factor.
        let sizes = vec![10, 20, 40, 80, 160];
        let times: Vec<f64> = sizes
            .iter()
            .map(|&s| {
                let n = s as f64;
                n * n.ln()
            })
            .collect();
        assert_eq!(
            fit_complexity(&sizes, &times),
            ComplexityClass::Linearithmic
        );
    }

    #[test]
    fn fit_complexity_classifies_exact_cubic() {
        // times grow exactly as n³.
        let sizes = vec![10, 20, 40, 80, 160];
        let times: Vec<f64> = sizes
            .iter()
            .map(|&s| {
                let n = s as f64;
                n * n * n
            })
            .collect();
        assert_eq!(fit_complexity(&sizes, &times), ComplexityClass::Cubic);
    }

    // ---- 34–36: comparison / serialisation / misc ----

    #[test]
    fn test_compare_sorts() {
        let data = test_vec(100, 30);
        let strategies = vec![SortStrategy::Quick, SortStrategy::Merge, SortStrategy::Heap];
        let report = compare_sorts(&data, &strategies);
        assert_eq!(report.algorithm, "sort");
        assert_eq!(report.results.len(), 3);
        for r in &report.results
        {
            assert!(r.correct);
        }
        assert!(!report.winner.is_empty());
    }

    #[test]
    fn test_compare_searches() {
        let data = test_vec(100, 31);
        let target = data[data.len() / 2];
        let strategies = vec![SearchStrategy::Linear, SearchStrategy::HashMap];
        let report = compare_searches(&data, target, &strategies);
        assert_eq!(report.algorithm, "search");
        assert_eq!(report.results.len(), 2);
    }

    #[test]
    fn test_statistical_compare() {
        let data = test_vec(80, 32);
        let stats = statistical_compare_sorts(&data, &[SortStrategy::Quick, SortStrategy::Heap], 5);
        assert_eq!(stats.len(), 2);
        assert!(stats[0].1 > 0.0);
        assert!(stats[0].2 >= 0.0);
    }

    #[test]
    fn test_characterize() {
        let data: Vec<i64> = (0..100).collect();
        let c = characterize(&data);
        assert_eq!(c.size, 100);
        assert!(c.sortedness > 0.99);
        assert_eq!(c.range, (0, 99));
        assert!(c.uniqueness > 0.99);
    }

    #[test]
    fn test_characterize_empty() {
        let c = characterize(&[]);
        assert_eq!(c.size, 0);
        assert_eq!(c.sortedness, 1.0);
    }

    #[test]
    fn test_sort_strategy_serde() {
        let s = SortStrategy::Quick;
        let json = serde_json::to_string(&s).unwrap();
        let s2: SortStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(s, s2);
    }

    #[test]
    fn test_graph_serde() {
        let mut g = Graph::new(3, true);
        g.add_edge(0, 1, Some(2.5));
        let json = serde_json::to_string(&g).unwrap();
        let g2: Graph = serde_json::from_str(&json).unwrap();
        assert_eq!(g2.vertices, 3);
        assert_eq!(g2.edges.len(), 1);
    }

    #[test]
    fn test_evolve_sort() {
        let data = test_vec(80, 33);
        let (strategy, _fitness) = evolve_sort(&data, 5, 15);
        assert!(SortStrategy::all().contains(&strategy));
    }

    #[test]
    fn test_benchmark_sort() {
        let data = test_vec(200, 34);
        let t = benchmark_sort(&data, SortStrategy::Quick);
        assert!(t > 0);
    }

    #[test]
    fn test_closest_pair_1d() {
        let pts = vec![5, 2, 8, 1, 9, 3];
        let result = closest_pair_1d(&pts);
        assert!(result.is_some());
        let (_, _, d) = result.unwrap();
        assert_eq!(d, 1);
    }

    #[test]
    fn test_dfs_path() {
        let mut g = Graph::new(4, false);
        g.add_edge(0, 1, None);
        g.add_edge(0, 2, None);
        g.add_edge(2, 3, None);
        let dist = shortest_path(&g, 0, PathStrategy::DFS);
        assert_eq!(dist[0], Some(0.0));
        assert!(dist[3].is_some());
    }

    #[test]
    fn test_floyd_warshall() {
        let mut g = Graph::new(3, false);
        g.add_edge(0, 1, Some(4.0));
        g.add_edge(1, 2, Some(3.0));
        g.add_edge(0, 2, Some(10.0));
        let dist = shortest_path(&g, 0, PathStrategy::FloydWarshall);
        assert_eq!(dist[0], Some(0.0));
        assert_eq!(dist[1], Some(4.0));
        assert_eq!(dist[2], Some(7.0));
    }

    #[test]
    fn test_select_search() {
        let data: Vec<i64> = (0..200).collect();
        let s = select_search(&data, 50, &DataDistribution::Sorted);
        assert_eq!(s, SearchStrategy::Binary);
        let s = select_search(&data, 50, &DataDistribution::Skewed { factor: 2.0 });
        assert_eq!(s, SearchStrategy::HashMap);
    }

    #[test]
    fn measure_complexity_reports_the_requested_size_schedule() {
        // measure_complexity is a wall-clock timing harness, so the *times* it
        // returns are non-deterministic — asserting on their fitted complexity
        // class is flaky (a sum over 80..400 i64s takes tens of nanoseconds, so
        // the measurement is dominated by clock/cache noise and can fit Constant
        // just as well as Linear). We therefore pin only the deterministic part
        // of its contract: the size schedule it samples and the shape of its
        // output. The classification logic itself is exercised exactly by the
        // fit_complexity_classifies_exact_* tests above, which feed synthetic
        // closed-form timings instead of a noisy clock.
        let (sizes, times) = measure_complexity(|d| d.iter().sum::<i64>(), 400, 5);
        // size = max_size * (step + 1) / steps for step in 0..5.
        assert_eq!(sizes, vec![80, 160, 240, 320, 400]);
        assert_eq!(times.len(), 5);
        assert!(times.iter().all(|t| t.is_finite() && *t >= 0.0));
    }
}
