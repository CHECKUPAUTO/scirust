use std::path::Path;

pub struct PretrainDataset {
    data: Vec<u32>,
    position: usize,
    seq_len: usize,
    vocab_size: usize,
}

impl PretrainDataset {
    pub fn from_slice(data: &[u32], seq_len: usize, vocab_size: usize) -> Self {
        Self {
            data: data.to_vec(),
            position: 0,
            seq_len,
            vocab_size,
        }
    }

    pub fn len(&self) -> usize {
        if self.data.len() <= self.seq_len + 1
        {
            0
        }
        else
        {
            (self.data.len() - 1) / self.seq_len
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reset(&mut self) {
        self.position = 0;
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Option<(Vec<usize>, Vec<usize>)> {
        let total_needed = batch_size * (self.seq_len + 1);
        if self.position + total_needed > self.data.len()
        {
            self.position = 0;
            if self.data.len() < total_needed
            {
                return None;
            }
        }

        let mut inputs = Vec::with_capacity(batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(batch_size * self.seq_len);

        for _ in 0..batch_size
        {
            let start = self.position;
            for j in 0..self.seq_len
            {
                let tok = self.data[start + j] as usize;
                inputs.push(tok.min(self.vocab_size - 1));
                let tgt = self.data[start + j + 1] as usize;
                targets.push(tgt.min(self.vocab_size - 1));
            }
            self.position += self.seq_len;
        }

        Some((inputs, targets))
    }

    pub fn shuffle(&mut self, seed: u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let n = self.len();
        if n <= 1
        {
            return;
        }
        let mut indices: Vec<usize> = (0..n).collect();

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let mut state = hasher.finish();
        for i in (1..n).rev()
        {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }

        let mut new_data = Vec::with_capacity(self.data.len());
        for &idx in &indices
        {
            let start = idx * self.seq_len;
            let end = (start + self.seq_len + 1).min(self.data.len());
            new_data.extend_from_slice(&self.data[start..end]);
        }
        self.data = new_data;
        self.position = 0;
    }
}

pub struct ShardLoader {
    buffer: Vec<u32>,
}

impl Default for ShardLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ShardLoader {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn load_bin<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {
        let bytes = std::fs::read(path.as_ref())?;
        let mut data = vec![0u32; bytes.len() / 4];
        for (i, chunk) in bytes.chunks_exact(4).enumerate()
        {
            data[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        self.buffer = data;
        Ok(())
    }

    pub fn load_dir<P: AsRef<Path>>(&mut self, dir: P) -> std::io::Result<()> {
        let mut all_data = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(dir.as_ref())?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in &entries
        {
            let bytes = std::fs::read(entry.path())?;
            for chunk in bytes.chunks_exact(4)
            {
                all_data.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        self.buffer = all_data;
        Ok(())
    }

    pub fn total_tokens(&self) -> usize {
        self.buffer.len()
    }

    pub fn into_dataset(self, seq_len: usize, vocab_size: usize) -> PretrainDataset {
        PretrainDataset::from_slice(&self.buffer, seq_len, vocab_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_basic() {
        let data: Vec<u32> = (0..20).collect();
        let mut ds = PretrainDataset::from_slice(&data, 4, 100);
        assert!(!ds.is_empty());
        let (inputs, targets) = ds.next_batch(1).unwrap();
        assert_eq!(inputs, vec![0, 1, 2, 3]);
        assert_eq!(targets, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_dataset_wraps_around() {
        let data: Vec<u32> = (0..10).collect();
        let mut ds = PretrainDataset::from_slice(&data, 4, 100);
        let _ = ds.next_batch(2);
        assert!(ds.next_batch(2).is_some() || ds.position == 0);
    }

    #[test]
    fn test_shuffle_changes_order() {
        let data: Vec<u32> = (0..50).collect();
        let mut ds1 = PretrainDataset::from_slice(&data, 5, 100);
        let mut ds2 = PretrainDataset::from_slice(&data, 5, 100);
        ds2.shuffle(12345);
        let b1 = ds1.next_batch(1).unwrap();
        let b2 = ds2.next_batch(1).unwrap();
        assert_ne!(b1.0, b2.0, "Shuffle should reorder data");
    }

    #[test]
    fn test_shuffle_deterministic() {
        let data: Vec<u32> = (0..50).collect();
        let mut ds1 = PretrainDataset::from_slice(&data, 5, 100);
        let mut ds2 = PretrainDataset::from_slice(&data, 5, 100);
        ds1.shuffle(42);
        ds2.shuffle(42);
        let b1 = ds1.next_batch(2).unwrap();
        let b2 = ds2.next_batch(2).unwrap();
        assert_eq!(b1.0, b2.0, "Same seed should produce same shuffle");
        assert_eq!(b1.1, b2.1);
    }

    #[test]
    fn test_clamp_vocab() {
        let data = vec![0u32, 1, 2, 200, 4];
        let mut ds = PretrainDataset::from_slice(&data, 4, 10);
        let (inputs, targets) = ds.next_batch(1).unwrap();
        assert_eq!(inputs[3], 9, "Token 200 should be clamped to vocab_size-1");
        assert_eq!(targets[3], 4, "Target 4 should be unchanged");
    }
}
