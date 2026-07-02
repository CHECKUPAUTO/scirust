# scirust-sciagent

Deterministic small language model for Rust code generation, trained from scratch on real Rust source code.

## Architecture

- **GQA** (Grouped Query Attention) with RoPE positional encoding
- **SwiGLU** feed-forward activations
- **RMSNorm** pre-normalization
- **Tied embeddings** (shared input/output projection)

### Configurations

| Config | Params | Vocab | d_model | Layers | Heads (KV) | Seq Len |
|--------|--------|-------|---------|--------|------------|---------|
| debug  | 106K   | 256   | 64      | 2      | 4 (2)      | 128     |
| small  | 1.6M   | 8192  | 128     | 4      | 4 (2)      | 256     |
| 350M   | 350M   | 32768 | 1024    | 24     | 16 (4)     | 8192    |
| 7B     | 7B     | 32768 | 4096    | 40     | 32 (8)     | 8192    |

## Training

### Data Pipeline

1. **Download crates**: `fetch-crates --count 1000 --output data/`
2. **Fetch HF datasets**: The Stack v2 Rust subset (9 parquet, 2.75GB)
3. **Train tokenizer**: `train-tokenizer --input corpus.txt --vocab-size 8192`
4. **Tokenize and shard**: `collect-data --input dir/ --tokenizer bpe.json --output shards/`
5. **Train**: `sciagent-train --model small --data-dir shards/ --total-steps 2000`

### Pretrained checkpoint (2000 steps, small model)

Location: `/tmp/scirust_small_2k/final/`
Loss: 9.03 → 8.90 (ln 8192 = 9.01 baseline)

### Usage

```bash
# Ask a prompt
cargo run --release --bin sciagent -- --model small \\
  --checkpoint /tmp/scirust_small_2k/final \\
  ask "fn main()" --max-tokens 100 --temperature 0.0

# Interactive chat
cargo run --release --bin sciagent -- --model small \\
  --checkpoint /tmp/scirust_small_2k/final chat

# Model info
cargo run --release --bin sciagent -- --model small \\
  --checkpoint /tmp/scirust_small_2k/final info
```

## Tokenizer

BPE tokenizer trained on 10MB sample of Rust code from crates.io + The Stack v2.
- Vocab: 8192 tokens (8000 learned merges + 192 reserved)
- Includes special tokens: `<pad>`(0), `<bos>`(1), `<eos>`(2), `<unk>`(3)
- Saved as JSON at `tokenizer/bpe.json`

## Project Structure

```
scirust-sciagent/
├── tokenizer/bpe.json      # BPE tokenizer (8192 vocab)
├── src/
│   ├── bin/
│   │   ├── sciagent.rs        # CLI: ask, chat, explain, generate, info
│   │   ├── sciagent-train.rs  # Training binary
│   │   ├── train-tokenizer.rs # BPE trainer
│   │   ├── collect-data.rs    # BPE tokenize + shard
│   │   ├── fetch-crates.rs    # Download crates from crates.io
│   │   └── byte-shard.rs      # Fast byte-level sharding
│   ├── config.rs           # Model architecture configs
│   ├── model.rs            # Model definition (embed, layers, head)
│   ├── block.rs            # Transformer block (attn + FFN)
│   ├── attention.rs        # GQA attention with RoPE + KV cache
│   ├── swiglu.rs           # SwiGLU activation FFN
│   ├── inference.rs        # Forward pass with KV caching
│   ├── generate.rs         # Autoregressive generation
│   ├── bpe.rs              # BPE tokenizer (train + encode + decode)
│   ├── norm.rs             # RMSNorm
│   ├── quantize.rs         # INT4 quantized inference
│   └── train/
│       ├── mod.rs          # Training loop + cross-entropy loss
│       ├── dataset.rs      # ShardLoader + PretrainDataset
│       ├── optimizer.rs    # Muon optimizer
│       ├── scheduler.rs    # Warmup + cosine LR schedule
│       ├── checkpoint.rs   # Save/load safetensors checkpoints
│       └── sft.rs          # SFT fine-tuning
└── tests/integration.rs    # Integration tests
```
