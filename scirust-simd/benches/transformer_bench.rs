// scirust-simd/benches/transformer_bench.rs
//
// Benchmarks criterion du bloc Transformer décodeur pre-norm virgule fixe
// (`fixed::transformer::TransformerBlock`), comparé à une baseline flottante
// `f32` naïve (référence non déterministe, même structure que
// `crate::transformer` mais sans dispatch AVX-512/GEMM tuilé — celui-ci est
// gardé par la feature optionnelle `transformer-inference`, non requise ici).
//
// * `transformer_prefill_32x64_h4_dff256` : débit du **préremplissage par
//   lot** (`TransformerBlock::forward`) sur une séquence de 32 tokens.
// * `transformer_decode_32_steps` : débit du **décodage incrémental**
//   (`TransformerBlock::forward_decode` + `KvCache`, un token à la fois).
//
// L'objectif est de situer le coût relatif du chemin quantifié déterministe
// face au flottant, pas de « battre » ce dernier : la virgule fixe apporte le
// déterminisme bit-à-bit (cf. `decode_matches_prefill_bit_exact` dans les
// tests), à un coût qui doit rester raisonnable.
//
// Lancement (cible AVX2 pour éviter la sur-détection AVX-512 en VM) :
//   RUSTFLAGS="-C target-cpu=x86-64-v3" \
//     cargo bench -p scirust-simd --features portable-simd --bench transformer_bench

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use scirust_simd::fixed::Q16_16;
use scirust_simd::fixed::kv_cache::KvCache;
use scirust_simd::fixed::layer::Linear;
use scirust_simd::fixed::transformer::TransformerBlock;

/// Séquence de 32 tokens, `d_model = 64`, `4` têtes, `d_ff = 256` : taille
/// type d'un petit bloc décodeur embarqué/edge.
const S: usize = 32;
const D: usize = 64;
const H: usize = 4;
const DFF: usize = 256;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn fixed_data(seed: u64, len: usize, scale: f64) -> Vec<Q16_16> {
    let mut rng = Lcg(seed);
    (0..len)
        .map(|_| Q16_16::try_from(rng.unit() * scale).unwrap())
        .collect()
}
fn f32_data(seed: u64, len: usize, scale: f32) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..len).map(|_| rng.unit() as f32 * scale).collect()
}

/// Construit un [`TransformerBlock`] avec des poids déterministes modérés
/// (évite tout débordement/quantification dégénérée en virgule fixe à
/// travers la profondeur de la composition).
fn build_fixed_block() -> TransformerBlock<16> {
    let zero = vec![Q16_16::zero(); D];
    TransformerBlock::new(
        D,
        H,
        DFF,
        Linear::new(fixed_data(0x10, D * D, 0.05), zero.clone(), D, D),
        Linear::new(fixed_data(0x11, D * D, 0.05), zero.clone(), D, D),
        Linear::new(fixed_data(0x12, D * D, 0.05), zero.clone(), D, D),
        Linear::new(fixed_data(0x13, D * D, 0.05), zero.clone(), D, D),
        Linear::new(
            fixed_data(0x14, D * DFF, 0.05),
            fixed_data(0x15, DFF, 0.05),
            DFF,
            D,
        ),
        Linear::new(fixed_data(0x16, DFF * D, 0.05), zero.clone(), D, DFF),
        vec![Q16_16::try_from(1.0).unwrap(); D],
        vec![Q16_16::try_from(1.0).unwrap(); D],
        Q16_16::try_from(1e-3).unwrap(),
        Q16_16::try_from(10000.0).unwrap(),
        true,
    )
}

// ------------------------------------------------------------------ //
//  Référence f32 naïve, indépendante (mêmes conventions que TransformerBlock)
// ------------------------------------------------------------------ //

/// `y = W·x + b`, `W` : `out×in` row-major (même convention que
/// [`Linear`]).
fn naive_linear_f32(
    x: &[f32],
    rows: usize,
    in_f: usize,
    w: &[f32],
    out_f: usize,
    b: &[f32],
) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out_f];
    for r in 0..rows
    {
        for o in 0..out_f
        {
            let mut acc = b[o];
            for i in 0..in_f
            {
                acc += x[r * in_f + i] * w[o * in_f + i];
            }
            y[r * out_f + o] = acc;
        }
    }
    y
}

fn naive_rmsnorm_f32(x: &[f32], rows: usize, d: usize, gamma: &[f32], eps: f32) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * d];
    for r in 0..rows
    {
        let row = &x[r * d..r * d + d];
        let ms: f32 = row.iter().map(|&v| v * v).sum::<f32>() / d as f32;
        let rms = (ms + eps).sqrt();
        for i in 0..d
        {
            y[r * d + i] = row[i] / rms * gamma[i];
        }
    }
    y
}

/// RoPE par tête (référence f32, mêmes conventions que
/// `fixed::transformer::rope_apply_heads`), en place.
fn naive_rope_heads_f32(
    x: &mut [f32],
    s: usize,
    h: usize,
    dh: usize,
    base: f32,
    pos_offset: usize,
) {
    let dm = h * dh;
    let half = dh / 2;
    for r in 0..s
    {
        let pos = (pos_offset + r) as f32;
        for hh in 0..h
        {
            let off = r * dm + hh * dh;
            for i in 0..half
            {
                let theta = base.powf(-2.0 * i as f32 / dh as f32);
                let (sn, cs) = (pos * theta).sin_cos();
                let a = x[off + 2 * i];
                let b = x[off + 2 * i + 1];
                x[off + 2 * i] = a * cs - b * sn;
                x[off + 2 * i + 1] = a * sn + b * cs;
            }
        }
    }
}

fn naive_silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Attention causale multi-tête sur tout le préfixe `0..=i` de chaque
/// requête `i` (référence f32, une passe en bloc sur toute la séquence).
fn naive_mha_causal_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    s: usize,
    h: usize,
    dh: usize,
) -> Vec<f32> {
    let dm = h * dh;
    let scale = 1.0 / (dh as f32).sqrt();
    let mut out = vec![0.0f32; s * dm];
    for hh in 0..h
    {
        let off = hh * dh;
        for i in 0..s
        {
            let mut row = vec![0.0f32; i + 1];
            for (j, r) in row.iter_mut().enumerate()
            {
                let mut acc = 0.0f32;
                for e in 0..dh
                {
                    acc += q[i * dm + off + e] * k[j * dm + off + e];
                }
                *r = scale * acc;
            }
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for r in row.iter_mut()
            {
                *r = (*r - m).exp();
                sum += *r;
            }
            for e in 0..dh
            {
                let mut acc = 0.0f32;
                for (j, &p) in row.iter().enumerate()
                {
                    acc += p * v[j * dm + off + e];
                }
                out[i * dm + off + e] = acc / sum;
            }
        }
    }
    out
}

/// Attention causale d'une seule requête (position `pos`) sur tout
/// l'historique `k_hist`/`v_hist` (`(pos+1)×dm`) déjà accumulé — référence
/// naïve du pas de décodage incrémental (pas de cache dédié, tout
/// l'historique est recalculé, comme un `Vec` de K/V à croissance simple).
fn naive_mha_decode_step_f32(
    q: &[f32],
    k_hist: &[f32],
    v_hist: &[f32],
    t: usize,
    h: usize,
    dh: usize,
) -> Vec<f32> {
    let dm = h * dh;
    let scale = 1.0 / (dh as f32).sqrt();
    let mut out = vec![0.0f32; dm];
    for hh in 0..h
    {
        let off = hh * dh;
        let mut row = vec![0.0f32; t];
        for (j, rr) in row.iter_mut().enumerate()
        {
            let mut acc = 0.0f32;
            for e in 0..dh
            {
                acc += q[off + e] * k_hist[j * dm + off + e];
            }
            *rr = scale * acc;
        }
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for r in row.iter_mut()
        {
            *r = (*r - m).exp();
            sum += *r;
        }
        for e in 0..dh
        {
            let mut acc = 0.0f32;
            for (j, &p) in row.iter().enumerate()
            {
                acc += p * v_hist[j * dm + off + e];
            }
            out[off + e] = acc / sum;
        }
    }
    out
}

struct NaiveWeights {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    norm1: Vec<f32>,
    norm2: Vec<f32>,
}

fn build_naive_weights() -> NaiveWeights {
    NaiveWeights {
        wq: f32_data(0x10, D * D, 0.05),
        wk: f32_data(0x11, D * D, 0.05),
        wv: f32_data(0x12, D * D, 0.05),
        wo: f32_data(0x13, D * D, 0.05),
        w1: f32_data(0x14, D * DFF, 0.05),
        b1: f32_data(0x15, DFF, 0.05),
        w2: f32_data(0x16, DFF * D, 0.05),
        norm1: vec![1.0f32; D],
        norm2: vec![1.0f32; D],
    }
}

const EPS_F32: f32 = 1e-3;
const BASE_F32: f32 = 10000.0;

/// Bloc décodeur, référence f32 naïve, **préremplissage par lot** (mêmes
/// étapes/ordre que `TransformerBlock::forward`).
fn naive_transformer_forward(x0: &[f32], s: usize, h: usize, w: &NaiveWeights) -> Vec<f32> {
    let dh = D / h;
    let zero_d = vec![0.0f32; D];
    let mut x = x0.to_vec();

    let hn = naive_rmsnorm_f32(&x, s, D, &w.norm1, EPS_F32);
    let mut q = naive_linear_f32(&hn, s, D, &w.wq, D, &zero_d);
    let mut k = naive_linear_f32(&hn, s, D, &w.wk, D, &zero_d);
    let v = naive_linear_f32(&hn, s, D, &w.wv, D, &zero_d);
    naive_rope_heads_f32(&mut q, s, h, dh, BASE_F32, 0);
    naive_rope_heads_f32(&mut k, s, h, dh, BASE_F32, 0);
    let attn = naive_mha_causal_f32(&q, &k, &v, s, h, dh);
    let o = naive_linear_f32(&attn, s, D, &w.wo, D, &zero_d);
    for i in 0..s * D
    {
        x[i] += o[i];
    }

    let hn2 = naive_rmsnorm_f32(&x, s, D, &w.norm2, EPS_F32);
    let mut f1 = naive_linear_f32(&hn2, s, D, &w.w1, DFF, &w.b1);
    for v in f1.iter_mut()
    {
        *v = naive_silu_f32(*v);
    }
    let f2 = naive_linear_f32(&f1, s, DFF, &w.w2, D, &zero_d);
    for i in 0..s * D
    {
        x[i] += f2[i];
    }
    x
}

/// Un pas de décodage incrémental, référence f32 naïve : normalise/projette
/// le token courant, l'ajoute à l'historique `k_hist`/`v_hist`, attend sur
/// tout l'historique, applique le FFN.
#[allow(clippy::too_many_arguments)]
fn naive_transformer_decode_step(
    x_t: &mut [f32],
    pos: usize,
    h: usize,
    w: &NaiveWeights,
    k_hist: &mut Vec<f32>,
    v_hist: &mut Vec<f32>,
) {
    let dh = D / h;
    let zero_d = vec![0.0f32; D];

    let hn = naive_rmsnorm_f32(x_t, 1, D, &w.norm1, EPS_F32);
    let mut q = naive_linear_f32(&hn, 1, D, &w.wq, D, &zero_d);
    let mut k = naive_linear_f32(&hn, 1, D, &w.wk, D, &zero_d);
    let v = naive_linear_f32(&hn, 1, D, &w.wv, D, &zero_d);
    naive_rope_heads_f32(&mut q, 1, h, dh, BASE_F32, pos);
    naive_rope_heads_f32(&mut k, 1, h, dh, BASE_F32, pos);

    k_hist.extend_from_slice(&k);
    v_hist.extend_from_slice(&v);
    let attn = naive_mha_decode_step_f32(&q, k_hist, v_hist, pos + 1, h, dh);

    let o = naive_linear_f32(&attn, 1, D, &w.wo, D, &zero_d);
    for (xi, oi) in x_t.iter_mut().zip(&o)
    {
        *xi += *oi;
    }

    let hn2 = naive_rmsnorm_f32(x_t, 1, D, &w.norm2, EPS_F32);
    let mut f1 = naive_linear_f32(&hn2, 1, D, &w.w1, DFF, &w.b1);
    for v in f1.iter_mut()
    {
        *v = naive_silu_f32(*v);
    }
    let f2 = naive_linear_f32(&f1, 1, DFF, &w.w2, D, &zero_d);
    for (xi, fi) in x_t.iter_mut().zip(&f2)
    {
        *xi += *fi;
    }
}

fn bench_transformer_prefill(c: &mut Criterion) {
    let block = build_fixed_block();
    let x0 = fixed_data(0x1, S * D, 0.5);
    let w = build_naive_weights();
    let fx0 = f32_data(0x1, S * D, 0.5);

    let mut g = c.benchmark_group("transformer_prefill_32x64_h4_dff256");
    g.throughput(Throughput::Elements((S * D) as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter_batched(
            || x0.clone(),
            |mut buf| {
                block
                    .forward(black_box(&mut buf), S)
                    .expect("rmsnorm bien défini");
                buf
            },
            criterion::BatchSize::SmallInput,
        )
    });
    g.bench_function(BenchmarkId::new("f32", "naive"), |bch| {
        bch.iter(|| naive_transformer_forward(black_box(&fx0), S, H, black_box(&w)))
    });
    g.finish();
}

fn bench_transformer_decode(c: &mut Criterion) {
    let block = build_fixed_block();
    let x0 = fixed_data(0x2, S * D, 0.5);
    let w = build_naive_weights();
    let fx0 = f32_data(0x2, S * D, 0.5);

    let mut g = c.benchmark_group("transformer_decode_32_steps");
    g.throughput(Throughput::Elements(S as u64));
    g.bench_function(BenchmarkId::new("fixed", "Q16_16"), |bch| {
        bch.iter(|| {
            let mut cache: KvCache<16> = KvCache::new(S, D);
            for t in 0..S
            {
                let mut row = x0[t * D..t * D + D].to_vec();
                block
                    .forward_decode(black_box(&mut row), t, &mut cache)
                    .expect("rmsnorm bien défini");
            }
        })
    });
    g.bench_function(BenchmarkId::new("f32", "naive"), |bch| {
        bch.iter(|| {
            let mut k_hist = Vec::with_capacity(S * D);
            let mut v_hist = Vec::with_capacity(S * D);
            for t in 0..S
            {
                let mut row = fx0[t * D..t * D + D].to_vec();
                naive_transformer_decode_step(
                    black_box(&mut row),
                    t,
                    H,
                    black_box(&w),
                    &mut k_hist,
                    &mut v_hist,
                );
            }
        })
    });
    g.finish();
}

criterion_group!(benches, bench_transformer_prefill, bench_transformer_decode);
criterion_main!(benches);
