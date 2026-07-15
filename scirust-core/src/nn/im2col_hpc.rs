//! Optimisation du layout mémoire des Convolutions (Chantier 3).

pub fn im2col_hpc(
    input: &[f32],
    c: usize,
    h: usize,
    w: usize,
    k: usize,
    s: usize,
    p: usize,
) -> Vec<f32> {
    // Guard the output-size geometry: s == 0 divides by zero and k > h+2p (or
    // > w+2p) underflows usize. This returns a Vec (not a Result), so assert.
    assert!(s > 0, "im2col_hpc: stride must be > 0");
    assert!(
        k > 0 && k <= h + 2 * p && k <= w + 2 * p,
        "im2col_hpc: kernel {k} too large for input {h}×{w} with pad {p}"
    );
    let ho = (h + 2 * p - k) / s + 1;
    let wo = (w + 2 * p - k) / s + 1;
    let n_cols = ho * wo;
    let mut out = vec![0.0; c * k * k * n_cols];

    for ic in 0..c
    {
        for kh in 0..k
        {
            for kw in 0..k
            {
                let row = (ic * k + kh) * k + kw;
                let row_off = row * n_cols;
                for iho in 0..ho
                {
                    let hi = (iho * s) as isize + kh as isize - p as isize;
                    if hi < 0 || hi >= h as isize
                    {
                        continue;
                    }
                    let in_hi_off = (ic * h + hi as usize) * w;
                    let out_ho_off = row_off + iho * wo;
                    for iwo in 0..wo
                    {
                        let wi = (iwo * s) as isize + kw as isize - p as isize;
                        if wi < 0 || wi >= w as isize
                        {
                            continue;
                        }
                        out[out_ho_off + iwo] = input[in_hi_off + wi as usize];
                    }
                }
            }
        }
    }
    out
}
