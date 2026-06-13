pub fn broadcast_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    // Simple same-size broadcast
    assert_eq!(
        a.len(),
        b.len(),
        "broadcast_add requires same-length slices"
    );
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
pub fn broadcast_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(
        a.len(),
        b.len(),
        "broadcast_mul requires same-length slices"
    );
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}
pub fn unbroadcast(grad: &[f32], shape: &[usize]) -> Vec<f32> {
    let total: usize = shape.iter().product();
    assert_eq!(grad.len(), total, "unbroadcast shape mismatch");
    grad.to_vec()
}
pub fn broadcast_get(data: &[f32], shape: &[usize], idx: &[usize]) -> f32 {
    assert_eq!(idx.len(), shape.len(), "broadcast_get index count mismatch");
    let mut flat_idx = 0usize;
    let mut stride = 1usize;
    for (&s, &ix) in shape.iter().zip(idx).rev()
    {
        flat_idx += ix * stride;
        stride *= s;
    }
    data[flat_idx]
}
