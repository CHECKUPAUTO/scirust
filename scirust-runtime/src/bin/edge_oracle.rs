// ORACLE : scirust_edge::infer (no_std) doit reproduire QModel::infer (std) bit-pour-bit.
use scirust_runtime::quant::{QLayer, QLinear, QModel};
use scirust_runtime::{fnv_fold_f32, fnv_init};

struct R(u64);
impl R {
    fn n(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

fn lin(r: &mut R, in_f: usize, out_f: usize, s_in: f32, relu: bool) -> QLayer {
    let scales = (0..out_f).map(|_| 0.001 + (r.n() % 100) as f32 / 50000.0).collect();
    let w_q = (0..in_f * out_f).map(|_| ((r.n() % 255) as i32 - 127) as i8).collect();
    let bias_i32 = (0..out_f).map(|_| (r.n() % 2001) as i32 - 1000).collect();
    QLayer::Linear(QLinear { in_f, out_f, s_in, relu_after: relu, scales, w_q, bias_i32 })
}

fn main() {
    let mut r = R(12345);
    let (b, d0, d1, d2) = (8usize, 784usize, 256usize, 10usize);
    let model = QModel { layers: vec![lin(&mut r, d0, d1, 0.02, true), lin(&mut r, d1, d2, 0.03, false)] };
    let bytes = model.to_bytes();
    let input: Vec<f32> = (0..b * d0).map(|_| (r.n() % 2000) as f32 / 1000.0 - 1.0).collect();

    let std_out = model.infer(&input, b);

    let (na, nacc, nout) = scirust_edge::buffer_requirements(&bytes, b).expect("buffer_requirements");
    let mut act_a = vec![0i8; na];
    let mut act_b = vec![0i8; na];
    let mut acc = vec![0i32; nacc];
    let mut out = vec![0.0f32; nout];
    let m = scirust_edge::infer(&bytes, &input, b, &mut act_a, &mut act_b, &mut acc, &mut out).expect("edge infer");
    let edge_out = &out[..m];

    let same = std_out.len() == edge_out.len()
        && std_out.iter().zip(edge_out).all(|(x, y)| x.to_bits() == y.to_bits());

    println!();
    println!("=== ORACLE no_std vs std (QSR1) ===");
    println!("dims        : batch={}  {}->{}->{}   artefact {} o", b, d0, d1, d2, bytes.len());
    println!("fp std      : {:#018x}", fnv_fold_f32(fnv_init(), &std_out));
    println!("fp no_std   : {:#018x}", fnv_fold_f32(fnv_init(), edge_out));
    println!("bit-identique std vs no_std : {}", if same { "OUI" } else { "NON" });
    println!("buffers      : act={} i8 (x2), acc={} i32, out={} f32 (sans allocation cote MCU)", na, nacc, nout);
}
