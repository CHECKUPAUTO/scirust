use scirust_runtime::proof;
use scirust_runtime::quant::{QLayer, QLinear, QModel};

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
    let scales = (0..out_f)
        .map(|_| 0.001 + (r.n() % 100) as f32 / 50000.0)
        .collect();
    let w_q = (0..in_f * out_f)
        .map(|_| ((r.n() % 255) as i32 - 127) as i8)
        .collect();
    let bias_i32 = (0..out_f).map(|_| (r.n() % 2001) as i32 - 1000).collect();
    QLayer::Linear(QLinear {
        in_f,
        out_f,
        s_in,
        relu_after: relu,
        scales,
        w_q,
        bias_i32,
    })
}
fn show(title: &str, checks: &[(String, bool)]) -> bool {
    println!("{title}");
    let mut all = true;
    for (label, ok) in checks
    {
        println!("  [{}] {}", if *ok { "PASS" } else { "FAIL" }, label);
        all &= *ok;
    }
    println!(
        "  => {}",
        if all
        {
            "BUNDLE VALIDE"
        }
        else
        {
            "BUNDLE REJETE"
        }
    );
    all
}
fn main() {
    let mut r = R(12345);
    let model = QModel {
        layers: vec![
            lin(&mut r, 784, 256, 0.02, true),
            lin(&mut r, 256, 10, 0.03, false),
        ],
    };
    let artifact = model.to_bytes();
    let batch = 8usize;

    let bundle = proof::ProofBundle::build(&artifact, batch, &[1, 2, 3, 4]);
    let text = bundle.to_canonical();
    let path = "target/proof_bundle.txt";
    std::fs::write(path, &text).expect("write bundle");
    println!(
        "=== BUNDLE DE PREUVES ({} octets, ecrit dans {}) ===",
        text.len(),
        path
    );
    print!("{text}");
    println!();

    let ok_clean = show(
        "=== VERIFICATION (artefact authentique) ===",
        &proof::verify_file(&text, &artifact),
    );
    println!();

    let mut bad = artifact.clone();
    let idx = 200usize;
    bad[idx] ^= 0xFF;
    let ok_bad = show(
        &format!(
            "=== VERIFICATION (artefact altere: octet {} retourne) ===",
            idx
        ),
        &proof::verify_file(&text, &bad),
    );
    println!();
    println!(
        "RESUME: authentique={}  altere_rejete={}",
        ok_clean, !ok_bad
    );
}
