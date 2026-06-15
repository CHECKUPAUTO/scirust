# SciRust Dokumentation 🦀

Willkommen bei der Dokumentation für **SciRust**, ein Framework für Deep Learning und wissenschaftliches Rechnen, das vollständig in **reinem Rust (pure Rust)** geschrieben ist.

## 1. Was ist SciRust?

SciRust ist eine Forschungs- und Entwicklungsplattform für Künstliche Intelligenz. Im Gegensatz zu vielen anderen Werkzeugen (wie PyTorch oder TensorFlow), die auf komplexen C++- oder Python-Bibliotheken basieren, wurde SciRust von Grund auf in Rust entwickelt.

**Warum ist das wichtig?**
- **Vollständige Transparenz**: Sie können jede Zeile des Rechencodes lesen, von der Netzwerkschicht bis zum mathematischen Kernel.
- **Sicherheit und Zuverlässigkeit**: Profitiert von den Speicher- und Sicherheitsgarantien von Rust.
- **Independenz**: Keine komplexen externen Abhängigkeiten (FFI) erforderlich.

## 2. Philosophie und Hauptvorteile

SciRust versucht nicht, die Branchenriesen zu ersetzen, sondern bietet einen anderen Ansatz, der auf **Vertrauen** und **Reproduzierbarkeit** setzt.

### Bit-exakter Determinismus (Bit-for-Bit Determinism)
In vielen Frameworks kann die Ausführung derselben Berechnung zweimal zu leicht unterschiedlichen Ergebnissen führen (aufgrund von Parallelität). SciRust garantiert **bit-exakten Determinismus**: Das Ergebnis ist strikt identisch, unabhängig von der Anzahl der verwendeten Prozessoren. Dies ist entscheidend für die Auditierbarkeit.

### Auditierbarkeit (Auditability)
Da alles in Rust geschrieben ist, lässt sich leicht überprüfen, ob der Code genau das tut, was er verspricht. Es gibt keine Software-"Blackbox".

### Validierungs-Orakel (Validation Oracles)
Jede mathematische Funktion in SciRust wird gegen ein „Validierungs-Orakel“ (eine vertrauenswürdige Referenz) validiert. Wir gehen nicht davon aus, dass das Ergebnis korrekt ist; wir messen es.

## 3. Anwendungsbereiche

SciRust ist besonders nützlich in Bereichen, in denen Präzision, Sicherheit und ein geringer Software-Fußabdruck kritisch sind:

- **Eingebettete Systeme (Edge AI)**: Dank des geringen Platzbedarfs und der Quantisierungsfähigkeiten (Reduzierung der Modellgröße) läuft es perfekt auf kleinen Geräten.
- **Regulierte Sektoren (Luft- und Raumfahrt, Medizin, Finanzen)**: Wo jede KI-Entscheidung aus Sicherheits- oder Compliance-Gründen reproduzierbar und erklärbar sein muss.
- **Wissenschaftliche Forschung**: Zur Entdeckung mathematischer Gesetze aus Daten mittels symbolischer Regression.
- **Sicherheits-Audit**: Für Unternehmen, die ihre gesamte Rechenkette zertifizieren müssen.

## 4. Was Sie erreichen können

SciRust deckt ein breites Spektrum moderner Techniken ab:

- **Deep Learning**: Aufbau neuronaler Netze (MLP, CNN, Transformer) mit automatischer Differenzierung (Autograd).
- **Reinforcement Learning (RL)**: Vollständige Stack-Unterstützung für Tabular Q-Learning, DQN und PPO mit Clipping.
- **Fortgeschrittene Computer Vision**: ResNet-18/34 Architekturen und Vision Transformer (ViT) mit Global Pooling.
- **Generative KI (VAE)**: Variationale Autoencoder mit Reparametrisierungstrick für latente Generierung.
- **Transformer und MoE**: Mixture of Experts-Layer mit Top-k-Routing für Modellskalierbarkeit.
- **Graph Neural Networks (GNN)**: Graph Convolutional Networks (GCN) für strukturierte Daten.
- **Speech AI und Audio**: Audio-Encoder und CTC-Loss-Funktion für Spracherkennung.
- **PEFT-Anpassung (LoRA)**: Low-Rank Adaptation für effizientes Fine-Tuning von vortrainierten Modellen.
- **Fortgeschrittenes wissenschaftliches Rechnen**: 1D-FEM (Finite-Elemente-Methode) Solver für physikalische Gleichungen.
- **Symbolische Regression**: Entdeckung mathematischer Formeln (z. B. `f(x) = sin(x) + x^2`) aus Beobachtungen.
- **Evolutionäre Optimierung**: Verwendung von von der Natur inspirierten Algorithmen (wie NSGA-II) zur Lösung komplexer Probleme.
- **int8-Quantisierung**: Verringerung der Modellgröße um das Vierfache, um auf kleine Prozessoren zu passen, ohne an Genauigkeit zu verlieren.
- **GPU-Beschleunigung**: Nutzung der Leistung von Grafikkarten über WebGPU (wgpu) oder NVIDIA Tensor Cores (cuBLAS).
- **Physics-Informed Neural Networks (PINN)**: Integration von physikalischen Gesetzen (Differenzialgleichungen) direkt in die Loss-Funktion.
- **Formale Invarianten-Verträge**: Mathematische Garantien (Fehlen von NaN/Inf) für kritische Anwendungen.

## 5. Befehlsübersicht

SciRust wird hauptsächlich über das Terminal mit `cargo`, dem Standardwerkzeug von Rust, verwendet.

### Installation
Fügen Sie dies zu Ihrer `Cargo.toml`-Datei hinzu:
```toml
[dependencies]
scirust-core = { path = "..." }
```

### Kompilieren und Testen
- **Projekt prüfen**: `cargo check --workspace`
- **Alle Tests ausführen** (über 250 Tests validieren das Framework): `cargo test --workspace`
- **Im optimierten Modus kompilieren** (empfohlen für KI): `cargo build --release`
- **GPU-Unterstützung aktivieren**: Fügen Sie `--features wgpu` zu Ihren Befehlen hinzu.

### Ausführungsbeispiele
- **MNIST-Training (handschriftliche Ziffern)**:
  ```bash
  cargo run --example mnist_classifier --release
  ```
- **Transformer-Kompressions-Demo**:
  ```bash
  cargo run -p transformer_compress --release
  ```
- **Matrixmultiplikations-Benchmark**:
  ```bash
  cargo run -p scirust-core --example bench_matmul --release
  ```

## 6. Codebeispiel (Schnellstart)

Hier erfahren Sie, wie Sie in wenigen Zeilen ein sehr einfaches Modell erstellen und trainieren:

```rust
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{Sequential, Linear, ReLU, PcgEngine};

fn main() {
    let mut rng = PcgEngine::new(42);

    // Erstellen eines einfachen Modells
    let mut model = Sequential::new()
        .push(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
        .push(ReLU)
        .push(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));

    // Trainingsschleife
    for epoch in 0..100 {
        let tape = Tape::new();
        // ... (Datenladen und Gradientenberechnung)
        println!("Epoche {}: Berechnung läuft...", epoch);
    }
}
```

## 7. scirust-tensor — Tensor-Algebra und Graph-Optimierung

Das Modul `scirust-tensor` führt eine High-Level-Abstraktionsschicht zur Manipulation komplexer Tensoren ein und gewährleistet gleichzeitig maximale Performance durch Graph-Kompilierung.

### Warum scirust-tensor verwenden?
- **Einsum**: Schreiben Sie komplexe Operationen (Multi-Head Attention, Kontraktionen) in einer einzigen, lesbaren Codezeile.
- **Operator Fusion**: Reduzieren Sie den Speicherzugriff, indem Sie Aktivierungen und Biases direkt in die Berechnungs-Kernels integrieren.
- **Garantierter Determinismus**: Wie bei ganz SciRust ist jede Berechnung Bit-für-Bit reproduzierbar.

### Beispiel: Multi-Head Attention
```rust
use scirust_tensor_einsum::einsum;

// Einstein-Signatur für Attention: Batch, Heads, SeqLen, Dim
// (b, h, i, d) , (b, h, j, d) -> (b, h, i, j)
let attention_scores = einsum("bhid,bhjd->bhij", &[&queries, &keys]).unwrap();
```

### Installation
Fügen Sie dies zu Ihrer `Cargo.toml` hinzu:
```toml
[dependencies]
scirust-tensor-core = { path = "scirust-tensor-core" }
scirust-tensor-einsum = { path = "scirust-tensor-einsum" }
```

## 8. Fazit

SciRust ist das Framework der Wahl für alle, die **Verständnis** und **Strenge** über rohe Geschwindigkeit oder die Einfachheit von Python stellen. Es ist ein leistungsstarkes Werkzeug für den Aufbau vertrauenswürdiger KI, von der Forschung bis hin zu eingebetteten Systemen.

---
*Weitere technische Details finden Sie im vollständigen Bericht unter `paper/SciRust-technical-report.md`.*

## 13. Forschung → Funktionen (N-D-Autograd-Erweiterungen)

Das N-D-Autograd-Band trägt nun einen vollständigen Deep-Learning-Stack, jedes
Element gestützt auf eine Forschungsarbeit und einen Test (Gradientenprüfung oder
Orakel). Siehe [`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md) (14/20 fertig).

- **Kausales Decoder-LM**, durchgängig trainiert (Token- + Positions-Embeddings,
  kausale Attention, fusionierte Softmax-Kreuzentropie); lernt eine Sequenz exakt.
- **LLaMA-Schichten**: RMSNorm, SwiGLU, LLaMA-Block, RoPE, gruppierte /
  Multi-Query-Attention (GQA/MQA).
- **Deterministische Optimierer**: Adam, AdamW, Lion, Muon (Newton–Schulz), Schedule-Free, AdEMAMix und SOAP (Adam in Shampoos Eigenbasis).
- **Zertifizierbare KI**: Interval Bound Propagation **und CROWN** (engere
  Schranken durch lineare Relaxation) — *beweisbare* Ausgabeschranken
  und Robustheitszertifikat.
- **Reproduzierbare Reduktionen**, reihenfolgenunabhängig (bit-identisch
  unabhängig von der Thread-Anzahl).
- **Exaktes spekulatives Decoding**; **FlashAttention** (Online-Softmax);
  **DeltaNet** (lineare Aufmerksamkeit mit Delta-Regel);
  **Neural ODE** (Backprop durch einen RK4-Löser).
- **Kompression**: Wanda-Pruning (aktivierungsbewusst), SmoothQuant, GPTQ (int8-Gewichtsquantisierung mit Fehler-Feedback zweiter Ordnung), AWQ (aktivierungsbewusste, suchbasierte int8-Gewichtsquantisierung).

Neue CLI-Befehle:
- `scirust certify [--seed N] [--eps E]` — beweisbare ReLU-MLP-Schranken (IBP **und** CROWN, die engeren Schranken durch lineare Relaxation, nebeneinander).
- `scirust lm [...] [--opt adam|adamw|lion|schedule-free|ademamix|soap]` — trainiert das N-D-Decoder-LM.
- `scirust deltanet [--seed N] [--steps S]` — trainiert eine einköpfige DeltaNet-Schicht (lineare Aufmerksamkeit mit Delta-Regel), um eine Sequenz zu fitten; gibt die MSE-Reduktion aus.
- `scirust conformal [--seed N] [--alpha A]` — konforme Intervalle mit garantierter, verteilungsfreier Überdeckung.
- `scirust gptq [--seed N] [--samples S] [--damp D]` — GPTQ-int8-Gewichtsquantisierung; gibt die Reduktion des Kalibrierungsfehlers gegenüber Round-to-Nearest aus.
- `scirust awq [--seed N] [--samples S] [--grid G]` — AWQ-aktivierungsbewusste int8-Gewichtsquantisierung; gibt den gewählten Skalierungsexponenten und die Reduktion des Kalibrierungsfehlers gegenüber Round-to-Nearest aus.
