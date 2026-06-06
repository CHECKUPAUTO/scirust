# SciRust Dokumentation 🦀

Willkommen bei der Dokumentation für **SciRust**, ein Framework für Deep Learning und wissenschaftliches Rechnen, das vollständig in **reinem Rust (pure Rust)** geschrieben ist.

## 1. Was ist SciRust?

SciRust ist eine Forschungs- und Entwicklungsplattform für Künstliche Intelligenz. Im Gegensatz zu vielen anderen Werkzeugen (wie PyTorch oder TensorFlow), die auf komplexen C++- oder Python-Bibliotheken basieren, wurde SciRust von Grund auf in Rust entwickelt.

**Warum ist das wichtig?**
- **Vollständige Transparenz**: Sie können jede Zeile des Rechencodes lesen, von der Netzwerkschicht bis zum mathematischen Kernel.
- **Sicherheit und Zuverlässigkeit**: Profitiert von den Speicher- und Sicherheitsgarantien von Rust.
- **Unabhängigkeit**: Keine komplexen externen Abhängigkeiten (FFI) erforderlich.

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
- **Symbolische Regression**: Entdeckung mathematischer Formeln (z. B. `f(x) = sin(x) + x^2`) aus Beobachtungen.
- **Evolutionäre Optimierung**: Verwendung von von der Natur inspirierten Algorithmen (wie NSGA-II) zur Lösung komplexer Probleme.
- **int8-Quantisierung**: Verringerung der Modellgröße um das Vierfache, um auf kleine Prozessoren zu passen, ohne an Genauigkeit zu verlieren.
- **GPU-Beschleunigung**: Nutzung der Leistung von Grafikkarten über WebGPU (wgpu) oder NVIDIA Tensor Cores (cuBLAS).

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

## 7. Fazit

SciRust ist das Framework der Wahl für alle, die **Verständnis** und **Strenge** über rohe Geschwindigkeit oder die Einfachheit von Python stellen. Es ist ein leistungsstarkes Werkzeug für den Aufbau vertrauenswürdiger KI, von der Forschung bis hin zu eingebetteten Systemen.

---
*Weitere technische Details finden Sie im vollständigen Bericht unter `paper/SciRust-technical-report.md`.*
