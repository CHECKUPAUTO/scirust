# SciRust ドキュメント 🦀

**SciRust** のドキュメントへようこそ。SciRust は、完全に **純粋な Rust (pure Rust)** で書かれたディープラーニングおよび科学計算フレームワークです。

## 1. SciRust とは？

SciRust は、人工知能の研究開発のためのプラットフォームです。複雑な C++ や Python のライブラリ（PyTorch や TensorFlow など）に依存する他の多くのツールとは異なり、SciRust は Rust でゼロから構築されています。

**なぜこれが重要なのか？**
- **完全な透明性**: ネットワーク層から数学カーネルまで、計算コードの全行を読むことができます。
- **安全性と信頼性**: Rust のメモリおよび安全性の保証の恩恵を受けられます。
- **独立性**: 複雑な外部依存関係 (FFI) は必要ありません。

## 2. 哲学と主な利点

SciRust は業界の巨人に取って代わろうとするのではなく、**信頼** と **再現性** に焦点を当てた異なるアプローチを提供します。

### ビット単位の決定論 (Bit-for-Bit Determinism)
多くのフレームワークでは、同じ計算を 2 回実行すると、並列処理などが原因でわずかに異なる結果が得られることがあります。SciRust は **ビット単位の決定論** を保証します。使用するプロセッサの数に関係なく、結果は厳密に同一になります。これは監査可能性にとって極めて重要です。

### 監査可能性 (Auditability)
すべてが Rust で書かれているため、コードが説明通りに動作しているかを確認することが容易です。ソフトウェアの「ブラックボックス」は存在しません。

### 検証オラクル (Validation Oracles)
SciRust のすべての数学関数は、「検証オラクル」（信頼できる参照）に対して検証されています。結果が正しいと仮定するのではなく、実際に測定します。

## 3. 適用分野

SciRust は、精度、セキュリティ、およびソフトウェアの設置面積の小ささが重要な分野で特に役立ちます。

- **組み込みシステム (Edge AI)**: 小さな設置面積と量子化機能（モデルサイズの削減）により、小型デバイスで完璧に動作します。
- **規制分野（航空宇宙、医療、金融）**: 安全性やコンプライアンスの理由から、AI のすべての決定が再現可能で説明可能でなければならない分野です。
- **科学研究**: 記号回帰を通じてデータから数学的法則を発見します。
- **セキュリティ監査**: 計算チェーン全体を認証する必要がある企業向け。

## 4. 実現できること

SciRust は、幅広い最新技術をカバーしています。

- **ディープラーニング**: 自動微分 (autograd) を備えたニューラルネットワーク (MLP、CNN、Transformers) の構築。
- **記号回帰**: 観測データから数学公式（例: `f(x) = sin(x) + x^2`）を発見。
- **進化最適化**: 自然にヒントを得たアルゴリズム（NSGA-II など）を使用して複雑な問題を解決。
- **int8 量子化**: 精度を落とさずにモデルサイズを 4 分の 1 に縮小し、小型プロセッサに適合。
- **GPU 加速**: WebGPU (wgpu) または NVIDIA Tensor Cores (cuBLAS) を介してグラフィックスカードのパワーを活用。
- **物理情報ニューラルネットワーク (PINN)**: 複雑な現象をモデリングするために、物理法則（微分方程式）を損失関数に直接統合します。
- **形式的な不変条件契約**: 重要なアプリケーション（医療、航空宇宙）向けの数学的保証（NaN/Infの欠如、値の境界）。
- **CSRテンソルとSpMMカーネル**: 組み込みターゲット上のスパースモデル向けのメモリと計算の最適化。
- **セキュアエンクレーブ実行 (TEE)**: OSアロケータを使用しない、分離実行（TrustZone/SGX）用の強化された#![no_std]互換ランタイム。

## 5. コマンドガイド

SciRust は主に、Rust の標準ツールである `cargo` を使用してターミナルから操作します。

### インストール
`Cargo.toml` ファイルに以下を追加します。
```toml
[dependencies]
scirust-core = { path = "..." }
```

### コンパイルとテスト
- **プロジェクトのチェック**: `cargo check --workspace`
- **全テストの実行**（250 以上のテストでフレームワークを検証）: `cargo test --workspace`
- **最適化モードでのコンパイル**（AI 推奨）: `cargo build --release`
- **GPU サポートの有効化**: コマンドに `--features wgpu` を追加します。

### 実行例
- **MNIST トレーニング（手書き数字）**:
  ```bash
  cargo run --example mnist_classifier --release
  ```
- **Transformer 圧縮デモ**:
  ```bash
  cargo run -p transformer_compress --release
  ```
- **行列計算ベンチマーク**:
  ```bash
  cargo run -p scirust-core --example bench_matmul --release
  ```

## 6. コード例（クイックスタート）

数行で非常にシンプルなモデルを作成してトレーニングする方法を以下に示します。

```rust
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{Sequential, Linear, ReLU, PcgEngine};

fn main() {
    let mut rng = PcgEngine::new(42);

    // シンプルなモデルの作成
    let mut model = Sequential::new()
        .push(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
        .push(ReLU)
        .push(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));

    // トレーニングループ
    for epoch in 0..100 {
        let tape = Tape::new();
        // ... (データのロードと勾配計算)
        println!("エポック {}: 計算中...", epoch);
    }
}
```

## 7. 結論

SciRust は、生の速度や Python の容易さよりも、**理解** と **厳格さ** を優先する人々にとって最適なフレームワークです。研究から組み込みシステムまで、信頼できる AI を構築するための強力なツールです。

---
*詳細な技術情報については、`paper/SciRust-technical-report.md` のフルレポートを参照してください。*
