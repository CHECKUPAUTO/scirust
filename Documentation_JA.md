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
- **強化学習 (RL)**: Tabular Q-Learning、DQN、およびクリッピング付き PPO のフルスタックサポート。
- **高度なコンピュータビジョン**: ResNet-18/34 アーキテクチャとグローバルプーリング付き Vision Transformer (ViT)。
- **生成 AI (VAE)**: 潜在空間生成のための再パラメータ化トリックを備えた変分オートエンコーダー。
- **Transformer と MoE**: モデルのスケーラビリティのための Top-k ルーティングを備えた Mixture of Experts レイヤー。
- **グラフニューラルネットワーク (GNN)**: 構造化データのためのグラフ畳み込みネットワーク (GCN)。
- **音声 AI とオーディオ**: 音声認識のためのオーディオエンコーダーと CTC 損失関数。
- **PEFT アダプテーション (LoRA)**: 事前学習済みモデルの効率的な微調整のための低ランク適応 (Low-Rank Adaptation)。
- **高度な科学計算**: 物理方程式のための 1D FEM (有限要素法) ソルバー。
- **記号回帰**: 観測データから数学公式（例: `f(x) = sin(x) + x^2`）を発見。
- **進化最適化**: 自然にヒントを得たアルゴリズム（NSGA-II など）を使用して複雑な問題を解決。
- **int8 量子化**: 精度を落とさずにモデルサイズを 4 分の 1 に縮小し、小型プロセッサに適合。
- **GPU 加速**: WebGPU (wgpu) または NVIDIA Tensor Cores (cuBLAS) を介してグラフィックスカードのパワーを活用。
- **AOT (Ahead-Of-Time) コンパイラ**: モデルを不変の Rust ソースコードに直接コンパイルすることで、超深度組み込みターゲットの実行時オーバーヘッドを排除。
- **Soft-Float 行列エンジン**: ソフトウェア定義の固定小数点エミュレーションにより、異なるアーキテクチャ（x86 vs ARM）間で厳密なビット単位の決定論を保証。
- **潜在アクティベーション・ステアリング (RepE)**: 隠れ層のアクティベーションをリアルタイムで遮断・操作し、エージェントの挙動を誘導。
- **量化感知学習 (QAT)**: 低精度シミュレータ（Fake Quantization）と STE (Straight-Through Estimator) を統合し、INT8 デプロイメント向けにモデルを最適化。
- **XAI エンジン (Integrated Gradients)**: 特徴量属性マップを生成し、ネットワークの予測を数学的に説明。

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
- **GPU サポートの有効化**: コマンドに `--features wgpu`を追加します。

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

## 13. 研究 → 機能（N-D 自動微分の拡張）

N-D 自動微分テープは、研究論文とテスト（勾配チェックまたはオラクル）に裏付け
られた完全な深層学習スタックを備えました。
[`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md) を参照（20 件中 14 件完了）。

- **因果デコーダ LM**、エンドツーエンドで学習（トークン＋位置埋め込み、因果
  アテンション、融合 softmax クロスエントロピー）；系列を厳密に過学習できる。
- **LLaMA 系レイヤ**：RMSNorm、SwiGLU、LLaMA ブロック、RoPE、グループ化/
  マルチクエリ注意（GQA/MQA）。
- **決定的オプティマイザ**：Adam、AdamW、Lion、Muon（Newton–Schulz）。
- **認証可能な AI**：区間境界伝播（IBP）——*証明可能な*出力境界とロバスト性証明書。
- **再現可能なリダクション**、順序非依存（スレッド数によらずビット単位で同一）。
- **厳密な投機的デコード**；**FlashAttention**（オンライン softmax）；
  **Neural ODE**（RK4 ソルバを通した逆伝播）。
- **圧縮**：Wanda 枝刈り（活性化考慮）、SmoothQuant。

新しい CLI コマンド：
- `scirust certify [--seed N] [--eps E]` —— ReLU MLP の証明可能な境界（IBP）。
- `scirust lm [...] [--opt adam|adamw|lion]` —— N-D デコーダ LM を学習。
