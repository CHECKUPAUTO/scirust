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

## 8. 産業・自動車向け監視 (v0.14-dev)

SciRust には、特に自動車分野における**産業生産ラインの監視**のための一連のクレートが含まれるようになりました。

### 8.1 信号処理 (`scirust-signal`)

振動解析と機械診断のための純粋 Rust 信号処理：

- **Radix-2 FFT**（Cooley-Tukey、順方向＋逆方向）
- **窓関数**: Hanning、Hamming、Blackman、Blackman-Harris、Flat-top
- **時間領域特徴**: RMS、波高率、尖度、歪度、ゼロ交差率、自己相関、エネルギー、エントロピー
- **周波数領域特徴**: PSD、スペクトル重心、広がり、スペクトルエントロピー、ロールオフ、帯域パワー、平坦度
- **ベアリング診断**: BPFO、BPFI、BSF、FTF 計算、包絡線スペクトルにおける故障周波数検出
- **次数分析**: 次数追跡、角度リサンプリング、可変速回転機械の次数スペクトル

### 8.2 OPC-UA コネクタ (`scirust-opcua`)

産業用 PLC/SCADA を SciRust パイプラインに接続：

- **`OpcuaClient` トレイト**: 変数読み取り、サブスクリプション、ブラウジングの抽象化
- **`SimulatedOpcuaClient`**: 8 つのシミュレーションセンサー（3 軸振動、モーター/冷却液温度、油圧、モーター電流、冷却液流量）
- **ブリッジ**: OPC-UA 値 → SciRust `EventStream` 変換
- フィーチャーフラグによる実際の OPC-UA スタック（`opcua` クレート）統合準備済み

### 8.3 MQTT パブリッシング (`scirust-mqtt`)

検出されたイベントを MQTT ブローカーに公開し、Industrie 4.0 に対応：

- **`MqttPublisher` トレイト**: パブリッシングの抽象化
- **SparkPlug B 形式**: Industrie 4.0 互換ペイロード
- **重大度**: Info / Warning / Critical（信頼度スコアから導出）
- **`SimulatedMqttPublisher`**: 実際のブローカー不要のテストバックエンド
- **`MonitoringStation`**: ステーション構成

### 8.4 予知保全 (`scirust-pdm`)

産業機械向け予知保全モジュール：

- **ヘルスインデックス**: 複数のセンサー指標を組み合わせた 0..1 スコア、EMA 平滑化、ISO 13374 分類（Good/Degraded/Warning/Critical/Failed）
- **RUL（残存耐用時間）**: 線形および指数推定器、95% 信頼区間
- **変化検出**: レジームシフト検出のための CUSUM（ISO 7870）および Page-Hinkley
- **専用検出器**: `ImbalanceDetector`、`MisalignmentDetector`、`BearingFaultDetector`、`CavitationDetector`

### 8.5 産業用 MLOps (`scirust-mlops`)

継続的な産業展開のための ML オペレーション：

- **ドリフト検出**: Population Stability Index（PSI）によるデータドリフト、相対 MAE によるモデルドリフト
- **シャドウデプロイメント**: 本番/候補モデルの並列実行、Promote/Keep/Inconclusive 推奨
- **署名付き OTA**: 暗号署名と整合性検証による Over-The-Air モデル配布

### 8.6 機能安全 (`scirust-func-safety`)

自動車 AI 向け ISO 26262 / IEC 61508 準拠：

- **ASIL A-D**: 完全性レベル、自動構成（ロックステップ、ウォッチドッグ、最大レイテンシ、冗長性）
- **要件トレーサビリティ**: 要件 → コード → テスト マトリックス、JSON エクスポート、認証レポート
- **故障注入**: 6 種類の故障（ビット反転、スタックアット、ノイズ、ゼロ化、スケールシフト、オーバーフロー）、バッチテスト
- **縮退モード**: 4 レベル（Full → Reduced → Safety → Emergency）、ヒステリシス、セーフステート
- **ハッシュチェーン監査ログ**: 不変の安全意思決定記録、チェーン整合性検証

### 8.7 統合キット (`scirust-integration`)

産業統合を簡素化する統一ライブラリ：

- **`Backend`**: フィーチャーフラグ付き統一 OPC-UA + MQTT 抽象化（`real-opcua`、`real-mqtt`）
- **`BackendFactory`**: 自動作成、シミュレーション → 実モード フォールバック
- **`PipelineConfig`**: 完全な JSON 構成（バックエンド、ステーション、センサー、ヘルスインデックス、RUL、ドリフト）
- **`Pipeline`**: 完全パイプライン Backend → Signal → Events → Health → RUL → MQTT → Audit
- **テンプレート**: プロジェクト生成（`minimal`、`automotive`、`bearing`、`pdm`）

### 8.8 産業用 CLI (`scirust-industrial`)

統合を容易にするコマンドラインツール：

```bash
scirust-industrial discover --simulated                    # 利用可能な PLC センサーを表示
scirust-industrial test-opcua --simulated --samples 5       # OPC-UA 接続テスト
scirust-industrial test-mqtt --simulated                    # MQTT 接続テスト
scirust-industrial gen-config --output config.json --template automotive --stations 3
scirust-industrial scaffold --name line3-monitor --template automotive
scirust-industrial run --config config.json --cycles 100 --report report.json
scirust-industrial doctor --config config.json             # 統合問題の診断
```

### 8.9 完全統合例 (`industrial-monitor`)

`industrial_monitor` サンプルが完全なチェーンを示します：

```
OPC-UA (PLC) → 信号処理 → イベント検出 → ヘルスインデックス
→ RUL 推定 → CUSUM → MQTT 公開 → 監査ログ → 機能安全 → MLOps ドリフト
```

```bash
cargo run -p industrial-monitor
```

## 9. 結論

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
- **決定的オプティマイザ**：Adam、AdamW、Lion、Muon（Newton–Schulz）、Schedule-Free、AdEMAMix、SOAP（Shampoo の固有基底における Adam）。
- **認証可能な AI**：区間境界伝播（IBP）**と CROWN**（線形緩和によるより厳しい境界）——*証明可能な*出力境界とロバスト性証明書。
- **再現可能なリダクション**、順序非依存（スレッド数によらずビット単位で同一）。
- **厳密な投機的デコード**；**FlashAttention**（オンライン softmax）；
  **DeltaNet**（デルタ則線形アテンション）；
  **Mamba**（選択的状態空間 / 選択的スキャン）；
  **RetNet**（リテンション / 線形アテンション）；
  **GLA**（ゲート付き線形アテンション）；
  **HGRN**（ゲート付き線形 RNN）；
  **Neural ODE**（RK4 ソルバを通した逆伝播）；損失に PDE 残差を組み込んで境界値問題を解く物理情報ニューラルネットワーク（PINN）。
- **圧縮**：Wanda 枝刈り（活性化考慮）、SmoothQuant、GPTQ（2 次の誤差フィードバックによる int8 重み量子化）、AWQ（活性化を考慮した探索ベースの int8 重み量子化）。

新しい CLI コマンド：
- `scirust certify [--seed N] [--eps E]` —— ReLU MLP の証明可能な境界（IBP **と** CROWN、線形緩和によるより厳しい境界を並べて表示）。
- `scirust lm [...] [--opt adam|adamw|lion|schedule-free|ademamix|soap|lookahead|lamb|adan|adafactor|shampoo|prodigy]` —— N-D デコーダ LM を学習。
- `scirust deltanet [--seed N] [--steps S]` —— 単一ヘッドの DeltaNet（デルタ則線形アテンション）層を学習して系列に適合させる；MSE の削減量を報告。
- `scirust mamba [--seed N] [--steps S]` —— Mamba 選択的状態空間（S6 スキャン）層を学習して系列に適合させる；MSE の削減量を報告。
- `scirust retnet [--seed N] [--steps S]` —— RetNet リテンション層（線形アテンション、再帰形式 ≡ 並列形式）を学習して系列に適合させる；MSE の削減量を報告。
- `scirust gla [--seed N] [--steps S]` —— ゲート付き線形アテンション GLA 層（データ依存の忘却ゲート）を学習して系列に適合させる；MSE の削減量を報告。
- `scirust hgrn [--seed N] [--steps S]` —— ゲート付き線形 RNN の HGRN トークンミキサ（下限付き忘却ゲート）を学習して系列に適合させる；MSE の削減量を報告。
- `scirust rwkv [--seed N] [--steps S]` —— RWKV 時間混合（WKV）層（チャネルごとの時間減衰＋ボーナス）を学習して系列に適合させる；MSE の削減量を報告。
- `scirust conformal [--seed N] [--alpha A]` —— 分布非依存で被覆率を保証する共形予測区間。
- `scirust calibrate [--seed N]` —— 温度スケーリング；精度を変えずに期待校正誤差（ECE）を下げるよう T を調整。
- `scirust pinn [--seed N] [--steps S]` —— 物理情報ネットワーク；BVP `u''=−u` を解き（損失に PDE 残差）、`sin x` と照合。
- `scirust gptq [--seed N] [--samples S] [--damp D]` —— GPTQ int8 重み量子化；round-to-nearest と比べた校正誤差の削減量を報告。
- `scirust awq [--seed N] [--samples S] [--grid G]` —— AWQ 活性化考慮 int8 重み量子化；選択されたスケーリング指数と、round-to-nearest と比べた校正誤差の削減量を報告。
- `scirust bitnet [--seed N]` —— BitNet b1.58 三値 {-1,0,+1} 重み量子化（約 1.58 ビット/重み）；乗算のない行列積を検証。

## 14. 産業用 CLI — 完全リファレンス

CLI `scirust-industrial` は SciRust と実際の産業システムとの統合を促進します。

### インストール

```bash
cargo install --path scirust-industrial   # `scirust-industrial` バイナリを提供
# またはその場で: cargo run -p scirust-industrial -- <コマンド>
```

### コマンド

| コマンド | 説明 | オプション |
|---------|------|----------|
| `discover` | OPC-UA サーバー上の利用可能なセンサーを一覧表示 | `--endpoint`、`--filter`、`--simulated` |
| `test-opcua` | OPC-UA 接続をテストし値を読み取り | `--endpoint`、`--simulated`、`--samples N` |
| `test-mqtt` | MQTT ブローカー接続をテストしメッセージを公開 | `--host`、`--port`、`--simulated`、`--topic` |
| `gen-config` | パイプライン設定ファイルを生成 | `--output`、`--template`、`--stations N`、`--line-id` |
| `scaffold` | 完全な監視プロジェクトを生成 | `--name`、`--output`、`--template` |
| `run` | 設定から監視パイプラインを実行 | `--config`、`--cycles N`、`--report` |
| `doctor` | 統合問題を診断 | `--config` |

### テンプレート

| テンプレート | 説明 |
|-------------|------|
| `minimal` | 1 ステーション、シミュレーションバックエンド、スパイク検出 |
| `automotive` | ベアリング診断、RUL、MQTT、監査付きマルチステーション自動車ライン |
| `bearing` | ベアリング故障検出（FFT 包絡線、BPFO/BPFI/BSF） |
| `pdm` | 予知保全（ヘルスインデックス、RUL、CUSUM） |

### 推奨統合フロー

```bash
# 1. プロジェクトを作成
scirust-industrial scaffold --name line3-monitor --template automotive

# 2. すべてが動作することを確認
cd line3-monitor
scirust-industrial doctor --config config.json

# 3. 設定をカスタマイズ
# config.json を編集: OPC-UA エンドポイント、MQTT ブローカー、センサー、しきい値

# 4. 実モードに切り替え（オプション）
# Cargo.toml を編集: real-opcua / real-mqtt フィーチャーのコメントを解除
# config.json を編集: backend_type "opcua"

# 5. 監視を開始
scirust-industrial run --config config.json --cycles 1000
```

### シミュレーションモードから実モードへの切り替え

シミュレーションモードはハードウェアなしで動作します。本番に移行するには：

1. **実 OPC-UA**: `Cargo.toml` で `scirust-integration` に `features = ["real-opcua"]` を追加し、依存関係 `opcua = "0.13"` を追加し、`config.json` で `backend_type` を `"opcua"` に変更。
2. **実 MQTT**: `features = ["real-mqtt"]` を追加し、`rumqttc = "0.24"` を追加し、ブローカーの `host`/`port` を設定。

`BackendFactory` が自動フォールバックを処理：実バックエンドが失敗した場合、シミュレーションモードにフォールバックします。

## 15. パターン検出

- **scirust-vision**: コンピュータビジョン — CNN 層、畳み込み、HOG、LBP、Haar、Canny エッジ検出、Otsu 閾値処理、連結成分、NMS
- **scirust-audio**: 音声認識 — MFCC、クロマ特徴、ピッチ追跡 (YIN)、オンセット検出、スペクトル特徴（重心、帯域幅、ロールオフ、平坦度、エントロピー）
- **scirust-graph**: グラフパターン — 部分グラフ同型性、グラフ同型性、モチーフ発見、コミュニティ検出（ラベル伝播、Girvan-Newman）、モジュラリティ、媒介中心性
- **scirust-sequential**: 系列パターン — HMM（前向き/後向き/Viterbi/Baum-Welch）、CRF、系列ラベリング (BIO)、編集距離、DTW、KMP、Boyer-Moore
- **scirust-multivariate**: 多変量解析 — PCA、ICA、K-Means++、マハラノビス距離、MDS、CCA、シルエットスコア
- **scirust-unsupervised**: 教師なし検出 — オートエンコーダ、Isolation Forest、DBSCAN、LOF、GMM（EM アルゴリズム）、One-Class SVM
- **scirust-seasonal**: 季節パターン — STL 分解、ACF/PACF、ピリオドグラム、フーリエ解析、Mann-Kendall 傾向検定、季節 CUSUM
- **scirust-nlp-advanced**: 高度な NLP — NER（ルールベース + 統計的）、LDA トピックモデリング、関係抽出、TextRank、RAKE、MinHash、NaiveBayes、文書類似度

## 16. アルゴリズム作成

- **scirust-automl**: AutoML — ハイパーパラメータ最適化（ランダム/グリッド/ベイズ GP）、t 検定によるモデル選択、アンサンブル（投票/平均化）、特徴量エンジニアリング、交差検証
- **scirust-synthesis**: プログラム合成 — 30 以上の式コンストラクタ、スケッチベース合成、ボトムアップ/トップダウン/GP/ビームサーチ、式書き換え、共通部分式除去
- **scirust-algogen**: アルゴリズム生成 — ソート（10 戦略）、探索（8 戦略）、グラフアルゴリズム（最短経路、全域木、最大流、彩色）、DP、分割統治、Big-O 複雑度解析
- **scirust-codetrans**: コード間変換 — 23 ノード型の AST、パターンマッチングエンジン、20 の最適化ルール（定数畳み込み、DCE、CSE、LICM、強度低減）、リファクタリング、Rust→Python/C トランスパイル
- **scirust-rl-algo**: RL アルゴリズム発見 — ベースライン付き REINFORCE、Actor-Critic、Q-Learning、焼きなまし法、ビームサーチ、漸進的拡大 MCTS、メタ学習、CEGAR 検証
- **scirust-scaffold**: アルゴリズムスキャフォールド — DSL ベースのアルゴリズム記述、コード生成（Rust/Python/C/疑似コード）、16 の組込みテンプレート、スキャフォールドジェネレータ、コード解析、ドキュメント生成
