# SciRust 文档 🦀

欢迎阅读 **SciRust** 文档，这是一个完全由 **纯 Rust** 编写的深度学习和科学计算框架。

## 1. 什么是 SciRust？

SciRust 是一个用于人工智能研究和开发的平台。与许多其他依赖复杂 C++ 或 Python 库的工具（如 PyTorch 或 TensorFlow）不同，SciRust 是完全从零开始使用 Rust 构建的。

**为什么这很重要？**
- **完全透明**：您可以阅读每一行计算代码，从网络层到数学内核。
- **安全性和可靠性**：受益于 Rust 的内存和安全保障。
- **独立性**：不需要复杂的外部依赖（FFI）。

## 2. 哲学与核心优势

SciRust 并不试图取代行业巨头，而是提供一种专注于 **信任** 和 **可复现性** 的不同方法。

### 位级确定性 (Bit-for-Bit Determinism)
在许多框架中，运行两次相同的计算可能会产生略有不同的结果（由于并行性）。SciRust 保证 **位级确定性**：无论使用多少个处理器，结果都将完全相同。这对于审计性至关重要。

### 可审计性 (Auditability)
由于一切都由 Rust 编写，因此很容易验证代码是否完全按照其说明执行。不存在软件“黑盒”。

### 验证预言机 (Validation Oracles)
SciRust 中的每个数学函数都根据“验证预言机”（受信任的参考）进行验证。我们不假设结果是正确的；我们衡量它。

## 3. 应用领域

SciRust 在精度、安全性和小型软件占用空间至关重要的领域特别有用：

- **嵌入式系统 (Edge AI)**：得益于其低占用空间和量化能力（减小模型大小），它可以在小型设备上完美运行。
- **受监管行业（航空航天、医疗、金融）**：出于安全或合规性原因，每个 AI决策都必须是可复现且可解释的。
- **科学研究**：通过符号回归从数据中发现数学规律。
- **安全审计**：适用于需要认证其整个计算链的公司。

## 4. 您可以实现的目标

SciRust 涵盖了广泛的现代技术：

- **深度学习**：构建具有自动微分 (autograd) 功能的神经网络（MLP、CNN、Transformers）。
- **强化学习 (RL)**: 提供对 Tabular Q-Learning、DQN 和带有 Clipping 的 PPO 的完整栈支持。
- **先进计算机视觉**: 支持 ResNet-18/34 架构和带有全局池化 (Global Pooling) 的 Vision Transformer (ViT)。
- **生成式 AI (VAE)**: 带有用于潜空间生成的重参数化技巧 (Reparameterization Trick) 的变分自编码器。
- **Transformers 和 MoE**: 带有用于模型可扩展性的 Top-k 路由的混合专家 (Mixture of Experts) 层。
- **图神经网络 (GNN)**: 用于结构化数据的图卷积网络 (GCN)。
- **语音 AI 与音频**: 用于语音识别的音频编码器和 CTC 损失函数。
- **PEFT 适配 (LoRA)**: 用于高效微调预训练模型的低秩适配 (Low-Rank Adaptation)。
- **先进科学计算**: 用于物理方程的 1D FEM (有限元法) 求解器。
- **符号回归**：从观察中发现数学公式（例如 `f(x) = sin(x) + x^2`）。
- **进化优化**：使用受自然启发的算法（如 NSGA-II）解决复杂问题。
- **int8 量化**：将模型大小缩小 4 倍，以在不损失精度的情况下适应小型处理器。
- **GPU 加速**：通过 WebGPU (wgpu) 或 NVIDIA Tensor Cores (cuBLAS) 利用显卡的性能。
- **AOT (Ahead-Of-Time) 编译器**：通过将模型直接编译为不可变的 Rust 源代码，消除超深度嵌入式目标的运行时开销。
- **Soft-Float 矩阵引擎**：通过软件定义的定点模拟，保证不同架构（x86 与 ARM）之间严格的位级确定性。
- **潜藏激活引导 (RepE)**：实时拦截和操纵隐藏层激活，以引导智能体行为。
- **量化感知训练 (QAT)**：集成低精度模拟器（伪量化）和 STE（直通估计器），优化 INT8 部署模型。
- **XAI 引擎 (积分梯度)**：生成特征归因图，从数学上解释网络预测。

## 5. 命令指南

SciRust 主要通过终端使用 `cargo`（Rust 的标准工具）运行。

### 安装
在您的 `Cargo.toml` 文件中添加以下内容：
```toml
[dependencies]
scirust-core = { path = "..." }
```

### 编译与测试
- **检查项目**：`cargo check --workspace`
- **运行所有测试**（超过 250 个测试验证框架）：`cargo test --workspace`
- **以优化模式编译**（推荐用于 AI）：`cargo build --release`
- **启用 GPU 支持**：在命令中添加 `--features wgpu`。

### 执行示例
- **MNIST 训练（手写数字）**：
  ```bash
  cargo run --example mnist_classifier --release
  ```
- **Transformer 压缩演示**：
  ```bash
  cargo run -p transformer_compress --release
  ```
- **矩阵乘法基准测试**：
  ```bash
  cargo run -p scirust-core --example bench_matmul --release
  ```

## 6. 代码示例（快速入门）

以下是如何在几行代码中创建并训练一个非常简单的模型：

```rust
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{Sequential, Linear, ReLU, PcgEngine};

fn main() {
    let mut rng = PcgEngine::new(42);

    // 创建一个简单的模型
    let mut model = Sequential::new()
        .push(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
        .push(ReLU)
        .push(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));

    // 训练循环
    for epoch in 0..100 {
        let tape = Tape::new();
        // ... (数据加载和梯度计算)
        println!("轮次 {}: 计算中...", epoch);
    }
}
```

## 7. scirust-tensor — 张量代数与图优化

`scirust-tensor` 模块引入了一个高级抽象层，用于操作复杂的张量，同时通过图编译确保最佳性能。

### 为什么使用 scirust-tensor？
- **Einsum**：仅需一行易读的代码即可编写复杂的运算（如 Multi-Head Attention、张量收缩）。
- **算子融合 (Operator Fusion)**：通过将激活函数和偏置直接合并到计算内核中来减少内存访问。
- **保证确定性**：与 SciRust 的所有组件一样，每次计算都是位对位 (bit-for-bit) 可复现的。

### 示例：多头注意力机制 (Multi-Head Attention)
```rust
use scirust_tensor_einsum::einsum;

// 注意力机制的爱因斯坦求和约定：Batch, Heads, SeqLen, Dim
// (b, h, i, d) , (b, h, j, d) -> (b, h, i, j)
let attention_scores = einsum("bhid,bhjd->bhij", &[&queries, &keys]).unwrap();
```

### 安装
在您的 `Cargo.toml` 中添加以下内容：
```toml
[dependencies]
scirust-tensor-core = { path = "scirust-tensor-core" }
scirust-tensor-einsum = { path = "scirust-tensor-einsum" }
```

## 8. 工业与汽车监控 (v0.14-dev)

SciRust 现包含一组用于**工业生产线监控**的 crate，特别是在汽车领域。

### 8.1 信号处理 (`scirust-signal`)

纯 Rust 信号处理，用于振动分析和机器诊断：

- **Radix-2 FFT**（Cooley-Tukey，正向 + 逆变换）
- **窗函数**：Hanning、Hamming、Blackman、Blackman-Harris、Flat-top
- **时域特征**：RMS、峰值因子、峭度、偏度、过零率、自相关、能量、熵
- **频域特征**：PSD、频谱质心、频谱展宽、频谱熵、滚降、带功率、平坦度
- **轴承诊断**：BPFO、BPFI、BSF、FTF 计算，包络谱中的故障频率检测
- **阶次分析**：阶次跟踪、角度重采样、变速旋转机械的阶次谱

### 8.2 OPC-UA 连接器 (`scirust-opcua`)

将工业 PLC/SCADA 连接到 SciRust 管道：

- **`OpcuaClient` trait**：变量读取、订阅、浏览的抽象
- **`SimulatedOpcuaClient`**：8 个模拟传感器（3 轴振动、电机/冷却液温度、液压压力、电机电流、冷却液流量）
- **桥接**：将 OPC-UA 值转换为 SciRust `EventStream`
- 通过 feature flag 支持真实 OPC-UA 协议栈集成（crate `opcua`）

### 8.3 MQTT 发布 (`scirust-mqtt`)

将检测到的事件发布到 MQTT 代理，用于工业 4.0：

- **`MqttPublisher` trait**：发布抽象
- **SparkPlug B 格式**：工业 4.0 兼容负载
- **严重程度**：Info / Warning / Critical（从置信度得分派生）
- **`SimulatedMqttPublisher`**：无真实代理的测试后端
- **`MonitoringStation`**：站点配置

### 8.4 预测性维护 (`scirust-pdm`)

工业机械的预测性维护模块：

- **健康指数**：0..1 分数，结合多个传感器指标，EMA 平滑，ISO 13374 分类（Good/Degraded/Warning/Critical/Failed）
- **RUL（剩余使用寿命）**：线性和指数估计器，95% 置信区间
- **变化检测**：CUSUM（ISO 7870）和 Page-Hinkley 用于工况切换检测
- **专用检测器**：`ImbalanceDetector`、`MisalignmentDetector`、`BearingFaultDetector`、`CavitationDetector`

### 8.5 工业 MLOps (`scirust-mlops`)

持续工业部署的 ML 操作：

- **漂移检测**：通过 Population Stability Index (PSI) 检测数据漂移，通过相对 MAE 检测模型漂移
- **影子部署**：并行生产/候选模型执行，Promote/Keep/Inconclusive 建议
- **签名 OTA**：带加密签名的空中模型分发和完整性验证

### 8.6 功能安全 (`scirust-func-safety`)

汽车 AI 的 ISO 26262 / IEC 61508 合规：

- **ASIL A-D**：完整性等级，自动配置（lockstep、watchdog、最大延迟、冗余）
- **需求追溯**：需求 → 代码 → 测试矩阵，JSON 导出，认证报告
- **故障注入**：6 种故障类型（位翻转、卡住、噪声、清零、缩放、溢出），批量测试
- **降级模式**：4 级（Full → Reduced → Safety → Emergency），迟滞，安全状态
- **哈希链审计日志**：不可变的安全决策日志，链完整性验证

### 8.7 集成套件 (`scirust-integration`)

简化工业集成的统一库：

- **`Backend`**：统一 OPC-UA + MQTT 抽象，带 feature flag（`real-opcua`、`real-mqtt`）
- **`BackendFactory`**：自动创建，模拟 → 真实回退
- **`PipelineConfig`**：完整 JSON 配置（后端、站点、传感器、健康指数、RUL、漂移）
- **`Pipeline`**：完整管道 Backend → Signal → Events → Health → RUL → MQTT → Audit
- **模板**：项目生成（`minimal`、`automotive`、`bearing`、`pdm`）

### 8.8 工业 CLI (`scirust-industrial`)

命令行工具，简化集成：

```bash
scirust-industrial discover --simulated                    # 浏览可用 PLC 传感器
scirust-industrial test-opcua --simulated --samples 5       # 测试 OPC-UA 连接
scirust-industrial test-mqtt --simulated                    # 测试 MQTT 连接
scirust-industrial gen-config --output config.json --template automotive --stations 3
scirust-industrial scaffold --name line3-monitor --template automotive
scirust-industrial run --config config.json --cycles 100 --report report.json
scirust-industrial doctor --config config.json             # 诊断集成问题
```

### 8.9 完整集成示例 (`industrial-monitor`)

`industrial_monitor` 示例展示了完整链路：

```
OPC-UA (PLC) → 信号处理 → 事件检测 → 健康指数
→ RUL 估计 → CUSUM → MQTT 发布 → 审计日志 → 功能安全 → MLOps 漂移
```

```bash
cargo run -p industrial-monitor
```

## 9. 结论

对于那些将 **理解** 和 **严谨性** 置于原始速度或 Python 的便利性之上的人来说，SciRust 是首选框架。它是构建值得信赖的人工智能（从研究到嵌入式系统）的强大工具。

---
*有关更多技术细节，请参阅 `paper/SciRust-technical-report.md` 中的完整报告。*

## 13. 研究 → 功能（N-D 自动微分扩展）

N-D 自动微分带现在承载了完整的深度学习栈，每个组件都有研究论文与测试支撑
（梯度检查或参考验证）。详见
[`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md)（已完成 14/20）。

- **因果解码器语言模型**，端到端训练（词元 + 位置嵌入、因果注意力、融合的
  softmax 交叉熵）；可精确过拟合一个序列。
- **LLaMA 系列层**：RMSNorm、SwiGLU、LLaMA 块、RoPE、分组/多查询注意力（GQA/MQA）。
- **确定性优化器**：Adam、AdamW、Lion、Muon（Newton–Schulz）、Schedule-Free、AdEMAMix 与 SOAP（Shampoo 特征基中的 Adam）。
- **可认证 AI**：区间界传播（IBP）**与 CROWN**（基于线性松弛的更紧界）——*可证明的*输出界与鲁棒性证书。
- **可复现归约**，与顺序无关（无论线程数多少都按位相同）。
- **精确投机解码**；**FlashAttention**（在线 softmax）；**DeltaNet**（delta 规则线性注意力）；**Mamba**（选择性状态空间 / 选择性扫描）；**RetNet**（保留 / 线性注意力）；**GLA**（门控线性注意力）；**HGRN**（门控线性 RNN）；**神经 ODE**
  （通过 RK4 求解器反向传播）；以及一个将 PDE 残差放入损失的物理信息神经网络（PINN），用于求解边值问题。
- **压缩**：Wanda 剪枝（感知激活）、SmoothQuant、GPTQ（二阶误差反馈的 int8 权重量化）、AWQ（感知激活的基于搜索的 int8 权重量化）。

新增 CLI 命令：
- `scirust certify [--seed N] [--eps E]` —— ReLU MLP 的可证明输出界（IBP **与** CROWN，基于线性松弛的更紧界，并排显示）。
- `scirust lm [...] [--opt adam|adamw|lion|schedule-free|ademamix|soap|lookahead|lamb|adan|adafactor|shampoo|prodigy]` —— 训练 N-D 解码器语言模型。
- `scirust deltanet [--seed N] [--steps S]` —— 训练单头 DeltaNet（delta 规则线性注意力）层以拟合序列；报告 MSE 的降低量。
- `scirust mamba [--seed N] [--steps S]` —— 训练 Mamba 选择性状态空间（S6 扫描）层以拟合序列；报告 MSE 的降低量。
- `scirust retnet [--seed N] [--steps S]` —— 训练 RetNet 保留层（线性注意力，递归形式 ≡ 并行形式）以拟合序列；报告 MSE 的降低量。
- `scirust gla [--seed N] [--steps S]` —— 训练门控线性注意力 GLA 层（数据依赖的遗忘门）以拟合序列；报告 MSE 的降低量。
- `scirust hgrn [--seed N] [--steps S]` —— 训练门控线性 RNN 的 HGRN token 混合器（下界受限的遗忘门）以拟合序列；报告 MSE 的降低量。
- `scirust rwkv [--seed N] [--steps S]` —— 训练 RWKV 时间混合（WKV）层（逐通道时间衰减 + 加成）以拟合序列；报告 MSE 的降低量。
- `scirust conformal [--seed N] [--alpha A]` —— 具有保证覆盖率的保形预测区间（无分布假设）。
- `scirust calibrate [--seed N]` —— 温度缩放；拟合 T 以降低期望校准误差（ECE）而不改变准确率。
- `scirust pinn [--seed N] [--steps S]` —— 物理信息网络；求解 BVP `u''=−u`（损失中含 PDE 残差），并与 `sin x` 对照。
- `scirust gptq [--seed N] [--samples S] [--damp D]` —— GPTQ int8 权重量化；报告相对 round-to-nearest 的校准误差降低量。
- `scirust awq [--seed N] [--samples S] [--grid G]` —— AWQ 感知激活的 int8 权重量化；报告所选的缩放指数以及相对 round-to-nearest 的校准误差降低量。
- `scirust bitnet [--seed N]` —— BitNet b1.58 三值 {-1,0,+1} 权重量化（约 1.58 比特/权重）；验证无乘法的矩阵乘法。

## 14. 工业 CLI — 完整参考

CLI `scirust-industrial` 促进 SciRust 与真实工业系统的集成。

### 安装

```bash
cargo install --path scirust-industrial   # 提供 `scirust-industrial` 二进制文件
# 或就地运行：cargo run -p scirust-industrial -- <command>
```

### 命令

| 命令 | 描述 | 选项 |
|------|------|------|
| `discover` | 列出 OPC-UA 服务器上的可用传感器 | `--endpoint`、`--filter`、`--simulated` |
| `test-opcua` | 测试 OPC-UA 连接并读取值 | `--endpoint`、`--simulated`、`--samples N` |
| `test-mqtt` | 测试 MQTT 代理连接并发布消息 | `--host`、`--port`、`--simulated`、`--topic` |
| `gen-config` | 生成管道配置文件 | `--output`、`--template`、`--stations N`、`--line-id` |
| `scaffold` | 生成完整的监控项目 | `--name`、`--output`、`--template` |
| `run` | 从配置运行监控管道 | `--config`、`--cycles N`、`--report` |
| `doctor` | 诊断集成问题 | `--config` |

### 模板

| 模板 | 描述 |
|------|------|
| `minimal` | 1 站点，模拟后端，尖峰检测 |
| `automotive` | 多站点汽车线，含轴承诊断、RUL、MQTT、审计 |
| `bearing` | 轴承故障检测（FFT 包络、BPFO/BPFI/BSF） |
| `pdm` | 预测性维护（健康指数、RUL、CUSUM） |

### 推荐集成流程

```bash
# 1. 创建项目
scirust-industrial scaffold --name line3-monitor --template automotive

# 2. 验证一切正常
cd line3-monitor
scirust-industrial doctor --config config.json

# 3. 自定义配置
# 编辑 config.json：OPC-UA 端点、MQTT 代理、传感器、阈值

# 4. 切换到真实模式（可选）
# 编辑 Cargo.toml：取消注释 real-opcua / real-mqtt 功能
# 编辑 config.json：backend_type "opcua"

# 5. 开始监控
scirust-industrial run --config config.json --cycles 1000
```

### 从模拟切换到真实模式

模拟模式无需任何硬件。要投入生产：

1. **真实 OPC-UA**：在 `Cargo.toml` 中为 `scirust-integration` 添加 `features = ["real-opcua"]`，添加依赖 `opcua = "0.13"`，并在 `config.json` 中将 `backend_type` 改为 `"opcua"`。
2. **真实 MQTT**：添加 `features = ["real-mqtt"]`，添加 `rumqttc = "0.24"`，并配置代理 `host`/`port`。

`BackendFactory` 自动处理回退：如果真实后端失败，会回退到模拟模式。

## 15. 模式检测

- **scirust-vision**: 计算机视觉 — CNN 层、卷积、HOG、LBP、Haar、Canny 边缘检测、Otsu 阈值分割、连通组件、NMS
- **scirust-audio**: 音频识别 — MFCC、色度特征、音高追踪 (YIN)、起始点检测、频谱特征（质心、带宽、滚降、平坦度、熵）
- **scirust-graph**: 图模式 — 子图同构、图同构、模体发现、社区检测（标签传播、Girvan-Newman）、模块度、介数中心性
- **scirust-sequential**: 序列模式 — HMM（前向/后向/Viterbi/Baum-Welch）、CRF、序列标注 (BIO)、编辑距离、DTW、KMP、Boyer-Moore
- **scirust-multivariate**: 多元分析 — PCA、ICA、K-Means++、马氏距离、MDS、CCA、轮廓系数
- **scirust-unsupervised**: 无监督检测 — 自编码器、隔离森林、DBSCAN、LOF、GMM（EM 算法）、One-Class SVM
- **scirust-seasonal**: 季节性模式 — STL 分解、ACF/PACF、周期图、傅里叶分析、Mann-Kendall 趋势检验、季节性 CUSUM
- **scirust-nlp-advanced**: 高级 NLP — NER（基于规则 + 统计）、LDA 主题建模、关系抽取、TextRank、RAKE、MinHash、NaiveBayes、文档相似度

## 16. 算法创建

- **scirust-automl**: AutoML — 超参数优化（随机/网格/贝叶斯 GP）、基于 t 检验的模型选择、集成方法（投票/平均）、特征工程、交叉验证
- **scirust-synthesis**: 程序合成 — 30 余种表达式构造器、基于草图的合成、自底向上/自顶向下/GP/束搜索、表达式重写、公共子表达式消除
- **scirust-algogen**: 算法生成 — 排序（10 种策略）、搜索（8 种策略）、图算法（最短路径、生成树、最大流、着色）、动态规划、分治法、大 O 复杂度分析
- **scirust-codetrans**: 代码到代码转换 — 含 23 种节点类型的 AST、模式匹配引擎、20 条优化规则（常量折叠、DCE、CSE、LICM、强度削减）、重构、Rust→Python/C 转译
- **scirust-rl-algo**: RL 算法发现 — REINFORCE 含基线、Actor-Critic、Q-Learning、模拟退火、束搜索、渐进扩展 MCTS、元学习、CEGAR 验证
- **scirust-scaffold**: 算法脚手架 — 基于 DSL 的算法描述、代码生成（Rust/Python/C/伪代码）、16 个内置模板、脚手架生成器、代码分析、文档生成
