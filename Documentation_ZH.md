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

## 8. 结论

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
- **精确投机解码**；**FlashAttention**（在线 softmax）；**DeltaNet**（delta 规则线性注意力）；**Mamba**（选择性状态空间 / 选择性扫描）；**神经 ODE**
  （通过 RK4 求解器反向传播）；以及一个将 PDE 残差放入损失的物理信息神经网络（PINN），用于求解边值问题。
- **压缩**：Wanda 剪枝（感知激活）、SmoothQuant、GPTQ（二阶误差反馈的 int8 权重量化）、AWQ（感知激活的基于搜索的 int8 权重量化）。

新增 CLI 命令：
- `scirust certify [--seed N] [--eps E]` —— ReLU MLP 的可证明输出界（IBP **与** CROWN，基于线性松弛的更紧界，并排显示）。
- `scirust lm [...] [--opt adam|adamw|lion|schedule-free|ademamix|soap]` —— 训练 N-D 解码器语言模型。
- `scirust deltanet [--seed N] [--steps S]` —— 训练单头 DeltaNet（delta 规则线性注意力）层以拟合序列；报告 MSE 的降低量。
- `scirust mamba [--seed N] [--steps S]` —— 训练 Mamba 选择性状态空间（S6 扫描）层以拟合序列；报告 MSE 的降低量。
- `scirust conformal [--seed N] [--alpha A]` —— 具有保证覆盖率的保形预测区间（无分布假设）。
- `scirust pinn [--seed N] [--steps S]` —— 物理信息网络；求解 BVP `u''=−u`（损失中含 PDE 残差），并与 `sin x` 对照。
- `scirust gptq [--seed N] [--samples S] [--damp D]` —— GPTQ int8 权重量化；报告相对 round-to-nearest 的校准误差降低量。
- `scirust awq [--seed N] [--samples S] [--grid G]` —— AWQ 感知激活的 int8 权重量化；报告所选的缩放指数以及相对 round-to-nearest 的校准误差降低量。
