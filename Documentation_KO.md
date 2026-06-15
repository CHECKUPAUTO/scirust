# SciRust 문서  Korean 🦀

**SciRust** 문서에 오신 것을 환영합니다. SciRust는 완전히 **순수 Rust(pure Rust)**로 작성된 딥러닝 및 과학 계산 프레임워크입니다.

## 1. SciRust란 무엇인가요?

SciRust는 인공지능 연구 및 개발을 위한 플랫폼입니다. 복잡한 C++ 또는 Python 라이브러리(예: PyTorch 또는 TensorFlow)에 의존하는 다른 많은 도구와 달리, SciRust는 Rust를 사용하여 기초부터 구축되었습니다.

**왜 이것이 중요한가요?**
- **완전한 투명성**: 네트워크 계층부터 수학 커널까지 모든 계산 코드를 읽을 수 있습니다.
- **보안 및 신뢰성**: Rust의 메모리 및 안전 보장 혜택을 누릴 수 있습니다.
- **독립성**: 복잡한 외부 종속성(FFI)이 필요하지 않습니다.

## 2. 철학 및 주요 장점

SciRust는 업계의 거대 기업을 대체하려는 것이 아니라, **신뢰**와 **재현성**에 초점을 맞춘 다른 접근 방식을 제공합니다.

### 비트 단위 결정론 (Bit-for-Bit Determinism)
많은 프레임워크에서 동일한 계산을 두 번 실행하면 병렬 처리 등으로 인해 약간 다른 결과가 나올 수 있습니다. SciRust는 **비트 단위 결정론**을 보장합니다. 사용하는 프로세서 수에 관계없이 결과는 엄격하게 동일합니다. 이는 감사 가능성에 있어 매우 중요합니다.

### 감사 가능성 (Auditability)
모든 것이 Rust로 작성되었기 때문에 코드가 설명된 대로 정확히 수행되는지 확인하기 쉽습니다. 소프트웨어 "블랙박스"가 존재하지 않습니다.

### 검증 오라클 (Validation Oracles)
SciRust의 모든 수학 함수는 "검증 오라클"(신뢰할 수 있는 참조)을 기준으로 검증됩니다. 결과가 정확하다고 가정하는 대신 실제로 측정합니다.

## 3. 응용 분야

SciRust는 정밀도, 보안 및 작은 소프트웨어 설치 공간이 중요한 분야에서 특히 유용합니다.

- **임베디드 시스템 (Edge AI)**: 작은 설치 공간과 양자화 기능(모델 크기 감소) 덕분에 소형 기기에서 완벽하게 작동합니다.
- **규제 분야 (항공우주, 의료, 금융)**: 안전 또는 규정 준수 이유로 모든 AI 결정이 재현 가능하고 설명 가능해야 하는 분야입니다.
- **과학 연구**: 기호 회귀를 통해 데이터에서 수학적 법칙을 발견합니다.
- **보안 감사**: 전체 계산 체인을 인증해야 하는 기업용.

## 4. 실현 가능한 것들

SciRust는 광범위한 현대 기술을 다룹니다.

- **딥러닝**: 자동 미분(autograd) 기능을 갖춘 신경망(MLP, CNN, Transformers) 구축.
- **강화 학습 (RL)**: Tabular Q-Learning, DQN 및 Clipping이 포함된 PPO에 대한 전체 스택 지원.
- **고급 컴퓨터 비전**: ResNet-18/34 아키텍처 및 글로벌 풀링(Global Pooling)이 포함된 Vision Transformer (ViT).
- **생성형 AI (VAE)**: 잠재 공간 생성을 위한 재매개변수화 트릭(Reparameterization Trick)이 포함된 변분 오토인코더.
- **Transformers 및 MoE**: 모델 확장성을 위한 Top-k 라우팅이 포함된 Mixture of Experts 레이어.
- **그래프 신경망 (GNN)**: 구조화된 데이터를 위한 그래프 컨볼루션 네트워크(GCN).
- **음성 AI 및 오디오**: 음성 인식을 위한 오디오 인코더 및 CTC 손실 함수.
- **PEFT 적응 (LoRA)**: 사전 훈련된 모델의 효율적인 미세 조정을 위한 저순위 적응(Low-Rank Adaptation).
- **고급 과학 계산**: 물리 방정식을 위한 1D FEM(유한요소법) 솔버.
- **기호 회귀**: 관측 데이터에서 수학 공식(예: `f(x) = sin(x) + x^2`) 발견.
- **진화 최적화**: 자연에서 영감을 얻은 알고리즘(예: NSGA-II)을 사용하여 복잡한 문제 해결.
- **int8 양자화**: 정밀도 손실 없이 모델 크기를 4배로 줄여 소형 프로세서에 적합하게 만듦.
- **GPU 가속**: WebGPU (wgpu) 또는 NVIDIA Tensor Cores (cuBLAS)를 통해 그래픽 카드의 성능 활용.
- **AOT (Ahead-Of-Time) 컴파일러**: 모델을 불변 Rust 소스 코드로 직접 컴파일하여 초심층 임베디드 타겟의 런타임 오버헤드를 제거합니다.
- **Soft-Float 행렬 엔진**: 소프트웨어 정의 고정 소수점 에뮬레이션을 통해 서로 다른 아키텍처(x86 대 ARM) 간의 엄격한 비트 단위 결정론을 보장합니다.
- **잠재 활성화 제어 (RepE)**: 실시간으로 히든 레이어 활성화를 가로채고 조작하여 에이전트 동작을 유도합니다.
- **양자화 인식 훈련 (QAT)**: 저정밀도 시뮬레이터(Fake Quantization)와 STE (Straight-Through Estimator)를 통합하여 INT8 배포용 모델을 최적화합니다.
- **XAI 엔진 (Integrated Gradients)**: 특징 기여도 맵을 생성하여 네트워크 예측을 수학적으로 설명합니다.

## 5. 명령 가이드

SciRust는 주로 Rust의 표준 도구인 `cargo`를 사용하여 터미널에서 조작합니다.

### 설치
`Cargo.toml` 파일에 다음을 추가하세요:
```toml
[dependencies]
scirust-core = { path = "..." }
```

### 컴파일 및 테스트
- **프로젝트 확인**: `cargo check --workspace`
- **모든 테스트 실행** (250개 이상의 테스트로 프레임워크 검증): `cargo test --workspace`
- **최적화 모드로 컴파일** (AI 권장): `cargo build --release`
- **GPU 지원 활성화**: 명령에 `--features wgpu`를 추가하세요.

### 실행 예시
- **MNIST 훈련 (손글씨 숫자)**:
  ```bash
  cargo run --example mnist_classifier --release
  ```
- **Transformer 압축 데모**:
  ```bash
  cargo run -p transformer_compress --release
  ```
- **행렬 곱셈 벤치마크**:
  ```bash
  cargo run -p scirust-core --example bench_matmul --release
  ```

## 6. 코드 예시 (퀵 스타트)

몇 줄의 코드로 매우 간단한 모델을 만들고 훈련하는 방법은 다음과 같습니다:

```rust
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{Sequential, Linear, ReLU, PcgEngine};

fn main() {
    let mut rng = PcgEngine::new(42);

    // 간단한 모델 생성
    let mut model = Sequential::new()
        .push(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
        .push(ReLU)
        .push(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));

    // 훈련 루프
    for epoch in 0..100 {
        let tape = Tape::new();
        // ... (데이터 로드 및 경사 계산)
        println!("에포크 {}: 계산 중...", epoch);
    }
}
```

## 7. 결론

SciRust는 가공되지 않은 속도나 Python의 편리함보다 **이해**와 **엄격함**을 우선시하는 사람들을 위한 최고의 프레임워크입니다. 연구에서 임베디드 시스템에 이르기까지 신뢰할 수 있는 AI를 구축하기 위한 강력한 도구입니다.

---
*자세한 기술 정보는 `paper/SciRust-technical-report.md`의 전체 보고서를 참조하세요.*

## 13. 연구 → 기능 (N-D 자동미분 확장)

N-D 자동미분 테이프는 이제 완전한 딥러닝 스택을 갖추었으며, 각 구성 요소는
연구 논문과 테스트(그래디언트 체크 또는 오라클)로 뒷받침됩니다.
[`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md) 참조 (20개 중 14개 완료).

- **인과적 디코더 LM**, 엔드투엔드 학습 (토큰 + 위치 임베딩, 인과적 어텐션,
  융합 softmax 교차 엔트로피); 시퀀스를 정확히 과적합.
- **LLaMA 계열 레이어**: RMSNorm, SwiGLU, LLaMA 블록, RoPE, 그룹/
  멀티쿼리 어텐션(GQA/MQA).
- **결정론적 옵티마이저**: Adam, AdamW, Lion, Muon(Newton–Schulz).
- **인증 가능한 AI**: 구간 경계 전파(IBP) **및 CROWN**(선형 완화 기반의 더 조밀한 경계) — *증명 가능한* 출력 경계와 견고성 인증서.
- **재현 가능한 리덕션**, 순서 무관 (스레드 수와 무관하게 비트 단위로 동일).
- **정확한 추측 디코딩**; **FlashAttention**(온라인 softmax); **Neural ODE**
  (RK4 솔버를 통한 역전파).
- **압축**: Wanda 가지치기(활성화 인식), SmoothQuant, GPTQ(2차 오차 피드백 기반 int8 가중치 양자화).

새 CLI 명령:
- `scirust certify [--seed N] [--eps E]` — ReLU MLP의 증명 가능한 경계(IBP **및** CROWN, 선형 완화 기반의 더 조밀한 경계를 나란히 표시).
- `scirust lm [...] [--opt adam|adamw|lion|schedule-free|ademamix]` — N-D 디코더 LM 학습.
- `scirust conformal [--seed N] [--alpha A]` — 분포 가정 없이 커버리지를 보장하는 컨포멀 예측 구간.
- `scirust gptq [--seed N] [--samples S] [--damp D]` — GPTQ int8 가중치 양자화; round-to-nearest 대비 보정 오차 감소량을 보고.
