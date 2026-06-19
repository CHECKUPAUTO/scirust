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

## 8. 산업 및 자동차 모니터링 (v0.14-dev)

SciRust는 이제 특히 자동차 도메인에서 **산업 생산 라인 모니터링**을 위한 크레이트 세트를 포함합니다.

### 8.1 신호 처리 (`scirust-signal`)

진동 분석 및 기계 진단을 위한 순수 Rust 신호 처리:

- **Radix-2 FFT** (Cooley-Tukey, 순방향 + 역방향)
- **윈도우**: Hanning, Hamming, Blackman, Blackman-Harris, Flat-top
- **시간 영역 특징**: RMS, 파고율, 첨도, 왜도, 제로 크로싱율, 자기상관, 에너지, 엔트로피
- **주파수 영역 특징**: PSD, 스펙트럼 중심, 확산, 스펙트럼 엔트로피, 롤오프, 대역 전력, 평탄도
- **베어링 진단**: BPFO, BPFI, BSF, FTF 계산, 포락선 스펙트럼에서의 결함 주파수 감지
- **차수 분석**: 차수 추적, 각도 리샘플링, 가변 속도 회전 기계를 위한 차수 스펙트럼

### 8.2 OPC-UA 커넥터 (`scirust-opcua`)

산업용 PLC/SCADA를 SciRust 파이프라인에 연결:

- **`OpcuaClient` 트레이트**: 변수 읽기, 구독, 브라우징을 위한 추상화
- **`SimulatedOpcuaClient`**: 8개의 시뮬레이션 센서 (3축 진동, 모터/냉각수 온도, 유압, 모터 전류, 냉각수 유량)
- **브리지**: OPC-UA 값 → SciRust `EventStream` 변환
- 기능 플래그를 통한 실제 OPC-UA 스택(`opcua` 크레이트) 통합 준비 완료

### 8.3 MQTT 게시 (`scirust-mqtt`)

감지된 이벤트를 MQTT 브로커에 게시하여 Industrie 4.0 지원:

- **`MqttPublisher` 트레이트**: 게시 추상화
- **SparkPlug B 형식**: Industrie 4.0 호환 페이로드
- **심각도**: Info / Warning / Critical (신뢰도 점수에서 파생)
- **`SimulatedMqttPublisher`**: 실제 브로커가 필요 없는 테스트 백엔드
- **`MonitoringStation`**: 스테이션 구성

### 8.4 예측 유지보수 (`scirust-pdm`)

산업 기계를 위한 예측 유지보수 모듈:

- **건강 지수**: 여러 센서 지표를 결합한 0..1 점수, EMA 평활화, ISO 13374 분류 (Good/Degraded/Warning/Critical/Failed)
- **RUL (잔여 수명)**: 95% 신뢰 구간을 가진 선형 및 지수 추정기
- **변화 감지**: 체제 전환 감지를 위한 CUSUM (ISO 7870) 및 Page-Hinkley
- **전문 감지기**: `ImbalanceDetector`, `MisalignmentDetector`, `BearingFaultDetector`, `CavitationDetector`

### 8.5 산업용 MLOps (`scirust-mlops`)

지속적인 산업 배포를 위한 ML 운영:

- **드리프트 감지**: Population Stability Index(PSI)를 통한 데이터 드리프트, 상대 MAE를 통한 모델 드리프트
- **섀도우 배포**: 프로덕션/후보 모델 병렬 실행, Promote/Keep/Inconclusive 권장
- **서명된 OTA**: 암호화 서명 및 무결성 검증을 통한 Over-The-Air 모델 배포

### 8.6 기능 안전 (`scirust-func-safety`)

자동차 AI용 ISO 26262 / IEC 61508 준수:

- **ASIL A-D**: 무결성 수준, 자동 구성 (lockstep, watchdog, 최대 지연, 중복)
- **요구사항 추적성**: 요구사항 → 코드 → 테스트 매트릭스, JSON 내보내기, 인증 보고서
- **결함 주입**: 6가지 결함 유형 (비트 플립, stuck-at, 노이즈, 제로화, 스케일 이동, 오버플로), 배치 테스트
- **성능 저하 모드**: 4단계 (Full → Reduced → Safety → Emergency), 히스테리시스, 안전 상태
- **해시 체인 감사 로그**: 불변의 안전 결정 저널, 체인 무결성 검증

### 8.7 통합 키트 (`scirust-integration`)

산업 통합을 단순화하는 통합 라이브러리:

- **`Backend`**: 기능 플래그(`real-opcua`, `real-mqtt`)가 있는 통합 OPC-UA + MQTT 추상화
- **`BackendFactory`**: 자동 생성, 시뮬레이션 → 실제 폴백
- **`PipelineConfig`**: 완전한 JSON 구성 (백엔드, 스테이션, 센서, 건강 지수, RUL, 드리프트)
- **`Pipeline`**: 전체 파이프라인 Backend → Signal → Events → Health → RUL → MQTT → Audit
- **템플릿**: 프로젝트 생성 (`minimal`, `automotive`, `bearing`, `pdm`)

### 8.8 산업용 CLI (`scirust-industrial`)

통합을 용이하게 하는 명령줄 도구:

```bash
scirust-industrial discover --simulated                    # 사용 가능한 PLC 센서 검색
scirust-industrial test-opcua --simulated --samples 5       # OPC-UA 연결 테스트
scirust-industrial test-mqtt --simulated                    # MQTT 연결 테스트
scirust-industrial gen-config --output config.json --template automotive --stations 3
scirust-industrial scaffold --name line3-monitor --template automotive
scirust-industrial run --config config.json --cycles 100 --report report.json
scirust-industrial doctor --config config.json             # 통합 문제 진단
```

### 8.9 전체 통합 예제 (`industrial-monitor`)

`industrial_monitor` 예제는 전체 체인을 보여줍니다:

```
OPC-UA (PLC) → 신호 처리 → 이벤트 감지 → 건강 지수
→ RUL 추정 → CUSUM → MQTT 게시 → 감사 로그 → 기능 안전 → MLOps 드리프트
```

```bash
cargo run -p industrial-monitor
```

## 9. 결론

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
- **결정론적 옵티마이저**: Adam, AdamW, Lion, Muon(Newton–Schulz), Schedule-Free, AdEMAMix 및 SOAP(Shampoo의 고유기저에서의 Adam).
- **인증 가능한 AI**: 구간 경계 전파(IBP) **및 CROWN**(선형 완화 기반의 더 조밀한 경계) — *증명 가능한* 출력 경계와 견고성 인증서.
- **재현 가능한 리덕션**, 순서 무관 (스레드 수와 무관하게 비트 단위로 동일).
- **정확한 추측 디코딩**; **FlashAttention**(온라인 softmax); **DeltaNet**(델타 규칙 선형 어텐션); **Mamba**(선택적 상태공간 / 선택적 스캔); **RetNet**(리텐션 / 선형 어텐션); **GLA**(게이트 선형 어텐션); **HGRN**(게이트 선형 RNN); **Neural ODE**
  (RK4 솔버를 통한 역전파); 손실에 PDE 잔차를 넣어 경계값 문제를 푸는 물리 정보 신경망(PINN).
- **압축**: Wanda 가지치기(활성화 인식), SmoothQuant, GPTQ(2차 오차 피드백 기반 int8 가중치 양자화), AWQ(활성화 인식 기반 탐색 방식 int8 가중치 양자화).

새 CLI 명령:
- `scirust certify [--seed N] [--eps E]` — ReLU MLP의 증명 가능한 경계(IBP **및** CROWN, 선형 완화 기반의 더 조밀한 경계를 나란히 표시).
- `scirust lm [...] [--opt adam|adamw|lion|schedule-free|ademamix|soap|lookahead|lamb|adan|adafactor|shampoo|prodigy]` — N-D 디코더 LM 학습.
- `scirust deltanet [--seed N] [--steps S]` — 단일 헤드 DeltaNet(델타 규칙 선형 어텐션) 레이어를 학습하여 시퀀스를 적합; MSE 감소량을 보고.
- `scirust mamba [--seed N] [--steps S]` — Mamba 선택적 상태공간(S6 스캔) 레이어를 학습하여 시퀀스를 적합; MSE 감소량을 보고.
- `scirust retnet [--seed N] [--steps S]` — RetNet 리텐션 레이어(선형 어텐션, 재귀 형식 ≡ 병렬 형식)를 학습하여 시퀀스를 적합; MSE 감소량을 보고.
- `scirust gla [--seed N] [--steps S]` — 게이트 선형 어텐션 GLA 레이어(데이터 의존적 망각 게이트)를 학습하여 시퀀스를 적합; MSE 감소량을 보고.
- `scirust hgrn [--seed N] [--steps S]` — 게이트 선형 RNN HGRN 토큰 믹서(하한이 있는 망각 게이트)를 학습하여 시퀀스를 적합; MSE 감소량을 보고.
- `scirust rwkv [--seed N] [--steps S]` — RWKV 시간 혼합(WKV) 레이어(채널별 시간 감쇠 + 보너스)를 학습하여 시퀀스를 적합; MSE 감소량을 보고.
- `scirust conformal [--seed N] [--alpha A]` — 분포 가정 없이 커버리지를 보장하는 컨포멀 예측 구간.
- `scirust calibrate [--seed N]` — 온도 스케일링; 정확도를 바꾸지 않고 기대 보정 오차(ECE)를 낮추도록 T를 적합.
- `scirust pinn [--seed N] [--steps S]` — 물리 정보 신경망; BVP `u''=−u`를 풀고(손실에 PDE 잔차), `sin x`와 대조.
- `scirust gptq [--seed N] [--samples S] [--damp D]` — GPTQ int8 가중치 양자화; round-to-nearest 대비 보정 오차 감소량을 보고.
- `scirust awq [--seed N] [--samples S] [--grid G]` — AWQ 활성화 인식 int8 가중치 양자화; 선택된 스케일링 지수와 round-to-nearest 대비 보정 오차 감소량을 보고.
- `scirust bitnet [--seed N]` — BitNet b1.58 삼치 {-1,0,+1} 가중치 양자화(약 1.58비트/가중치); 곱셈 없는 행렬곱을 검증.

## 14. 산업용 CLI — 전체 참조

CLI `scirust-industrial`은 SciRust와 실제 산업 시스템의 통합을 촉진합니다.

### 설치

```bash
cargo install --path scirust-industrial   # `scirust-industrial` 바이너리 제공
# 또는 제자리에서: cargo run -p scirust-industrial -- <명령>
```

### 명령

| 명령 | 설명 | 옵션 |
|------|-----|------|
| `discover` | OPC-UA 서버의 사용 가능한 센서 나열 | `--endpoint`, `--filter`, `--simulated` |
| `test-opcua` | OPC-UA 연결 테스트 및 값 읽기 | `--endpoint`, `--simulated`, `--samples N` |
| `test-mqtt` | MQTT 브로커 연결 테스트 및 메시지 게시 | `--host`, `--port`, `--simulated`, `--topic` |
| `gen-config` | 파이프라인 구성 파일 생성 | `--output`, `--template`, `--stations N`, `--line-id` |
| `scaffold` | 완전한 모니터링 프로젝트 생성 | `--name`, `--output`, `--template` |
| `run` | 구성에서 모니터링 파이프라인 실행 | `--config`, `--cycles N`, `--report` |
| `doctor` | 통합 문제 진단 | `--config` |

### 템플릿

| 템플릿 | 설명 |
|--------|------|
| `minimal` | 1개 스테이션, 시뮬레이션 백엔드, 스파이크 감지 |
| `automotive` | 베어링 진단, RUL, MQTT, 감사가 포함된 다중 스테이션 자동차 라인 |
| `bearing` | 베어링 결함 감지 (FFT 포락선, BPFO/BPFI/BSF) |
| `pdm` | 예측 유지보수 (건강 지수, RUL, CUSUM) |

### 권장 통합 흐름

```bash
# 1. 프로젝트 생성
scirust-industrial scaffold --name line3-monitor --template automotive

# 2. 모든 것이 작동하는지 확인
cd line3-monitor
scirust-industrial doctor --config config.json

# 3. 구성 사용자 정의
# config.json 편집: OPC-UA 엔드포인트, MQTT 브로커, 센서, 임계값

# 4. 실제 모드로 전환 (선택 사항)
# Cargo.toml 편집: real-opcua / real-mqtt 기능 주석 해제
# config.json 편집: backend_type "opcua"

# 5. 모니터링 시작
scirust-industrial run --config config.json --cycles 1000
```

### 시뮬레이션에서 실제 모드로 전환

시뮬레이션 모드는 하드웨어 없이 작동합니다. 프로덕션으로 전환하려면:

1. **실제 OPC-UA**: `Cargo.toml`에서 `scirust-integration`에 `features = ["real-opcua"]` 추가, 의존성 `opcua = "0.13"` 추가, `config.json`에서 `backend_type`을 `"opcua"`로 변경.
2. **실제 MQTT**: `features = ["real-mqtt"]` 추가, `rumqttc = "0.24"` 추가, 브로커 `host`/`port` 구성.

`BackendFactory`가 자동 폴백을 처리합니다: 실제 백엔드가 실패하면 시뮬레이션 모드로 폴백합니다.

## 15. 패턴 탐지

- **scirust-vision**: 컴퓨터 비전 — CNN 계층, 합성곱, HOG, LBP, Haar, Canny 에지 검출, Otsu 임계값, 연결 요소, NMS
- **scirust-audio**: 오디오 인식 — MFCC, 크로마 특징, 피치 추적 (YIN), 시작점 검출, 스펙트럼 특징 (중심, 대역폭, 롤오프, 평탄도, 엔트로피)
- **scirust-graph**: 그래프 패턴 — 부분 그래프 동형, 그래프 동형, 모티프 발견, 커뮤니티 검출 (레이블 전파, Girvan-Newman), 모듈성, 매개 중심성
- **scirust-sequential**: 순차 패턴 — HMM (전향/후향/Viterbi/Baum-Welch), CRF, 시퀀스 레이블링 (BIO), 편집 거리, DTW, KMP, Boyer-Moore
- **scirust-multivariate**: 다변량 분석 — PCA, ICA, K-Means++, 마할라노비스 거리, MDS, CCA, 실루엣 점수
- **scirust-unsupervised**: 비지도 탐지 — 오토인코더, Isolation Forest, DBSCAN, LOF, GMM (EM 알고리즘), One-Class SVM
- **scirust-seasonal**: 계절 패턴 — STL 분해, ACF/PACF, 페리오도그램, 푸리에 분석, Mann-Kendall 추세 검정, 계절 CUSUM
- **scirust-nlp-advanced**: 고급 NLP — NER (규칙 기반 + 통계적), LDA 토픽 모델링, 관계 추출, TextRank, RAKE, MinHash, NaiveBayes, 문서 유사도

## 16. 알고리즘 생성

- **scirust-automl**: AutoML — 하이퍼파라미터 최적화 (무작위/격자/베이지안 GP), t-검정 모델 선택, 앙상블 (투표/평균), 특징 공학, 교차 검증
- **scirust-synthesis**: 프로그램 합성 — 30개 이상의 표현식 생성자, 스케치 기반 합성, 상향식/하향식/GP/빔 탐색, 표현식 재작성, 공통 부분식 제거
- **scirust-algogen**: 알고리즘 생성 — 정렬 (10개 전략), 탐색 (8개 전략), 그래프 알고리즘 (최단 경로, 신장 트리, 최대 유량, 색칠), DP, 분할 정복, Big-O 복잡도 분석
- **scirust-codetrans**: 코드 간 변환 — 23개 노드 유형의 AST, 패턴 매칭 엔진, 20개 최적화 규칙 (상수 폴딩, DCE, CSE, LICM, 강도 감소), 리팩터링, Rust→Python/C 트랜스파일
- **scirust-rl-algo**: RL 알고리즘 발견 — 베이스라인 포함 REINFORCE, Actor-Critic, Q-Learning, 시뮬레이티드 어닐링, 빔 탐색, 점진적 확장 MCTS, 메타 학습, CEGAR 검증
- **scirust-scaffold**: 알고리즘 스캐폴드 — DSL 기반 알고리즘 설명, 코드 생성 (Rust/Python/C/의사코드), 16개 내장 템플릿, 스캐폴드 생성기, 코드 분석, 문서 생성
