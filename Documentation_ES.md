# Documentación SciRust 🦀

Bienvenido a la documentación de **SciRust**, un framework de aprendizaje profundo (Deep Learning) y computación científica escrito íntegramente en **Rust puro**.

## 1. ¿Qué es SciRust?

SciRust es una plataforma de investigación y desarrollo para la inteligencia artificial. A diferencia de muchas otras herramientas (como PyTorch o TensorFlow) que se basan en bibliotecas complejas en C++ o Python, SciRust está construido desde cero en Rust.

**¿Por qué es esto importante?**
- **Transparencia total**: Puedes leer cada línea de código del cálculo, desde la capa de red hasta el núcleo matemático.
- **Seguridad y Fiabilidad**: Aprovecha las garantías de memoria y seguridad de Rust.
- **Independencia**: No se requieren dependencias externas complejas (FFI).

## 2. Filosofía y Ventajas Clave

SciRust no intenta reemplazar a los gigantes de la industria, sino que ofrece un enfoque diferente centrado en la **confianza** y la **reproducibilidad**.

### Determinismo Bit a Bit
En muchos frameworks, ejecutar el mismo cálculo dos veces puede dar resultados ligeramente diferentes (debido al paralelismo). SciRust garantiza un **determinismo bit a bit**: el resultado será estrictamente idéntico, sin importar el número de procesadores utilizados. Esto es crucial para la auditabilidad.

### Auditabilidad
Como todo está en Rust, es fácil verificar que el código hace exactamente lo que dice. No hay "cajas negras" de software.

### Oracles de Validación
Cada función matemática en SciRust se valida contra un "oracle" (una referencia de confianza). No asumimos que el resultado es correcto, lo medimos.

## 3. Dominios de Aplicación

SciRust es particularmente útil en áreas donde la precisión, la seguridad y el pequeño tamaño del software son críticos:

- **Sistemas Embebidos (Edge AI)**: Gracias a su baja huella y capacidad de cuantización (reducción del tamaño del modelo), funciona perfectamente en dispositivos pequeños.
- **Sectores Regulados (Aeroespacial, Médico, Finanzas)**: Donde cada decisión de la IA debe ser reproducible y explicable por razones de seguridad o cumplimiento.
- **Investigación Científica**: Para descubrir leyes matemáticas a partir de datos mediante regresión simbólica.
- **Auditoría de Seguridad**: Para empresas que necesitan certificar toda su cadena de cálculo.

## 4. Qué se puede lograr

SciRust cubre una amplia gama de técnicas modernas:

- **Aprendizaje Profundo (Deep Learning)**: Construcción de redes neuronales (MLP, CNN, Transformers) con diferenciación automática (autograd).
- **Aprendizaje por Refuerzo (RL)**: Soporte completo para Tabular Q-Learning, DQN y PPO con clipping.
- **Visión Artificial Avanzada**: Arquitecturas ResNet-18/34 y Vision Transformer (ViT) con pooling global.
- **IA Generativa (VAE)**: Auto-encodeadores variacionales con truco de reparametrización para generación latente.
- **Transformers y MoE**: Capas de Mixture of Experts con enrutamiento Top-k para escalabilidad de modelos.
- **Redes Neuronales de Grafos (GNN)**: Graph Convolutional Networks (GCN) para datos estructurados.
- **Speech AI y Audio**: Codificadores de audio y función de pérdida CTC para reconocimiento de voz.
- **Adaptación PEFT (LoRA)**: Low-Rank Adaptation para un ajuste eficiente de modelos pre-entrenados.
- **Computación Científica Avanzada**: Solucionador FEM (Método de Elementos Finitos) 1D para ecuaciones físicas.
- **Regresión Simbólica**: Descubrimiento de fórmulas matemáticas (ej: `f(x) = sin(x) + x^2`) a partir de observaciones.
- **Optimización Evolutiva**: Uso de algoritmos inspirados en la naturaleza (como NSGA-II) para resolver problemas complejos.
- **Cuantización int8**: Dividir por 4 el tamaño de los modelos para que quepan en procesadores pequeños sin perder precisión.
- **Aceleración GPU**: Uso de la potencia de las tarjetas gráficas mediante WebGPU (wgpu) o NVIDIA Tensor Cores (cuBLAS).
- **Physics-Informed Neural Networks (PINN)**: Integración de leyes físicas (ecuaciones diferenciales) directamente en la función de pérdida.
- **Contratos de Invariantes Formales**: Garantías matemáticas (ausencia de NaN/Inf) para aplicaciones críticas.

## 5. Guía de Comandos

SciRust se utiliza principalmente a través de la terminal con `cargo`, la herramienta estándar de Rust.

### Instalación
Añade esto a tu archivo `Cargo.toml`:
```toml
[dependencies]
scirust-core = { path = "..." }
```

### Compilar y Probar
- **Verificar el proyecto**: `cargo check --workspace`
- **Ejecutar todas las pruebas**: `cargo test --workspace`
- **Compilar en modo optimizado**: `cargo build --release`
- **Activar soporte GPU**: Añade `--features wgpu` a tus comandos.

## 6. Ejemplo de Código

```rust
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::{Sequential, Linear, ReLU, PcgEngine};

fn main() {
    let mut rng = PcgEngine::new(42);

    let mut model = Sequential::new()
        .push(Linear::new(2, 8, &KaimingNormal, &Zeros, &mut rng))
        .push(ReLU)
        .push(Linear::new(8, 2, &KaimingNormal, &Zeros, &mut rng));

    for epoch in 0..100 {
        let tape = Tape::new();
        // ...
        println!("Época {}: cálculo en curso...", epoch);
    }
}
```

## 7. scirust-tensor — Álgebra Tensorial y Optimización de Grafos

El módulo `scirust-tensor` introduce una capa de abstracción de alto nivel para manipular tensores complejos garantizando el máximo rendimiento mediante la compilación de grafos.

### ¿Por qué usar scirust-tensor?
- **Einsum**: Escribe operaciones complejas (Multi-Head Attention, contracciones) en una sola línea de código legible.
- **Fusión de Operadores**: Reduce los accesos a memoria fusionando activaciones y sesgos directamente en los kernels de cálculo.
- **Determinismo Garantizado**: Como todo en SciRust, cada cálculo es reproducible bit a bit.

### Ejemplo: Multi-Head Attention
```rust
use scirust_tensor_einsum::einsum;

// Firma de Einstein para la atención: Batch, Heads, SeqLen, Dim
// (b, h, i, d) , (b, h, j, d) -> (b, h, i, j)
let attention_scores = einsum("bhid,bhjd->bhij", &[&queries, &keys]).unwrap();
```

### Instalación
Añade esto a tu `Cargo.toml`:
```toml
[dependencies]
scirust-tensor-core = { path = "scirust-tensor-core" }
scirust-tensor-einsum = { path = "scirust-tensor-einsum" }
```

## 8. Monitorización Industrial y Automoción (v0.14-dev)

SciRust ahora incluye un conjunto de crates para la **monitorización de líneas de producción industrial**, particularmente en el dominio automotriz.

### 8.1 Procesamiento de Señales (`scirust-signal`)

Procesamiento de señales en Rust puro para análisis vibratorio y diagnóstico de máquinas:

- **FFT radix-2** (Cooley-Tukey, directa + inversa)
- **Ventanas**: Hanning, Hamming, Blackman, Blackman-Harris, Flat-top
- **Características temporales**: RMS, factor de cresta, kurtosis, asimetría, tasa de cruces por cero, autocorrelación, energía, entropía
- **Características espectrales**: PSD, centroide espectral, dispersión, entropía espectral, rolloff, potencia de banda, flatness
- **Diagnóstico de rodamientos**: BPFO, BPFI, BSF, FTF con detección de fallos en el espectro de envolvente
- **Análisis de orden**: seguimiento de órdenes, remuestreo angular, espectro de orden para máquinas de velocidad variable

### 8.2 Conector OPC-UA (`scirust-opcua`)

Conecta PLCs/SCADA industriales al pipeline de SciRust:

- **Trait `OpcuaClient`**: abstracción para lectura de variables, suscripción, navegación
- **`SimulatedOpcuaClient`**: 8 sensores simulados (vibración 3 ejes, temperatura motor/refrigerante, presión hidráulica, corriente motor, caudal refrigerante)
- **Bridge**: convierte valores OPC-UA → `EventStream` de SciRust
- Listo para integración de stack OPC-UA real (crate `opcua`) mediante feature flags

### 8.3 Publicación MQTT (`scirust-mqtt`)

Publica eventos detectados a brokers MQTT para Industria 4.0:

- **Trait `MqttPublisher`**: abstracción de publicación
- **Formato SparkPlug B**: payloads compatibles con Industria 4.0
- **Severidad**: Info / Warning / Critical (derivada del score de confianza)
- **`SimulatedMqttPublisher`**: backend de prueba sin broker real
- **`MonitoringStation`**: configuración de estación

### 8.4 Mantenimiento Predictivo (`scirust-pdm`)

Módulos de mantenimiento predictivo para maquinaria industrial:

- **Health Index**: score 0..1 que combina múltiples indicadores de sensores, con suavizado EMA y clasificación ISO 13374 (Good/Degraded/Warning/Critical/Failed)
- **RUL (Remaining Useful Life)**: estimadores lineal y exponencial con intervalos de confianza del 95%
- **Detección de cambios**: CUSUM (ISO 7870) y Page-Hinkley para detección de cambio de régimen
- **Detectores especializados**: `ImbalanceDetector`, `MisalignmentDetector`, `BearingFaultDetector`, `CavitationDetector`

### 8.5 MLOps Industrial (`scirust-mlops`)

Operaciones ML para despliegue industrial continuo:

- **Detección de deriva**: Data drift vía Population Stability Index (PSI), Model drift vía MAE relativo
- **Shadow deployment**: ejecución paralela de modelo producción/candidato, recomendación Promote/Keep/Inconclusive
- **OTA firmado**: distribución de modelos Over-The-Air con firma criptográfica y verificación de integridad

### 8.6 Seguridad Funcional (`scirust-func-safety`)

Cumplimiento ISO 26262 / IEC 61508 para IA automotriz:

- **ASIL A-D**: niveles de integridad, auto-configuración (lockstep, watchdog, latencia máx, redundancia)
- **Trazabilidad de requisitos**: matriz requisitos → código → tests, exportación JSON, informe de certificación
- **Inyección de fallos**: 6 tipos de fallos (bit-flip, stuck-at, ruido, zero-out, scale-shift, overflow), pruebas por lotes
- **Modo degradado**: 4 niveles (Full → Reduced → Safety → Emergency), histéresis, safe state
- **Audit log con cadena hash**: diario inmutable de decisiones de seguridad, verificación de integridad de cadena

### 8.7 Kit de Integración (`scirust-integration`)

Librería unificadora para simplificar la integración industrial:

- **`Backend`**: abstracción unificada OPC-UA + MQTT con feature flags (`real-opcua`, `real-mqtt`)
- **`BackendFactory`**: creación automática, fallback simulado → real
- **`PipelineConfig`**: configuración JSON completa (backend, estaciones, sensores, Health Index, RUL, deriva)
- **`Pipeline`**: pipeline completo Backend → Signal → Events → Health → RUL → MQTT → Audit
- **Plantillas**: generación de proyectos (`minimal`, `automotive`, `bearing`, `pdm`)

### 8.8 CLI Industrial (`scirust-industrial`)

Herramienta de línea de comandos para facilitar la integración:

```bash
scirust-industrial discover --simulated                    # Explorar sensores PLC disponibles
scirust-industrial test-opcua --simulated --samples 5       # Probar conexión OPC-UA
scirust-industrial test-mqtt --simulated                    # Probar conexión MQTT
scirust-industrial gen-config --output config.json --template automotive --stations 3
scirust-industrial scaffold --name line3-monitor --template automotive
scirust-industrial run --config config.json --cycles 100 --report report.json
scirust-industrial doctor --config config.json             # Diagnosticar problemas de integración
```

### 8.9 Ejemplo de Integración Completa (`industrial-monitor`)

El ejemplo `industrial_monitor` demuestra la cadena completa:

```
OPC-UA (PLC) → Procesamiento de Señales → Detección de Eventos → Health Index
→ Estimación RUL → CUSUM → Publicación MQTT → Audit Log → Seguridad Funcional → MLOps Drift
```

```bash
cargo run -p industrial-monitor
```

## 9. Conclusión

SciRust es el framework de elección para quienes priorizan la **comprensión** y el **rigor** sobre la velocidad bruta o la facilidad de Python. Es una herramienta poderosa para construir una IA de confianza, desde la investigación hasta el entorno embebido.

## 13. Investigación → Funciones (extensiones del grafo N-D)

El grafo de autodiferenciación N-D ahora incluye una pila completa de aprendizaje
profundo, cada pieza respaldada por un artículo de investigación y una prueba
(comprobación de gradiente u oráculo). Véase
[`docs/RESEARCH_ROADMAP.md`](docs/RESEARCH_ROADMAP.md) (14/20 entregados).

- **LM decodificador causal**, entrenado de extremo a extremo (embeddings de
  token + posición, atención causal, entropía cruzada con softmax fusionado);
  memoriza una secuencia exactamente.
- **Capas estilo LLaMA**: RMSNorm, SwiGLU, bloque LLaMA, RoPE, atención agrupada /
  multi-consulta (GQA/MQA).
- **Optimizadores deterministas**: Adam, AdamW, Lion, Muon (Newton–Schulz), Schedule-Free, AdEMAMix y SOAP (Adam en la base propia de Shampoo).
- **IA certificable**: Interval Bound Propagation **y CROWN** (cotas más
  ajustadas por relajación lineal) — cotas de salida *demostrables*
  y certificado de robustez.
- **Reducciones reproducibles**, independientes del orden (bit a bit idénticas sin
  importar el número de hilos).
- **Decodificación especulativa exacta**; **FlashAttention** (softmax en línea);
  **DeltaNet** (atención lineal con regla delta);
  **Mamba** (espacio de estados selectivo / escaneo selectivo);
  **RetNet** (retención / atención lineal);
  **GLA** (atención lineal con compuerta);
  **HGRN** (RNN lineal con compuerta);
  **Neural ODE** (retropropagación a través de un solucionador RK4); una red neuronal informada por la física (PINN) que resuelve un problema de valores en la frontera con el residuo de la EDP en la función de pérdida.
- **Compresión**: poda Wanda (consciente de activaciones), SmoothQuant, GPTQ (cuantización int8 de pesos por retroalimentación de error de segundo orden), AWQ (cuantización int8 de pesos basada en búsqueda y consciente de activaciones).

Nuevos comandos CLI:
- `scirust certify [--seed N] [--eps E]` — cotas demostrables de un MLP ReLU (IBP **y** CROWN, las cotas más ajustadas por relajación lineal, en paralelo).
- `scirust lm [...] [--opt adam|adamw|lion|schedule-free|ademamix|soap|lookahead|lamb|adan|adafactor|shampoo|prodigy]` — entrena el LM decodificador N-D.
- `scirust deltanet [--seed N] [--steps S]` — entrena una capa DeltaNet (atención lineal con regla delta) de una sola cabeza para ajustar una secuencia; informa la reducción del MSE.
- `scirust mamba [--seed N] [--steps S]` — entrena una capa Mamba de espacio de estados selectivo (escaneo S6) para ajustar una secuencia; informa la reducción del MSE.
- `scirust retnet [--seed N] [--steps S]` — entrena una capa de retención RetNet (atención lineal, forma recurrente ≡ forma paralela) para ajustar una secuencia; informa la reducción del MSE.
- `scirust gla [--seed N] [--steps S]` — entrena una capa de atención lineal con compuerta GLA (compuerta de olvido dependiente de los datos) para ajustar una secuencia; informa la reducción del MSE.
- `scirust hgrn [--seed N] [--steps S]` — entrena un mezclador de tokens HGRN de RNN lineal con compuerta (compuerta de olvido acotada inferiormente) para ajustar una secuencia; informa la reducción del MSE.
- `scirust rwkv [--seed N] [--steps S]` — entrena una capa de mezcla temporal RWKV (WKV; decaimiento temporal por canal + bono) para ajustar una secuencia; informa la reducción del MSE.
- `scirust conformal [--seed N] [--alpha A]` — intervalos conformes con cobertura garantizada, sin supuestos de distribución.
- `scirust calibrate [--seed N]` — escalado de temperatura; ajusta T para reducir el error de calibración esperado (ECE) sin cambiar la exactitud.
- `scirust pinn [--seed N] [--steps S]` — red informada por la física; resuelve el BVP `u''=−u` (residuo de la EDP en la pérdida), verificado frente a `sin x`.
- `scirust gptq [--seed N] [--samples S] [--damp D]` — cuantización int8 de pesos GPTQ; informa la reducción del error de calibración frente a round-to-nearest.
- `scirust awq [--seed N] [--samples S] [--grid G]` — cuantización int8 de pesos AWQ consciente de activaciones; informa el exponente de escalado seleccionado y la reducción del error de calibración frente a round-to-nearest.
- `scirust bitnet [--seed N]` — cuantización ternaria {-1,0,+1} de pesos BitNet b1.58 (~1,58 bit/peso); verifica la multiplicación de matrices sin multiplicaciones.

## 14. CLI Industrial — Referencia Completa

El CLI `scirust-industrial` facilita la integración de SciRust con sistemas industriales reales.

### Instalación

```bash
cargo install --path scirust-industrial   # proporciona el binario `scirust-industrial`
# o en su lugar: cargo run -p scirust-industrial -- <comando>
```

### Comandos

| Comando | Descripción | Opciones |
|----------|-------------|---------|
| `discover` | Lista los sensores disponibles en el servidor OPC-UA | `--endpoint`, `--filter`, `--simulated` |
| `test-opcua` | Prueba la conexión OPC-UA y lee valores | `--endpoint`, `--simulated`, `--samples N` |
| `test-mqtt` | Prueba la conexión MQTT y publica un mensaje | `--host`, `--port`, `--simulated`, `--topic` |
| `gen-config` | Genera un archivo de configuración de pipeline | `--output`, `--template`, `--stations N`, `--line-id` |
| `scaffold` | Genera un proyecto de monitorización completo | `--name`, `--output`, `--template` |
| `run` | Ejecuta un pipeline de monitorización desde config | `--config`, `--cycles N`, `--report` |
| `doctor` | Diagnostica problemas de integración | `--config` |

### Plantillas

| Plantilla | Descripción |
|-----------|-------------|
| `minimal` | 1 estación, backend simulado, detección de spikes |
| `automotive` | Línea automotriz multi-estación con diagnóstico de rodamientos, RUL, MQTT, auditoría |
| `bearing` | Detección de fallos de rodamientos (envolvente FFT, BPFO/BPFI/BSF) |
| `pdm` | Mantenimiento predictivo (Health Index, RUL, CUSUM) |

### Flujo de Integración Recomendado

```bash
# 1. Crear un proyecto
scirust-industrial scaffold --name line3-monitor --template automotive

# 2. Verificar que todo funciona
cd line3-monitor
scirust-industrial doctor --config config.json

# 3. Personalizar configuración
# Editar config.json: endpoint OPC-UA, broker MQTT, sensores, umbrales

# 4. Cambiar a modo real (opcional)
# Editar Cargo.toml: descomentar features real-opcua / real-mqtt
# Editar config.json: backend_type "opcua"

# 5. Iniciar monitorización
scirust-industrial run --config config.json --cycles 1000
```

### Cambio de Modo Simulado a Real

El modo simulado funciona sin ningún hardware. Para pasar a producción:

1. **OPC-UA real**: Añadir `features = ["real-opcua"]` a `scirust-integration` en `Cargo.toml`, añadir dependencia `opcua = "0.13"`, y cambiar `backend_type` a `"opcua"` en `config.json`.
2. **MQTT real**: Añadir `features = ["real-mqtt"]`, añadir `rumqttc = "0.24"`, y configurar `host`/`port` del broker.

El `BackendFactory` gestiona automáticamente el fallback: si el backend real falla, cambia al modo simulado.
