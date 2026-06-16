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

## 8. Conclusión

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
  **Neural ODE** (retropropagación a través de un solucionador RK4); una red neuronal informada por la física (PINN) que resuelve un problema de valores en la frontera con el residuo de la EDP en la función de pérdida.
- **Compresión**: poda Wanda (consciente de activaciones), SmoothQuant, GPTQ (cuantización int8 de pesos por retroalimentación de error de segundo orden), AWQ (cuantización int8 de pesos basada en búsqueda y consciente de activaciones).

Nuevos comandos CLI:
- `scirust certify [--seed N] [--eps E]` — cotas demostrables de un MLP ReLU (IBP **y** CROWN, las cotas más ajustadas por relajación lineal, en paralelo).
- `scirust lm [...] [--opt adam|adamw|lion|schedule-free|ademamix|soap|lookahead]` — entrena el LM decodificador N-D.
- `scirust deltanet [--seed N] [--steps S]` — entrena una capa DeltaNet (atención lineal con regla delta) de una sola cabeza para ajustar una secuencia; informa la reducción del MSE.
- `scirust mamba [--seed N] [--steps S]` — entrena una capa Mamba de espacio de estados selectivo (escaneo S6) para ajustar una secuencia; informa la reducción del MSE.
- `scirust conformal [--seed N] [--alpha A]` — intervalos conformes con cobertura garantizada, sin supuestos de distribución.
- `scirust calibrate [--seed N]` — escalado de temperatura; ajusta T para reducir el error de calibración esperado (ECE) sin cambiar la exactitud.
- `scirust pinn [--seed N] [--steps S]` — red informada por la física; resuelve el BVP `u''=−u` (residuo de la EDP en la pérdida), verificado frente a `sin x`.
- `scirust gptq [--seed N] [--samples S] [--damp D]` — cuantización int8 de pesos GPTQ; informa la reducción del error de calibración frente a round-to-nearest.
- `scirust awq [--seed N] [--samples S] [--grid G]` — cuantización int8 de pesos AWQ consciente de activaciones; informa el exponente de escalado seleccionado y la reducción del error de calibración frente a round-to-nearest.
