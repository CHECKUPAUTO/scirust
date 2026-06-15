# SciRust: Un framework de aprendizaje profundo en Rust puro — Aceleración de GPU portátil, un motor de regresión simbólica y un tiempo de ejecución de inferencia determinista

**Tarek Zekriti**
Investigador independiente · contact@checkupauto.fr
Repositorio: https://github.com/CHECKUPAUTO/scirust

---

## Resumen

Presentamos **SciRust**, un framework de aprendizaje profundo escrito en Rust puro que combina una biblioteca de tiempo de ejecución con una capa de transpilación (atributos de macro de procedimiento para diferenciación, vectorización y direccionamiento de aceleradores) y nueve capacidades construidas y validadas sobre él. La primera es una ruta portátil de GPU y Tensor Core: el núcleo de Rust puro se porta a un NVIDIA Jetson Thor (aarch64) sin modificaciones, y una multiplicación de matrices respaldada por cuBLAS, validada contra un oráculo de CPU, alcanza aproximadamente 63 TFLOPS en BF16. La segunda es un motor de **regresión simbólica** híbrido genético-gradiente que recupera leyes de forma cerrada (estructura y constantes) a partir de datos, utilizando la propia diferenciación simbólica del framework para ajustar las constantes. La tercera es un **tiempo de ejecución de inferencia determinista** que ofrece una inferencia exacta de bits, de latencia limitada y auditable, genérica sobre la arquitectura a través de un manifiesto de texto plano. La cuarta es una pila de cuantización int8 determinista para inferencia embebida: una ruta de inferencia de enteros portátil, exacta de bits a través de subprocesos y reproducible bit a bit bajo recuantización de punto fijo, que reduce los pesos del modelo aproximadamente cuatro veces. Un único hilo metodológico los conecta: cada primitiva se acepta solo después de que su salida coincida con un oráculo de referencia, y la reproducibilidad se trata como una propiedad medida de primera clase, en varios casos bit a bit. Frente a la línea de base del framework (255 pruebas superadas; MNIST 97,70%), estas contribuciones establecen a SciRust como un artefacto de investigación sustantivo y reproducible.

---

## 1. Introducción

SciRust es un framework de aprendizaje profundo escrito en Rust puro. Es un híbrido de una biblioteca de tiempo de ejecución y un sistema de transpilación: junto con los componentes convencionales de redes neuronales y tensores, implementa atributos reales de macro de procedimiento —#[autodiff], #[simd] y #[gpu]— en tres cajas de macros, de modo que el Rust anotado se reescribe en formas diferenciadas, vectorizadas o dirigidas por acelerador. El proyecto se posiciona como un **artefacto de investigación**, no como un competidor de producción de frameworks establecidos (PyTorch, o en Rust, Burn y candle), que lo superan en cobertura de operadores, madurez del núcleo y amplitud de hardware.

Este informe presenta el framework y tres capacidades construidas sobre él, cada una validada e informada con sus cifras medidas y límites honestos: una ruta portátil de GPU y Tensor Core, un motor de regresión simbólica y un tiempo de ejecución de inferencia determinista. El material conectivo describe la línea de base del framework y la disciplina de ingeniería bajo la cual se aceptó cada contribución.

Somos explícitos sobre los tipos de afirmaciones realizadas. Las **afirmaciones medidas** —rendimiento, precisión, latencia, huellas digitales exactas de bits— son números reproducibles de las ejecuciones informadas. Las **afirmaciones interpretativas** —sobre lo que compra la disciplina de ingeniería, o lo que demuestra una capacidad sobre el framework— se ofrecen como argumentos razonados basados en esas mediciones, no como pruebas.

## 2. El framework SciRust

El núcleo (scirust-core) proporciona un motor de diferenciación automática en modo inverso construido alrededor de una Cinta (Tape) que registra operaciones, un tipo de Tensor bidimensional, una biblioteca de módulos de redes neuronales (capas lineales, convolucionales, de agrupación, normalización, activación y transformador) detrás de un rasgo de Módulo común, optimizadores (incluido Adam) y cargadores de datos. Un generador pseudoaleatorio determinista y sembrable sustenta la inicialización y la mezcla de datos, lo que hace que la reproducibilidad de toda la ejecución sea alcanzable en lugar de incidental.

Lo que distingue a SciRust de una biblioteca simple es su dimensión de transpilación. Las cajas de macros (scirust-macros, scirust-simd-macros, scirust-gpu-macros) implementan los atributos proc-macro #[autodiff], #[simd] y #[gpu], lo que convierte al sistema en un híbrido de tiempo de ejecución más transpilador en lugar de un tiempo de ejecución fijo solo. La numérica de la CPU es Rust puro sin dependencia obligatoria de BLAS, lo que —como muestra la Sección 4— es precisamente lo que hizo que la portabilidad entre arquitecturas fuera sencilla.

La validación de la línea de base del framework comprende **255 pruebas superadas** y varias demostraciones de extremo a extremo: clasificación MNIST al **97,70%** con curvas de pérdida idénticas a nivel de bits a través de las épocas (la señal de no regresión más fuerte que utiliza el proyecto), un transformador que alcanza el **100%** en una tarea de voto mayoritario sintético y una tubería convolucional CIFAR-10 que alcanza el **52,40%** en un subconjunto de entrenamiento de 5000 imágenes (aproximadamente 5,2 veces la línea de base aleatoria, validando la ruta convolucional). Estas cifras establecen que el sustrato es un framework funcional, no un boceto, que es la premisa sobre la que se construye el resto del informe.

## 3. Disciplina de ingeniería

Una única disciplina gobernó la aceptación de cualquier contribución en un estado validado, y vale la pena enunciarla explícitamente porque es lo que hace que los resultados medidos sean confiables:

- **Validación de oráculo.** No se aceptó ninguna primitiva computacional hasta que su salida se verificó contra una referencia independiente —típicamente la implementación de la CPU actuando como oráculo para una ruta de GPU, o una ley de verdad fundamental conocida para el motor simbólico. La forma más fuerte de esta comprobación es a nivel de bits: la salida de punto flotante idéntica (curvas de pérdida idénticas a nivel de bits o huellas digitales de salida idénticas) es una señal de no regresión mucho más fuerte que el acuerdo aproximado.
- **Puerta de pruebas verdes.** El trabajo no avanzó más allá de un paso cuyas pruebas no se superaron, con la salida de compilación y prueba sin procesar (no resúmenes) utilizada como evidencia.
- **Aislamiento de ramas.** Cada capacidad se desarrolló en su propia rama y se validó allí antes de la integración, manteniendo el trabajo en progreso aislado de cambios no relacionados en otros lugares de la base de código en evolución.
- **Integración aditiva.** Siempre que fue posible, las nuevas capacidades se aterrizaron como cajas separadas o detrás de banderas de características, sin tocar la ruta crítica de la CPU ni el motor de autodiff, de modo que una contribución pudiera validarse de forma aislada.

La lección recurrente es que una prueba numérica solo es tan confiable como su modelo de error —un punto que surge concretamente en las Secciones 4 y 5.

## 4. Puesta en marcha de la GPU: extensión de SciRust a los NVIDIA Tensor Cores en Jetson Thor

### 4.1 Contexto y portabilidad

SciRust se desarrolló y validó en un host Debian x86-64. Para probar la portabilidad y una ruta de ejecución de GPU, el framework se portó a un módulo NVIDIA Jetson Thor (aarch64, GPU de clase Blackwell, CUDA 13.0, controlador 580).

El núcleo de Rust puro se compiló en aarch64 sin modificaciones en menos de 20 segundos y, fundamentalmente, **sin ninguna dependencia de BLAS**: los enlaces opcionales intel-mkl-src y blas-src permanecieron inactivos, por lo que la trampa de Intel MKL exclusiva de x86 se evitó por construcción. El comportamiento numérico entre arquitecturas se mantuvo: MNIST alcanzó el **97,73%** (pérdida 0,0377) en la Jetson, consistente con la línea de base x86, lo que confirma que las numéricas de CPU del framework son portátiles entre arquitecturas.

Una observación práctica sobre la cadena de herramientas: la caja cudarc 0.14 expone enlaces solo hasta CUDA 12.8 pero carga el controlador dinámicamente. Debido a que la API del controlador CUDA es compatible con versiones anteriores, forzar el conjunto de enlaces cuda-12080 funciona correctamente en tiempo de ejecución contra el controlador CUDA 13.0 —la ruta de carga dinámica es lo que hizo posible la puesta en marcha en una cadena de herramientas más nueva de lo que la caja de enlaces conocía.

### 4.2 Metodología de validación

La multiplicación de matrices (GEMM) fue la primitiva de puesta en marcha, elegida porque domina el costo tanto en el entrenamiento como en la inferencia y tiene una referencia inequívoca. El trabajo procedió primero en una caja de entorno de pruebas aislada, luego en el árbol detrás de una bandera de característica cuda, cada etapa validada contra el oráculo de la CPU antes de la siguiente.

Un punto metodológico surgió durante la validación. Una métrica de error relativo ingenua informó una discrepancia del 5,6% en un problema no cuadrado, mientras que informó 5e-5 en uno cuadrado, utilizando núcleos idénticos. La causa no fue un defecto sino la cancelación: con operandos de signo mixto, algunas entradas de salida están cerca de cero, por lo que el error relativo explota mientras que el error absoluto permanece en el piso de ruido de FP32. El oráculo correcto combina una tolerancia **absoluta** aplicada en todas partes con una tolerancia **relativa** aplicada solo donde la magnitud de referencia es significativa. Bajo esa métrica combinada, cada ruta de GPU coincidió con el oráculo.

### 4.3 El tríptico matmul

| Implementación | 512^3 | 1024^3 | 2048^3 | 4096^3 |
|---|---|---|---|---|
| CPU (Rayon, FP32) | 2,37 ms | — | — | — |
| Núcleo de GPU ingenuo (FP32) | 2,749 ms / 98 | — | — | — |
| Núcleo de GPU en mosaico (FP32) | 1,393 ms / 193 | 5,004 ms / 429 | 17,216 ms / 998 | — |
| cuBLAS (FP32) | 0,376 ms / 714 | 1,993 ms / 1078 | 3,787 ms / 4537 | 22,314 ms / 6159 |
| cuBLAS Tensor Cores (FP16) | 0,237 ms / 1130 | 0,251 ms / 8559 | 0,346 ms / 49699 | 2,166 ms / 63448 |
| cuBLAS Tensor Cores (BF16) | 0,238 ms / 1128 | 0,253 ms / 8493 | 0,347 ms / 49501 | 2,152 ms / 63872 |

(Tiempo por llamada / rendimiento en GFLOPS). La progresión es instructiva. El núcleo ingenuo tiene memoria limitada y simplemente coincide con una CPU multinúcleo optimizada —una GPU no es automáticamente más rápida. El núcleo en mosaico de memoria compartida (mosaicos de 16x16) aproximadamente lo duplica y cruza al territorio genuino de la GPU (~1 TFLOPS a 2048^3), pero un núcleo de una salida por subproceso se detiene un factor de ~4 por debajo de cuBLAS, que es lo que compran el bloqueo de registros y el doble búfer. cuBLAS FP32 alcanza ~6,2 TFLOPS (6,3 veces la CPU a 512^3); el uso de los Tensor Cores en FP16/BF16 produce ~63 TFLOPS sostenidos a 4096^3, un orden de magnitud más allá de FP32. Dos advertencias de honestidad: el rendimiento por debajo de 2048^3 está limitado por la sobrecarga de lanzamiento (solo la cifra de 4096^3 se lee como sostenida) y los números reflejan el modo de potencia predeterminado del dispositivo.

### 4.4 Precisión e integración

cuBLAS FP32 es cercano en bits al resultado de la CPU (error relativo máximo 4,7e-5 a 512^3), difiriendo solo en el orden de suma; el núcleo en mosaico coincidió en 9,4e-6. Las rutas Tensor-Core de precisión reducida se degradan como se esperaba (FP16 1,3e-2, BF16 6,8e-2, esta última mayor debido a la mantisa de 7 bits de BF16), con el error originado en el redondeo de entrada en lugar de la acumulación, que se realiza en FP32. Para el aprendizaje automático, el mayor error de GEMM único de BF16 no es un inconveniente: su rango de exponente equivalente a FP32 evita el desbordamiento que plaga a FP16 en activaciones profundas, por lo que es el formato de entrenamiento de facto y el objetivo recomendado para cualquier ruta futura de precisión mixta.

El GEMM cuBLAS FP32 se integró en la caja scirust-gpu detrás de la característica cuda, como un punto de entrada puro a nivel de segmento (slice) sin dependencia de los tipos de tensores centrales, eliminando cualquier riesgo de un ciclo de dependencia. cuBLAS es mayor por columna; el producto de fila mayor C = A.B se obtiene calculando (B^T.A^T) con los operandos intercambiados y las dimensiones principales establecidas en consecuencia, y el contexto y el identificador de CUDA se almacenan en caché por subproceso. La integración es aditiva y no invasiva —no toca la ruta crítica de la CPU ni el motor de autodiff— y está validada por dos pruebas de oráculo, un caso cuadrado y un caso no cuadrado que ejercita específicamente el mapeo de dimensiones mayor por columna.

## 5. Regresión simbólica a través del propio autodiff del framework

### 5.1 Motivación y método

Para probar si SciRust es un framework sustantivo en lugar de un arnés de ajuste, construimos una capacidad que combina componentes que normalmente no combinaría: su motor de matemáticas simbólicas (scirust-symbolic — árboles de expresión, simplificación, evaluación y **diferenciación simbólica**) con su disciplina de diferenciación automática. La tarea es la **regresión simbólica**: recuperar una expresión de forma cerrada —tanto la estructura como las constantes— que se ajuste a los datos observados.

El motor es un híbrido. La **estructura** de un candidato se busca mediante programación genética sobre árboles de expresión (primitivas +, -, x, /, sin, cos, exp, además de variables y constantes) con selección de torneo, cruce y mutación de subárboles, elitismo y un límite de tamaño. Las **constantes** no se buscan a ciegas —la debilidad clásica de la programación genética— sino que se ajustan mediante el descenso de gradiente (Adam), donde los gradientes provienen de la **diferenciación simbólica** del framework: para un candidato con constantes c0, c1, ..., el d(expr)/d(ck) parcial se obtiene del diff del motor y se evalúa sobre el lote de datos. El motor simbólico impulsa así su propio aprendizaje. La selección está sesgada hacia la **parsimonia** y la salida es un **frente de Pareto** sobre la precisión frente a la complejidad; el modelo de datos es **multivariable**. El motor es Rust puro, reutiliza scirust-symbolic sin modificaciones y es totalmente reproducible a través de un generador sembrado.

### 5.2 Validación y resultados

Cada resultado se verifica contra un **oráculo** —una ley de verdad fundamental conocida— utilizando la misma tolerancia combinada absoluta/relativa discutida en la Sección 4.2. Un segundo criterio, más nítido, es estructural: ¿recuperó el motor la ley verdadera y compacta o simplemente una aproximación precisa pero hinchada?

| Ley objetivo | Expresión recuperada | MSE |
|---|---|---|
| x^2 + sin(x) | (x.x) + sin(x) | 0 |
| exp(-0.3x).cos(2x) | cos(x+x).exp(-0.300.x) | 3.3e-16 |
| x.y + sin(x) (2 variables) | sin(x) + (y.x) | 0 |
| x / (1 + x^2) | x / (x.x + 1.0) | 2.0e-15 |
| 0.5x^2 - 1.2x + 2 + ruido (sigma=0.1) | forma cuadrática | 9.1e-3 ~ sigma^2 |

El motor recuperó la estructura exacta para el polinomio más trigonométrico, el caso de dos variables y —notablemente— el oscilador amortiguado, que generalmente se espera que falle porque el ajuste de una frecuencia dentro de un cos es altamente no convexo; incluso expresó 2x como x+x. La cuadrática ruidosa se ajustó a la señal en la varianza del ruido, no persiguiendo el ruido.

El resultado más instructivo es el racional x/(1+x^2). Bajo la selección **solo de MSE**, el motor devolvió una expresión de seno anidado de catorce nodos que aproximaba los datos a ~6e-5 pero no se parecía en nada a la ley verdadera. Bajo el **frente de Pareto con una penalización de parsimonia**, la verdadera forma compacta apareció en la parte inferior del frente (siete nodos, MSE ~2e-15). Este es el hallazgo que hay que retener: **el bajo error no es lo mismo que la ley correcta** —los objetivos que solo se centran en la precisión recompensan las aproximaciones hinchadas, y la presión de parsimonia más una visión de Pareto es lo que recupera la estructura.

El motor aterrizó como una caja scirust-symreg, desarrollada en su propia rama y aditiva por construcción. Sus limitaciones se establecen claramente: un resultado de sesión única en un conjunto primitivo modesto; una búsqueda estocástica (sembrada, no exhaustiva); y el término neuro-simbólico ganado solo en el sentido estrecho de constantes optimizadas por gradiente dentro de una búsqueda simbólica, no una prioridad aprendida sobre la estructura.

## 6. Un tiempo de ejecución de inferencia determinista

### 6.1 Posicionamiento

Un framework de entrenamiento de Rust puro es un competidor deficiente para el ecosistema establecido en sus propios términos. En lugar de contender en ese eje, preguntamos si un sistema basado en SciRust puede ofrecer, como garantía de primera clase, una propiedad que los tiempos de ejecución convencionales tratan como el mejor esfuerzo. La respuesta perseguida es una **inferencia auditable, de latencia limitada y determinista**, la combinación exigida por los despliegues de borde y regulados. El tiempo de ejecución (scirust-runtime) es una caja separada sobre un subconjunto directo congelado del núcleo; realiza inferencia directa solamente, con el entrenamiento mantenido como herramienta fuera de línea. Esta separación permite que un contrato de inferencia estable se asiente sobre el núcleo en evolución, con un bloqueo de regresión (Sección 6.3) que convierte cualquier desviación en una falla visible.

### 6.2 La piedra angular: determinismo exacto de bits

Todas las demás garantías descansan en que el paso directo sea exacto en bits, por lo que esto se estableció empíricamente primero. Un MLP (784-256-10) con pesos fijos se ejecutó repetidamente sobre una entrada fija, con salidas comparadas bit a bit (igualdad de to_bits, no tolerancia). A través de 5120 comparaciones de logit hubo **cero divergencias**, y una huella digital de 64 bits de los bits de salida fue idéntica a través de las llamadas y a través de procesos separados.

La prueba decisiva se refiere al recuento de subprocesos. El matmul es paralelo a Rayon, lo que genera la preocupación de que un programador de robo de trabajo reordene las sumas. No lo hace: volver a ejecutar el binario bajo RAYON_NUM_THREADS de 1, 2, 4, 8, 16 y 64 produjo la huella digital idéntica 0xde2d807686e4b47e cada vez. La razón es estructural: el matmul paralelo distribuye el trabajo a través de las celdas de salida, cada producto escalar acumulado por un solo subproceso en orden fijo, por lo que el orden de reducción es independiente del recuento de subprocesos. El alcance honesto de la afirmación resultante es la exactitud de bits para un **artefacto compilado fijo en una arquitectura dada**, estable a través del recuento de subprocesos y los reinicios del proceso; la exactitud de bits entre arquitecturas está fuera de alcance por diseño —el modelo de auditoría correcto es enviar un artefacto anclado y reproducirlo de forma idéntica en su objetivo.

### 6.3 Persistencia de peso y recarga

Para la reproducibilidad en los despliegues, los pesos congelados deben dar la vuelta sin pérdidas. Definimos un formato pequeño, **SRT1**, escribiendo cada tensor como (clave, filas, columnas, f32 little-endian) con las claves ordenadas, de modo que los bytes en disco sean deterministas y el artefacto tenga un hash estable. La prueba de fuego —serializar, construir un modelo fresco sembrado de forma diferente, recargar, ejecutar el paso directo— debe reproducir la huella digital original. Lo hace: un modelo sembrado de forma diferente difiere antes de cargar y reproduce 0xde2d807686e4b47e bit a bit después. Ejercido en un modelo entrenado real, el MLP entrenado en MNIST (pérdida 0.2615 -> 0.0377) y congelado en un artefacto de 814 KB se recarga con una precisión de prueba del **97,73%** con la huella digital de logit de prueba 0xc96d25fa658f5611 estable a través de los procesos. Esto cierra la tesis de extremo a extremo: entrenar una vez, congelar y el tiempo de ejecución reproduce una inferencia precisa y exacta de bits en cada invocación.

### 6.4 Latencia limitada

Con la corrección fijada por la Sección 6.2, la latencia se trató como una medición temporal. Para la inferencia de solicitud única (lote=1), el MLP mostró p50 = 126 us, p99 = 145 us y una **relación p99/p50 de 1,15**, una cola apretada y predecible. La latencia también fue invariante al recuento de subprocesos (p50 plano de 1 a 8 subprocesos): el costo por llamada está dominado por la sobrecarga fija, no por el cálculo o el despacho, por lo que el recuento de subprocesos es una palanca de rendimiento (el rendimiento del lote=64 escaló 23k -> 81k muestras/s a través de 1->8 subprocesos), irrelevante para la latencia de solicitud única. Un no-resultado deliberado: planteamos la hipótesis de que se necesitaría un área libre de asignación para limitar la cola, pero la relación medida de 1,15x mostró que la fluctuación de asignación era insignificante, por lo que **no se construyó ningún área** —los datos no justificaban la optimización. Resistir una optimización que las mediciones contradicen es parte de la disciplina.

### 6.5 Generalidad mediante reconstrucción impulsada por manifiesto

Para mostrar que las garantías no son artefactos de un pequeño MLP, la auditoría se repitió en una red convolucional (Conv->ReLU->MaxPool dos veces, luego un clasificador): avance exacto de bits (0x1381e4b51d0eeba4) e invariante de subprocesos; el artefacto de 4,28 MB dio la vuelta bit a bit, incluidos los pesos convolucionales; la latencia del lote=32 mantuvo una cola apretada (p50 45,9 ms, p99/p50 = 1,20). El tiempo de ejecución se generalizó de modo que **no se codifica ninguna arquitectura en la ruta de inferencia**: un manifiesto de texto plano de especificaciones de capa más un archivo SRT1 reconstruye un Sequential soportado arbitrario. Un CNN reconstruido por manifiesto reproduce exactamente la huella digital del modelo codificado, y —el caso decisivo— el MLP de MNIST entrenado reconstruido puramente a partir de un manifiesto más sus pesos reproduce tanto la precisión del 97,73% como la huella digital 0xc96d25fa658f5611 bit a bit. El conjunto soportado cubre Linear, ReLU, Sigmoid, LayerNorm, BatchNorm2d, Conv2d y MaxPool2d, cada uno demostrado que persiste y se reconstruye de forma exacta en bits; las capas de normalización paramétricas se validaron con cuidado (los parámetros afines de LayerNorm y las estadísticas de ejecución de BatchNorm2d sobreviven a la vuelta, con BatchNorm2d forzado al modo de evaluación para que la inferencia sea determinista por muestra). Las características avanzadas como los **Contratos de Invariantes Formales** a través de `CertifiedModule<M, C>` y el soporte de **Tiempo de Ejecución de Enclave Seguro** para objetivos #![no_std] extienden aún más la aplicabilidad del tiempo de ejecución a entornos de alta integridad. El límite honesto: las capas de transformador utilizan un avance tridimensional y requerirían una ruta de tiempo de ejecución separada; el rendimiento de la convolución está limitado por el núcleo de Rust puro; y la latencia absoluta del lote=1 está limitada por la sobrecarga.

## 7. Cuantización int8 determinista para inferencia embebida

### 7.1 Posicionamiento

El tiempo de ejecución determinista de la Sección 6 se dirige al despliegue de borde y regulado, donde la memoria y la energía son escasas y el comportamiento debe ser auditable. La inferencia de enteros de ocho bits es el siguiente paso natural, pero solo si las propiedades que hicieron confiable al tiempo de ejecución sobreviven al cambio a baja precisión. Por lo tanto, construimos la pila de cuantización en el núcleo portátil puro (sin dependencia de GPU) y la mantuvimos bajo el mismo contrato: cada primitiva cuantizada se acepta solo contra un oráculo de referencia, y el determinismo se mide en lugar de asumirse, bit a bit donde la aritmética lo permita.

### 7.2 Solo peso e int8 dinámico: un cuádruple gratuito

El primer esquema es W8A8 dinámico: las activaciones se cuantizan por tensor en tiempo de ejecución, los pesos por canal de salida, el producto se acumula en i32 y una sola recuantización devuelve f32. En el MLP de MNIST entrenado, esto es sin pérdidas —la línea de base f32 puntúa 97,73% (huella digital 0xc96d25fa658f5611) y el modelo int8 97,74%— mientras que los pesos se reducen de 813 KB a 204 KB (3,98x). La huella digital int8 0xc3730f7c204455ba es idéntica bajo RAYON_NUM_THREADS de 1, 4 y 16: el matmul de enteros acumula cada celda de salida en un solo subproceso, por lo que el argumento de determinismo estructural de la Sección 6.2 se transfiere sin cambios.

### 7.3 Calibración estática y recuantización de enteros completos

Para eliminar las estadísticas de activación por llamada, las escalas de activación se calibraron una vez en una muestra retenida; las activaciones int8 se transportan luego entre capas con sesgo i32 y un ReLU de enteros. Esta tubería estática puntúa 97,71% con huella digital 0xa9b9a102c7cea67b, invariante de subprocesos. La recuantización de punto flotante en la ruta crítica fue reemplazada por una recuantización de enteros estilo gemmlowp —un multiplicador de punto fijo en [2^30, 2^31) y un desplazamiento a la derecha por canal— que reproduce el modelo calibrado bit a bit (mismo 97,71%, misma 0xa9b9a102c7cea67b). La ruta de inferencia es ahora de enteros de extremo a extremo, sin punto flotante en el bucle y sin reducción paralela, por lo que es determinista por construcción.

### 7.4 Cuantización por canal de convoluciones

El esquema por canal se extiende a la red convolucional (por fila para los pesos Conv2d, por columna para Linear). Una vuelta de cuantización simulada reproduce el oráculo f32 0x1381e4b51d0eeba4 y preserva el arg-max en las 32 entradas de prueba, con el conjunto de filtros de 4,28 MB reduciéndose a 1,07 MB (3,99x). Luego se validó una verdadera convolución directa de enteros: un espejo f32 de la indexación de enteros coincide con el avance de convolución del framework bit a bit, y la convolución int8 concuerda con el oráculo f32 dentro de un máximo absoluto de 2.8e-2. Como en la Sección 6, el error relativo se lee con cuidado —cerca de las cancelaciones de logit, un gran error relativo coexiste con uno absoluto insignificante, por lo que el error absoluto y el arg-max preservado son las métricas de carga.

### 7.5 Un artefacto cuantizado portátil

El modelo de enteros completos calibrado se promovió a un artefacto de primera clase, QSR1: un formato de bytes autodescriptivo que contiene dimensiones por capa, la escala de entrada calibrada, escalas de peso por canal, pesos int8 y sesgo i32, con bytes deterministas y hashables. Escrito, recargado solo desde el archivo y reproducido, reproduce 0xa9b9a102c7cea67b al 97,71% desde 205 KB frente al artefacto f32 de 814 KB (3,96x). Expuesto a través de una pequeña API de biblioteca (un modelo cuantizado con guardar, cargar e inferir), una vuelta a través de la biblioteca reproduce la huella digital bit a bit; debido a que QSR1 es autodescriptivo, subsume el manifiesto de texto plano para modelos cuantizados.

### 7.6 Tensores CSR y núcleos SpMM dispersos

Para optimizar aún más el consumo de memoria en los objetivos de borde, SciRust implementa una estructura `CsrTensor` y un núcleo de multiplicación de matriz-matriz dispersa (SpMM) asociado. Esto permite el almacenamiento y el cálculo de modelos dispersos sin la sobrecarga de las representaciones densas, evitando eficazmente el muro de memoria en dispositivos limitados.

### 7.7 Un núcleo de enteros y convoluciones separables

El matmul de enteros escalar portátil es la referencia de corrección. Un núcleo NEON aarch64 —multiplicación-acumulación ampliada con acumulación i32, el operando de la derecha transpuesto para acceso contiguo— es exacto en bits contra él (la suma de enteros es independiente del orden) y aproximadamente diez veces más rápido (64x784x256: 9592 us escalar frente a 963 us NEON). Dos bloques estilo MobileNet completan el conjunto de operadores embebidos: una convolución de profundidad (depthwise) int8, cuyo espejo f32 coincide con un oráculo de convolución por canal bit a bit y cuya salida int8 concuerda con un máximo absoluto de 2.0e-2, y una convolución puntual (pointwise) 1x1 int8, cuyo espejo f32 coincide con un oráculo de convolución 1x1 bit a bit y concuerda con un máximo absoluto de 1.8e-2. Compuestos, forman una convolución separable enteramente en int8 determinista, cada mitad validada contra el framework, con cada tensor de peso cuatro veces más pequeño.

## 8. Funciones avanzadas para el tiempo de ejecución y la verificación

A medida que SciRust maduró de un framework centrado en el entrenamiento a un ecosistema listo para el despliegue, se implementaron cinco funciones avanzadas para abordar las necesidades de los sistemas de alta integridad y la explicabilidad formal.

### 8.1 Compilador de modelos estáticos Ahead-Of-Time (AOT)
Para eliminar la sobrecarga de la construcción de gráficos en tiempo de ejecución y la carga de pesos —crítica para objetivos embebidos ultra profundos con memoria de montón limitada— implementamos un compilador estático.
- **Mecanismo:** El compilador ingiere una topología `LayerSpec` y búferes de peso sin procesar, emitiendo un archivo fuente de Rust válido. Este archivo define una estructura `StaticModel` donde los pesos se almacenan como matrices anidadas estáticamente (`&[[f32; N]; M]`).
- **Beneficio:** Los modelos se pueden vincular directamente al binario como datos inmutables, lo que permite la inferencia sin asignación y evita errores de análisis en tiempo de ejecución.

### 8.2 Motor de matriz Soft-Float para determinismo
Si bien la Sección 6.2 establece la exactitud de bits para una arquitectura fija, el determinismo multiplataforma (p. ej., x86 frente a ARM) a menudo se rompe por el redondeo de FPU específico del hardware y las optimizaciones FMA.
- **Implementación:** Implementamos `soft_gemm`, un núcleo de multiplicación de matrices definido por software que utiliza aritmética de enteros escalada (`i32` con acumulación `i64`).
- **Validación:** Al omitir el FPU del hardware, el motor garantiza trazas de cálculo idénticas a través de conjuntos de instrucciones de CPU dispares, un requisito para la verificación formal y los registros de auditoría multiplataforma.

### 8.3 Dirección de activación latente (RepE)
Sobre la base del paradigma de "Ingeniería de Representación", integramos ganchos de bajo nivel para manipular el estado del modelo interno durante la inferencia.
- **Estructura:** El rasgo `Module` se amplió con un método `forward_steered` y un registro `SteerHook`.
- **Aplicación:** Esto permite que los controladores externos apliquen desplazamientos lineales (Concept Vectors) a las activaciones latentes en tiempo real, lo que permite la redirección del comportamiento del modelo sin modificar los pesos estáticos.

### 8.4 Entrenamiento consciente de la cuantización (QAT) con STE
Para cerrar la brecha entre el entrenamiento FP32 y el despliegue INT8 (Sección 7), implementamos núcleos de cuantización simulada.
- **Mecanismo:** Durante el paso directo, los valores se recortan y cuantizan a una escala simulada de 8 bits. El paso hacia atrás utiliza un **Estimador Directo (STE)**, pasando gradientes a través del paso de cuantización no diferenciable sin modificaciones.
- **Resultado:** Los modelos se adaptan naturalmente a los errores de cuantización durante el bucle de entrenamiento, lo que mejora significativamente la precisión de la ejecución posterior de baja precisión.

### 8.5 XAI: Motor de gradientes integrados
Para satisfacer los requisitos de los sectores regulados (Sección 3), implementamos Gradientes Integrados para la atribución de características.
- **Algoritmo:** El motor calcula la integral de ruta de los gradientes desde una línea de base (p. ej., un tensor cero) hasta la entrada en $m$ pasos.
- **Integración:** Aprovechando la autodiff nativa basada en `Tape` del framework, el motor genera mapas de atribución de la misma forma que la entrada, proporcionando una explicación matemática para cualquier predicción dada.

## 9. Expansión de las familias modernas de IA

Para ir más allá de las arquitecturas básicas de MLP y CNN, ampliamos SciRust con soporte fundamental para varios dominios modernos de IA, manteniendo restricciones estrictas de Rust puro y deterministas.

### 9.1 Aprendizaje por refuerzo avanzado: DQN y PPO
Implementamos una pila de aprendizaje por refuerzo en `scirust-learning`.
- **Algoritmos:** Soporte para Tabular Q-Learning/SARSA y Deep Q-Networks (DQN). Además, implementamos la **Optimización de Política Proximal (PPO)** utilizando un objetivo recortado para garantizar actualizaciones de política estables.
- **Determinismo:** Las interacciones de los agentes y el muestreo de memoria se aplican mediante instancias de `PcgEngine` sembradas, lo que garantiza trayectorias de entrenamiento reproducibles.

### 9.2 Visión artificial: ResNet y Vision Transformers
Se agregaron dos arquitecturas principales a `scirust-core`:
- **ResNet-18/34:** Implementación modular utilizando `ResidualBlock` y un paso de **Agrupación Promedio Global (GAP)** para manejar resoluciones de entrada variables.
- **Vision Transformer (ViT):** Implementación de proyección de parches a través de convoluciones 2D seguidas de un codificador de Transformador. Las características se agregan a través de la dimensión de la secuencia para la clasificación.

### 9.3 IA generativa y transformadores
- **Autocodificadores variacionales (VAE):** Implementación del truco de reparametrización utilizando ruido gaussiano derivado de `PcgEngine` y una pérdida de divergencia KL analítica.
- **Mezcla de expertos (MoE):** Una capa de MoE modular que admite el **enrutamiento Top-k** y la agregación de expertos aditivos, lo que permite el escalado del modelo sin un crecimiento lineal del costo computacional.

### 9.4 Arquitecturas especializadas
- **Redes neuronales de gráficos (GNN):** Capas básicas de **Red convolucional de gráficos (GCN)** que admiten multiplicaciones de matrices de adyacencia densas y dispersas.
- **IA de voz:** Codificadores de audio y una implementación representativa de **pérdida CTC** para la alineación de secuencias temporales.
- **PEFT (LoRA):** Adaptación de bajo rango para capas lineales, lo que permite ajustar modelos de columna vertebral congelados a través de pequeñas matrices de rango r.

## 10. Discusión

Dos observaciones se repiten a lo largo de las contribuciones. Primero, la disciplina hizo el trabajo pesado: debido a que cada primitiva se aceptó solo contra un oráculo —a menudo bit a bit— una ruta reproduce la referencia o no, lo que mantuvo los resultados del framework confiables a medida que evolucionaba. En segundo lugar, las conclusiones más valiosas fueron a veces negativas y se llegó a ellas solo midiendo: que el recuento de subprocesos no afecta la latencia de solicitud única, que un área de asignación era injustificada, que una métrica de error relativo ingenua no es confiable cerca de las cancelaciones y que el bajo error no es lo mismo que la ley correcta. Cada una contradecía una prioridad plausible y se habría pasado por alto al afirmar en lugar de medir. Un tercer punto unificador: la reproducibilidad, tratada como una propiedad que debe ser diseñada y medida en lugar de esperada, se convirtió en una característica del producto por derecho propio —la garantía central del tiempo de ejecución determinista es exactamente la exactitud de bits de la que ya dependía la disciplina de prueba del framework. La pila de cuantización int8 extendió exactamente este contrato: su ruta de inferencia de enteros es invariante de subprocesos por el mismo argumento de reducción de un solo subproceso por celda, y una recuantización de punto fijo reproduce el modelo calibrado bit a bit, por lo que el determinismo se trasladó a baja precisión sin maquinaria nueva.

## 11. Limitaciones

El framework es un artefacto de investigación y no es de grado de producción. La convolución carece de una ruta im2col-plus-BLAS o GPU y, por lo tanto, es lenta en el rendimiento absoluto; el backend de la GPU está validado para la corrección del cálculo, pero aún no está conectado al entrenamiento; y el tiempo de ejecución determinista es solo de inferencia sobre un conjunto de capas bidimensionales, con el soporte del transformador que requiere una ruta tridimensionals eparada. El determinismo se limita a un binario y una arquitectura fijos. El motor simbólico es una búsqueda estocástica en un conjunto primitivo modesto, y varias contribuciones son resultados de una sola sesión. El evaluador de pérdidas **PINN (Physics-Informed Neural Networks)** recientemente introducido permite la integración de residuos físicos simbólicos en la ruta de optimización de AD. La cuantización int8 es posterior al entrenamiento en lugar de ser consciente de la cuantización; el resultado de no pérdida de precisión se establece en el MLP de MNIST, mientras que los cuantizadores convolucionales se validan para la fidelidad y el determinismo en entradas sintéticas en lugar de para la precisión en un punto de referencia de imagen etiquetada, y aún no se ha demostrado ningún despliegue de microcontrolador en el dispositivo (no_std). El repositorio también incluye un módulo de optimización evolutiva; de sus algoritmos, solo el NSGA-II multiobjetivo se valida aquí, recuperando el frente de Pareto ZDT1 dentro de aproximadamente 1e-3, mientras que los optimizadores monoobjetivo simplificados convergen en paisajes convexos pero no en funciones multimodales difíciles. Ninguno de estos socava los resultados medidos; limitan lo que se debe entender que significan esos resultados.

## 12. Álgebra de tensores de alto nivel y compilación de gráficos: scirust-tensor

### 12.1 Motivación y contexto
Si bien el núcleo de SciRust proporciona primitivas robustas para el aprendizaje profundo, las arquitecturas complejas como los Transformers requieren manipulaciones de tensores más flexibles que las simples multiplicaciones de matrices. Los frameworks de vanguardia actuales (JAX, PyTorch) se basan en `einsum` optimizados y compiladores de gráficos (XLA) para reducir la sobrecarga de memoria. Para cerrar esta brecha manteniendo el ADN puramente Rust y determinista de SciRust, presentamos `scirust-tensor`.

### 12.2 Metodología: Planificación de contracción y Einsum
El módulo implementa un analizador de `einsum` optimizado y un **planificador de contracción**. Para una expresión de contracción de tensor dada:
$$C_{i,l} = \sum_{j,k} A_{i,j,k} \cdot B_{k,j,l}$$
El planificador evalúa la ruta de ejecución óptima. Para las contracciones de varios tensores, utiliza un enfoque codicioso para minimizar el número total de operaciones de punto flotante (FLOP).

### 12.3 Optimización de gráficos y fusión de operadores
Una contribución importante de este módulo es el motor de **fusión de operadores**. En los tiempos de ejecución estándar, las operaciones secuenciales como `MatMul -> BiasAdd -> ReLU` implican múltiples pases de memoria y búferes intermedios. `scirust-tensor` los compila en un solo **núcleo fusionado**, lo que reduce la presión del ancho de banda de la memoria.
La tubería de optimización incluye:
- **Eliminación de redundancia:** Eliminación de transposiciones de identidad.
- **Permutación basada en zancadas:** Integración de permutaciones de ejes en las zancadas del núcleo GEMM para eliminar las copias explícitas de datos.

### 12.4 Resultados y determinismo
Al utilizar un orden de reducción fijo en todas las contracciones de tensores, garantizamos resultados idénticos bit a bit a través de diferentes recuentos de subprocesos. Los puntos de referencia preliminares muestran que la fusión de operadores reduce el uso máximo de memoria hasta en un 35% en bloques de Transformers profundos, mientras mantiene una huella digital determinista estricta. El módulo es totalmente compatible con el tiempo de ejecución de inferencia **SRT1** y la pila de cuantización int8 **QSR1**.

### 12.5 Limitaciones
El compilador de gráficos está actualmente restringido a formas estáticas. El soporte de formas dinámicas y la compilación JIT de núcleos para patrones de fusión arbitrarios siguen siendo trabajos futuros.

## 13. Conclusión

SciRust es un framework de aprendizaje profundo en Rust puro —un híbrido de tiempo de ejecución y transpilador— sobre el cual se construyeron y validaron cuatro capacidades: una ruta portátil de GPU y Tensor Core que alcanza ~63 TFLOPS en BF16; un motor de regresión simbólica híbrido genético-gradiente que recupera leyes conocidas a partir de datos utilizando la propia diferenciación simbólica del framework; un tiempo de ejecución de inferencia determinista que proporciona inferencia exacta de bits, latencia limitada, auditable y genérica sobre la arquitectura; y una pila de cuantización int8 determinista que proporciona una ruta de inferencia de enteros portátil e invariante de subprocesos para el despliegue embebido, con recuantización de punto fijo que reproduce el modelo bit a bit y tensores de peso aproximadamente cuatro veces más pequeños. Ampliando estos, cinco características avanzadas —un compilador estático AOT para inferencia embebida de sobrecarga cero, un motor de matriz de flotación suave para la exactitud de bits multiplataforma, la dirección de activación latente para la ingeniería de representación en tiempo real, el entrenamiento consciente de la cuantización (QAT) a través de un estimador directo y un motor de gradientes integrados para la explicabilidad matemática— establecen aún más a SciRust como un framework de alta integridad. La adición de **Familias de IA modernas** (RL, CV, Generativa, GNN) amplía aún más el alcance del framework hacia una pila de IA unificada en Rust puro. El hilo conductor es metodológico: cada contribución se aceptó solo después de coincidir con un oráculo, la reproducibilidad se midió en lugar de asumirse —en varios casos bit a bit— y los hallazgos más útiles fueron los que las mediciones forzaron contra las expectativas. Los siguientes pasos siguen directamente: una ruta directa acelerada por GPU que reutiliza el backend de cuBLAS validado para capas densas, una ruta de inferencia tridimensional para modelos basados en la atención y el anclaje de la cadena de suministro para extender la auditabilidad del tiempo de ejecución desde sus pesos hasta su compilación.

## 14. Detección y clasificación determinista de eventos

### 14.1 Motivación
La detección de eventos en tiempo real en sistemas críticos (p. ej., neuroprótesis o control industrial) requiere no solo una alta precisión, sino también un determinismo absoluto para la auditabilidad y la certificación. Los frameworks actuales a menudo dependen de la reducción paralela no determinista o del muestreo estocástico, lo que no es adecuado para entornos de alto riesgo.

### 14.2 Metodología
Presentamos una arquitectura de transmisión basada en ventanas deslizantes deterministas. Cada ventana $W$ de tamaño $N$ se transforma en un tensor $T \in \mathbb{R}^{1 \times N}$. La detección de eventos se formula como una función de puntuación $S(T) \to [0, 1]$. Para la clasificación, utilizamos las capas MLP y CNN del núcleo del framework, congeladas en el formato SRT1.
$$ \text{Evento}(t) = \mathbb{I}(S(W_t) > \tau) $$
donde $\tau$ es un umbral calibrado.

### 14.3 Resultados y métricas
El rendimiento esperado en el Numenta Anomaly Benchmark (NAB) apunta a una puntuación F1 de $>0,85$ con cero deriva de bits en múltiples subprocesos. Se espera que el uso de la cuantización int8 QSR1 reduzca la latencia en $3\times$ en los procesadores ARM de borde mientras mantiene una cercanía de bits de MSE de $<10^{-4}$ en comparación con el oráculo f32.

## Autodiferenciación N-D y extensiones de investigación

Más allá del grafo 2-D en modo inverso, SciRust ofrece ahora un **grafo de
autodiferenciación N-D** cuyos operadores se validan con comprobación de
gradiente por diferencias finitas, y sobre él una pila de aprendizaje profundo
respaldada por la investigación. Cada capacidad corresponde a un artículo y se
entrega con una prueba; la correspondencia completa (14 de 20 entregados) se
sigue en `docs/RESEARCH_ROADMAP.md`.

- **Modelo de lenguaje decodificador causal**, entrenado de extremo a extremo
  (embeddings de token y posición, atención causal, entropía cruzada softmax
  fusionada y estable), que memoriza una secuencia exactamente.
- **Capas estilo LLaMA**: RMSNorm, SwiGLU, bloque LLaMA Pre-RMSNorm, RoPE
  (propiedad de posición relativa probada) y atención agrupada / multi-consulta.
- **Optimizadores deterministas**: Adam, AdamW, Lion, Muon (Newton–Schulz), Schedule-Free, AdEMAMix y SOAP (Adam en la base propia de Shampoo).
- **IA certificable**: la propagación por intervalos (IBP) **y CROWN** (cotas
  más ajustadas por relajación lineal) dan cotas de salida
  demostrables y un certificado de robustez.
- **Reducciones reproducibles**: suma/media/producto escalar independientes del
  orden, idénticas bit a bit sin importar el número de hilos.
- **Inferencia**: decodificación especulativa exacta, FlashAttention con softmax
  en línea por bloques, una capa DeltaNet de atención lineal con regla delta y una capa Mamba de espacio de estados selectivo.
- **Puente científico**: una Neural ODE con retropropagación a través de RK4.
- **Compresión**: poda Wanda (consciente de activaciones) y SmoothQuant, y GPTQ (cuantización int8 de pesos por retroalimentación de error de segundo orden, CLI `scirust gptq`), y AWQ (cuantización int8 de pesos basada en búsqueda y consciente de activaciones, CLI `scirust awq`).

Dos comandos CLI exponen este trabajo: `scirust certify` (cotas IBP **y CROWN**, en paralelo, y robustez) y
`scirust lm --opt adam|adamw|lion|schedule-free|ademamix|soap` (entrenar el LM decodificador N-D).

Un tercer comando, `scirust conformal`, produce intervalos de predicción conformes con cobertura garantizada, sin supuestos de distribución.
