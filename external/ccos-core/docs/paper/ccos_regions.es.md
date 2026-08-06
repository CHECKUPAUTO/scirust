# Introducción

Un agente de código basado en LLM que opera sobre un repositorio debe
decidir repetidamente *qué poner en la ventana de contexto*. El patrón
dominante es la recuperación: incrustar fragmentos de código,
clasificarlos frente a la tarea, y concatenar el top-$k$ hasta agotar un
presupuesto de tokens . Esto trata el contexto como una lista
clasificada unidimensional. De ahí se siguen dos modos de fallo.
Primero, la *dilución*: a medida que las tareas se alargan, la ventana
acumula código globalmente saliente pero irrelevante para la tarea, y
los modelos atienden mal al medio de los contextos largos . Segundo, la
*incoherencia*: los $k$ fragmentos mejor clasificados no tienen por qué
formar una unidad conexa de razonamiento — una función puede paginarse
sin sus llamadores, sus sitios de error o los datos de los que depende.

Los sistemas operativos afrontaron el problema análogo hace décadas y
respondieron con el *working set*: el conjunto de páginas referenciadas
recientemente que deben permanecer residentes . CCOS adopta la analogía
para el contexto de LLM : el código fuente se analiza en un grafo
causal, los nodos se puntúan, y una «ventana de contexto» acotada se
pagina dentro y fuera como RAM $\leftrightarrow$ VRAM. A diferencia de
una pila de recuperación, cada transición se registra en un registro de
eventos encadenado por hash, de modo que la memoria no es una caja negra
sino un artefacto *auditable y reproducible*. Nuestras contribuciones
son:

1.  Un **modelo formal y determinista de regiones de contexto**
    (§<a href="#sec:regions" data-reference-type="ref"
    data-reference="sec:regions">4</a>,
    §<a href="#sec:determinism" data-reference-type="ref"
    data-reference="sec:determinism">5</a>): la distancia causal como
    camino más corto ponderado, la pertenencia a una región como
    componente conexa del grafo de enlaces causales entre archivos, y un
    teorema de determinismo — las regiones y la ventana paginada son una
    función pura del grafo, así que una sesión se reconstruye bit a bit
    desde su registro de eventos encadenado por hash (reproducción a
    prueba de manipulación).

2.  **Sesiones de agente event-sourced con depuración por viaje en el
    tiempo** (§<a href="#sec:timetravel" data-reference-type="ref"
    data-reference="sec:timetravel">6</a>): cada operación cognitiva
    (ingesta, señal de fallo, recuerdo) queda registrada; el estado
    exacto del contexto en cualquier paso es reconstruible; y un
    recuerdo puede *reproducirse bajo parámetros distintos* para
    preguntar si el agente habría decidido mejor — la capacidad que a
    una pila de recuperación probabilística le falta estructuralmente.
    Hasta donde sabemos, es el primer tratamiento del *ensamblaje del
    contexto en sí* como un subsistema reproducible y depurable a
    posteriori — una *caja negra de la atención de un agente* — con un
    *punto de interrupción de desalojo* que nombra el paso y la
    operación exactos en que la causa verdadera fue expulsada de la
    ventana acotada.

3.  Un **arnés de validación honesto y un resultado negativo**
    (§<a href="#sec:protocol" data-reference-type="ref"
    data-reference="sec:protocol">10</a>): sobre $70$ commits reales de
    corrección, la selección causal *no* supera a un recuperador léxico
    TF-IDF al colocar los archivos de una corrección en la ventana
    (empata, y pierde con un presupuesto ajustado), y un pivote por
    traza de fallo es superado por un RAG-sobre-el-mensaje-de-error. Lo
    informamos con claridad — reubica el valor de CCOS de la
    *recuperación* a la *auditabilidad*.

4.  Una **medición de localidad sin LLM**
    (§<a href="#sec:eval" data-reference-type="ref"
    data-reference="sec:eval">8</a>) con cifras reales y reproducibles,
    y un protocolo falsable de condición *suficiente* (Fase 4) que
    especificamos pero dejamos abierto.

# Trabajo relacionado

#### Generación aumentada por recuperación.

RAG aumenta un modelo paramétrico con un almacén no paramétrico y
recupera pasajes por consulta . Self-RAG añade tokens de reflexión que
regulan la recuperación y critican las generaciones . Estos operan sobre
fragmentos *independientes*; la coherencia entre elementos recuperados
no se modela.

#### Recuperación consciente de grafos y estructura.

GraphRAG construye un grafo de conocimiento de entidades y responde
consultas globales resumiendo la estructura comunitaria . Para código,
los grafos de propiedades unifican AST, flujo de control y flujo de
datos . Las regiones de CCOS están en este linaje pero apuntan a la
*paginación*: qué subestructura conexa hacer residente para una tarea,
bajo un presupuesto de tokens, con desalojo determinista.

#### Memoria de agente.

MemGPT presenta el LLM como un SO que pagina entre una ventana en
contexto y almacenamiento externo ; los Generative Agents recuperan
recuerdos puntuados por recencia, importancia y relevancia  — los mismos
factores que CCOS agrega en una temperatura de región. LangGraph 
estructura los agentes como grafos de pasos con estado; orquesta el
flujo de *control*, mientras que CCOS estructura la *memoria*. Ambos son
complementarios.

#### Gestión de la ventana de contexto.

El desalojo de la caché KV mantiene residentes los tokens «grandes
impactadores» o sumidero , y PagedAttention aplica la paginación de SO a
la caché KV . Estas técnicas actúan a nivel de token dentro de un único
paso hacia adelante; CCOS actúa al nivel *semántico* a lo largo de una
sesión de agente.

# El sustrato de contexto causal

CCOS analiza el código fuente en un *grafo de memoria causal* dirigido
$G=(V,E)$. Un nodo $v\in V$ es un archivo, módulo, símbolo o dependencia
externa, con campos escalares $\mathrm{imp}(v)$ (importancia base),
$\mathrm{fail}(v)\in[0,1]$ (relevancia de fallo) y
$\mathrm{rec}(v)\in[0,1]$ (recencia). Una arista $e=(u\!\to\!w)\in E$
lleva un peso $w(e)\in(0,1]$ y un tipo (contención, dependencia,
referencia, causación). El núcleo asigna a cada nodo una puntuación
causal
$$\mathrm{score}(v) = \mathrm{clamp}\big(0.15\,\mathrm{imp}(v) + 0.50\,\mathrm{fail}(v)
  + 0.30\,\mathrm{rec}(v) + 0.05\ln(1{+}\mathrm{acc}(v)),\,0,1\big),
\label{eq:score}$$ donde $\mathrm{acc}(v)$ es el conteo de accesos. Los
fallos se propagan a lo largo de las aristas, $\mathrm{fail}$ decae con
un reloj lógico, y cada transición de estado se añade a un registro de
eventos encadenado por hash que permite una reproducción determinista .
Los identificadores de nodo tienen espacio de nombres (`file:p`,
`mod:p:n`, `use:p:path`, `sym:p:n`, `dep:root`); el archivo propietario
de un nodo es recuperable desde su identificador. CCOS v0.2 pagina los
nodos de mayor $\mathrm{score}$: una política plana y 1-D. Ahora hacemos
espacial la selección.

# Regiones de contexto

## Distancia causal

<div class="definition">

**Definición 1** (Distancia causal). *Sea $\hat G$ el multigrafo no
dirigido sobre $V$ inducido por $E$, y asignemos a cada arista el coste
$c(e) = -\ln w(e) \ge 0$, de modo que un enlace causal más fuerte sea un
paso más corto. La *distancia causal* $d_{\mathrm{c}}(u,v)$ es el coste
total mínimo sobre todos los caminos $u$–$v$ en $\hat G$, y $+\infty$ si
no existe ninguno. La distancia en saltos no ponderada
$\mathrm{hops}(u,v)$ se define análogamente con costes unitarios.*

</div>

$c(e)\ge 0$ puesto que $w(e)\le 1$, así que $d_{\mathrm{c}}$ es una
verdadera métrica de camino más corto en cada componente conexa (no
negativa, simétrica, desigualdad triangular). El *vecindario causal a
$k$ saltos* de un objetivo $t$ es
$$\mathcal{N}_k(t) = \{\, v \in V : \mathrm{hops}(t,v) \le k \,\}.$$
$\mathcal{N}_k(t)$ es la verdad-base «lo que es causalmente relevante
para una tarea en $t$» usada en la evaluación
(§<a href="#sec:eval" data-reference-type="ref"
data-reference="sec:eval">8</a>).

## Pertenencia a una región

Los concentradores de dependencias externas (p. ej. `dep:std`) conectan
casi todo y no deben colapsar el grafo en una sola componente. Por eso
los separamos.

<div class="definition">

**Definición 2** (Enlace causal entre archivos). *Para un nodo no
externo $v$ sea $\phi(v)$ su archivo propietario. Dos archivos $f,g$
están *directamente enlazados*, escrito $f \approx g$, si y solo si
existe una arista $(u\!\to\!w)\in E$ con $\phi(u)=f$, $\phi(w)=g$,
$f\neq g$, y ni $u$ ni $w$ es un nodo de dependencia externa.*

</div>

<div id="def:region" class="definition">

**Definición 3** (Región). *Sea $\approx^{*}$ la clausura
reflexiva-transitiva de $\approx$ sobre el conjunto de claves de
archivo. Una *región* es el conjunto de todos los nodos cuyos archivos
están en una misma clase de equivalencia de $\approx^{*}$; todos los
nodos de dependencia externa forman una región adicional. Dos nodos
pertenecen a la misma región si y solo si sus archivos están en la misma
componente conexa del grafo de enlaces causales entre archivos.*

</div>

<div class="proposition">

**Proposición 1** (Las regiones particionan los nodos). *Las regiones de
la Definición <a href="#def:region" data-reference-type="ref"
data-reference="def:region">3</a> forman una partición de $V$: cada nodo
está en exactamente una región.*

</div>

<div class="proof">

*Proof.* $\approx^{*}$ es una relación de equivalencia sobre las claves
de archivo, así que sus clases particionan las claves; el cubo externo
es una clase distinta por construcción. Aplicar cada nodo no externo a
la clase de su archivo y cada nodo externo a la clase externa es una
función total hacia bloques disjuntos, y por tanto una partición. 0◻ ◻

</div>

Por defecto (solo aristas de contención e importación) cada archivo
fuente es su propia región. Una verdadera dependencia entre archivos o
un fallo propagado fusiona archivos en una única región multiarchivo —
la «zona de conocimiento» que un agente debería despertar junta.

## Escalares de región

Para una región $R$ con conjunto de miembros $M$: $$\begin{aligned}
\mathrm{heat}(v)        &= 0.5\,\mathrm{score}(v) + 0.3\,\mathrm{fail}(v) + 0.2\,\mathrm{rec}(v), \\
\mathrm{temp}(R)        &= \mathrm{clamp}\!\Big(\tfrac{1}{|M|}\textstyle\sum_{v\in M}\mathrm{heat}(v),\,0,1\Big), \label{eq:temp}\\
\mathrm{dens}(R)        &= \frac{|\{\,e\in E : \text{ambos extremos} \in M\,\}|}{|M|}. \label{eq:dens}
\end{aligned}$$ La temperatura es cuán «despierta» está una región; la
densidad es su cohesión causal interna (aristas internas por miembro).
La activación calienta una región ($\mathrm{temp}\mathrel{+}=0.25$, con
tope) y registra un tic lógico; un paso de enfriamiento multiplica las
temperaturas por un factor de decaimiento y desaloja las regiones por
debajo de un umbral.

## Política de admisión dinámica

El umbral estático $0.6$ se convierte en una función de la presión de
tokens $u\in[0,1]$ (la fracción usada del presupuesto) y la complejidad
de la tarea $\kappa\in[0,1]$: $$\begin{aligned}
\theta(u,\kappa)       &= \mathrm{clamp}(0.6 + 0.3\,u - 0.2\,\kappa,\; 0.05,\; 0.95), \\
a(R)                   &= 0.55\,\mathrm{temp}(R) + 0.30\,\frac{\mathrm{dens}(R)}{1+\mathrm{dens}(R)} + 0.15\,\kappa, \\
\mathrm{admit}(R)      &\iff a(R) \ge \theta(u,\kappa).
\end{aligned}$$ Una región caliente y cohesiva puede admitirse aun
cuando el $0.6$ estático la rechazaría; una ventana casi llena eleva
$\theta$ para que solo entren las regiones más calientes.

# Determinismo y reproducción

<div class="theorem">

**Teorema 1** (Determinismo regional). *La partición en regiones, cada
escalar de región de las
ecuaciones <a href="#eq:temp" data-reference-type="eqref"
data-reference="eq:temp">[eq:temp]</a>–<a href="#eq:dens" data-reference-type="eqref"
data-reference="eq:dens">[eq:dens]</a>, y el historial de activación son
funciones puras, independientes del orden, del grafo $G$ y de la
secuencia de eventos de activación (lógicamente marcados con tiempo). En
consecuencia, dado el registro de eventos de una sesión, el estado del
motor se reconstruye idénticamente: si $G'$ es el grafo reconstruido
desde el registro y $L$ sus eventos de región, entonces
$\mathrm{replay\_from}(G',L)$ es igual al motor en vivo que produjo
$L$.*

</div>

<div class="proof">

*Esbozo de prueba.* El agrupamiento enumera nodos y aristas en orden
ordenado y calcula las componentes conexas mediante una búsqueda en
anchura ordenada, de modo que su salida es independiente del orden de
iteración del hash. Las ecuaciones
<a href="#eq:score" data-reference-type="eqref"
data-reference="eq:score">[eq:score]</a>,
<a href="#eq:temp" data-reference-type="eqref"
data-reference="eq:temp">[eq:temp]</a> y
<a href="#eq:dens" data-reference-type="eqref"
data-reference="eq:dens">[eq:dens]</a> son aritmética determinista sobre
los campos de los nodos y un conteo de aristas que cualifican. La
activación lee un reloj lógico (un contador), nunca el tiempo de pared,
y el evento `RegionActivated` emitido registra el tic exacto. La
reconstrucción reagrupa $G'$ (estado base idéntico, ya que $G'\!=\!G$
estructuralmente) y aplica las activaciones registradas en el orden del
registro; cada paso es la misma función pura de las mismas entradas, así
que el estado resultante es idéntico. La prueba de integración
`replay_reconstructs_identical_engine` verifica
$\mathrm{engine}=\mathrm{replay\_from}(G',L)$ por igualdad estructural.
0◻ ◻

</div>

Esto extiende las garantías existentes de CCOS: el registro de eventos
principal está encadenado por hash y es a prueba de manipulación, de
modo que el historial de regiones es auditable, y una prueba de 10 000
ciclos no exhibe deriva alguna en el número de regiones ni en las
temperaturas.

#### Una frontera de entrada auditable.

El mismo sustrato determinista y reproducible endurece el texto que un
agente ingiere. Los vectores de inyección por caracteres ocultos —
anulaciones bidireccionales (el ataque *Trojan Source*, CVE-2021-42574),
formato de ancho cero y el bloque Unicode *Tags* usado para el
contrabando de ASCII invisible — se des-ofuscan en la ingesta a
literales explícitos y visibles (`[U+202E RLO]`) en lugar de eliminarse
en silencio, y los hallazgos se registran en el mismo registro
encadenado por hash, de modo que una reproducción reconstruye el estado
limpio. Un clasificador lineal en espacio logarítmico posterior — la
forma cerrada del Naïve Bayes multinomial,
$\mathrm{logit}=b+W\!\cdot\!X$ sobre un vector de características por
*hashing trick*, con sus pesos fijados en un blob verificado por suma de
comprobación — añade una señal de inyección *determinista y
descomponible forensemente* (un red-team sobre datos reservados mide
$F_1=0{,}90$; precisión $0{,}87$, exhaustividad $0{,}93$). Lo afirmamos
de forma deliberada: es una *señal, no una solución* — un modelo de
bolsa de características se evade con paráfrasis, y ninguna pasada a
nivel de carácter aborda la inyección semántica; la mitigación real es
la separación de privilegios en el host. La contribución no es un
detector novedoso, sino que la des-ofuscación y la puntuación heredan la
propiedad distintiva de CCOS: toda decisión de seguridad es reproducible
y auditable hasta la característica exacta que la inclinó.

# Sesiones event-sourced y depuración por viaje en el tiempo

El determinismo no es solo una propiedad de corrección; es la base de la
capacidad que, como argumentará nuestra evaluación, distingue a CCOS de
una pila de recuperación. Una `AgentSession` registra cada operación
cognitiva que un agente realiza sobre su memoria — `Ingest`,
`SignalFailure`, `Recall` — como una línea de tiempo ordenada. Dado que
cada operación es una función determinista del estado previo, la memoria
tras las primeras $n$ operaciones se reconstruye exactamente
reproduciendo las que mutan (`replay_to(n)`): se puede *rebobinar la
mente del agente* a cualquier paso.

La operación que importa para la depuración es la contrafactual.
`recall_what_if(`$n$`, `$q$`, `$b$`)` reproduce la memoria hasta el paso
$n$ y vuelve a ejecutar un recuerdo con una consulta $q$ o un
presupuesto de tokens $b$ distintos, devolviendo la ventana que el
agente *habría* visto. Cuando un agente emite un mal parche en el paso
15, un operador puede reproducir su contexto exacto en el paso 14,
ampliar el presupuesto o cambiar el ancla, y observar si la decisión
mejora — un depurador de bucle cerrado y reproducible para la memoria de
trabajo de un agente.

#### Un punto de interrupción sobre la atención.

La reproducción también localiza la *deriva*. El punto de observación
`missing` recorre la línea de tiempo en busca del paso exacto en que un
nodo dado — típicamente la causa verdadera de un fallo — fue expulsado
de la ventana acotada por la presión competidora, nombrando la operación
desencadenante y la diferencia de tokens; una vista `energy`
complementaria revela la migración de calor causal por el grafo que un
diff a nivel de archivo pasa por alto. Es, en efecto, un *punto de
interrupción sobre la atención de un agente*: el instante preciso en que
la información correcta abandonó la ventana. Una pila de recuperación
opaca solo puede mostrar el contexto final, corrupto; CCOS puede señalar
el paso — y la operación — en que se torció.

Una pila RAG o de marco muta su almacén de forma probabilística a lo
largo de una sesión y no conserva ningún registro de transiciones
canónico y reproducible, así que la misma pregunta (*¿por qué, y cuándo,
se corrompió la representación?*) no tiene respuesta allí. No afirmamos
que esto mejore el éxito en las tareas; afirmamos que es una propiedad
que las alternativas no tienen, y que es el lugar honesto de la
contribución de CCOS (§<a href="#sec:protocol" data-reference-type="ref"
data-reference="sec:protocol">10</a>).

# Implementación

El motor son $\approx\!600$ líneas de Rust ligero en dependencias, en
capa sobre el grafo, sin cambios en el analizador, el guardián, el
constructor incremental o el núcleo event-sourcing. La superficie
pública es: `ContextRegionEngine` (`cluster_nodes`,
`initialize_regions`, `activate_region`, `tick_cooldown`,
`replay_from`), los tipos de datos `ContextRegion` / `ContextPoint` /
`ContextWindow`, la `ContextPolicy`, y cinco nuevas variantes de evento
(`RegionCreated`, `RegionActivated`, `RegionMerged`, `RegionEvicted`,
`ContextWindowGenerated`). Una CLI `ccos regions` expone el
agrupamiento, la activación y el informe de localidad. Toda la crate
compila sin advertencias bajo `clippy -D warnings` y pasa 364 pruebas.

# Evaluación: lo que podemos medir hoy

Separamos los resultados *medidos* (esta sección, totalmente
reproducible y sin LLM) de las ganancias *hipotéticas* a nivel de agente
(§<a href="#sec:protocol" data-reference-type="ref"
data-reference="sec:protocol">10</a>).

#### Montaje.

Operamos sobre el propio árbol de fuentes de CCOS ($|V|=705$ nodos,
$|E|=822$ aristas, dando $26$ regiones). Para un nodo objetivo $t$
tomamos $\mathcal{N}_k(t)$ con $k=2$ como verdad-base y comparamos dos
estrategias de selección al presupuesto de tamaño de la región:
**plana** — los $|R|$ mejores nodos por $\mathrm{score}$ (CCOS v0.2 /
recuperación clasificada clásica) — y **región** — los miembros de la
región de $t$. Reportamos precisión causal $|S\cap\mathcal{N}_k|/|S|$,
exhaustividad $|S\cap\mathcal{N}_k|/|\mathcal{N}_k|$, y los tokens que
cada estrategia necesita para *cubrir* $\mathcal{N}_k$. Todas las cifras
las emite `scripts/region_benchmark.sh` y son deterministas.

<div id="tab:locality">

| Estrategia                 | precisión causal | exhaustividad causal |
|:---------------------------|:----------------:|:--------------------:|
| plana (v0.2 / clasificada) |      0.021       |        0.347         |
| **región (v0.3)**          |    **0.057**     |      **0.972**       |

Localidad sobre `src/` ($k=2$, 12 objetivos). La selección por región
cubre el $97\%$ del vecindario causal de una tarea frente al $35\%$ de
la selección plana, con $\approx\!48\%$ menos tokens para igual
cobertura.

</div>

#### Cohesión y coste.

Las regiones alcanzan una **densidad causal media de $0{,}955$**
(Ec. <a href="#eq:dens" data-reference-type="ref"
data-reference="eq:dens">[eq:dens]</a>, normalizada) — están casi por
completo conectadas internamente, es decir, agrupaciones causales
genuinas y no grupos de archivos arbitrarios. Construir el mapa de
regiones para $705$ nodos lleva $\approx\!20$ ms ($54{,}5$
construcciones/s); una sola activación son $\approx\!20$ ms.

#### Lectura honesta.

La precisión absoluta es baja ($0{,}06$) porque el analizador v0.2 solo
emite aristas de contención e importación, así que $\mathcal{N}_k(t)$ es
diminuto (a menudo $2$–$3$ nodos) mientras que una región abarca un
archivo entero. Las ganancias robustas y reales son la *exhaustividad*
($0{,}97$ frente a $0{,}35$) y la *eficiencia en tokens*
($\approx\!48\%$): una región contiene de forma fiable el vecindario
causal de una tarea y paga menos tokens por hacerlo. Aristas semánticas
más ricas (grafo de llamadas / flujo de datos, ítem P1.3 de la hoja de
ruta) afinarían $\mathcal{N}_k$ y son la palanca principal para elevar
la precisión; lo señalamos en vez de ocultarlo.

# Simulación de hipótesis bajo un oráculo declarado (sin LLM)

La tesis central — *¿ayuda la memoria causal por regiones a un agente en
tareas largas multiarchivo?* — en última instancia necesita despliegues
de LLM (§<a href="#sec:protocol" data-reference-type="ref"
data-reference="sec:protocol">10</a>). Pero su *condición necesaria* es
comprobable ya, sin LLM: **un agente no puede resolver una tarea cuyo
contexto causal requerido está ausente de su ventana**. Medimos, bajo un
oráculo explícito, si cada estrategia de selección *coloca el contexto
causal requerido en la ventana*.

#### Montaje.

Generamos repositorios sintéticos modulares: $S$ subsistemas
independientes, cada uno un conjunto de archivos enlazados por una
cadena causal entre archivos más símbolos *señuelo* de alta puntuación;
no hay aristas entre subsistemas, así que cada subsistema es una región
causal acotada. La estructura causal está *desacoplada* de la estructura
léxica — un vecino de la cadena no comparte ningún token de
identificador con el objetivo, solo una arista — modelando una
dependencia causalmente esencial pero léxicamente distinta. Una tarea de
*diámetro* $d$ requiere la ventana $\pm d$ a lo largo de su cadena
(hasta $2d{+}1$ archivos); el presupuesto contiene $\approx$ un
subsistema, mucho menos que el repositorio. El **oráculo** es
$R(t)\subseteq S$.

Crucialmente, los *pipelines* de recuperación localizan código a partir
de una *consulta* textual, mientras que una memoria de tipo SO se
*ancla* en la señal del espacio de trabajo (el archivo activo, una
prueba que falla). Modelamos ambos: cada tarea lleva una consulta (una
bolsa de tokens) y un ancla (el nodo del archivo activo), y ejecutamos
dos escenarios — **limpio** (la consulta apunta al objetivo) y
**ruidoso** (un señuelo trampa en un subsistema no relacionado supera
léxicamente al objetivo). Seis estrategias comparten el presupuesto:
`rag-dense` (top-$k$ léxico), `rag-hybrid` (léxico $+$ puntuación
causal), `graphrag-1hop` (mejor acierto $+$ un salto), `graphrag-bfs`
(expansión no acotada desde el mejor acierto), `ccos-from-query` (región
CCOS del mejor acierto *léxico* — una ablación), y `ccos-region` (región
CCOS del *ancla*). Con semilla y deterministas; reproducir con
`ccos experiment`.

<div id="tab:sim">

|                                                                | éxito al diámetro $d$ |          |          |          |  global  |          |
|:---------------------------------------------------------------|:---------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 2-5(lr)6-7 Estrategia                                          |        $d{=}1$        | $d{=}2$  | $d{=}3$  | $d{=}4$  |  éxito   |   cob.   |
| *Consulta limpia (apunta al objetivo)*                         |                       |          |          |          |          |          |
| `rag-dense`                                                    |         0.00          |   0.00   |   0.00   |   0.00   |   0.00   |   0.19   |
| `rag-hybrid`                                                   |         1.00          |   0.00   |   0.00   |   0.00   |   0.23   |   0.65   |
| `graphrag-1hop`                                                |         1.00          |   0.00   |   0.00   |   0.00   |   0.23   |   0.58   |
| `graphrag-bfs`                                                 |         1.00          |   1.00   |   1.00   |   1.00   |   1.00   |   1.00   |
| `ccos-from-query`                                              |         1.00          |   1.00   |   1.00   |   1.00   |   1.00   |   1.00   |
| `ccos-region`                                                  |         1.00          |   1.00   |   1.00   |   1.00   |   1.00   |   1.00   |
| *Consulta ruidosa (un señuelo supera léxicamente al objetivo)* |                       |          |          |          |          |          |
| `rag-dense`                                                    |         0.00          |   0.00   |   0.00   |   0.00   |   0.00   |   0.19   |
| `rag-hybrid`                                                   |         1.00          |   0.00   |   0.00   |   0.00   |   0.23   |   0.65   |
| `graphrag-1hop`                                                |         0.00          |   0.00   |   0.00   |   0.00   |   0.00   |   0.19   |
| `graphrag-bfs`                                                 |         0.00          |   0.00   |   0.00   |   0.00   |   0.00   |   0.00   |
| `ccos-from-query`                                              |         0.00          |   0.00   |   0.00   |   0.00   |   0.00   |   0.00   |
| **`ccos-region`**                                              |       **1.00**        | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |

Simulación de hipótesis ($800$ tareas, semilla $42$, presupuesto
$\approx$ un subsistema). Éxito $=$ el conjunto causal requerido $R(t)$
está dentro de la ventana. Bajo una consulta limpia, todo método
consciente de la estructura empata; bajo una consulta engañosa, solo
sobrevive la región anclada al espacio de trabajo.

</div>

#### Hallazgos (y sus límites honestos).

**(1)** La recuperación léxica (`rag-dense`) *falla* en tareas causales
entre archivos ($0\%$; cobertura $0{,}19$): la similitud no puede hacer
aflorar contexto causalmente esencial pero léxicamente distinto — la
premisa de la hipótesis. (Las victorias en $d{=}1$ de `rag-hybrid`
vienen de la *puntuación causal* que toma prestada, no de la similitud
léxica, razón por la cual persisten bajo ruido.) **(2)** El valor de la
selección consciente de la estructura *crece con el diámetro*: la
expansión a un salto solo resuelve $d{=}1$, mientras que la paginación
causal completa (`graphrag-bfs`, `ccos-region`) resuelve todos los $d$ —
la dirección de H2. **(3) El empate limpio.** Bajo una consulta limpia,
`graphrag-bfs`, `ccos-from-query` y `ccos-region` alcanzan todas
$1{,}00$: la palanca es la *estructura* causal, **no CCOS en sí**. **(4)
La separación ruidosa.** Bajo una consulta *engañosa*, todo método que
localiza código léxicamente colapsa a $0\%$ — incluido el fuerte
`graphrag-bfs` *y* la ablación `ccos-from-query` (CCOS sembrado en la
consulta). Solo `ccos-region`, que se ancla en la señal del espacio de
trabajo en vez de la consulta, sobrevive a $1{,}00$. La ablación aísla
el diferenciador: es la *fuente del ancla* (una señal estructural del
espacio de trabajo frente a una consulta léxica), no la maquinaria de
regiones. Este es el régimen realista — una descripción de tarea rara
vez nombra la causa distante exacta, y una memoria de tipo SO que
rastrea el working set activo es robusta donde la recuperación en tiempo
de consulta es engañada. **(5) Supuestos, declarados.** Esto acredita a
CCOS con un ancla fiable (el archivo activo / la prueba que falla), que
una memoria a nivel de SO tiene pero un *pipeline* de pura recuperación
no; y supone repositorios *modulares* con regiones separables (densidad
$0{,}955$ en código real, §<a href="#sec:eval" data-reference-type="ref"
data-reference="sec:eval">8</a>) — una región gigante monolítica que
exceda el presupuesto colapsa la ventaja (observado, reportado). **(6)**
En todo momento esto es una *simulación bajo un oráculo declarado*:
prueba la condición *necesaria* (recuperación), no la *suficiente*
(generación) de la §<a href="#sec:protocol" data-reference-type="ref"
data-reference="sec:protocol">10</a>.

# Evaluación con LLM real: una primera medición en dispositivo

La simulación (§<a href="#sec:sim" data-reference-type="ref"
data-reference="sec:sim">9</a>) estableció la condición necesaria. La
condición suficiente — que un agente LLM *resuelva* luego más tareas con
memoria por regiones que con recuperación por fragmentos — requiere
despliegues de modelo. **Implementamos el arnés** (`ccos eval`) y
reportamos una **primera ejecución con LLM real** contra un modelo
local.

#### Arnés implementado.

Cada tarea es un diminuto proyecto multiarchivo que codifica una *cadena
causal aritmética*: una constante base en un archivo se transforma a
través de una cadena de funciones de una línea en archivos separados, y
la pregunta «¿qué entero devuelve la última función?» solo es
respondible *leyendo* toda la cadena (la causa distante incluida). La
calificación es por tanto coincidencia exacta sobre un entero — objetiva
y automatizable, sin ejecución de código. Las mismas seis estrategias de
la §<a href="#sec:sim" data-reference-type="ref"
data-reference="sec:sim">9</a> ensamblan la ventana de contexto bajo un
presupuesto de tokens (con la división de consulta limpia/ruidosa), la
ventana se envía a un modelo (cualquier endpoint compatible con OpenAI,
Anthropic-Messages u Ollama), y registramos tres métricas por diámetro:
**éxito de tarea** (entero correcto), **cobertura del oráculo** (cadena
requerida $\subseteq$ ventana — independiente del modelo), y
**alucinación de símbolo** (la respuesta cita una función ausente del
proyecto).

#### Montaje.

Ejecutamos `ccos eval` en dispositivo sobre una NVIDIA Jetson AGX Thor
contra `qwen2.5:7b-instruct` servido por Ollama (temperatura $0$), con
$20$ tareas, semilla $7$, un presupuesto de $600$ tokens, diámetros de
$1$ a $4$, en ambos regímenes limpio y ruidoso. La
Tabla <a href="#tab:real" data-reference-type="ref"
data-reference="tab:real">3</a> reporta la ejecución.

<div id="tab:real">

|                                                                | éxito al diámetro $d$ |          |          |          |          |         |
|:---------------------------------------------------------------|:---------------------:|:--------:|:--------:|:--------:|:--------:|--------:|
| 2-5 Estrategia                                                 |        $d{=}1$        | $d{=}2$  | $d{=}3$  | $d{=}4$  |   cob.   |    tok. |
| *Consulta limpia (nombra la función objetivo)*                 |                       |          |          |          |          |         |
| `rag-dense`                                                    |         0.12          |   0.00   |   0.00   |   0.00   |   0.00   |     519 |
| `rag-hybrid`                                                   |         0.00          |   0.00   |   0.00   |   0.00   |   0.00   |     521 |
| `graphrag-1hop`                                                |         1.00          |   0.00   |   0.00   |   0.00   |   0.40   |     270 |
| `graphrag-bfs`                                                 |         1.00          |   0.00   |   0.17   |   0.00   |   1.00   |     402 |
| `ccos-from-query`                                              |         1.00          |   0.00   |   0.17   |   0.00   |   1.00   |     402 |
| `ccos-region`                                                  |         1.00          |   0.00   |   0.17   |   0.00   |   1.00   |     402 |
| *Consulta ruidosa (un señuelo supera léxicamente al objetivo)* |                       |          |          |          |          |         |
| `rag-dense`                                                    |         0.00          |   0.00   |   0.00   |   0.00   |   0.00   |     531 |
| `rag-hybrid`                                                   |         0.00          |   0.00   |   0.00   |   0.00   |   0.00   |     521 |
| `graphrag-1hop`                                                |         0.00          |   0.00   |   0.00   |   0.00   |   0.00   |     165 |
| `graphrag-bfs`                                                 |         0.00          |   0.00   |   0.00   |   0.00   |   0.00   |     165 |
| `ccos-from-query`                                              |         0.00          |   0.00   |   0.00   |   0.00   |   0.00   |     165 |
| **`ccos-region`**                                              |       **1.00**        | **0.00** | **0.17** | **0.00** | **1.00** | **402** |

Primera ejecución con LLM real: `qwen2.5:7b-instruct` vía Ollama, en
dispositivo (NVIDIA Jetson AGX Thor), $20$ tareas, semilla $7$,
presupuesto $600$ tokens. «cob.» es la cobertura del oráculo
independiente del modelo; «tok.» la media de tokens de entrada. El éxito
es una respuesta entera por coincidencia exacta.

</div>

#### Hallazgos.

**(1) La cobertura se transfiere de la simulación al texto real.** La
columna de cobertura independiente del modelo reproduce la
Tabla <a href="#tab:sim" data-reference-type="ref"
data-reference="tab:sim">2</a> sobre contenido de archivo real: el RAG
léxico cubre $0$, los métodos conscientes de la estructura cubren
$1{,}00$ bajo una consulta limpia, y bajo ruido *solo* `ccos-region`
mantiene $1{,}00$ — la condición necesaria, confirmada fuera del
simulador. **(2) El éxito está acotado por el modelo, no por el
contexto.** Donde la cobertura es $1{,}00$, el modelo de $7$B aún
resuelve solo $d{=}1$ ($1{,}00$) y parte de $d{=}3$ ($0{,}17$), fallando
$d{=}2,4$: con toda la cadena en la ventana, el fallo residual es el
*razonamiento aritmético* — un techo de capacidad del modelo, no un
fallo de selección. Crucialmente, ningún método tiene éxito jamás *sin*
cobertura (éxito $\subseteq$ cobertura), que es la afirmación que el
arnés está construido para probar. **(3) La separación ruidosa se
mantiene en un LLM real.** Bajo una consulta engañosa, todo método
impulsado por la consulta colapsa a $0\%$ de éxito — incluidos
`graphrag-bfs` y la ablación `ccos-from-query` — mientras que
`ccos-region`, anclada en la señal del espacio de trabajo, es la *única*
estrategia con éxito no nulo ($d{=}1$ a $1{,}00$). Los métodos
impulsados por la consulta incluso parecen *más baratos* bajo ruido
($165$ frente a $402$ tokens) precisamente porque recuperan con
confianza el archivo equivocado, más pequeño; `ccos-region` gasta su
presupuesto en la cadena causalmente correcta. **(4) Límites honestos.**
$20$ tareas ($\approx 5$ por diámetro) es una muestra pequeña, y un
modelo de $7$B pone en el suelo la condición suficiente. Un modelo de
frontera (`deepseek-v4-pro`, en curso) o un modelo local de $70$B se
espera que eleve el éxito allí donde la cobertura es $1{,}00$,
*agudizando* — no cambiando — la separación, ya que la cobertura fija el
techo.

#### Hacia benchmarks externos.

La suite de cadena aritmética aísla la pregunta de la selección; las
pruebas externas decisivas son la resolución de incidencias de
SWE-bench  (un parche pasa las pruebas ocultas) y una suite controlada
de *errores multiarchivo* donde el sitio de la falla, su causa y su
radio de impacto residen en archivos distintos. Las referencias
comparten el LLM base y el presupuesto de tokens: RAG clásico  (top-$k$
coseno sobre fragmentos), GraphRAG  (resúmenes de comunidades sobre un
grafo de código), MemGPT  (memoria paginada de tipo SO), un agente
LangGraph  con un almacén vectorial, y las regiones de CCOS. Métricas:
tasa de resolución, tokens de entrada hasta el éxito, y una tasa de
alucinación de fundamentación de citas verificada contra el grafo
verdad-base. Hemos comenzado esto sobre historia real:
`scripts/causal_validation` extrae commits de corrección, recupera el
árbol previo a la corrección, inyecta la falla, y puntúa
$R_{\mathrm{cov}} = |F_{\mathrm{target}} \cap \mathrm{WorkingSet}_K| / |F_{\mathrm{target}}|$
— la fracción de los archivos que una corrección tocó que el working set
acotado recupera. Sobre la historia de este repositorio, la primera
ejecución expuso una limitación y su corrección, ambas medidas: con
propagación de fallo solo aguas abajo, $R_{\mathrm{cov}}$ era plano en
$0{,}33$ (solo el archivo semilla recuperado, ya que los archivos
co-modificados son importadores *aguas arriba* alcanzados solo por
concentradores de dependencias). Resolver las importaciones intra-crate
en aristas archivo$\to$archivo y propagar el fallo *bidireccionalmente*
corrige esto. En tres crates maduras — `fd`, `bat` e `hyperfine`, $70$
commits de corrección extraídos — el efecto es consistente: a
presupuesto suficiente ($K{\ge}50$) los dos cambios elevan
$R_{\mathrm{cov}}$ a $0{,}85$–$1{,}0$ (desde $0{,}50$–$0{,}84$ solo
aguas abajo), mientras se *diluye* a $0{,}19$–$0{,}28$ a un presupuesto
ajustado $K{=}20$. **Pero frente a la referencia obvia, esto no es una
victoria.** Ejecutando un RAG léxico clásico (coseno TF-IDF sobre el
texto de los archivos, consultado por el archivo de la falla) al *mismo
presupuesto de archivos*, $R_{\mathrm{cov}}$ para CCOS / RAG es
$0{,}92/0{,}94$, $1{,}00/0{,}98$, $0{,}87/0{,}92$ a $K{=}50$ y empata a
$K{=}100$, mientras que a $K{=}20$ el RAG va claramente por delante
($0{,}20/0{,}56$ en `bat`, $0{,}20/0{,}73$ en `hyperfine`). La selección
causal no tiene por tanto *ninguna ventaja neta de cobertura* sobre la
similitud léxica aquí, y es peor a presupuesto ajustado: en errores
reales los archivos de una corrección se parecen léxicamente entre sí,
así que el TF-IDF también los recupera — la premisa de la
§<a href="#sec:sim" data-reference-type="ref"
data-reference="sec:sim">9</a> de una causa léxicamente distinta de su
síntoma no se reproduce en estos repositorios. La lectura honesta es que
la condición *necesaria* se cumple para la gran mayoría pero *no* es
específica de CCOS; una ventaja real tendría que venir de los regímenes
que este montaje no prueba — una consulta degradada/ausente
(§<a href="#sec:sim" data-reference-type="ref"
data-reference="sec:sim">9</a>), un síntoma (prueba que falla)
léxicamente lejos de su causa (que necesita la siembra real guiada por
la prueba, no la heurística de mayor grado), o la condición *suficiente*
(Fase 4). Los espacios de trabajo multi-crate, ahora enlazables, también
quedan por medir a escala.

#### Condición suficiente (Fase 4): la resolución empata, la eficiencia no.

Ejecutamos la mitad de generación sobre correcciones reales de un solo
archivo de `fd`: a un agente se le da un contexto de igual presupuesto
construido de dos formas — la región causal de CCOS frente a un RAG
léxico top-$k$ — se le pide reescribir el archivo con el error, y se
califica según si `cargo test` pasa, con un reintento *compilador en el
bucle* (al fallar, una *falla de página de contexto* analiza el error
con la capa de traza, inyecta presión en los archivos que fallan, y
vuelve a indicar con una ventana refrescada). Con un modelo débil de
$7$B y un solo intento, ambos resuelven $0/15$; con `qwen3-coder-30B` y
tres intentos el bucle eleva ambos a $2/10$ ($20\%$) — es la
retroalimentación por falla de página, no el recuperador, lo que
desbloquea la resolución, y CCOS no muestra **ninguna ventaja de
resolución** sobre el RAG, consistente con el resultado de recuperación.
La *eficiencia* los separa: a igual resolución, la región auto-acotada
de CCOS promedia $776$ tokens de contexto frente a los $5366$ que llenan
el presupuesto del recuperador léxico — una reducción de
$\mathbf{6{,}9\times}$. Lo mismo se cumple a escala sin modelo: en $51$
escenarios de corrección de un solo archivo de `fd`, `bat` e
`hyperfine`, CCOS ensambla $700$–$1600$ tokens de contexto frente a los
$\approx\!6000$ del RAG, una reducción de
$\mathbf{4{,}1}$–$\mathbf{9{,}1\times}$. Este es el único eje en el que
CCOS domina: no *lo que* recupera (una referencia lo iguala) sino *cuán
poco* necesita, porque la región causal se detiene en el working set en
lugar de rellenar un top-$k$ hasta el presupuesto. Enunciamos las
salvedades: la muestra de *resolución* es diminuta ($n{=}10$, dos
pasadas), y la referencia llena el presupuesto por construcción (un RAG
con $k$ cuidadosamente ajustado también sería más disperso). La
afirmación defendible es más estrecha pero real: CCOS *se autocalibra* —
se acota en la región causal sin $k$ ni presupuesto que ajustar — así
que nunca desperdicia la ventana, que es precisamente el propósito de la
paginación bajo demanda.

#### Hipótesis.

**H1** (eficiencia): CCOS alcanza un éxito de tarea igual con menos
tokens de entrada que RAG/GraphRAG, porque una región cubre el
vecindario causal con mayor exhaustividad
(Tabla <a href="#tab:locality" data-reference-type="ref"
data-reference="tab:locality">1</a>). **H2** (éxito a largo horizonte):
en tareas multiarchivo, la tasa de resolución de CCOS supera al RAG por
fragmentos por un margen que crece con el diámetro causal de la tarea.
**H3** (fundamentación): CCOS reduce la tasa de alucinación de símbolos,
porque la ventana admitida es un subgrafo conexo de nodos *reales*.

#### Amenazas a la validez.

La calidad de una región está acotada por la calidad de las aristas
(§<a href="#sec:eval" data-reference-type="ref"
data-reference="sec:eval">8</a>); los resultados confundirían la
*política* de región con el *grafo* subyacente. Las ablaciones deben,
por tanto, fijar el grafo y variar solo la estrategia de selección, y
reportar el diámetro causal por tarea para que las ganancias se
atribuyan a la coherencia, no a recuperar más texto. La ablación
`ccos-from-query` de la
Tabla <a href="#tab:real" data-reference-type="ref"
data-reference="tab:real">3</a> es exactamente este control — mismo
grafo y presupuesto, solo el ancla intercambiada — y colapsa con las
referencias léxicas bajo ruido. Un resultado positivo en H1–H3
constituiría la contribución de investigación; un resultado nulo aún
validaría la infraestructura determinista y auditable.

# Limitaciones

Somos deliberadamente explícitos sobre lo que *no* se muestra. **(1)
Ninguna ventaja de cobertura sobre RAG en errores reales.** La métrica
principal sobre datos reales ($R_{\mathrm{cov}}$,
§<a href="#sec:protocol" data-reference-type="ref"
data-reference="sec:protocol">10</a>) empata con un simple recuperador
léxico TF-IDF a presupuesto suficiente y pierde ante él a uno ajustado;
en commits de corrección reales los archivos que una corrección toca se
parecen léxicamente, así que la premisa de la simulación de una causa
léxicamente distinta no se sostiene. Los fuertes números absolutos son
una condición *necesaria* que una referencia también satisface, no una
victoria de CCOS. (Una medición complementaria sobre la propia fuente
del motor aísla una relación distinta, definida estructuralmente: un
recuperador léxico TF-IDF recupera solo el $49\%$ (exhaustividad@10) de
las dependencias de importación resueltas por AST — pares que
rutinariamente no comparten vocabulario, coseno de dependencia $0{,}53$
frente a $0{,}43$ aleatorio — así que el punto ciego del RAG es real,
pero queda fuera del eje de cohesión de la corrección que mide la
métrica principal, donde los archivos de una corrección sí comparten
vocabulario. La capa estructural recupera esas aristas por construcción;
el AST, ahora por defecto, es lo que hace completa esa recuperación.)
**(2) Necesario $\neq$ suficiente.** Nunca medimos si una región *ayuda
a un agente a corregir* un error (Fase 4: un parche que pasa las pruebas
ocultas) — solo si los archivos relevantes son recuperables. **(3)
Heurística de siembra.** El arnés inyecta la falla en el archivo
modificado de mayor grado de salida, no en la prueba realmente fallida,
así que no ejercita el régimen donde un síntoma está léxicamente lejos
de su causa — el régimen más favorable a la selección causal. **(4)
Escala y estadística.** Tres repositorios de una sola crate,
$n\!\approx\!20$ cada uno; las diferencias se reportan con su desviación
estándar pero no se validan de forma cruzada. **(5) Motor.** El
analizador ahora usa por defecto un AST `syn` real — medido un
$36{,}5\%$ más preciso que la antigua heurística de línea sobre la
propia fuente del motor (dos tercios frente a exhaustividad de
importación completa), con la heurística conservada solo como reserva
para entradas que no son Rust — y la ingesta se endureció para
reconstruir de forma bit-idéntica entre procesos (resolución de
importaciones ordenada; centralidad invariante al orden). Las regiones
siguen siendo de granularidad de archivo y solo se fusionan en aristas
entre archivos explícitas; los pesos de puntuación están ajustados a
mano, aunque ya están disponibles refinamientos, desactivados por
defecto, de centralidad de autovalor y de ciclo de vida de nodo
(Stable/Working/Orphan). Nada de esto afecta a las propiedades
*probadas* (partición, determinismo, reproducción) ni a la localidad
medida — pero el caso empírico de un beneficio para el agente aguas
abajo está, en este punto, *sin probar*.

# Conclusión

Nos propusimos mostrar que organizar la memoria de un agente por
*regiones causales* recupera el contexto a largo horizonte mejor que la
recuperación por fragmentos, dimos a la construcción una definición
precisa y determinista, y probamos que se reconstruye bit a bit en la
reproducción. Luego probamos la afirmación de recuperación contra la
referencia obvia sobre datos reales — y no se sostuvo: en $70$ commits
reales de corrección de errores, un simple recuperador léxico TF-IDF
empata con la selección causal y la supera a presupuesto ajustado, y un
pivote por traza de fallo pierde ante un RAG-sobre-el-mensaje-de-error,
porque en código real los archivos de una corrección y sus mensajes de
error comparten vocabulario. Reportamos el resultado negativo en vez de
enterrarlo. Lo que sobrevive no es un mejor recuperador sino un tipo de
objeto distinto: una memoria de trabajo *determinista, reproducible,
auditable* en la que el estado exacto del contexto de un agente puede
rebobinarse y reproducirse bajo parámetros distintos — depuración por
viaje en el tiempo que una pila de recuperación probabilística no puede
ofrecer. Si esa auditabilidad, o una ventaja de contexto estructurado en
tiempo de *generación* (Fase 4), produce una ganancia medible aguas
abajo, queda abierto. Publicamos el motor, el arnés de validación
honesto y este artículo para que la distinción entre lo que está
probado, lo que está medido y lo que está refutado pueda comprobarse en
vez de afirmarse.

<div class="thebibliography">

99 P. Lewis et al. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. *NeurIPS*, 2020. arXiv:2005.11401.
A. Asai et al. Self-RAG: Learning to Retrieve, Generate and Critique
through Self-Reflection. *ICLR*, 2024. arXiv:2310.11511. D. Edge et al.
From Local to Global: A Graph RAG Approach to Query-Focused
Summarization. 2024. arXiv:2404.16130. F. Yamaguchi et al. Modeling and
Discovering Vulnerabilities with Code Property Graphs. *IEEE S&P*, 2014.
C. Packer et al. MemGPT: Towards LLMs as Operating Systems. 2023.
arXiv:2310.08560. J. S. Park et al. Generative Agents: Interactive
Simulacra of Human Behavior. *UIST*, 2023. arXiv:2304.03442. LangChain.
LangGraph: Building Stateful, Multi-Actor Applications with LLMs. Marco
de software, 2023–2024. <https://github.com/langchain-ai/langgraph>.
P. J. Denning. The Working Set Model for Program Behavior.
*Communications of the ACM*, 11(5), 1968. N. F. Liu et al. Lost in the
Middle: How Language Models Use Long Contexts. *TACL*, 2024.
arXiv:2307.03172. W. Kwon et al. Efficient Memory Management for Large
Language Model Serving with PagedAttention. *SOSP*, 2023.
arXiv:2309.06180. S. Haber and W. S. Stornetta. How to Time-Stamp a
Digital Document. *Journal of Cryptology*, 3(2), 1991. C. E. Jimenez et
al. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
*ICLR*, 2024. arXiv:2310.06770.

</div>

[^1]: Causal Context Operating System (CCOS), un prototipo de
    investigación abierto. Código fuente, guiones de reproducción y este
    artículo: <https://github.com/CHECKUPAUTO/CCOS>.
