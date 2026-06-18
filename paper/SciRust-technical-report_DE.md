# SciRust: Ein Pure-Rust Deep Learning Framework — Portable GPU-Beschleunigung, eine symbolische Regressions-Engine und eine deterministische Inferenz-Runtime

**Tarek Zekriti**
Unabhängiger Forscher · contact@checkupauto.fr
Repository: https://github.com/CHECKUPAUTO/scirust

---

## Zusammenfassung

Wir präsentieren **SciRust**, ein vollständig in Rust geschriebenes Deep-Learning-Framework, das eine Runtime-Bibliothek mit einer Transpiler-Schicht (Prozedurale Makro-Attribute für Differenzierung, Vektorisierung und Accelerator-Targeting) kombiniert, sowie neun darauf aufgebaute und validierte Fähigkeiten. Die erste ist ein portabler GPU- und Tensor-Core-Pfad: Der Pure-Rust-Kern lässt sich ohne Modifikationen auf ein NVIDIA Jetson Thor (aarch64) portieren, und eine cuBLAS-gestützte Matrixmultiplikation erreicht, validiert gegen ein CPU-Orakel, rund 63 TFLOPS in BF16. Die zweite ist eine hybride genetisch-gradientenbasierte **symbolische Regressions-Engine**, die geschlossene Gesetze – Struktur und Konstanten – aus Daten wiedergewinnt, wobei die frameworkeigene symbolische Differenzierung zur Anpassung der Konstanten genutzt wird. Die dritte ist eine **deterministische Inferenz-Runtime**, die bit-exakte, latenzgebundene und auditierbare Inferenz bietet, generisch über Architekturen via ein Klartext-Manifest. Die vierte ist ein deterministischer Int8-Quantisierungsstack für eingebettete Inferenz: Ein portabler Integer-Inferenzpfad, bit-exakt über Threads hinweg und bit-genau reproduzierbar unter Festkomma-Requantisierung, der die Modellgewichte um etwa das Vierfache schrumpft. Ein gemeinsamer methodischer roter Faden verbindet sie: Jedes Primitiv wird erst akzeptiert, nachdem seine Ausgabe einem Referenzorakel entspricht, und Reproduzierbarkeit wird als erstklassige, gemessene Eigenschaft behandelt – in mehreren Fällen bit-genau. Gegenüber der Baseline des Frameworks (255 bestandene Tests; MNIST 97,70 %) etablieren diese Beiträge SciRust als substanzielles, reproduzierbares Forschungsartefakt.

---

## 1. Einleitung

SciRust ist ein in reinem Rust geschriebenes Deep-Learning-Framework. Es ist ein Hybrid aus einer Runtime-Bibliothek und einem Transpiler-System: Neben konventionellen Tensor- und neuronalen Netzwerkkomponenten implementiert es echte prozedurale Makro-Attribute – #[autodiff], #[simd] und #[gpu] – über drei Makro-Crates hinweg, so dass annotierter Rust-Code in differenzierte, vektorisierte oder Accelerator-orientierte Formen umgeschrieben wird. Das Projekt ist als **Forschungsartefakt** positioniert, nicht als Produktionskonkurrent zu etablierten Frameworks (PyTorch oder in Rust Burn und candle), die es in Operatorenabdeckung, Kernel-Reife und Hardware-Breite übertreffen.

Dieser Bericht stellt das Framework und drei darauf aufbaute Fähigkeiten vor, die jeweils mit ihren gemessenen Zahlen und ehrlichen Grenzen validiert und berichtet werden: Ein portabler GPU- und Tensor-Core-Pfad, eine symbolische Regressions-Engine und eine deterministische Inferenz-Runtime. Das verbindende Material beschreibt die Framework-Baseline und die Engineering-Disziplin, unter der jeder Beitrag akzeptiert wurde.

Wir sind explizit bezüglich der Art der gemachten Aussagen. **Gemessene Aussagen** – Durchsatz, Genauigkeit, Latenz, bit-exakte Fingerabdrücke – sind reproduzierbare Zahlen aus den berichteten Läufen. **Interpretative Aussagen** – darüber, was die Engineering-Disziplin bringt oder was eine Fähigkeit über das Framework aussagt – werden als begründete Argumente angeboten, die auf diesen Messungen basieren, nicht als Beweise.

## 2. Das SciRust-Framework

Der Kern (scirust-core) bietet eine Reverse-Mode-Engine für automatisches Differenzieren, die um ein Tape herum aufgebaut ist, das Operationen aufzeichnet, einen zweidimensionalen Tensor-Typ, eine Bibliothek von neuronalen Netzwerkmodulen (lineare, faltungsbasierte, Pooling-, Normalisierungs-, Aktivierungs- und Transformer-Schichten) hinter einem gemeinsamen Module-Trait, Optimierer (einschließlich Adam) und Datenlader. Ein deterministischer, seedbarer Pseudozufallszahlengenerator bildet die Grundlage für Initialisierung und Datenmischung, was die Reproduzierbarkeit über den gesamten Lauf erreichbar statt zufällig macht.

Was SciRust von einer reinen Bibliothek unterscheidet, ist seine Transpiler-Dimension. Die Makro-Crates (scirust-macros, scirust-simd-macros, scirust-gpu-macros) implementieren die Proc-Makro-Attribute #[autodiff], #[simd] und #[gpu], was das System zu einem Hybrid aus Runtime und Transpiler macht, statt einer reinen festen Runtime. Die CPU-Numerik ist reiner Rust-Code ohne obligatorische BLAS-Abhängigkeit, was – wie Abschnitt 4 zeigt – genau das ist, was die architekturübergreifende Portabilität einfach machte.

Die Baseline-Validierung des Frameworks umfasst **255 bestandene Tests** und mehrere End-to-End-Demonstrationen: MNIST-Klassifizierung bei **97,70 %** mit bit-identischen Verlustkurven über Epochen hinweg (das stärkste Nicht-Regressions-Signal, das das Projekt verwendet), ein Transformer, der **100 %** bei einer synthetischen Mehrheitsentscheidungs-Aufgabe erreicht, und eine CIFAR-10-Faltungs-Pipeline, die **52,40 %** auf einer 5000-Bilder-Trainings-Teilmenge erreicht (etwa das 5,2-fache der Zufalls-Baseline, was den Faltungs-Pfad validiert). Diese Zahlen belegen, dass das Substrat ein funktionierendes Framework ist, kein Platzhalter, worauf der Rest des Berichts aufbaut.

## 3. Engineering-Disziplin

Eine einzige Disziplin regelte die Akzeptanz jedes Beitrags in einen validierten Zustand, und es lohnt sich, sie explizit zu nennen, da sie es ist, die die gemessenen Ergebnisse vertrauenswürdig macht:

- **Orakel-Validierung.** Kein Rechenprimitiv wurde akzeptiert, bis seine Ausgabe gegen eine unabhängige Referenz geprüft wurde – typischerweise die CPU-Implementierung, die als Orakel für einen GPU-Pfad fungiert, oder ein bekanntes Ground-Truth-Gesetz für die symbolische Engine. Die stärkste Form dieser Prüfung ist auf Bit-Ebene: Identische Gleitkomma-Ausgabe (bit-identische Verlustkurven oder identische Ausgabe-Fingerabdrücke) ist ein weitaus stärkeres Nicht-Regressions-Signal als eine ungefähre Übereinstimmung.
- **Grüne-Tests-Gate.** Die Arbeit schritt nicht über einen Schritt hinaus, dessen Tests nicht bestanden wurden, wobei rohe Build- und Test-Ausgaben (keine Zusammenfassungen) als Beweis dienten.
- **Branch-Isolierung.** Jede Fähigkeit wurde in ihrem eigenen Branch entwickelt und dort vor der Integration validiert, wodurch laufende Arbeiten von unabhängigen Änderungen an anderer Stelle in der sich entwickelnden Codebasis isoliert blieben.
- **Additive Integration.** Wo möglich, wurden neue Fähigkeiten als separate Crates oder hinter Feature-Flags gelandet, wobei weder der CPU-Hot-Path noch die Autodiff-Engine berührt wurden, so dass ein Beitrag isoliert validiert werden konnte.

Die wiederkehrende Lektion ist, dass ein numerischer Test nur so vertrauenswürdig ist wie sein Fehlermodell – ein Punkt, der in den Abschnitten 4 und 5 konkret wird.

## 4. GPU-Inbetriebnahme: Erweiterung von SciRust auf NVIDIA Tensor Cores auf Jetson Thor

### 4.1 Kontext und Portabilität

SciRust wurde auf einem x86-64 Debian-Host entwickelt und validiert. Um die Portabilität und einen GPU-Ausführungspfad zu untersuchen, wurde das Framework auf ein NVIDIA Jetson Thor Modul (aarch64, Blackwell-Klasse GPU, CUDA 13.0, Treiber 580) portiert.

Der Pure-Rust-Kern kompilierte auf aarch64 ohne Modifikation in unter 20 Sekunden, und entscheidend **ohne jede BLAS-Abhängigkeit**: Die optionalen intel-mkl-src und blas-src Bindings blieben inaktiv, so dass die x86-only Intel MKL Falle konstruktionsbedingt vermieden wurde. Das architekturübergreifende numerische Verhalten hielt stand: MNIST erreichte **97,73 %** (Verlust 0,0377) auf dem Jetson, konsistent mit der x86-Baseline, was bestätigt, dass die CPU-Numerik des Frameworks architekturportabel ist.

Eine praktische Beobachtung zur Toolchain: Die cudarc 0.14 Crate bietet Bindings nur bis zu CUDA 12.8 an, lädt den Treiber jedoch dynamisch. Da die CUDA-Treiber-API abwärtskompatibel ist, funktioniert das Erzwingen des cuda-12080 Binding-Sets zur Laufzeit korrekt gegen den CUDA 13.0 Treiber – der dynamische Ladepfad ermöglichte die Inbetriebnahme auf einer Toolchain, die neuer war als die Binding-Crate wusste.

### 4.2 Validierungsmethode

Die Matrixmultiplikation (GEMM) war das Inbetriebnahme-Primitiv, gewählt, weil sie sowohl beim Training als auch bei der Inferenz die Kosten dominiert und eine eindeutige Referenz hat. Die Arbeit schritt zuerst in einer isolierten Sandbox-Crate voran, dann in-tree hinter einem cuda-Feature-Flag, wobei jede Stufe vor der nächsten gegen das CPU-Orakel validiert wurde.

Ein methodischer Punkt tauchte während der Validierung auf. Eine naive Metrik für den relativen Fehler meldete eine Abweichung von 5,6 % bei einem nicht quadratischen Problem, während sie bei einem quadratischen Problem 5e-5 meldete, unter Verwendung identischer Kernel. Die Ursache war kein Defekt, sondern Auslöschung: Bei Operanden mit gemischten Vorzeichen liegen einige Ausgabeeinträge nahe bei Null, so dass der relative Fehler explodiert, während der absolute Fehler auf dem FP32-Rauschniveau bleibt. Das korrekte Orakel kombiniert eine **absolute** Toleranz, die überall angewendet wird, mit einer **relativen** Toleranz, die nur dort angewendet wird, wo die Referenzgröße signifikant ist. Unter dieser kombinierten Metrik entsprach jeder GPU-Pfad dem Orakel.

### 4.3 Das Matmul-Triptychon

| Implementierung | 512^3 | 1024^3 | 2048^3 | 4096^3 |
|---|---|---|---|---|
| CPU (Rayon, FP32) | 2,37 ms | — | — | — |
| GPU naiver Kernel (FP32) | 2,749 ms / 98 | — | — | — |
| GPU getachelter Kernel (FP32) | 1,393 ms / 193 | 5,004 ms / 429 | 17,216 ms / 998 | — |
| cuBLAS (FP32) | 0,376 ms / 714 | 1,993 ms / 1078 | 3,787 ms / 4537 | 22,314 ms / 6159 |
| cuBLAS Tensor Cores (FP16) | 0,237 ms / 1130 | 0,251 ms / 8559 | 0,346 ms / 49699 | 2,166 ms / 63448 |
| cuBLAS Tensor Cores (BF16) | 0,238 ms / 1128 | 0,253 ms / 8493 | 0,347 ms / 49501 | 2,152 ms / 63872 |

(Zeit pro Aufruf / Durchsatz in GFLOPS.) Die Progression ist aufschlussreich. Der naive Kernel ist speichergebunden und erreicht lediglich das Niveau einer optimierten Multi-Core-CPU – eine GPU ist nicht automatisch schneller. Der getachelte Kernel mit Shared-Memory (16x16 Kacheln) verdoppelt dies etwa und stößt in echtes GPU-Territorium vor (~1 TFLOPS bei 2048^3), aber ein Kernel mit einer Ausgabe pro Thread stagniert um den Faktor ~4 unter cuBLAS, was Register-Blocking und Double-Buffering ausmachen. cuBLAS FP32 erreicht ~6,2 TFLOPS (6,3-fache der CPU bei 512^3); der Einsatz der Tensor Cores in FP16/BF16 liefert ~63 TFLOPS nachhaltig bei 4096^3, eine Größenordnung über FP32. Zwei ehrliche Einschränkungen: Der Durchsatz unter 2048^3 ist durch den Launch-Overhead begrenzt (nur die 4096^3-Zahl liest sich als nachhaltig), und die Zahlen spiegeln den Standard-Power-Modus des Geräts wider.

### 4.4 Präzision und Integration

cuBLAS FP32 ist bit-nah am CPU-Ergebnis (maximaler relativer Fehler 4,7e-5 bei 512^3), wobei es sich nur in der Summationsreihenfolge unterscheidet; der getachelte Kernel stimmte bis auf 9,4e-6 überein. Die Tensor-Core-Pfade mit reduzierter Präzision degradieren wie erwartet (FP16 1,3e-2, BF16 6,8e-2, letzterer größer aufgrund der 7-Bit-Mantisse von BF16), wobei der Fehler aus der Eingangsrundung statt aus der Akkumulation stammt, die in FP32 durchgeführt wird. Für maschinelles Lernen ist der größere Single-GEMM-Fehler von BF16 kein Nachteil: Sein dem FP32 entsprechender Exponentenbereich vermeidet den Überlauf, der FP16 in tiefen Aktivierungen plagt, weshalb es das De-facto-Trainingsformat und das empfohlene Ziel für jeden zukünftigen Mixed-Precision-Pfad ist.

Das cuBLAS FP32 GEMM wurde in die scirust-gpu Crate hinter dem cuda-Feature integriert, als reiner Slice-Level-Einstiegspunkt ohne Abhängigkeit von den Kern-Tensor-Typen, wodurch jedes Risiko eines Abhängigkeitszyklus eliminiert wurde. cuBLAS ist spaltenorientiert (column-major); das zeilenorientierte (row-major) Produkt C = A.B wird durch Berechnung von (B^T.A^T) mit vertauschten Operanden und entsprechend gesetzten führenden Dimensionen erhalten, und der CUDA-Kontext sowie das Handle werden pro Thread gecacht. Die Integration ist additiv und nicht-invasiv – sie berührt weder den CPU-Hot-Path noch die Autodiff-Engine – und wird durch zwei Orakel-Tests validiert, einen quadratischen Fall und einen nicht quadratischen Fall, der speziell das column-major Dimensions-Mapping ausübt.

## 5. Symbolische Regression via frameworkeigener Autodiff

### 5.1 Motivation und Methode

Um zu untersuchen, ob SciRust ein substanzielles Framework statt eines Fitting-Gerüsts ist, haben wir eine Fähigkeit aufgebaut, die Komponenten kombiniert, die normalerweise nicht kombiniert würden: Seine symbolische Mathematik-Engine (scirust-symbolic – Ausdrucksbäume, Vereinfachung, Auswertung und **symbolische Differenzierung**) mit seiner Automatic-Differentiation-Disziplin. Die Aufgabe ist **symbolische Regression**: Die Wiedergewinnung eines geschlossenen Ausdrucks – sowohl Struktur als auch Konstanten –, der zu beobachteten Daten passt.

Die Engine ist ein Hybrid. Die **Struktur** eines Kandidaten wird durch genetische Programmierung über Ausdrucksbäume (Primitive +, -, x, /, sin, cos, exp plus Variablen und Konstanten) mit Turnier-Selektion, Subtree-Crossover und Mutation, Elitismus und einer Größenbeschränkung gesucht. Die **Konstanten** werden nicht blind gesucht – die klassische Schwäche der genetischen Programmierung –, sondern durch Gradientenabstieg (Adam) angepasst, wobei die Gradienten aus der **symbolischen Differenzierung** des Frameworks stammen: Für einen Kandidaten mit Konstanten c0, c1, ... wird das partielle d(expr)/d(ck) aus dem Diff der Engine gewonnen und über dem Daten-Batch ausgewertet. Die symbolische Engine treibt somit ihr eigenes Lernen voran. Die Selektion ist auf **Parsimonie** (Sparsamkeit) ausgerichtet und die Ausgabe ist eine **Pareto-Front** über Genauigkeit gegen Komplexität; das Datenmodell ist **multivariabel**. Die Engine ist reiner Rust-Code, verwendet scirust-symbolic unverändert wieder und ist über einen seedbaren Generator voll reproduzierbar.

### 5.2 Validierung und Ergebnisse

Jedes Ergebnis wird gegen ein **Orakel** geprüft – ein bekanntes Ground-Truth-Gesetz – unter Verwendung derselben kombinierten absoluten/relativen Toleranz, die in Abschnitt 4.2 diskutiert wurde. Ein zweites, schärferes Kriterium ist strukturell: Hat die Engine das wahre, kompakte Gesetz wiedergewonnen oder lediglich eine genaue, aber aufgeblähte Annäherung?

| Zielgesetz | Wiedergewonnener Ausdruck | MSE |
|---|---|---|
| x^2 + sin(x) | (x.x) + sin(x) | 0 |
| exp(-0.3x).cos(2x) | cos(x+x).exp(-0.300.x) | 3,3e-16 |
| x.y + sin(x) (2 Variablen) | sin(x) + (y.x) | 0 |
| x / (1 + x^2) | x / (x.x + 1,0) | 2,0e-15 |
| 0,5x^2 - 1,2x + 2 + Rauschen (sigma=0,1) | quadratische Form | 9,1e-3 ~ sigma^2 |

Die Engine gewann die exakte Struktur für das Polynom-plus-Trigonometrie-Gesetz, den Fall mit zwei Variablen und – bemerkenswerterweise – den gedämpften Oszillator wieder, von dem man normalerweise erwartet, dass er scheitert, da das Fitten einer Frequenz innerhalb eines Cosinus hochgradig nicht-konvex ist; sie drückte sogar 2x als x+x aus. Die verrauschte quadratische Funktion wurde an das Signal bei der Rauschvarianz angepasst, ohne dem Rauschen nachzujagen.

Das aufschlussreichste Ergebnis ist die rationale Funktion x/(1+x^2). Unter **reiner MSE-Selektion** lieferte die Engine einen verschachtelten Sinus-Ausdruck mit vierzehn Knoten, der die Daten auf ~6e-5 annäherte, aber keinerlei Ähnlichkeit mit dem wahren Gesetz aufwies. Unter der **Pareto-Front mit einer Parsimonie-Strafe** erschien die wahre kompakte Form am unteren Ende der Front (sieben Knoten, MSE ~2e-15). Dies ist die zu merkende Erkenntnis: **Geringer Fehler ist nicht gleichbedeutend mit dem korrekten Gesetz** – rein auf Genauigkeit ausgerichtete Ziele belohnen aufgeblähte Annäherungen, und Parsimonie-Druck plus eine Pareto-Sichtweise sind das, was die Struktur wiedergewinnt.

Die Engine landete als scirust-symreg Crate, entwickelt in ihrem eigenen Branch und konstruktionsbedingt additiv. Ihre Grenzen werden offen dargelegt: Ein Einzelsitzungs-Ergebnis auf einem bescheidenen Primitiven-Set; eine stochastische (seedbare, nicht exhaustive) Suche; und der Begriff neuro-symbolisch ist nur im engen Sinne von gradientenoptimierten Konstanten innerhalb einer symbolischen Suche verdient, nicht als gelernter Prior über die Struktur.

## 6. Eine deterministische Inferenz-Runtime

### 6.1 Positionierung

Ein Pure-Rust Trainings-Framework ist ein schlechter Konkurrent für das etablierte Ökosystem zu dessen eigenen Bedingungen. Statt auf dieser Achse zu konkurrieren, fragten wir uns, ob ein auf SciRust basierendes System eine Eigenschaft als erstklassige Garantie bieten kann, die Mainstream-Runtimes als Best-Effort behandeln. Die verfolgte Antwort ist **deterministische, latenzgebundene, auditierbare Inferenz** – die Kombination, die von Edge- und regulierten Deployments gefordert wird. Die Runtime (scirust-runtime) ist eine separate Crate über einer eingefrorenen Forward-Teilmenge des Kerns; sie führt nur Forward-Inferenz durch, wobei das Training als Offline-Tooling beibehalten wird. Diese Trennung erlaubt es, dass ein stabiler Inferenz-Vertrag auf dem sich entwickelnden Kern sitzt, wobei eine Regressionssperre (Abschnitt 6.3) jede Drift in einen sichtbaren Fehler verwandelt.

### 6.2 Der Schlussstein: Bit-exakter Determinismus

Jede andere Garantie ruht darauf, dass der Forward-Pass bit-exakt ist, daher wurde dies zuerst empirisch etabliert. Ein MLP (784-256-10) mit festen Gewichten wurde wiederholt über eine feste Eingabe laufen gelassen, wobei die Ausgaben bit-genau verglichen wurden (to_bits-Gleichheit, keine Toleranz). Über 5120 Logit-Vergleiche hinweg gab es **Null Abweichungen**, und ein 64-Bit-Fingerabdruck der Ausgabebits war über Aufrufe und über separate Prozesse hinweg identisch.

Der entscheidende Test betrifft die Thread-Anzahl. Das Matmul ist Rayon-parallel, was die Sorge weckt, dass ein Work-Stealing-Scheduler die Summationen umordnet. Das tut er nicht: Das erneute Ausführen des Binaries unter RAYON_NUM_THREADS von 1, 2, 4, 8, 16 und 64 erzeugte jedes Mal den identischen Fingerabdruck 0xde2d807686e4b47e. Der Grund ist strukturell – das parallele Matmul verteilt die Arbeit über Ausgabezellen, wobei jedes Skalarprodukt von einem einzelnen Thread in fester Reihenfolge akkumuliert wird, so dass die Reduktionsreihenfolge unabhängig von der Thread-Anzahl ist. Der ehrliche Umfang des resultierenden Anspruchs ist Bit-Exaktheit für ein **festes kompiliertes Artefakt auf einer gegebenen Architektur**, stabil über Thread-Anzahl und Prozess-Neustarts hinweg; architekturübergreifende Bit-Exaktheit liegt konstruktionsbedingt außerhalb des Umfangs – das korrekte Audit-Modell besteht darin, ein gepinntes Artefakt auszuliefern und es identisch auf seinem Ziel abzuspielen.

### 6.3 Gewichts-Persistenz und Reload

Für die Reproduzierbarkeit über Deployments hinweg müssen eingefrorene Gewichte verlustfrei hin- und herwandern. Wir definierten ein kleines Format, **SRT1**, das jeden Tensor als (Key, Zeilen, Spalten, f32 little-endian) mit sortierten Schlüsseln schreibt, so dass die Bytes auf der Festplatte deterministisch sind und das Artefakt einen stabilen Hash hat. Der tragende Goldstandard-Test – serialisieren, ein frisches, anders geseedetes Modell konstruieren, neu laden, Forward-Pass ausführen – muss den ursprünglichen Fingerabdruck reproduzieren. Das tut er: Ein anders geseedetes Modell unterscheidet sich vor dem Laden und reproduziert 0xde2d807686e4b47e danach bit-genau. An einem real trainierten Modell exerziert, lädt das auf MNIST trainierte MLP (Verlust 0,2615 -> 0,0377) und in ein 814 KB Artefakt eingefroren mit **97,73 %** Testgenauigkeit wieder, wobei der Test-Logit-Fingerabdruck 0xc96d25fa658f5611 über Prozesse hinweg stabil bleibt. Dies schließt die These end-to-end: Einmal trainieren, einfrieren, und die Runtime spielt bei jeder Invokation eine genaue, bit-exakte Inferenz ab.

### 6.4 Gebundene Latenz

Da die Korrektheit durch Abschnitt 6.2 fixiert war, wurde die Latenz als zeitliche Messung behandelt. Für Single-Request-Inferenz (Batch=1) zeigte das MLP p50 = 126 us, p99 = 145 us und ein **p99/p50-Verhältnis von 1,15** – ein enger, vorhersagbarer Tail. Die Latenz war auch invariant gegenüber der Thread-Anzahl (flaches p50 von 1 bis 8 Threads): Die Kosten pro Aufruf werden von festem Overhead dominiert, nicht von Berechnung oder Dispatch, so dass die Thread-Anzahl ein Durchsatz-Hebel ist (Batch=64 Durchsatz skaliert von 23k -> 81k Samples/s über 1->8 Threads), irrelevant für die Single-Request-Latenz. Ein bewusstes Nicht-Ergebnis: Wir hypothetisierten, dass eine allokationsfreie Arena benötigt würde, um den Tail zu binden, aber das gemessene 1,15x-Verhältnis zeigte, dass der Allokations-Jitter vernachlässigbar war, so dass **keine Arena gebaut wurde** – die Daten rechtfertigten die Optimierung nicht. Einer Optimierung zu widerstehen, der die Messungen widersprechen, ist Teil der Disziplin.

### 6.5 Allgemeingültigkeit via manifestgesteuerte Rekonstruktion

Um zu zeigen, dass die Garantien keine Artefakte eines einzelnen kleinen MLPs sind, wurde das Audit an einem faltungsbasierten Netzwerk wiederholt (Conv->ReLU->MaxPool zweimal, dann ein Klassifizierer): Forward bit-exakt (0x1381e4b51d0eeba4) und thread-invariant; das 4,28 MB Artefakt wanderte bit-genau hin und zurück, einschließlich der Faltungsgewichte; Batch=32 Latenz behielt einen engen Tail bei (p50 45,9 ms, p99/p50 = 1,20). Die Runtime wurde dann so generalisiert, dass **keine Architektur im Inferenzpfad hartcodiert ist**: Ein Klartext-Manifest von Schichtspezifikationen plus eine SRT1-Datei rekonstruiert ein beliebiges unterstütztes Sequential. Ein manifest-rekonstruiertes CNN reproduziert den Fingerabdruck des hartcodierten Modells exakt, und – der entscheidende Fall – das trainierte MNIST MLP, rein aus einem Manifest plus seinen Gewichten wiederaufgebaut, reproduziert sowohl die 97,73 % Genauigkeit als auch den Fingerabdruck 0xc96d25fa658f5611 bit-genau. Das unterstützte Set deckt Linear, ReLU, Sigmoid, LayerNorm, BatchNorm2d, Conv2d und MaxPool2d ab, von denen jeweils gezeigt wurde, dass sie bit-exakt persistieren und rekonstruieren; parametrische Normalisierungsschichten wurden mit Sorgfalt validiert (LayerNorm-Affine-Parameter und BatchNorm2d-Running-Statistics überleben beide den Round-Trip, wobei BatchNorm2d in den Evaluationsmodus gezwungen wird, so dass die Inferenz pro Sample deterministisch ist). Fortgeschrittene Features wie **formale Invarianten-Verträge** durch `CertifiedModule<M, C>` und Unterstützung für **Secure Enclave Runtime** für #![no_std]-Targets erweitern die Anwendbarkeit der Runtime auf Umgebungen mit hoher Integrität weiter. Die ehrliche Grenze: Transformer-Schichten verwenden ein dreidimensionales Forward und würden einen separaten Runtime-Pfad erfordern; der Faltungs-Durchsatz ist durch den Pure-Rust-Kernel begrenzt; und die absolute Batch=1 Latenz ist overhead-gebunden.

## 7. Deterministische Int8-Quantisierung für eingebettete Inferenz

### 7.1 Positionierung

Die deterministische Runtime aus Abschnitt 6 zielt auf Edge- und regulierte Deployments ab, wo Speicher und Energie knapp sind und das Verhalten auditierbar sein muss. Acht-Bit-Integer-Inferenz ist der natürliche nächste Schritt, aber nur, wenn die Eigenschaften, die die Runtime vertrauenswürdig machten, den Wechsel zu geringer Präzision überleben. Wir haben daher den Quantisierungsstack im reinen portablen Kern (keine GPU-Abhängigkeit) aufgebaut und ihn an denselben Vertrag gebunden: Jedes quantisierte Primitiv wird nur gegen ein Referenzorakel akzeptiert, und Determinismus wird gemessen statt angenommen – bit-genau, wo immer die Arithmetik es erlaubt.

### 7.2 Weight-only und dynamisches Int8: Ein kostenloser Vierfach-Gewinn

Das erste Schema ist dynamisches W8A8: Aktivierungen werden pro Tensor zur Laufzeit quantisiert, Gewichte pro Ausgangskanal, das Produkt akkumuliert in i32 und eine einzige Requantisierung gibt f32 zurück. Auf dem trainierten MNIST MLP ist dies verlustfrei – die f32 Baseline erreicht 97,73 % (Fingerabdruck 0xc96d25fa658f5611) und das Int8-Modell 97,74 % –, während die Gewichte von 813 KB auf 204 KB (3,98x) schrumpfen. Der Int8-Fingerabdruck 0xc3730f7c204455ba ist unter RAYON_NUM_THREADS von 1, 4 und 16 identisch: Das Integer-Matmul akkumuliert jede Ausgabezelle in einem einzelnen Thread, so dass das strukturelle Determinismus-Argument aus Abschnitt 6.2 unverändert übernommen wird.

### 7.3 Statische Kalibrierung und Full-Integer-Requantisierung

Um Aktivierungsstatistiken pro Aufruf zu entfernen, wurden die Aktivierungsskalen einmal an einer zurückbehaltenen Stichprobe kalibriert; Int8-Aktivierungen werden dann zwischen den Schichten mit i32-Bias und einem Integer-ReLU weitergereicht. Diese statische Pipeline erreicht 97,71 % mit dem Fingerabdruck 0xa9b9a102c7cea67b, thread-invariant. Die Gleitkomma-Requantisierung im Hot-Path wurde dann durch eine Integer-Requantisierung im gemmlowp-Stil ersetzt – ein Festkomma-Multiplikator in [2^30, 2^31) und ein pro-Kanal Rechtsshift –, was das kalibrierte Modell bit-genau reproduziert (gleiche 97,71 %, gleiches 0xa9b9a102c7cea67b). Der Inferenzpfad ist nun durchgehend ganzzahlig, ohne Gleitkomma in der Schleife und ohne parallele Reduktion, so dass er konstruktionsbedingt deterministisch ist.

### 7.4 Pro-Kanal-Quantisierung von Faltungen

Das Pro-Kanal-Schema erstreckt sich auf das faltungsbasierte Netzwerk (pro Zeile für Conv2d-Gewichte, pro Spalte für Linear). Ein Fake-Quantized-Round-Trip reproduziert das f32-Orakel 0x1381e4b51d0eeba4 und bewahrt das Arg-Max auf allen 32 Test-Eingaben, wobei das 4,28 MB Filterset auf 1,07 MB (3,99x) schrumpft. Eine echte ganzzahlige direkte Faltung wurde dann validiert: Ein f32-Spiegel der ganzzahligen Indizierung entspricht dem Faltungs-Forward des Frameworks bit-genau, und die Int8-Faltung stimmt mit dem f32-Orakel bis auf max-abs 2,8e-2 überein. Wie in Abschnitt 6 wird der relative Fehler mit Vorsicht gelesen – nahe bei Logit-Auslöschungen koexistiert ein großer relativer Fehler mit einem vernachlässigbaren absoluten Fehler, so dass der absolute Fehler und das bewahrte Arg-Max die tragenden Metriken sind.

### 7.5 Ein portables quantisiertes Artefakt

Das kalibrierte Full-Integer-Modell wurde zu einem erstklassigen Artefakt erhoben, QSR1: Ein selbstbeschreibendes Byte-Format, das Dimensionen pro Schicht, die kalibrierte Eingangsskala, pro-Kanal Gewichtsskalen, Int8-Gewichte und i32-Bias enthält, mit deterministischen, hashbaren Bytes. Geschrieben, allein aus der Datei neu geladen und abgespielt, reproduziert es 0xa9b9a102c7cea67b bei 97,71 % aus 205 KB gegenüber dem 814 KB f32-Artefakt (3,96x). Über eine kleine Bibliotheks-API (ein quantisiertes Modell mit Save, Load und Infer) exponiert, reproduziert ein Round-Trip durch die Bibliothek den Fingerabdruck bit-genau; da QSR1 selbstbeschreibend ist, ersetzt es das Klartext-Manifest für quantisierte Modelle.

### 7.6 CSR-Tensoren und Sparse-SpMM-Kernel

Um den Speicherverbrauch auf Edge-Targets weiter zu optimieren, implementiert SciRust eine `CsrTensor`-Struktur und einen zugehörigen Sparse-Matrix-Matrix-Multiplikations-Kernel (SpMM). Dies ermöglicht die Speicherung und Berechnung von dünnbesetzten (sparse) Modellen ohne den Overhead dichter Repräsentationen, wodurch die Speicherwand auf ressourcenbeschränkten Geräten effektiv umgangen wird.

### 7.7 Ein Integer-Kernel und separierbare Faltungen

Das portable skalare Integer-Matmul ist die Korrektheitsreferenz. Ein aarch64 NEON-Kernel – verbreiternde Multiplikations-Akkumulation mit i32-Akkumulation, der rechte Operand für kontigueren Zugriff transponiert – ist bit-exakt dagegen (ganzzahlige Summation ist reihenfolgeunabhängig) und etwa zehnmal schneller (64x784x256: 9592 us skalar gegenüber 963 us NEON). Zwei Blöcke im MobileNet-Stil vervollständigen das eingebettete Operatorenset: Eine Int8-Depthwise-Faltung, deren f32-Spiegel einem Pro-Kanal-Faltungs-Orakel bit-genau entspricht und deren Int8-Ausgabe bis auf max-abs 2,0e-2 übereinstimmt, und eine Int8-Pointwise-1x1-Faltung, deren f32-Spiegel einem 1x1-Faltungs-Orakel bit-genau entspricht und bis auf max-abs 1,8e-2 übereinstimmt. Zusammengesetzt bilden sie eine separierbare Faltung vollständig in deterministischem Int8, wobei jede Hälfte gegen das Framework validiert wurde und jeder Gewichtstensor viermal kleiner ist.

## 8. Fortgeschrittene Features für Runtime und Verifizierung

Während SciRust von einem trainingsfokussierten Framework zu einem einsatzbereiten Ökosystem heranreifte, wurden fünf fortgeschrittene Features implementiert, um den Anforderungen von Systemen mit hoher Integrität und formaler Erklärbarkeit gerecht zu werden.

### 8.1 Ahead-Of-Time (AOT) statischer Modell-Compiler
Um den Overhead der Runtime-Graphkonstruktion und des Ladens von Gewichten zu eliminieren – kritisch für ultra-tief eingebettete Ziele mit begrenztem Heap-Speicher –, haben wir einen statischen Compiler implementiert.
- **Mechanismus:** Der Compiler liest eine `LayerSpec`-Topologie und rohe Gewichts-Puffer ein und gibt eine valide Rust-Quelldatei aus. Diese Datei definiert eine `StaticModel`-Struktur, in der die Gewichte als statisch verschachtelte Arrays (`&[[f32; N]; M]`) gespeichert sind.
- **Nutzen:** Modelle können direkt als unveränderliche Daten in das Binary gelinkt werden, was allokationsfreie Inferenz ermöglicht und Runtime-Parsing-Fehler verhindert.

### 8.2 Soft-Float-Matrix-Engine für Determinismus
Während Abschnitt 6.2 Bit-Exaktheit für eine feste Architektur etabliert, wird plattformübergreifender Determinismus (z. B. x86 vs. ARM) oft durch hardware-spezifische FPU-Rundungen und FMA-Optimierungen gebrochen.
- **Implementierung:** Wir haben `soft_gemm` implementiert, einen softwaredefinierten Matrixmultiplikations-Kernel, der skalierte ganzzahlige Arithmetik verwendet (`i32` mit `i64`-Akkumulation).
- **Validierung:** Durch Umgehung der Hardware-FPU garantiert die Engine identische Rechenspuren über disparate CPU-Instruktionssätze hinweg, eine Anforderung für formale Verifizierung und plattformübergreifende Audit-Logs.

### 8.3 Latente Aktivierungssteuerung (RepE)
Aufbauend auf dem Paradigma des „Representation Engineering“ haben wir Low-Level-Hooks integriert, um den internen Modellzustand während der Inferenz zu manipulieren.
- **Struktur:** Der `Module`-Trait wurde um eine `forward_steered`-Methode und eine `SteerHook`-Registry erweitert.
- **Anwendung:** Dies erlaubt es externen Controllern, lineare Verschiebungen (Concept Vectors) auf latente Aktivierungen in Echtzeit anzuwenden, was die Umleitung des Modellverhaltens ermöglicht, ohne statische Gewichte zu modifizieren.

### 8.4 Quantisierungsbewusstes Training (QAT) mit STE
Um die Lücke zwischen FP32-Training und INT8-Einsatz (Abschnitt 7) zu schließen, haben wir Fake-Quantisierungs-Kernel implementiert.
- **Mechanismus:** Während des Forward-Passes werden Werte geclamped und auf eine simulierte 8-Bit-Skala quantisiert. Der Backward-Pass nutzt einen **Straight-Through Estimator (STE)**, der Gradienten unverändert durch den nicht-differenzierbaren Quantisierungsschritt leitet.
- **Ergebnis:** Modelle passen sich während der Trainingsschleife natürlich an Quantisierungsfehler an, was die Genauigkeit des nachfolgenden Low-Precision-Einsatzes signifikant verbessert.

### 8.5 XAI: Integrated Gradients Engine
Um den Anforderungen regulierter Sektoren (Abschnitt 3) gerecht zu werden, haben wir Integrated Gradients für die Feature-Attribution implementiert.
- **Algorithmus:** Die Engine berechnet das Pfadintegral der Gradienten von einer Baseline (z. B. einem Null-Tensor) zum Input über $m$ Schritte.
- **Integration:** Unter Nutzung der frameworkeigenen `Tape`-basierten Autodiff generiert die Engine Attributionskarten derselben Form wie der Input und liefert so eine mathematische Erklärung für jede gegebene Vorhersage.

## 9. Expansion auf moderne KI-Familien

Um über grundlegende MLP- und CNN-Architekturen hinauszugehen, haben wir SciRust um grundlegende Unterstützung für mehrere moderne KI-Domänen erweitert, wobei wir strikte Pure-Rust- und Determinismus-Beschränkungen beibehalten haben.

### 9.1 Fortgeschrittenes Reinforcement Learning: DQN und PPO
Wir haben einen Reinforcement-Learning-Stack in `scirust-learning` implementiert.
- **Algorithmen:** Unterstützung für Tabular Q-Learning/SARSA und Deep Q-Networks (DQN). Darüber hinaus haben wir **Proximal Policy Optimization (PPO)** unter Verwendung eines geclippten Ziels implementiert, um stabile Policy-Updates zu gewährleisten.
- **Determinismus:** Agenteninteraktionen und Speicher-Sampling werden unter Verwendung von geseedeten `PcgEngine`-Instanzen erzwungen, was reproduzierbare Trainings-Trajektorien garantiert.

### 9.2 Computer Vision: ResNet und Vision Transformer
Zwei Hauptarchitekturen wurden zu `scirust-core` hinzugefügt:
- **ResNet-18/34:** Modulare Implementierung unter Verwendung von `ResidualBlock` und einem **Global Average Pooling (GAP)** Schritt, um variierende Eingangsauflösungen zu handhaben.
- **Vision Transformer (ViT):** Implementierung der Patch-Projektion via 2D-Faltungen, gefolgt von einem Transformer-Encoder. Features werden über die Sequenzdimension für die Klassifizierung aggregiert.

### 9.3 Generative KI und Transformer
- **Variationale Autoencoder (VAE):** Implementierung des Reparametrisierungstricks unter Verwendung von `PcgEngine`-basiertem Gaußschem Rauschen und einem analytischen KL-Divergenz-Verlust.
- **Mixture of Experts (MoE):** Eine modulare MoE-Schicht, die **Top-k-Routing** und additive Experten-Aggregation unterstützt, was Modell-Skalierung ohne lineares Anwachsen der Rechenkosten ermöglicht.

### 9.4 Spezialisierte Architekturen
- **Graph Neural Networks (GNN):** Grundlegende **Graph Convolutional Network (GCN)** Schichten, die dünnbesetzt-dichte Adjazenzmatrix-Multiplikationen unterstützen.
- **Speech AI:** Audio-Encoder und eine repräsentative **CTC-Loss** Implementierung für die zeitliche Sequenz-Ausrichtung.
- **PEFT (LoRA):** Low-Rank Adaptation für lineare Schichten, was es ermöglicht, eingefrorene Backbone-Modelle über kleine Rang-r-Matrizen feinabzustimmen.

## 10. Diskussion

Zwei Beobachtungen kehren in den Beiträgen wieder. Erstens leistete die Disziplin die Hauptarbeit: Da jedes Primitiv nur gegen ein Orakel akzeptiert wurde – oft bit-genau –, reproduziert ein Pfad entweder die Referenz oder nicht, was die Ergebnisse des Frameworks während seiner Entwicklung vertrauenswürdig hielt. Zweitens waren die wertvollsten Schlussfolgerungen manchmal negativ und kamen erst durch Messen zustande: Dass die Thread-Anzahl die Single-Request-Latenz nicht beeinflusst, dass eine Allokations-Arena ungerechtfertigt war, dass eine naive Metrik für den relativen Fehler nahe bei Auslöschungen unzuverlässig ist und dass geringer Fehler nicht gleichbedeutend mit dem korrekten Gesetz ist. Jedes widersprach einem plausiblen Prior und wäre durch Behaupten statt Messen übersehen worden. Ein dritter, verbindender Punkt: Reproduzierbarkeit, als eine zu entwickelnde und zu messende Eigenschaft behandelt statt erhofft, wurde zu einem eigenständigen Produktmerkmal – die zentrale Garantie der deterministischen Runtime ist genau die Bit-Exaktheit, von der die Testdisziplin des Frameworks bereits abhing. Der Int8-Quantisierungsstack erweiterte genau diesen Vertrag: Sein ganzzahliger Inferenzpfad ist thread-invariant durch dasselbe Argument der Single-Thread-Reduktion pro Zelle, und eine Festkomma-Requantisierung reproduziert das kalibrierte Modell bit-genau, so dass sich der Determinismus ohne neue Maschinerie auf geringe Präzision übertrug.

## 11. Einschränkungen

Das Framework ist ein Forschungsartefakt und nicht für den Produktionseinsatz geeignet. Der Faltung fehlt ein im2col-plus-BLAS- oder GPU-Pfad und sie ist daher im absoluten Durchsatz langsam; das GPU-Backend ist für die Korrektheit der Berechnung validiert, aber noch nicht in das Training eingebunden; und die deterministische Runtime ist nur für Inferenz über einem zweidimensionalen Schichten-Set ausgelegt, wobei Transformer-Unterstützung einen separaten dreidimensionalen Pfad erfordern würde. Determinismus ist auf ein festes Binary und eine feste Architektur beschränkt. Die symbolische Engine ist eine stochastische Suche auf einem bescheidenen Primitiven-Set, und mehrere Beiträge sind Einzelsitzungs-Ergebnisse.
Der neu eingeführte **PINN (Physics-Informed Neural Networks)** Loss-Evaluator ermöglicht die Integration symbolischer physikalischer Residuen in den AD-Optimierungspfad.
Die Int8-Quantisierung ist Post-Training statt quantisierungsbewusst; ein Ergebnis ohne Genauigkeitsverlust ist auf dem MNIST MLP etabliert, während die faltungsbasierten Quantisierer auf synthetischen Eingaben auf Treue und Determinismus validiert wurden, statt auf Genauigkeit auf einem gelabelten Bild-Benchmark, und ein On-Device (no_std) Mikrocontroller-Einsatz ist noch nicht demonstriert. Das Repository enthält auch ein Modul für evolutionäre Optimierung; von dessen Algorithmen ist hier nur das multi-objektive NSGA-II validiert, das die ZDT1-Pareto-Front bis auf etwa 1e-3 wiedergewinnt, während die vereinfachten ein-objektiven Optimierer auf konvexen Landschaften konvergieren, aber nicht auf schwierigen multimodalen Funktionen. Keines dieser Elemente entwertet die gemessenen Ergebnisse; sie grenzen ein, wie diese Ergebnisse zu verstehen sind.

## 12. High-Level Tensor Algebra und Graph-Kompilierung: scirust-tensor

### 12.1 Motivation und Kontext
Während der Kern von SciRust robuste Primitive für Deep Learning bietet, erfordern komplexe Architekturen wie Transformer flexiblere Tensor-Manipulationen als einfache Matrixmultiplikationen. Aktuelle State-of-the-Art-Frameworks (JAX, PyTorch) setzen auf optimiertes `einsum` und Graph-Compiler (XLA), um den Speicher-Overhead zu reduzieren. Um diese Lücke zu schließen und gleichzeitig die Pure-Rust- und deterministische DNA von SciRust zu bewahren, haben wir `scirust-tensor` eingeführt.

### 12.2 Methodik: Einsum und Contraction Planning
Das Modul implementiert einen optimierten `einsum`-Parser und einen **Contraction-Planner**. Für einen gegebenen Tensor-Kontraktions-Ausdruck:
$$C_{i,l} = \sum_{j,k} A_{i,j,k} \cdot B_{k,j,l}$$
evaluiert der Planner den optimalen Ausführungspfad. Für Kontraktionen mit mehreren Tensoren verwendet er einen gierigen Ansatz, um die Gesamtzahl der Gleitkommaoperationen (FLOPs) zu minimieren.

### 12.3 Graph-Optimierung und Operator-Fusion
Ein wesentlicher Beitrag dieses Moduls ist die **Operator-Fusion** Engine. In Standard-Runtimes beinhalten sequentielle Operationen wie `MatMul -> BiasAdd -> ReLU` mehrere Speicherdurchläufe und Zwischenpuffer. `scirust-tensor` kompiliert diese in einen einzigen **fused kernel**, was den Druck auf die Speicherbandbreite reduziert.
Die Optimierungs-Pipeline umfasst:
- **Redundanz-Eliminierung:** Entfernen von Identitäts-Transpositionen.
- **Stride-basierte Permutation:** Integration von Achsen-Permutationen in die GEMM-Kernel-Strides, um explizite Datenkopien zu eliminieren.

### 12.4 Ergebnisse und Determinismus
Durch Verwendung einer festen Reduktionsreihenfolge in allen Tensor-Kontraktionen stellen wir bit-genau identische Ergebnisse über verschiedene Thread-Anzahlen hinweg sicher. Vorläufige Benchmarks zeigen, dass Operator-Fusion die Spitzen-Speichernutzung bei tiefen Transformer-Blöcken um bis zu 35 % reduziert, während ein strikter deterministischer Fingerabdruck beibehalten wird. Das Modul ist voll kompatibel mit der **SRT1** Inferenz-Runtime und dem **QSR1** Int8-Quantisierungsstack.

### 12.5 Einschränkungen
Der Graph-Compiler ist derzeit auf statische Formen (shapes) beschränkt. Dynamische Shape-Unterstützung und JIT-Kompilierung von Kerneln für beliebige Fusionsmuster bleiben zukünftige Arbeiten.

## 13. Fazit

SciRust ist ein Pure-Rust Deep-Learning-Framework – ein Hybrid aus Runtime und Transpiler –, auf dem vier Fähigkeiten aufgebaut und validiert wurden: Ein portabler GPU- und Tensor-Core-Pfad, der ~63 TFLOPS in BF16 erreicht; eine hybride genetisch-gradientenbasierte symbolische Regressions-Engine, die bekannte Gesetze aus Daten unter Verwendung der frameworkeigenen symbolischen Differenzierung wiedergewinnt; eine deterministische Inferenz-Runtime, die bit-exakte, latenzgebundene, auditierbare und architektur-generische Inferenz bietet; und ein deterministischer Int8-Quantisierungsstack, der einen portablen, thread-invarianten ganzzahligen Inferenzpfad für den eingebetteten Einsatz bietet, mit Festkomma-Requantisierung, die das Modell bit-genau reproduziert und Gewichtstensoren etwa viermal kleiner macht. Darauf aufbauend etablieren fünf fortgeschrittene Features – ein statischer AOT-Compiler für Zero-Overhead-Inferenz, eine Soft-Float-Matrix-Engine für plattformübergreifende Bit-Exaktheit, latente Aktivierungssteuerung für Echtzeit-Representation-Engineering, quantisierungsbewusstes Training (QAT) via STE und eine Integrated Gradients Engine für mathematische Erklärbarkeit – SciRust weiter als Framework mit hoher Integrität. Die Hinzufügung **moderner KI-Familien** (RL, CV, Generative, GNN) verbreitert den Umfang des Frameworks weiter in Richtung eines einheitlichen Pure-Rust KI-Stacks. Der rote Faden ist methodisch: Jeder Beitrag wurde erst nach Übereinstimmung mit einem Orakel akzeptiert, Reproduzierbarkeit wurde gemessen statt angenommen – in mehreren Fällen bit-genau – und die nützlichsten Erkenntnisse waren diejenigen, die die Messungen gegen die Erwartung erzwangen. Die nächsten Schritte folgen direkt: Ein GPU-beschleunigter Forward-Pfad unter Wiederverwendung des validierten cuBLAS-Backends für dichte Schichten, ein dreidimensionaler Inferenzpfad für auf Aufmerksamkeit basierende Modelle und Supply-Chain-Pinning, um die Auditierbarkeit der Runtime von ihren Gewichten auf ihren Build auszuweiten.

## 14. Deterministische Ereignis-Erkennung und -Klassifizierung

### 14.1 Motivation
Echtzeit-Ereignis-Erkennung in kritischen Systemen (z. B. Neuroprothesen oder industrielle Steuerung) erfordert nicht nur hohe Genauigkeit, sondern auch absoluten Determinismus für Auditierbarkeit und Zertifizierung. Aktuelle Frameworks verlassen sich oft auf nicht-deterministische parallele Reduktion oder stochastisches Sampling, was für hochriskante Umgebungen ungeeignet ist.

### 14.2 Methodik
Wir führen eine Streaming-Architektur ein, die auf deterministischen gleitenden Fenstern basiert. Jedes Fenster $W$ der Größe $N$ wird in einen Tensor $T \in \mathbb{R}^{1 \times N}$ transformiert. Ereignis-Erkennung wird als Score-Funktion $S(T) \to [0, 1]$ formuliert. Für die Klassifizierung nutzen wir die Kern-MLP- und CNN-Schichten des Frameworks, eingefroren im SRT1-Format.
$$ \text{Ereignis}(t) = \mathbb{I}(S(W_t) > \tau) $$
wobei $\tau$ ein kalibrierter Schwellenwert ist.

### 14.3 Ergebnisse und Metriken
Erwartete Leistung auf dem Numenta Anomaly Benchmark (NAB) zielt auf einen F1-Score von $>0,85$ bei Null Bit-Drift über mehrere Threads ab. Die Verwendung der QSR1 Int8-Quantisierung wird voraussichtlich die Latenz auf eingebetteten ARM-Prozessoren um das 3-fache reduzieren, während eine MSE-Bit-Nähe von $<10^{-4}$ im Vergleich zum f32-Orakel beibehalten wird.

## N-D-Autograd und forschungsgetriebene Erweiterungen

Über das 2-D-Rückwärtsband hinaus bietet SciRust nun ein **N-D-Autograd-Band**,
dessen Operatoren jeweils per Finite-Differenzen-Gradientenprüfung validiert sind,
und darauf einen forschungsgestützten Deep-Learning-Stack. Jede Fähigkeit
entspricht einer bestimmten Arbeit und wird mit einem ehrlichen Test geliefert
(eine Gradientenprüfung für einen Operator, ein Korrektheits-/Orakel-Test für
eine Garantie); die vollständige Zuordnung – nun sind **alle 80 von 80
Kandidaten-Arbeiten fertig** – führt `docs/RESEARCH_ROADMAP.md`.

- **Kausales Decoder-Sprachmodell & effizientes Decoding**: ein durchgängig trainierter Decoder (Token- und gelernte Positions-Embeddings, kausale Multi-Head-Attention, eine fusionierte numerisch stabile Softmax-Kreuzentropie), der eine feste Sequenz exakt lernt; dazu exaktes (ausgabeerhaltendes) spekulatives Decoding, Medusa-Multi-Head- und EAGLE-Feature-Level-Drafting sowie eine Paged-KV-Cache-Attention (vLLM-Stil), die unter Fragmentierung bit-identisch ist.
- **LLaMA-Familie & Attention**: RMSNorm, SwiGLU, ein Pre-RMSNorm-Block, rotierende Positions-Embeddings (`RoPE`, getestete Relativpositions-Eigenschaft), gruppierte/Multi-Query-Attention via Batched-Matmul-Broadcasting, ALiBi-lineare Positions-Verzerrung, eine gekachelte Online-Softmax-FlashAttention und YaRN-Kontexterweiterung.
- **Effiziente Sequenzmodelle** (jeweils auf dem Band abgerollt und gradientengeprüft): Mamba mit selektivem Zustandsraum und Mamba-2/SSD (die Dualität Zustandsraum ↔ Attention), S4/S4D und S5 (MIMO mit einem parallelen assoziativen Scan), RWKV, RetNet, GLA, HGRN, DeltaNet, xLSTM (sLSTM + mLSTM) und Hyena (implizite lange Faltung).
- **Deterministische Optimierer** (alle bit-für-bit reproduzierbar): Adam, AdamW, Lion, Muon (Newton–Schulz-orthogonalisiertes Momentum), Schedule-Free, AdEMAMix, SOAP, Shampoo, Adafactor, LAMB, Adan, Prodigy, Lookahead, SAM, GaLore (niedrigrangig projizierte Zustände) und Sophia (geclippte diagonale Hessian-Methode zweiter Ordnung).
- **Quantisierung, Kompression & PEFT** (jeweils gegen Round-to-Nearest getestet): SmoothQuant, `GPTQ`, AWQ, NF4, SqueezeLLM, SpQR, KVQuant, LLM.int8(), OmniQuant, ternäres BitNet b1.58, QuIP# (Hadamard-Inkohärenz + E8-Gitter) und AQLM (additive Multi-Codebuch-Quantisierung); Wanda-/Magnitude-/Lottery-Pruning; sowie LoRA- / DoRA-Adapter.
- **Zertifizierbare KI und vollständige Verifizierung**: Interval Bound Propagation, CROWN, Zonotope (AI²/DeepZ), DeepPoly (relationale Polyeder), Randomized Smoothing, GloRo-Lipschitz-Schranken und CROWN-IBP **zertifiziertes Training** (eine differenzierbare IBP-Schranke vergrößert den zertifizierten Radius); dazu **vollständige** Verifizierer – Branch-and-Bound, eine exakte MILP-Formulierung und eine Reluplex-artige Lazy-SMT-Suche –, die über Robustheit entscheiden und konkrete Gegenbeispiele liefern (für kleine ReLU-Netze).
- **Unsicherheit & Kalibrierung**: verteilungsfreie konforme Prädiktion (split, CQR, adaptives APS/RAPS, risikokontrollierendes RCPS, Learn-then-Test, online ACI), Temperatur-Skalierung und Deep Ensembles mit epistemischer Unsicherheit.
- **Wissenschaftliche Brücke**: ein Neural ODE mit Backprop durch einen RK4-Löser, ein physikinformiertes neuronales Netz (PINN), ein Fourier Neural Operator (FNO), der einen Operator lernt und generalisiert, DeepONet und ein Kolmogorov–Arnold-Netz (KAN).
- **Reproduzierbarkeit, Privatsphäre & Audit**: reihenfolgenunabhängige Gleitkomma-Summe/-Mittel/-Skalarprodukt (bit-identisch unabhängig von der Thread-Anzahl), DP-SGD mit einem Rényi-DP-Accountant, ein LLM-Wasserzeichen mit einem Erkennungs-z-Test, **DiFR** (Verifikation einer Inferenz trotz Gleitkomma-Nichtdeterminismus, via einer fundierten FP-Fehler-Hülle um die reproduzierbare Referenz) und ein Argument für **verifizierbare Inferenz** (Finite-Field-Freivalds + ein Modell-Commitment + Fiat-Shamir – kryptographische Korrektheit, nicht Zero-Knowledge).
- **Quanten- / Tensor-Netzwerk-Simulation**: ein MPS-Quantenschaltkreis-Simulator (Matrix Product State / Tensor Train) (`quantum::Mps`), der einen n-Qubit-Zustand als Kette von Rang-3-Tensoren speichert – O(n·χ³) statt 2ⁿ, solange die Verschränkung moderat bleibt – mit Ein- und Zwei-Qubit-Gattern und Bond-Dimensions-Abschneidung über die hauseigene reine Rust-Implementierung der abgeschnittenen SVD (kein FFI); validiert, einen dichten Zustandsvektor exakt zu reproduzieren (zufälliger 5-Qubit-Schaltkreis, Bell, GHZ). Dieselbe Kontraktions- und abgeschnittene-SVD-Maschinerie liegt der Tensor-Train-Gewichtskompression (`tn::tt_decompose`) zugrunde.

CLI-Befehle erschließen einen Großteil dieser Arbeit, darunter `scirust certify` (IBP-, CROWN-, Zonotop-, DeepPoly- und Randomized-Smoothing-Schranken nebeneinander, dazu eine vollständige Branch-and-Bound-Entscheidung), `scirust lm --opt …` (Training des N-D-Decoder-LM mit einem beliebigen der obigen Optimierer), `scirust conformal`, `scirust calibrate`, die Sequenzmodell-Demos (`mamba`, `deltanet`, `retnet`, `gla`, `hgrn`, `rwkv`), die Quantisierer (`gptq`, `awq`, `bitnet`), `scirust pinn` und `scirust quantum` (der MPS-Quantenschaltkreis-Simulator).
