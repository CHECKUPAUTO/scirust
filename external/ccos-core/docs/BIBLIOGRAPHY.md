# CCOS — Annotated Bibliography

A curated, **verified** reading list (~60 papers across 12 themes) underpinning the
ideas in CCOS. Every entry was confirmed to exist (title / authors / year) with a
primary link (arXiv, ACL, DOI, USENIX, IETF). For the design write-up that uses
these ideas, see [`PAPER.md`](PAPER.md); for the code map, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

Each entry ends with a one-line *CCOS:* note tying it to a concrete mechanism in
the system.

## How the themes map to CCOS modules

| CCOS module / concept | Themes |
| --------------------- | ------ |
| `memory`, `scheduler`, `select_context_window`, `enforce_paging` | §1 Long-context · §2 Virtual memory & caching · §3 Agent memory |
| `llm`, `parser`, `incremental`, the causal graph | §4 Retrieval-augmented generation · §5 Code representations & graphs |
| `agents` | §6 SE agents & benchmarks · §7 Agent reasoning & tool use |
| `guard`, `consensus`, `adversarial` | §8 Guardrails & structured output · §9 Consensus & ensembles · §10 Adversarial & chaos |
| `event_log`, `distributed_event_log`, `propagate_failure` | §11 Tamper-evident logs & replay · §12 Causality & fault localization |

---

## Part I — Context as Memory: paging, eviction & the OS analogy

### §1. Long-context & context-window management

- **Efficient Streaming Language Models with Attention Sinks** — Xiao et al., *ICLR* 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
  <br>*CCOS:* the "attention sink" finding — keep a few anchor tokens when evicting the KV cache — justifies CCOS always keeping high-importance anchor nodes resident while paging out the rest.
- **H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs** — Zhang et al., *NeurIPS* 2023. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
  <br>*CCOS:* score-based eviction of all but the "heavy-hitter" tokens is the direct analog of CCOS paging nodes by causal score (importance · recency · access).
- **Lost in the Middle: How Language Models Use Long Contexts** — Liu et al., *TACL* 2024. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
  <br>*CCOS:* the U-shaped accuracy curve motivates actively ranking/positioning the most causally-relevant nodes rather than dumping a long flat context.
- **Efficient Memory Management for LLM Serving with PagedAttention** — Kwon et al., *SOSP* 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
  <br>*CCOS:* borrowing OS virtual-memory paging for the KV cache is the foundational precedent for CCOS's whole "manage context like an OS" architecture.
- **LLMLingua: Compressing Prompts for Accelerated Inference of LLMs** — Jiang et al., *EMNLP* 2023. [arXiv:2310.05736](https://arxiv.org/abs/2310.05736)
  <br>*CCOS:* budget-controlled prompt compression is how CCOS can fit selected high-score nodes into a bounded window without losing semantics.

### §2. OS virtual memory, caching & the working set (the core analogy)

- **The Working Set Model for Program Behavior** — Denning, *CACM* 1968. [doi:10.1145/363095.363141](https://doi.org/10.1145/363095.363141)
  <br>*CCOS:* defines the working set — recently referenced pages that must stay resident — exactly CCOS's model for which context to keep paged in vs. evict.
- **A Study of Replacement Algorithms for a Virtual-Storage Computer (MIN/optimal)** — Belády, *IBM Systems Journal* 1966. [doi:10.1147/sj.52.0078](https://doi.org/10.1147/sj.52.0078)
  <br>*CCOS:* the offline-optimal "evict the farthest future use" bound is the yardstick against which CCOS's deterministic eviction policy can be measured.
- **An Anomaly in Space-Time Characteristics of Certain Programs Running in a Paging Machine (Bélády's anomaly)** — Bélády, Nelson & Shedler, *CACM* 1969. [doi:10.1145/363011.363155](https://doi.org/10.1145/363011.363155)
  <br>*CCOS:* shows more memory can cause *more* faults under FIFO — a caution motivating CCOS's score/recency-based eviction over naive FIFO.
- **The LRU-K Page Replacement Algorithm for Database Disk Buffering** — O'Neil, O'Neil & Weikum, *SIGMOD* 1993. [doi:10.1145/170035.170081](https://doi.org/10.1145/170035.170081)
  <br>*CCOS:* using the last K reference times to estimate reuse is a deterministic, frequency-aware eviction strategy CCOS can adopt for its `access_count` term.
- **LIRS: An Efficient Low Inter-reference Recency Set Replacement Policy** — Jiang & Zhang, *SIGMETRICS* 2002. [doi:10.1145/511334.511340](https://doi.org/10.1145/511334.511340)
  <br>*CCOS:* a scan-resistant policy that keeps high-reuse pages and quickly evicts one-offs — a model for retaining the genuine working set under churn.

### §3. Memory architectures for LLM agents

- **MemGPT: Towards LLMs as Operating Systems** — Packer et al., 2023. [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
  <br>*CCOS:* OS-inspired tiered memory paging data between an in-context window and external storage is the closest prior art to CCOS's core premise.
- **Generative Agents: Interactive Simulacra of Human Behavior** — Park et al., *UIST* 2023. [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
  <br>*CCOS:* its memory-stream retrieval scored by recency · importance · relevance is essentially CCOS's multi-factor node-scoring function.
- **A-MEM: Agentic Memory for LLM Agents** — Xu et al., 2025. [arXiv:2502.12110](https://arxiv.org/abs/2502.12110)
  <br>*CCOS:* Zettelkasten-style interlinked, dynamically-indexed memory notes parallel CCOS's causal graph of linked, scored code nodes.
- **MemoryBank: Enhancing LLMs with Long-Term Memory** — Zhong et al., *AAAI* 2024. [arXiv:2305.10250](https://arxiv.org/abs/2305.10250)
  <br>*CCOS:* Ebbinghaus-curve forgetting/reinforcement (decay or strengthen by time and significance) mirrors CCOS's recency and access terms.
- **Augmenting Language Models with Long-Term Memory (LongMem)** — Wang et al., *NeurIPS* 2023. [arXiv:2306.07174](https://arxiv.org/abs/2306.07174)
  <br>*CCOS:* a decoupled cache-and-retrieve side-network models how CCOS holds a large external code-graph memory and pages fresh slices into the active window.

## Part II — Retrieval & the Causal Code Graph

### §4. Retrieval-augmented generation (incl. code / repo-level)

- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** — Lewis et al., *NeurIPS* 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
  <br>*CCOS:* the parametric-plus-non-parametric memory split is the basis for treating the code graph as an external store feeding a bounded context.
- **REALM: Retrieval-Augmented Language Model Pre-Training** — Guu et al., *ICML* 2020. [arXiv:2002.08909](https://arxiv.org/abs/2002.08909)
  <br>*CCOS:* a learned latent retriever selecting documents to attend over prefigures CCOS's scored selection of which nodes to surface.
- **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** — Asai et al., *ICLR* 2024. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
  <br>*CCOS:* reflection tokens that critique and gate generation map onto CCOS's guard layer validating output before committing a transition.
- **Active Retrieval Augmented Generation (FLARE)** — Jiang et al., *EMNLP* 2023. [arXiv:2305.06983](https://arxiv.org/abs/2305.06983)
  <br>*CCOS:* re-retrieving when upcoming tokens are low-confidence is the failure-relevance signal CCOS uses to page in additional nodes when the agent stalls.
- **RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation** — Zhang et al., *EMNLP* 2023. [arXiv:2303.12570](https://arxiv.org/abs/2303.12570)
  <br>*CCOS:* an iterative retrieve-then-generate loop over whole-repo context is the code-specific precedent for paging repo-level graph nodes into a window.

### §5. Code representations, code graphs & program analysis

- **Learning to Represent Programs with Graphs** — Allamanis et al., *ICLR* 2018. [arXiv:1711.00740](https://arxiv.org/abs/1711.00740)
  <br>*CCOS:* validates encoding source as a graph whose edges capture syntactic/semantic relations — the same node/edge vocabulary CCOS builds.
- **code2vec: Learning Distributed Representations of Code** — Alon et al., *POPL* 2019. [arXiv:1803.09473](https://arxiv.org/abs/1803.09473)
  <br>*CCOS:* AST-path decomposition into fixed-length embeddings is a candidate feature signal for CCOS's per-symbol node scores.
- **CodeBERT: A Pre-Trained Model for Programming and Natural Languages** — Feng et al., *Findings of EMNLP* 2020. [arXiv:2002.08155](https://arxiv.org/abs/2002.08155)
  <br>*CCOS:* NL-PL bimodal embeddings the coder/reviewer agents could use to rank and retrieve relevant nodes.
- **GraphCodeBERT: Pre-training Code Representations with Data Flow** — Guo et al., *ICLR* 2021. [arXiv:2009.08366](https://arxiv.org/abs/2009.08366)
  <br>*CCOS:* injecting data-flow edges improves code understanding — motivating CCOS's dependency edges as first-class structure rather than plain text.
- **Modeling and Discovering Vulnerabilities with Code Property Graphs** — Yamaguchi et al., *IEEE S&P* 2014. [doi:10.1109/SP.2014.44](https://doi.org/10.1109/SP.2014.44)
  <br>*CCOS:* the Code Property Graph (AST + CFG + PDG) is the archetype for CCOS's security agent traversing the causal graph for vulnerable patterns.

## Part III — Agents over Code

### §6. Software-engineering agents & repo-level benchmarks

- **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** — Jimenez et al., *ICLR* 2024. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
  <br>*CCOS:* the canonical benchmark for CCOS's exact task — editing a real codebase to resolve an issue — a standard evaluation harness for its agents.
- **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering** — Yang et al., *NeurIPS* 2024. [arXiv:2405.15793](https://arxiv.org/abs/2405.15793)
  <br>*CCOS:* a purpose-built repo view/search/edit interface boosts success — paralleling CCOS exposing the graph as a structured action surface.
- **AutoCodeRover: Autonomous Program Improvement** — Zhang et al., *ISSTA* 2024. [arXiv:2404.05427](https://arxiv.org/abs/2404.05427)
  <br>*CCOS:* AST-aware, structure-based code-search APIs (retrieve methods/classes, not string matches) are essentially CCOS's symbol-graph navigation.
- **Agentless: Demystifying LLM-based Software Engineering Agents** — Xia et al., *ICSE/PACMSE* 2025. [arXiv:2407.01489](https://arxiv.org/abs/2407.01489)
  <br>*CCOS:* a localize→repair→validate pipeline arguing precise localization beats elaborate tooling — supporting CCOS's bet on a high-quality causal graph.
- **CodePlan: Repository-level Coding using LLMs and Planning** — Bairi et al., *FSE* 2024. [arXiv:2309.12499](https://arxiv.org/abs/2309.12499)
  <br>*CCOS:* incremental dependency analysis and change-may-impact propagation across a repo graph is the direct analog of CCOS's incremental O(Δ) updates.

### §7. LLM agent reasoning, tool use & self-improvement

- **ReAct: Synergizing Reasoning and Acting in Language Models** — Yao et al., *ICLR* 2023. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
  <br>*CCOS:* the interleaved reason-then-act loop is the control pattern CCOS's agents follow when querying the graph and applying edits.
- **Reflexion: Language Agents with Verbal Reinforcement Learning** — Shinn et al., *NeurIPS* 2023. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
  <br>*CCOS:* episodic self-reflection memory maps onto how the reviewer agent feeds verbal critiques back to the coder agent across iterations.
- **Toolformer: Language Models Can Teach Themselves to Use Tools** — Schick et al., *NeurIPS* 2023. [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
  <br>*CCOS:* grounds the premise that agents should call structured tools (graph queries, scorers) rather than reason from raw text.
- **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** — Yao et al., *NeurIPS* 2023. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
  <br>*CCOS:* search-with-backtracking over reasoning branches offers a planning strategy for agents exploring alternative edit paths over the graph.
- **Voyager: An Open-Ended Embodied Agent with Large Language Models** — Wang et al., *TMLR* 2024. [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)
  <br>*CCOS:* an ever-growing skill library of reusable code with self-verification models how CCOS could accumulate and re-apply analysis routines.

## Part IV — Reliability & Safety

### §8. Output validation, guardrails, structured generation & hallucination

- **NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails** — Rebedea et al., *EMNLP (System Demos)* 2023. [arXiv:2310.10501](https://arxiv.org/abs/2310.10501)
  <br>*CCOS:* programmable, model-independent runtime rails that filter/constrain output map directly onto CCOS's guard layer.
- **Efficient Guided Generation for Large Language Models (Outlines)** — Willard & Louf, 2023. [arXiv:2307.09702](https://arxiv.org/abs/2307.09702)
  <br>*CCOS:* finite-state-machine indexing over the vocabulary is the canonical mechanism for guaranteeing schema-valid JSON — exactly what the guard must emit.
- **Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning** — Geng et al., *EMNLP* 2023. [arXiv:2305.13971](https://arxiv.org/abs/2305.13971)
  <br>*CCOS:* context-free-grammar enforcement at decode time underpins the guard's rejection of malformed / over-nested payloads.
- **SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection** — Manakul et al., *EMNLP* 2023. [arXiv:2303.08896](https://arxiv.org/abs/2303.08896)
  <br>*CCOS:* sampling-and-comparison (consistent = trustworthy, divergent = hallucinated) doubles as a guard check and a consistency signal for consensus weighting.
- **Survey of Hallucination in Natural Language Generation** — Ji et al., *ACM Computing Surveys* 2023. [arXiv:2202.03629](https://arxiv.org/abs/2202.03629)
  <br>*CCOS:* the taxonomy of hallucination failure modes is exactly what the guard catches and what the adversarial harness injects as a fault class.

### §9. Multi-model consensus, ensembles & LLM-as-judge

- **Self-Consistency Improves Chain-of-Thought Reasoning** — Wang et al., *ICLR* 2023. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
  <br>*CCOS:* sample-many-paths-then-majority-vote is the direct theoretical basis for the consensus module's majority voting.
- **LLM-Blender: Ensembling LLMs with Pairwise Ranking and Generative Fusion** — Jiang et al., *ACL* 2023. [arXiv:2306.02561](https://arxiv.org/abs/2306.02561)
  <br>*CCOS:* PairRanker + GenFuser is a concrete realization of confidence-weighted resolution across models.
- **Mixture-of-Agents Enhances Large Language Model Capabilities** — Wang et al., *ICLR* 2025. [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
  <br>*CCOS:* a layered architecture where agents aggregate prior-layer outputs is a recent template for combining several models before consensus.
- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — Zheng et al., *NeurIPS Datasets & Benchmarks* 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
  <br>*CCOS:* establishes (and characterizes the biases of) using a strong LLM as evaluator — the mechanism behind confidence-weighted arbitration.
- **Improving Factuality and Reasoning through Multiagent Debate** — Du et al., *ICML* 2024. [arXiv:2305.14325](https://arxiv.org/abs/2305.14325)
  <br>*CCOS:* model instances debating to convergence is an alternative consensus protocol, and its black-box-only requirement matches CCOS's model-agnostic resolution.

### §10. Adversarial robustness, prompt injection & chaos engineering

- **Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection** — Greshake et al., *ACM AISec* 2023. [arXiv:2302.12173](https://arxiv.org/abs/2302.12173)
  <br>*CCOS:* defines the indirect prompt-injection threat model the adversarial harness replicates as a chaos-testing vector.
- **Universal and Transferable Adversarial Attacks on Aligned Language Models** — Zou et al., 2023. [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)
  <br>*CCOS:* automatically-generated transferable jailbreak suffixes are precisely the payloads the harness injects to stress-test the guard.
- **Ignore Previous Prompt: Attack Techniques for Language Models (PromptInject)** — Perez & Ribeiro, *NeurIPS ML Safety Workshop* 2022. [arXiv:2211.09527](https://arxiv.org/abs/2211.09527)
  <br>*CCOS:* goal-hijacking and prompt-leaking attacks provide ready-made injection cases for the adversarial chaos suite.
- **Chaos Engineering** — Basiri et al., *IEEE Software* 2016. [doi:10.1109/MS.2016.60](https://doi.org/10.1109/MS.2016.60) · [arXiv:1702.05843](https://arxiv.org/abs/1702.05843)
  <br>*CCOS:* the principles of controlled fault-injection experiments to verify resilience are the methodological blueprint for the adversarial harness (timeouts, corruption…).
- **Explaining and Harnessing Adversarial Examples** — Goodfellow et al., *ICLR* 2015. [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
  <br>*CCOS:* the seminal framing of worst-case perturbations and adversarial training grounds CCOS's robustness goals and fault-injection hardening.

## Part V — Provenance, Determinism & Causality

### §11. Tamper-evident logs, Merkle structures & deterministic replay

- **How to Time-Stamp a Digital Document** — Haber & Stornetta, *Journal of Cryptology* 1991. [doi:10.1007/BF00196791](https://doi.org/10.1007/BF00196791)
  <br>*CCOS:* the original linked-timestamping scheme hash-chaining each document to its predecessor is the direct ancestor of CCOS's append-only hash-chained event log.
- **A Digital Signature Based on a Conventional Encryption Function (Merkle trees)** — Merkle, *CRYPTO* 1987. [doi:10.1007/3-540-48184-2_32](https://doi.org/10.1007/3-540-48184-2_32)
  <br>*CCOS:* introduces the Merkle hash tree underpinning logarithmic-size integrity proofs of any event's membership without re-reading the log.
- **Efficient Data Structures for Tamper-Evident Logging** — Crosby & Wallach, *USENIX Security* 2009. [usenix.org](https://www.usenix.org/conference/usenixsecurity09/technical-sessions/presentation/efficient-data-structures-tamper-evident)
  <br>*CCOS:* its history tree with O(log n) membership and incremental consistency proofs against an untrusted logger is essentially CCOS's integrity-verification design.
- **Certificate Transparency (RFC 6962)** — Laurie, Langley & Kasper, *IETF* 2013. [rfc-editor.org](https://www.rfc-editor.org/info/rfc6962)
  <br>*CCOS:* a production append-only Merkle log with signed tree heads and consistency proofs — a real-world blueprint for proving the log is append-only and unforked.
- **Bitcoin: A Peer-to-Peer Electronic Cash System** — Nakamoto, 2008. [bitcoin.org](https://bitcoin.org/bitcoin.pdf)
  <br>*CCOS:* the canonical hash-chained ledger where each record commits to the prior one mirrors CCOS's tamper-evident chain enabling deterministic event-sourced replay.

### §12. Causality, fault localization & change-impact analysis

- **Causal Inference in Statistics: An Overview** — Pearl, *Statistics Surveys* 2009. [doi:10.1214/09-SS057](https://doi.org/10.1214/09-SS057)
  <br>*CCOS:* the formal do-calculus / structural-causal-model foundation for distinguishing causation from correlation grounds CCOS's causal scoring.
- **Visualization of Test Information to Assist Fault Localization (Tarantula)** — Jones, Harrold & Stasko, *ICSE* 2002. [doi:10.1145/581339.581397](https://doi.org/10.1145/581339.581397)
  <br>*CCOS:* the spectrum-based suspiciousness metric ranking statements by failing-vs-passing participation is directly analogous to CCOS's node scoring.
- **A Survey on Software Fault Localization** — Wong et al., *IEEE TSE* 2016. [doi:10.1109/TSE.2016.2521368](https://doi.org/10.1109/TSE.2016.2521368)
  <br>*CCOS:* a taxonomy of fault-localization techniques that situates and justifies CCOS's choice of scoring functions for failure propagation.
- **Simplifying and Isolating Failure-Inducing Input (Delta Debugging)** — Zeller & Hildebrandt, *IEEE TSE* 2002. [doi:10.1109/32.988498](https://doi.org/10.1109/32.988498)
  <br>*CCOS:* the ddmin algorithm narrowing a failure to its minimal cause is the basis for isolating the minimal set of events/changes behind a propagated failure.
- **Chianti: A Tool for Change Impact Analysis of Java Programs** — Ren et al., *OOPSLA* 2004. [doi:10.1145/1028976.1029012](https://doi.org/10.1145/1028976.1029012)
  <br>*CCOS:* decomposing an edit into atomic changes mapped to affected tests is the change-impact model CCOS uses to propagate effects along the dependency graph.

---

## Further reading (verified alternates)

- **Scissorhands: Exploiting the Persistence of Importance for KV Cache Compression** — Liu et al., *NeurIPS* 2023. [arXiv:2305.17118](https://arxiv.org/abs/2305.17118) — its "persistence of importance" hypothesis maps onto CCOS's importance scoring.
- **LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens** — Ding et al., *ICML* 2024. [arXiv:2402.13753](https://arxiv.org/abs/2402.13753) — positional-window extension.
- **Leave No Context Behind: Efficient Infinite Context Transformers (Infini-attention)** — Munkhdalai et al., 2024. [arXiv:2404.07143](https://arxiv.org/abs/2404.07143) — compressive memory for unbounded context.
- **Improving Language Models by Retrieving from Trillions of Tokens (RETRO)** — Borgeaud et al., *ICML* 2022. [arXiv:2112.04426](https://arxiv.org/abs/2112.04426) — chunked retrieval at scale.
- **ARC: A Self-Tuning, Low Overhead Replacement Cache** — Megiddo & Modha, *USENIX FAST* 2003. [usenix.org](https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache) — adaptive recency/frequency balancing for eviction.

---

*Verification note:* every citation above was confirmed to exist via web search
(title, authors, year) with a working primary link. A few venues evolved from a
preprint to a conference/journal version; where a paper is best known by its arXiv
ID, that ID is given. Corrections and additions welcome — see
[`CONTRIBUTING.md`](../CONTRIBUTING.md).
