# CCOS research paper (arXiv source)

[`ccos_regions.tex`](ccos_regions.tex) — *The Cognitive MMU: Deterministic, Auditable
Working Memory for LLM Coding Agents (and an honest account of why causal retrieval
does not beat RAG)*. Self-contained LaTeX (`article` class, standard packages, inline
`thebibliography`), ready for arXiv submission. The project README and docs describe
**what CCOS is today**; this paper carries the **research story** — the original
hypothesis, the bug-mining harness, and the measurements against RAG / GraphRAG.

## Language variants

The canonical paper is English; faithful translations carry the same structure,
equations, tables and bibliography (only the prose is translated).

Every variant also ships as **GitHub-flavoured Markdown** (`*.md`), and the English and French
additionally as **PDF** — see *Markdown and PDF renditions* below.

| LaTeX source | Language | LaTeX engine | Markdown · PDF |
| --- | --- | --- | --- |
| [`ccos_regions.tex`](ccos_regions.tex)       | English (canonical) | `pdflatex`           | [`.md`](ccos_regions.md) · [`.en.pdf`](ccos_regions.en.pdf) |
| [`ccos_regions.fr.tex`](ccos_regions.fr.tex) | Français            | `pdflatex` (babel)   | [`.fr.md`](ccos_regions.fr.md) · [`.fr.pdf`](ccos_regions.fr.pdf) |
| [`ccos_regions.es.tex`](ccos_regions.es.tex) | Español             | `pdflatex` (babel)   | [`.es.md`](ccos_regions.es.md) |
| [`ccos_regions.zh.tex`](ccos_regions.zh.tex) | 中文 (Chinese)      | `xelatex` (ctex)     | [`.zh.md`](ccos_regions.zh.md) |
| [`ccos_regions.ko.tex`](ccos_regions.ko.tex) | 한국어 (Korean)     | `xelatex` (kotex)    | [`.ko.md`](ccos_regions.ko.md) |
| [`ccos_regions.ar.tex`](ccos_regions.ar.tex) | العربية (Arabic)    | `xelatex` (polyglossia, RTL) | [`.ar.md`](ccos_regions.ar.md) |

## Build

```bash
# English / French / Spanish (Latin script):
pdflatex ccos_regions.fr.tex
pdflatex ccos_regions.fr.tex      # second pass resolves cross-references

# Chinese / Korean / Arabic (need XeLaTeX + the CJK/Arabic fonts installed):
xelatex ccos_regions.zh.tex
xelatex ccos_regions.zh.tex
```

Any standard TeX distribution (TeX Live / MiKTeX) or the arXiv build system works for
the Latin-script versions; no external `.bib` or non-standard packages are required.
The CJK and Arabic versions require XeLaTeX with the appropriate fonts (e.g. a CJK font
such as Noto/Source Han for `zh`/`ko`, and Amiri for `ar`). The translations have not
been compile-tested in this environment; a build pass may be needed, especially for the
right-to-left Arabic typesetting.

## Markdown and PDF renditions

The committed `*.md` and the English/French `*.pdf` are generated **without a TeX
installation** — `pandoc` for Markdown, and LaTeX→HTML→headless-Chromium for PDF:

```bash
# Markdown (all six languages), GitHub-flavoured, equations as $...$:
pandoc ccos_regions.fr.tex -f latex -t gfm -o ccos_regions.fr.md

# PDF (English & French): HTML with MathML (renders offline) → Chromium print-to-PDF:
pandoc ccos_regions.fr.tex -f latex -t html5 --standalone --mathml -o /tmp/p.html
chromium --headless --no-pdf-header-footer --print-to-pdf=ccos_regions.fr.pdf /tmp/p.html
```

The `.tex` remains canonical for arXiv and citations: pandoc maps `\ref` to anchors and
elides the inline `thebibliography` markers, which the Markdown does not reproduce.

## What it claims (and what it doesn't)

The paper is deliberately explicit about the boundary between proven and hypothesised
results:

- **Proven / measured** (reproducible, LLM-free): the formal region definition and
  causal-distance metric (§4), the determinism + replay theorem (§5), the locality
  evaluation (§7) — region recall 0.97 vs flat 0.35 at ≈48% fewer tokens, regions
  95.5% internally connected — and the **hypothesis simulation** (§8) under a stated
  retrieval oracle: lexical RAG 0% vs structure-aware regional selection 100% on
  cross-file causal tasks (a strong graph-BFS baseline ties the regional method).
  Regenerate with [`../../scripts/region_benchmark.sh`](../../scripts/region_benchmark.sh)
  and `ccos experiment`.
- **The honest negative result** (§9): on 70 real bug-fix commits, causal selection
  ties a plain lexical TF-IDF retriever and loses at a tight budget; on real code a
  fix's files share vocabulary. The one axis CCOS wins is **efficiency** — 4–9× fewer
  context tokens, because the causal region self-bounds instead of padding a top-k.
- **Still hypothesised** (a falsifiable protocol, *not yet run* in full): the real-LLM
  comparison against RAG, GraphRAG, MemGPT and LangGraph on long-horizon tasks (§9),
  with explicit hypotheses H1–H3 and threats to validity.

See [`../context_regions.md`](../context_regions.md) for the engineering-level
description of the same system.
