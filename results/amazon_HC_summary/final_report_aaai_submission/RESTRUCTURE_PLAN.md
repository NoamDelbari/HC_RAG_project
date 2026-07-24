# AAAI-27 Restructuring Plan (Professor Feedback)

> **STATUS — IMPLEMENTED & VERIFIED (main.tex).** All five feedback items are
> landed in `main.tex`. Compiled clean under a separate jobname
> (`latexmk -pdf -jobname=main_check`): exit 0, **no undefined references or
> citations**, **7 pages total** with the References list at the tail of p.7
> (so non-reference content is ≤ 7 pages → AAAI-27 compliant). Verified section
> numbering: 1 Introduction (1.1 sparse detection … 1.5 Our Contributions) ·
> 2 Related Work · 3 Methods (3.1 HC Retrieval Algorithm + Algorithm 1
> pseudocode, 3.2 Methods and Pipelines) · 4 Experiments (4.1 Datasets/Table 1,
> 4.2 Phase 1, 4.3 Phase 2 — all tables/figures here) · 5 Conclusions ·
> 6 Ethics. System diagram = Figure 1 (p.2). `supplement.tex` unchanged
> (`rag_pipeline.png` kept in its appendix). Build artifacts (`main_check.*`)
> were removed; **`main.pdf` was never touched** (user owns it).
> Note: page count went 6 → 7 vs. the pre-restructure PDF; still within the
> 7-content-page cap.

This plan covers the round of restructuring requested by the two advisors. It
is scoped **only** to `results/amazon_HC_summary/final_report_aaai_submission/`
and edits `main.tex` (and, minimally, `supplement.tex`). Nothing outside this
folder is touched.

## Ground rules (from the user)
1. **Everything must be grounded** in the paper's existing results/content.
   No new claim, number, or capability is invented. New prose only *reorganizes
   or lightly rewords* material that is already in `main.tex`/`supplement.tex`.
2. **Minimal edits.** This is a restructuring pass: move existing sections/parts,
   rename headings, and add only the new scaffolding the feedback asks for.
   Do not rewrite content that does not need to move.
3. **AAAI-27 compliance is preserved** — checked against *both* sources: the
   Author Kit at `/Users/lihuzur/Downloads/AuthorKit27/` **and** the submission
   rules pasted by the user. Concretely:
   - AAAI **two-column** style via `aaai2027.sty` / `aaai2027.bst` (Author Kit);
     no forbidden packages (the kit's `.sty` hard-errors on hyperref, geometry,
     titlesec, fullpage, setspace, etc. — we add only `algorithm`/`algorithmic`,
     `booktabs`, `array`, `amsmath`, `amssymb`, `enumitem`, none forbidden).
   - **Anonymous / double-blind:** `\author{Anonymous Submission}`, no
     affiliations, no identity strings, scrubbed `\pdfinfo` (already done).
   - **US Letter**, high-resolution PDF, **Type 1/TrueType fonts only** (no
     Type 3 — the class loads newtx/Helvetica/Courier; figures are raster PNG).
   - **Page budget:** main PDF ≤ 9 pages total, with **pp. 8–9 reserved
     exclusively for references** ⇒ ≤ **7 pages of non-reference content**.
   - **Acknowledgements omitted** in the review version; the **Ethics
     Statement stays inside the 7 content pages**.
   - Only the **PDF** is required at submission for review (source later if
     accepted); we still keep the source compiling clean (`latexmk`, no
     undefined refs/citations).
4. When something is not 100% clear, ask before deciding.

## Decisions locked with the user
- **System diagram:** *author a NEW, general conceptual diagram* for the front
  of the paper (no dataset-specific terms — no "CRAG", no "FAISS"). The existing
  `figures/rag_pipeline.png` **stays in the supplement appendix** unchanged.
- **Related Work:** *build it from references already cited in the paper first.*
  If, in my judgment, that is not sufficient to make a credible Related Work
  section, I will stop and ask before adding any new reference.

---

## Feedback → task mapping

| # | Advisor request | What we do |
|---|-----------------|-----------|
| 1 | Start with an **Introduction**, ending with an **"Our Contributions"** subsection (may keep 1.1, 1.2, …). | Rename/adapt current front matter into `Introduction`; move the existing `Contributions` enumerate block to the **end** of the Introduction as a subsection. |
| 2 | Add a **Related Work** section (papers are cited in-line but no dedicated section exists). | New `Related Work` section after the Introduction, consolidating already-cited work into a narrative. |
| 3 | **Separate Methods from Experiments.** Methods describes the method/algorithm/system and holds **no tables**; Experiments holds the tables/results. | Split current `Methodology` + `Phase 1` + `Phase 2` into a table-free `Methods` section and an `Experiments` section that carries every table and figure of results. |
| 4 | Add a **system diagram** near the beginning summarizing the general idea. | New general conceptual figure (authored by me) placed in the Introduction. |
| 5 | Add a small **pseudo-code** block for the HCT-Ret procedure. | `algorithm`-style pseudocode placed in the Methods section next to the HC Retrieval Algorithm text. |

---

## FINAL structure decision (confirmed with user)
- **Introduction keeps 1.1, 1.2, and 1.3** (Sparse Signal Detection, HC as a
  Cutoff Rule / formula, The Null Distribution). Per the advisor, 1.1/1.2 stay in
  the Introduction; 1.3 stays with them because it describes *our own* null (the
  paper's central object), not prior art — Related Work is the wrong home for it.
- **Methods** = HC Retrieval Algorithm (+ new pseudocode) and Methods and
  Pipelines. It is the procedural/system description and holds **no tables**.
- **Experiments** carries the datasets/setup table and all Phase 1/Phase 2
  result tables and figures.

## Proposed new section structure

Current order:
```
Abstract
1  Background and Motivation
   1.1 Retrieval as Sparse Signal Detection        (Fig: signal detection)
   1.2 Higher Criticism as a Cutoff Rule           (HC formula)
   1.3 The Null Distribution
   1.4 Research Question
2  Methodology
   2.1 HC Retrieval Algorithm
   2.2 Methods and Pipelines
   2.3 Datasets and Setup                           (Table 1: datasets)
3  Contributions                                    (enumerate 1–3)
4  Integrating HC across Retrieval Pipelines        (Tables 2–4)
5  A Controlled Study on Amazon                     (Tables 5–6, Figs)
6  Conclusions and Future Work
7  Ethics Statement
References
```

Proposed order:
```
Abstract                                            (unchanged)
1  Introduction
   1.x Retrieval as Sparse Signal Detection         (Fig: signal detection)
   [NEW] system-overview figure near the top
   1.x Higher Criticism as a Cutoff Rule            (HC formula, stays here)
   1.x The Null Distribution                        (stays here)
   1.x Research Question
   1.x Our Contributions                            (moved from old §3)
2  Related Work                                      (NEW; existing citations)
3  Methods                                           (NO tables)
   3.x HC Retrieval Algorithm  +  [NEW] pseudocode
   3.x Methods and Pipelines                        (moved from 2.2)
4  Experiments
   4.x Datasets and Setup                           (Table 1, moved from 2.3)
   4.x Integrating HC across Retrieval Pipelines    (old §4, Tables 2–4)
   4.x A Controlled Study on Amazon                 (old §5, Tables 5–6, Figs)
5  Conclusions and Future Work                       (unchanged)
6  Ethics Statement                                  (unchanged)
References                                           (unchanged)
```

### The HC-formula / Null-Distribution placement — RESOLVED
Both stay in the **Introduction** (see "FINAL structure decision" above). They
are conceptual background on our method, not prior art, so they do not move to
Related Work; keeping them in the Introduction is also the most minimal edit.

---

## Task detail

### T1 — Introduction + Our Contributions
- Rename `\section{Background and Motivation}` → `\section{Introduction}`.
- Keep the opening motivation paragraph and `Retrieval as Sparse Signal
  Detection` (with its existing Fig. 1).
- Keep `Research Question` (fix its forward-reference wording so it points to the
  new Experiments section, "(see Experiments)"/`\ref` as appropriate).
- Move the whole old `Contributions` enumerate into a final subsection
  `\subsection{Our Contributions}` of the Introduction. Text unchanged except
  any cross-references (e.g. table labels) that must still resolve.

### T2 — Related Work (new, existing citations only)
Draft a compact section grouping the already-cited work into the same three
families the paper already uses, so every sentence is backed by a citation
already in `refs.bib`:
- **Fixed and geometric cutoffs:** Top-$k$; gap-based Adaptive-K
  `adaptivek2025`; clustering CAR `xu2025car`; complexity routing Adaptive-RAG
  `jeong2024adaptiverag`.
- **Multiple-testing / goodness-of-fit rules:** Bonferroni `bonferroni1961dunn`;
  Benjamini–Hochberg `benjamini1995fdr`; Berk–Jones `berkjones1979`; Higher
  Criticism `donohojin2004hc`.
- **RAG systems, frameworks, benchmarks:** RAG `lewis2020rag`; FlashRAG
  `jin2025flashrag`; BEIR/FiQA `thakur2021beir`; HotpotQA `yang2018hotpotqa`;
  MuSiQue `trivedi2022musique`; HoloBench `maekawa2024holobench`; BGE
  `bgeembeddings2023`; FAISS `johnson2019faiss`; CRAG `crag2024`.

Some of these citations currently only live in the supplement's roadmap; using
them in the main paper's Related Work is fine (they are already in `refs.bib`).
**Checkpoint:** if this reads too thin to pass as a Related Work section, I stop
and ask about adding new references (per the locked decision).

### T3 — Methods (table-free) vs. Experiments (tables/results)
- New `\section{Methods}` containing: HC Retrieval Algorithm (+ pseudocode T5)
  and Methods and Pipelines. **No tables here.**
- New `\section{Experiments}` containing: Datasets and Setup (Table 1, the
  datasets table), then the old Phase 1 section (Tables 2–4) and old Phase 2
  section (Tables 5–6 and the result figures) as subsections.
- Only tables/figures of *results and setup* live under Experiments; Methods has
  none. Update every `\ref{}`/`\label{}` and any "Section~\ref{...}" prose so
  cross-references still resolve.

### T4 — New general system diagram  [figure DONE, awaiting wire-in]
- A clean, **general** schematic (no dataset/library names — no "CRAG", no
  "FAISS"). Content, strictly reflecting the method already described in the
  paper: `Query` + `Corpus` → `Embed & similarity scoring` → `Candidate pool
  (sorted similarities $s_1\ge\cdots\ge s_N$)` → **HC cutoff box** (`scores →
  p-values under the null`, `k* = argmax_i HC_i`) → `Top-k* passages` → `LLM →
  Answer`. The single **Global null (irrelevant query–doc pairs)** cell sits to
  the left, labelled *offline calibration*, and feeds the HC box by a clean
  horizontal arrow. Phases are marked with light right-side brackets: *online
  retrieval* (Query…Top-k*) and *generation* (LLM→Answer) — no enclosing box.
- **How to regenerate (documented for future edits):**
  ```bash
  source ~/Documents/scripts/scripts_venv/bin/activate
  cd results/amazon_HC_summary/final_report_aaai_submission/figures
  python _gen_system_overview.py      # writes fig_system_overview.png
  ```
  Generator: `figures/_gen_system_overview.py` (matplotlib only; matches the
  existing `figures/_gen_*.py` convention). Output:
  `figures/fig_system_overview.png` (220 dpi, portrait, single-column friendly).
  Design history: v1 was a single horizontal chain (too cramped for one
  column); v2 a top-to-bottom flow; v3 wrapped the online steps in one big
  panel (rejected — heavy, and wrongly swept LLM/Answer into "retrieval"); v4
  (current) uses side brackets to mark phases instead.
- Placed as the first figure in the Introduction.

### T5 — HCT-Ret pseudocode
- A small `algorithm`/`algorithmic`-style block (or a lightweight
  `verbatim`/tabular fallback if the AAAI class does not bundle
  `algorithm2e`/`algpseudocode` — I will confirm what the style file supports
  before choosing) transcribing the **already-described** offline+online
  procedure: calibrate null offline; per query: retrieve pool → z-score →
  p-values via null CDF → sort → compute HC over the $\gamma$ window → return
  top-$k^\star$. No new algorithmic content.
- Placed in Methods beside the HC Retrieval Algorithm text.

### Supplement
- `supplement.tex` keeps `figures/rag_pipeline.png` (Appendix B) unchanged.
- Only adjust the supplement if a cross-reference to it changes wording.

---

## Page-budget note (AAAI-27: ≤ 7 content pages)
`main.pdf` is currently 6 pages. The additions (system diagram ≈ ¼ col,
Related Work ≈ ½ col, pseudocode ≈ ⅓ col, extra section headers) add roughly
half a page, landing near 6.5–7 content pages. This should fit, with references
allowed to continue onto pp. 8–9. I will keep the new prose tight and, if it
crosses 7 content pages, first tighten wording (not delete results) and flag it
to you before moving anything to the supplement.

## Verification (after edits)
> **`main.pdf` is owned by the user.** They compile/produce the final
> `main.pdf` themselves. To verify my edits build, I compile under a **separate
> jobname** (`latexmk -pdf -jobname=main_check main.tex` → `main_check.pdf`) so
> `main.pdf` is never overwritten. Same for the supplement if needed.

1. `latexmk -pdf -jobname=main_check main.tex` exits 0 with **no undefined
   references/citations** (inspect `main_check.log`).
2. `latexmk -pdf supplement.tex` still exits 0.
3. Content pages ≤ 7; references only after content.
4. No Type 3 fonts; anonymity intact (no author/identity strings).
5. Re-read the diff to confirm only intended moves + the five new pieces changed.
