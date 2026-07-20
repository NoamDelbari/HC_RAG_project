# AAAI-27 Compliance Fix — Plan and Checklist

This document lists every discrepancy between the previous project-report PDF
(`main_original.tex`) and the AAAI-27 Main Technical Track submission rules
(https://aaai.org/conference/aaai/aaai-27/submission-instructions/, and the
Author Kit at `/Users/lihuzur/Downloads/AuthorKit27/`), and the concrete fix
we apply for each. Every item is checked when the fix is landed.

The end state is:

- `main.tex` — anonymous AAAI-27 submission (≤ 7 content pages, references
  exclusively on pp. 8–9), built with `aaai2027.sty` / `aaai2027.bst`.
- `supplement.tex` — separate "Technical Appendix" PDF for OpenReview's
  Supplementary Document upload (also anonymous, same class).
- `refs.bib` — cleaned bibliography compatible with `aaai2027.bst`
  (author-year, full author lists, proper venues).
- `README_AAAI27_FIX.md` (this file) — record of what changed and why.
- `main_original.tex` — untouched snapshot of the previous version, kept for
  the audit trail.

---

## 0. Prep

- [x] **P0.1** Snapshot the previous submission as `main_original.tex` and
  `refs_original.bib` (never edited afterwards).
- [x] **P0.2** Copy the AAAI-27 style files into the submission folder:
  `aaai2027.sty`, `aaai2027.bst`.
- [x] **P0.3** Remove obsolete files that AAAI-27 forbids in the source
  archive (`aaai.sty` — the 2013 style — and `fixbib.sty`; the old
  `aaai.bst` is also unused). They are deleted from the folder that
  ships to OpenReview.

---

## 1. Blocking (double-blind / desk-reject risks)

- [x] **B1. Author identity removed from the LaTeX source.**
  Old author block (`\author{Lihu Zur \and Noam Delbari …@post.runi.ac.il}`)
  is replaced by the AAAI-27 anonymous form:
  `\author{Anonymous Submission}` and empty `\affiliations{}`.
- [x] **B2. PDF metadata scrubbed.** Removed `/Title`, `/Author`, `/Keywords`
  from `\pdfinfo`; kept only the required
  `\pdfinfo{ /TemplateVersion (2027.1) }`.
- [x] **B3. File-header author comment removed / neutralised** (was
  `% Authors: Ben Volovelsky, Noam Delbari.`, contradicting the title-page
  authors and constituting an identity leak).
- [x] **B4. De-anonymising phrases removed.**
  - Subtitle "Final Project Report" removed.
  - References to "our final project presentation" / "the presentation
    feedback" rewritten as neutral, self-contained motivation (Background
    §Research Question, Phase 2 §Answering feedback, Appendix E preamble).

## 2. Formatting rule violations

- [x] **F5. Style file swapped to the AAAI-27 kit.**
  Preamble becomes `\usepackage[submission]{aaai2027}` (the `submission`
  option hides the copyright block, sets the anonymous name, and enforces
  the review-time metadata). The 2013 `aaai.sty` is no longer loaded.
- [x] **F6. Numeric-citation override removed.**
  The entire `\makeatletter … \makeatother` block that redefined
  `\@biblabel`, `\@cite`, `\@citex`, `\thebibliography`, and
  `\bibliographystyle` is deleted. `aaai2027.sty` uses `natbib` (loaded
  automatically) and sets `aaai2027.bst`, giving the required author-year
  citations.
- [x] **F7. `\bibliographystyle{plain}` and unused `aaai.bst` removed.**
  Only `\bibliography{refs}` remains; the style is set by the class file.
- [x] **F.extra Font packages removed.**
  `\usepackage{times}`, `\usepackage{helvet}`, `\usepackage{courier}`,
  `\usepackage{float}` are removed — all forbidden by `aaai2027.sty` (and
  the required serif/sans/monospace fonts are loaded by the class).
- [x] **F.extra Manual page dimensions removed.**
  `\setlength{\pdfpagewidth}{…}` / `\setlength{\pdfpageheight}{…}` are
  removed; the class sets US Letter automatically. `\setcounter{secnumdepth}{2}`
  is left at `0` (AAAI-27 default; the previous value violated the class
  instruction that only 0/1/2 are supported and its own tables in the
  template use 0).

## 3. Page-layout rule violation

- [x] **L8. Pages 8–9 reserved exclusively for references.**
  All five appendices (Roadmap, HC-RAG Pipeline, Comparison Methods,
  CrossEntityQA Construction, Null-Distribution Diagnostics) are moved
  out of `main.tex` into a standalone `supplement.tex` intended for
  OpenReview's Supplementary Document slot (deadline: 3 days after the
  full-paper deadline). In `main.tex`, all `\ref{app:*}` cross-references
  become plain "(see supplement)" pointers so they do not need to resolve
  across files. Main paper ends with `\bibliography{refs}`; references
  are the last thing in the PDF.

## 4. Reproducibility / process requirements

- [x] **R9. Reproducibility checklist wired in.**
  README documents the requirement; the AAAI-27
  `ReproducibilityChecklist.tex` from the Author Kit is to be filled and
  uploaded to the dedicated OpenReview field (it is *not* included in the
  main PDF, per the submission instructions). We ship a starter copy at
  `ReproducibilityChecklist.tex` for the authors to complete.
- [x] **R10. Ethics/responsible-use statement.**
  A short, non-identifying "Ethics Statement" subsection is added inside
  the 7 content pages, disclosing use of LLM APIs (`gpt-4o-mini`,
  `gpt-4o`, `gpt-5-mini`) and public datasets (BEIR, HotpotQA, MuSiQue,
  Wikidata/Wikipedia) with licence-compatible use.
- [x] **R11. Generative-AI writing disclosure.**
  A one-line disclosure that generative-AI tools were used to assist with
  copy-editing (not with research design, execution, or analysis) is
  included in the Ethics Statement, per AAAI-27's "Appropriate use of
  Generative AI by Authors".

## 5. Content / citation hygiene

- [x] **C12. `maekawa2024holobench` full author list restored** in
  `refs.bib` (was `author = "… and others"`, which mangles author-year
  citations to "(Maekawa et al. 2024)" with no first-name resolution).
- [x] **C13. `xu2025car` cleaned up.** Given as `@misc` with
  `howpublished = {arXiv preprint arXiv:2511.14769}`, full author list,
  and year 2025. Verified the arXiv ID / year cited pre-dates the AAAI-27
  regular-paper deadline (July 28, 2026), so it is a valid citation.
- [x] **C.extra `bge` and `crag` entries** made valid `@misc` with URL and
  year; author lists are complete.
- [x] **C.extra No `and others`, no bare arXiv IDs, no missing years** —
  a scan of `refs.bib` confirms all entries have the fields
  `aaai2027.bst` expects (author, year, and either journal+volume+pages,
  booktitle, or howpublished).

## 6. Minor / verify

- [x] **M14. `\pubnote` not used** — correct for the review version
  (submission option handles copyright); no change needed.
- [x] **M15. Manual `\pdfpagewidth` / `\pdfpageheight` removed** (see F.extra).
- [x] **M16. `\setcounter{secnumdepth}` at default (0)** (see F.extra).
- [x] **M17. `\renewcommand{\thesection}{Appendix~\Alph{section}}`
  removed** — was in the appendix block, no longer needed since appendices
  have moved to `supplement.tex`, which uses its own numbering.
- [x] **M18. No `\acknowledgments` section** in the review-version PDF —
  correct; nothing to change.

## 7. Verification (final)

- [x] **V1.** `latexmk -pdf main.tex` and `latexmk -pdf supplement.tex`
  both exit 0 with **no undefined references or citations** in the final
  `main.log` / `supplement.log`.
- [x] **V2.** `main.pdf` = **6 pages** (well under the 7-page non-reference
  cap; references occupy the tail of p. 6, leaving pp. 7–9 unused, which
  is compliant since the "pp. 8–9 reserved exclusively for references"
  rule only bites when those pages exist).
- [x] **V3.** `supplement.pdf` = **3 pages**, standalone anonymous
  document ready for the OpenReview Supplementary Document field.
- [x] **V4.** No **Type 3** font references in either PDF
  (`grep /Type3` returns 0); the AAAI-27 style loads Times/newtx (Type 1)
  and Helvetica/Courier URW.
- [x] **V5.** PDF metadata scan (raw dictionary parse) shows
  `/Title`, `/Author`, `/Keywords`, `/Subject` **not present** in either
  PDF. `/Creator = TeX`, `/Producer = pdfTeX-1.40.29` — no identifying
  strings.
- [x] **V6.** `grep -InE "Lihu|Zur|Noam|Delbari|Volovelsky|post\.runi"`
  on `main.tex`, `supplement.tex`, `refs.bib` returns **zero matches**.
- [x] **V7.** All references to former appendices in the main paper are
  rewritten as "(see supplement)"; no dangling `\ref{app:*}` remains in
  `main.tex`.

---

## File map (final)

```
final_report_aaai_submission/
├── README_AAAI27_FIX.md           ← this file
├── main.tex                       ← anonymous AAAI-27 main paper
├── supplement.tex                 ← anonymous AAAI-27 technical appendix
├── refs.bib                       ← cleaned bibliography (author-year)
├── aaai2027.sty                   ← AAAI-27 style (from Author Kit)
├── aaai2027.bst                   ← AAAI-27 bibstyle (from Author Kit)
├── ReproducibilityChecklist.tex   ← to be filled and uploaded separately
├── figures/                       ← unchanged
└── main_original.tex, refs_original.bib   ← pre-fix snapshots
```
