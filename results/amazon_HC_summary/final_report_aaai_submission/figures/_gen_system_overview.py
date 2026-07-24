#!/usr/bin/env python3
"""Generate the general HC-RAG system-overview diagram for the AAAI-27 paper.

Deliberately GENERAL: no dataset- or library-specific names (no CRAG, no FAISS).
Every element reflects the method already described in main.tex:
  offline: calibrate a single global null from irrelevant query-document pairs;
  online:  query -> embed & score -> candidate pool -> HC cutoff (scores to
           p-values under the null, k* = argmax_i HC_i) -> top-k* passages;
  then a generator (LLM) consumes the selected passages.
Phases are marked with light side brackets/labels, not enclosing boxes.
Output: fig_system_overview.png (single-column friendly, portrait aspect).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- palette (soft, print-friendly) --------------------------------------
C_IO    = "#dfe7f3"   # query / corpus / answer
C_PROC  = "#f6d9cf"   # embedding / retrieval steps
C_POOL  = "#d9ead3"   # candidate pool / selected passages
C_NULL  = "#e6dcf0"   # global null (offline)
C_HC    = "#ffe08a"   # HC cutoff (highlighted core)
EDGE    = "#5b5b5b"
GREY    = "#777777"

fig, ax = plt.subplots(figsize=(6.6, 6.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, text, fc, fontsize=10.0, weight="normal", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=2",
                                fc=fc, ec=EDGE, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, zorder=5)


def arrow(x1, y1, x2, y2, lw=1.7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=15,
                                 lw=lw, color=EDGE, shrinkA=0, shrinkB=0,
                                 zorder=4))


def vbracket(x, y_lo, y_hi, label, tick=2.2):
    """Vertical phase bracket with ticks pointing left and a rotated label."""
    ax.plot([x, x], [y_lo, y_hi], color=GREY, lw=1.3, zorder=2)
    ax.plot([x - tick, x], [y_hi, y_hi], color=GREY, lw=1.3, zorder=2)
    ax.plot([x - tick, x], [y_lo, y_lo], color=GREY, lw=1.3, zorder=2)
    ax.text(x + 2.4, (y_lo + y_hi) / 2, label, ha="center", va="center",
            rotation=90, fontsize=9.5, style="italic", color=GREY)


cx = 56  # center x of the online vertical flow

# ---- online retrieval flow (top -> bottom) --------------------------------
box(36, 80, 18, 8, "Query", C_IO)
box(58, 80, 18, 8, "Corpus", C_IO)
box(38, 67, 36, 8, "Embed & similarity scoring", C_PROC, fontsize=9.5)
arrow(45, 80, 49, 75)                      # query  -> embed
arrow(67, 80, 63, 75)                      # corpus -> embed

box(35, 54, 42, 8,
    "Candidate pool\n(sorted similarities $s_1\\!\\geq\\!\\cdots\\!\\geq\\!s_N$)",
    C_POOL, fontsize=9.0)
arrow(cx, 67, cx, 62)                      # embed -> pool

# HC cutoff (highlighted core)
hc_x, hc_y, hc_w, hc_h = 38, 34, 36, 15
ax.add_patch(FancyBboxPatch((hc_x, hc_y), hc_w, hc_h,
                            boxstyle="round,pad=0.02,rounding_size=2",
                            fc=C_HC, ec=EDGE, lw=1.9, zorder=3))
ax.text(cx, hc_y + hc_h - 3.4, "Higher Criticism cutoff",
        ha="center", va="center", fontsize=10.0, weight="bold", zorder=5)
ax.text(cx, hc_y + hc_h / 2 - 1.6,
        "$s_i \\rightarrow p$-values under the null\n"
        "$k^\\star=\\arg\\max_i\\,\\mathrm{HC}_i$",
        ha="center", va="center", fontsize=9.5, zorder=5)
arrow(cx, 54, cx, hc_y + hc_h)             # pool -> HC

box(41, 21, 30, 8, "Top-$k^\\star$ passages", C_POOL, fontsize=9.5)
arrow(cx, hc_y, cx, 29)                    # HC -> top-k*

# ---- downstream generation ------------------------------------------------
box(41, 7, 30, 8, "LLM  $\\rightarrow$  Answer", C_IO, fontsize=9.5)
arrow(cx, 21, cx, 15)                      # top-k* -> LLM

# ---- offline calibration (single cell, left, feeds the HC cutoff) ---------
box(2, 34, 24, 13, "Global null\n(irrelevant\nquery-doc pairs)",
    C_NULL, fontsize=9.0)
ax.text(14, 49.5, "offline calibration", ha="center", va="bottom",
        fontsize=9.5, style="italic", color=GREY)
arrow(26, 40.5, hc_x, 40.5)                # global null -> HC (clean gap)

# ---- phase brackets on the right ------------------------------------------
vbracket(82, 21, 88, "online retrieval")   # query .. top-k* passages
vbracket(82, 7, 15, "generation")          # LLM -> answer

plt.tight_layout(pad=0.4)
fig.savefig("fig_system_overview.png", dpi=220, bbox_inches="tight")
print("wrote fig_system_overview.png")
