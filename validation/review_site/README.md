# Review presentation site

**Open:** double-click `index.html` - a single self-contained file (no server,
no internet, no build step at view time). Size 1.9 MB.

**Present:** press **P** for presenter mode (one section per screen), arrow keys /
space to advance, **Esc** to exit; slide counter bottom-right. Click any figure for
a full-screen lightbox. Scorecard rows expand on click; each table has a
"show only FAIL / PARTIAL" filter. **Print to PDF** from the browser for a clean
handout (one section per page); `handout.pdf` beside this file was produced that way
if a headless browser was available.

**Rebuild:** `python validation/analysis/build_review_site.py` (idempotent).

**What it reads (nothing is typed by hand):** validation/results/*.csv,
validation/figures/review/README.md + *.svg, data/DATASET_CARD.md,
report/validation_report.md (sections 1, 5b, 7, 8), validation/validation_plan.md (S/P
claims), validation/system_audit.md (pipeline diagram), validation/README.md
(reproduction commands), and git (branch/commit). The build asserts that every
numeric token in the rendered text originates from one of these sources
(structural whitelist: section indices, build date, git hash, the fixed
"Project 74" label), that banned claim tokens (oracle_v4 / reward /
screenshot) appear only inside the "Do not claim" box, disclaimer-marked text
(figure caveats, the retracted-claims list) or verbatim source quotes (the
pipeline diagram), and that the file is under 15 MB.
