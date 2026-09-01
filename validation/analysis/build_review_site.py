"""Build the self-contained review presentation site.

  python validation/analysis/build_review_site.py
      -> validation/review_site/index.html   (single file, no dependencies)
      -> validation/review_site/README.md

Every number shown on the page is read at build time from the declared data
sources (validation/results/*.csv, validation/figures/review/README.md,
data/DATASET_CARD.md, report/validation_report.md, validation/validation_plan.md,
validation/system_audit.md, validation/README.md, git). Numbers emitted by this
script go through fmt()/reg() which register their formatted tokens; whole
source texts register all their tokens. The build ends with:
  1. a numeric-provenance assertion - every numeric token in the rendered
     visible text must originate from a registered source token (structural
     whitelist: section indices, build date, git hash, the fixed project label);
  2. a banned-claims assertion - 'oracle_v4', 'screenshot' and the word
     'reward' may appear only inside explicitly allowed disclaimer boxes;
  3. a size check (< 15 MB).
The build fails loudly on any violation. Idempotent: re-running overwrites.
"""
from __future__ import annotations

import datetime
import html as html_mod
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "validation/results"
FIGDIR = ROOT / "validation/figures/review"
OUT = ROOT / "validation/review_site"
OUT.mkdir(parents=True, exist_ok=True)

PROJECT_LABEL = "Project 74"  # fixed page label, whitelisted (not a data claim)

# ---------------------------------------------------------------- provenance
ALLOWED: set[str] = set()
NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


def _norm(tok: str) -> str:
    return tok.replace(",", "").rstrip(".")


def reg_text(text: str) -> str:
    """Register every numeric token in a source-derived text; returns text."""
    for t in NUM_RE.findall(text):
        ALLOWED.add(_norm(t))
    return text


def read_source(path: Path) -> str:
    return reg_text(path.read_text(encoding="utf-8"))


def fmt(value: float, spec: str) -> str:
    """Format a source-derived value and register the result."""
    s = format(value, spec)
    return reg_text(s)


# structural whitelist: section indices + project label number
for tok in list("0123456789") + ["74"]:
    ALLOWED.add(tok)

# ---------------------------------------------------------------- sources
report = read_source(ROOT / "report/validation_report.md")
plan = read_source(ROOT / "validation/validation_plan.md")
card_md = read_source(ROOT / "data/DATASET_CARD.md")
fig_readme = read_source(FIGDIR / "README.md")
sysaudit = read_source(ROOT / "validation/system_audit.md")
val_readme = read_source(ROOT / "validation/README.md")

env_sc = pd.read_csv(RES / "environment_scorecard.csv")
agent_sc = pd.read_csv(RES / "agent_scorecard.csv")
claims_audit = pd.read_csv(RES / "claim_audit.csv")
a3 = pd.read_csv(RES / "a3_oracle_value.csv")
for df in (env_sc, agent_sc, claims_audit, a3):
    # cell-by-cell: registering the raw CSV text would merge comma-adjacent
    # values into one token and miss the individual numbers
    for col in df.columns:
        reg_text(str(col))
    for v in df.to_numpy().ravel():
        reg_text(str(v))

git_branch = reg_text(subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip())
git_hash = reg_text(subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip())
build_date = reg_text(datetime.date.today().isoformat())

# ---------------------------------------------------------------- parsing
def esc(s: str) -> str:
    return html_mod.escape(s, quote=False)


def md_inline(s: str) -> str:
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)  # links -> plain text
    return s


def md_to_html(md: str) -> str:
    out, in_ul, table = [], False, []

    def flush_table():
        nonlocal table
        if not table:
            return
        head, *rows = table
        out.append("<table><thead><tr>" +
                   "".join(f"<th>{md_inline(c)}</th>" for c in head) +
                   "</tr></thead><tbody>")
        for r in rows:
            out.append("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in r) + "</tr>")
        out.append("</tbody></table>")
        table = []

    for line in md.splitlines():
        ls = line.strip()
        if ls.startswith("|"):
            cells = [c.strip() for c in ls.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            table.append(cells)
            continue
        flush_table()
        if ls.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{md_inline(ls[2:])}</li>")
            continue
        if in_ul:
            out.append("</ul>")
            in_ul = False
        m = re.match(r"(#{1,4})\s+(.*)", ls)
        if m:
            lvl = len(m.group(1)) + 1
            out.append(f"<h{lvl}>{md_inline(m.group(2))}</h{lvl}>")
        elif ls:
            out.append(f"<p>{md_inline(ls)}</p>")
    if in_ul:
        out.append("</ul>")
    flush_table()
    return "\n".join(out)


def section_of(md: str, start_pat: str, end_pat: str) -> str:
    m = re.search(start_pat + r"(.*?)" + end_pat, md, re.S)
    if not m:
        raise SystemExit(f"BUILD FAIL: could not extract section {start_pat!r}")
    return m.group(1).strip()


# verdict badges (letters read from report section 1)
mv = re.search(r"\*\*SIMULATOR: ([A-E]) — ([^*]+?)\.\*\*", report)
ma = re.search(r"\*\*AGENTS: ([A-E]) — ([^*]+?)\.\*\*", report)
sim_grade, sim_phrase = mv.group(1), mv.group(2)
ag_grade, ag_phrase = ma.group(1), ma.group(2)

# section 8 verbatim paragraphs + do-not-claim
sec8 = section_of(report, r"## 8\.", r"## 9\.")
sim_claim = re.search(r"\*\*Simulator:\*\*\s*(\".*?\")", sec8, re.S).group(1)
ag_claim = re.search(r"\*\*Agents:\*\*\s*(\".*?\")", sec8, re.S).group(1)
do_not_claim = re.search(r"Do not claim:(.*?)(?:\n\n|\Z)", sec8, re.S).group(1).strip()

# section 7 Q&A
sec7 = section_of(report, r"## 7\.", r"## 8\.")
qa = re.findall(r"^\d+\.\s+\*\*(.+?)\*\*\s*(.*?)(?=^\d+\.\s+\*\*|\Z)", sec7, re.S | re.M)

# section 5b: E6 table + C2 paragraph
sec5b = section_of(report, r"## 5b\.", r"## 6\.")
e6_table = "\n".join(l for l in sec5b.splitlines() if l.strip().startswith("|"))
c2_par = re.search(r"\*\*C2 —.*?(?=\n\n|\Z)", sec5b, re.S).group(0)

# plan section 1: claims S and P
sp = re.search(r"\*\*\(S\)\*\*(.*?);\s*\*\*\(P\)\*\*(.*?)\.", plan, re.S)
claim_S, claim_P = sp.group(1).strip(), sp.group(2).strip()

# system-audit pipeline diagram (existing text diagram; nothing new drawn)
pipeline = section_of(sysaudit, r"## 1\. The actual pipeline\s*```", r"```")

# reproduction commands
repro = section_of(val_readme, r"```\n", r"```")

# retrieval numbers (computed from source CSVs, registered via fmt)
r_mem = a3[(a3.comparison == "oracle_v3 - oracle_v3_no_memory") & (a3.metric == "final_mrr")].iloc[0]
r_orc = a3[(a3.comparison == "oracle_v3 - boardroom") & (a3.metric == "final_mrr")].iloc[0]
mem_mean = fmt(r_mem.mean_diff / 1e3, ",.1f")
mem_lo = fmt(r_mem.ci95_lo / 1e3, ",.1f")
mem_hi = fmt(r_mem.ci95_hi / 1e3, ",.1f")
mem_p = fmt(r_mem.wilcoxon_p, ".4f")
mem_share = fmt(100 * r_mem.mean_diff / r_orc.mean_diff, ".1f")

# claim-audit summary counts
n_repro = fmt((claims_audit.status == "REPRODUCED").sum(), "d")
n_other = fmt((claims_audit.status != "REPRODUCED").sum(), "d")
retracted = claims_audit[claims_audit.status != "REPRODUCED"][["claim", "status"]]

# diagnosis text (from the C1 scorecard row - a data source)
c1_row = agent_sc[agent_sc.test.str.startswith("C1 hold-arm")].iloc[0]
diagnosis = c1_row.interpretation

# ---------------------------------------------------------------- figures
def inline_svg(name: str) -> str:
    svg = (FIGDIR / f"{name}.svg").read_text(encoding="utf-8")
    svg = re.sub(r"<\?xml.*?\?>\s*", "", svg, flags=re.S)
    svg = re.sub(r"<!DOCTYPE.*?>\s*", "", svg, flags=re.S)
    key = name.split("_")[0]
    svg = svg.replace('id="', f'id="{key}-')
    svg = svg.replace('url(#', f'url(#{key}-')
    svg = svg.replace('href="#', f'href="#{key}-')
    svg = re.sub(r'(<svg[^>]*?)\swidth="[^"]+"\sheight="[^"]+"',
                 r'\1 style="width:100%;height:auto"', svg, count=1)
    return svg


FIG_META = {}
for row in re.findall(r"^\|\s*`(f\d+[^`]*)`\s*\|(.+)$", fig_readme, re.M):
    cells = [c.strip() for c in row[1].strip("|").split("|")]
    FIG_META[row[0]] = dict(claim=cells[0], source=cells[1], n=cells[2], caveat=cells[3])


def fig_verdict(test_prefix: str, frame: pd.DataFrame) -> str:
    hits = frame[frame.test.str.startswith(test_prefix)]
    return hits.verdict.iloc[0] if len(hits) else ""


FIG_VERDICT = {
    "f3_growth_distribution_sim_vs_edgar": fig_verdict("E1", env_sc),
    "f4_growth_deceleration": fig_verdict("E3", env_sc),
    "f5_policy_baselines": fig_verdict("A2", agent_sc),
    "f6_oracle_paired_gain": fig_verdict("A3r", agent_sc),
    "f7_post_shock_r40_recovery": fig_verdict("A8", agent_sc),
    "f8_backtest_retrodiction": fig_verdict("C1 hold-arm", agent_sc),
    "f9_robustness_grid": fig_verdict("A7", agent_sc),
    "f10_memory_ablation": fig_verdict("A3 v3 vs", agent_sc),
    "f11_action_ladders": fig_verdict("A1", agent_sc),
}

BADGE_CLASS = {"PASS": "pass", "PARTIAL": "partial", "FAIL": "fail"}


def badge(v: str) -> str:
    if not v:
        return ""
    return f'<span class="badge {BADGE_CLASS.get(v, "other")}">{esc(v)}</span>'


def figure_block(name: str, simulator_internal: bool = False) -> str:
    m = FIG_META[name]
    v = FIG_VERDICT.get(name, "")
    tag = ' <span class="siminternal">simulator-internal</span>' if simulator_internal else ""
    caveat = (f' <span class="disclaimer">Caveat: {md_inline(m["caveat"])}</span>'
              if m["caveat"] else "")
    return f"""
<figure class="figwrap" id="{name}">
  <div class="figtitle">{md_inline(m['claim'])} {badge(v)}{tag}</div>
  <div class="figsvg">{inline_svg(name)}</div>
  <figcaption>n = {md_inline(m['n'])} &middot; source: {md_inline(m['source'])}.{caveat}</figcaption>
</figure>"""


def scorecard_table(df: pd.DataFrame, table_id: str, cols: list[str],
                    interp_col: str) -> str:
    rows = []
    for _, r in df.iterrows():
        cells = "".join(f"<td>{md_inline(str(r[c]))}</td>" for c in cols)
        rows.append(
            f'<tr class="scrow" data-verdict="{esc(str(r.verdict))}">'
            f"{cells}<td>{badge(str(r.verdict))}</td></tr>"
            f'<tr class="detail"><td colspan="{len(cols)+1}">'
            f"{md_inline(str(r[interp_col]))}</td></tr>")
    head = "".join(f"<th>{esc(c)}</th>" for c in cols) + "<th>verdict</th>"
    return f"""
<div class="tablewrap">
  <label class="filterbox"><input type="checkbox" class="failfilter" data-table="{table_id}">
    show only FAIL / PARTIAL</label>
  <table id="{table_id}" class="scorecard">
    <thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody>
  </table>
  <p class="hint">click a row to expand its interpretation</p>
</div>"""


# ---------------------------------------------------------------- page
CSS = """
:root { --ink:#1a202c; --mut:#4a5568; --line:#d9dee6; --accent:#1f77b4; --bg:#ffffff; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:17px/1.55 "Segoe UI", system-ui, sans-serif; }
main { max-width:1150px; margin:0 auto; padding:1rem 2rem 4rem 15rem; }
nav#sidenav { position:fixed; left:0; top:0; bottom:0; width:13.5rem; padding:1rem .9rem;
  border-right:1px solid var(--line); background:#f7f9fb; overflow-y:auto; font-size:.85rem; }
nav#sidenav a { display:block; color:var(--mut); text-decoration:none; padding:.28rem .4rem;
  border-radius:4px; }
nav#sidenav a.current { background:#e3edf7; color:var(--ink); font-weight:600; }
section.slide, section.appendix { padding:2.2rem 0 1rem; border-bottom:1px solid var(--line); }
h1 { font-size:1.7rem; margin:.2rem 0; } h2 { font-size:1.35rem; margin:.4rem 0 .8rem; }
h3 { font-size:1.05rem; } p, li { max-width:62rem; }
.kicker { color:var(--mut); font-size:.85rem; }
.grades { display:flex; gap:1rem; margin:.8rem 0; }
.grade { border:2px solid var(--line); border-radius:10px; padding:.6rem 1.1rem; }
.grade b { font-size:2rem; display:block; }
.grade.gB b { color:#2f855a; } .grade.gC b { color:#b7791f; }
.grade.gA b { color:#2f855a; } .grade.gD b, .grade.gE b { color:#c53030; }
.badge { display:inline-block; padding:.05rem .5rem; border-radius:9px; font-size:.75rem;
  font-weight:700; vertical-align:middle; }
.badge.pass { background:#def7e5; color:#22543d; } .badge.partial { background:#fef3d8; color:#7b5c11; }
.badge.fail { background:#fde3e3; color:#822727; } .badge.other { background:#e7ebf0; color:#4a5568; }
.siminternal { font-size:.72rem; background:#e8eef7; color:#2c5282; border-radius:9px;
  padding:.05rem .5rem; font-weight:600; }
.figwrap { margin:1.4rem 0; width:min(100%, 78vw); cursor:zoom-in; }
.figtitle { font-weight:650; margin-bottom:.3rem; }
figcaption { font-size:.83rem; color:var(--mut); margin-top:.25rem; }
table { border-collapse:collapse; margin:.7rem 0; font-size:.86rem; }
th, td { border:1px solid var(--line); padding:.32rem .55rem; text-align:left; vertical-align:top; }
th { background:#eef2f6; }
tr.detail { display:none; } tr.detail.open { display:table-row; }
tr.detail td { background:#f7f9fb; color:var(--mut); }
tr.scrow { cursor:pointer; }
.hint { font-size:.75rem; color:var(--mut); }
.filterbox { font-size:.82rem; color:var(--mut); }
.callout { border:1px solid var(--line); border-left:5px solid var(--accent);
  background:#f4f8fc; padding:.7rem 1rem; margin:1rem 0; max-width:62rem; }
.donotclaim { border:2px solid #c53030; border-radius:8px; background:#fff7f7;
  padding:.8rem 1.1rem; margin:1.2rem 0; max-width:62rem; }
.donotclaim h3 { color:#822727; margin:0 0 .4rem; }
.bigclaims p { font-size:1.28rem; line-height:1.6; max-width:60rem; }
pre { background:#f4f6f8; border:1px solid var(--line); border-radius:6px;
  padding:.8rem 1rem; overflow-x:auto; font-size:.78rem; }
details { margin:.4rem 0; max-width:62rem; } summary { cursor:pointer; font-weight:600; }
#lightbox { display:none; position:fixed; inset:0; background:rgba(15,20,28,.93); z-index:60;
  padding:2vh 3vw; cursor:zoom-out; }
#lightbox.open { display:flex; align-items:center; justify-content:center; }
#lightbox svg { max-width:94vw; max-height:94vh; width:auto; height:auto; background:#fff;
  border-radius:6px; }
#counter { display:none; position:fixed; right:1.1rem; bottom:.9rem; font-size:1rem;
  color:var(--mut); z-index:55; }
body.present { font-size:22px; }
body.present nav#sidenav { display:none; }
body.present main { padding:2.5rem 3.5rem; max-width:none; }
body.present section { display:none; border:none; }
body.present section.active { display:block; min-height:96vh; }
body.present .figwrap { width:min(100%, 88vw); }
body.present figcaption { font-size:.95rem; }
body.present #counter { display:block; }
body.present section.appendix { display:none !important; }
@media print {
  nav#sidenav, #counter, .filterbox, .hint { display:none !important; }
  main { padding:0 .5in; max-width:none; }
  section.slide, section.appendix { page-break-after:always; border:none; }
  .figwrap { break-inside:avoid; width:100%; }
  .figsvg svg { max-height:7.2in; }
  tr.detail { display:table-row; }
  body { font-size:11pt; }
}
"""

JS = """
(function () {
  var slides = Array.prototype.slice.call(document.querySelectorAll('section.slide'));
  var cur = 0, presenting = false;
  var counter = document.getElementById('counter');
  function show(i) {
    cur = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach(function (s, k) { s.classList.toggle('active', k === cur); });
    counter.textContent = (cur + 1) + ' / ' + slides.length;
  }
  function setPresent(on) {
    presenting = on;
    document.body.classList.toggle('present', on);
    if (on) show(cur); else slides.forEach(function (s) { s.classList.remove('active'); });
  }
  document.addEventListener('keydown', function (e) {
    if (e.key === 'p' || e.key === 'P') { setPresent(!presenting); return; }
    if (!presenting) return;
    if (e.key === 'Escape') setPresent(false);
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') { e.preventDefault(); show(cur + 1); }
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); show(cur - 1); }
  });
  // lightbox
  var lb = document.getElementById('lightbox');
  document.querySelectorAll('.figwrap .figsvg').forEach(function (f) {
    f.parentElement.addEventListener('click', function () {
      lb.innerHTML = f.innerHTML; lb.classList.add('open');
    });
  });
  lb.addEventListener('click', function () { lb.classList.remove('open'); });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') lb.classList.remove('open');
  });
  // scorecard row expansion
  document.querySelectorAll('tr.scrow').forEach(function (r) {
    r.addEventListener('click', function () {
      r.nextElementSibling.classList.toggle('open');
    });
  });
  // FAIL/PARTIAL filter
  document.querySelectorAll('.failfilter').forEach(function (cb) {
    cb.addEventListener('change', function () {
      var t = document.getElementById(cb.getAttribute('data-table'));
      t.querySelectorAll('tr.scrow').forEach(function (r) {
        var keep = !cb.checked || r.dataset.verdict === 'FAIL' || r.dataset.verdict === 'PARTIAL';
        r.style.display = keep ? '' : 'none';
        if (!keep) r.nextElementSibling.classList.remove('open');
      });
    });
  });
  // nav highlight
  var links = document.querySelectorAll('#sidenav a');
  var obs = new IntersectionObserver(function (es) {
    es.forEach(function (en) {
      if (en.isIntersecting) {
        links.forEach(function (l) {
          l.classList.toggle('current', l.getAttribute('href') === '#' + en.target.id);
        });
      }
    });
  }, { rootMargin: '-40% 0px -55% 0px' });
  document.querySelectorAll('section[id]').forEach(function (s) { obs.observe(s); });
})();
"""

NAV_ITEMS = [
    ("s0", "0 · Header & verdicts"), ("s1", "1 · Problem & system"),
    ("s2", "2 · The dataset"), ("s3", "3 · Is the simulator realistic?"),
    ("s4", "4 · Do the agents work?"), ("s5", "5 · What does not work"),
    ("s6", "6 · Defensible claims"), ("s7", "7 · Next step"),
    ("appendix", "Appendix"),
]
nav_html = "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in NAV_ITEMS)

qa_html = "".join(
    f"<details><summary>{md_inline(q.strip())}</summary><p>{md_inline(a.strip())}</p></details>"
    for q, a in qa)

retracted_html = "".join(
    f"<li>{md_inline(r.claim)} — <strong>{esc(r.status)}</strong></li>"
    for _, r in retracted.iterrows())

env_cols = ["test", "dimension", "policy_arm", "edgar_n", "sim_n"]
agent_sc_sorted = agent_sc.assign(_tid=agent_sc.test.str.extract(r"^(A\d+r?|C\d+)")[0]) \
                          .sort_values(["_tid"], kind="stable")
agent_cols = ["test", "dimension", "baseline", "n", "effect"]

page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Validation review — startup simulator &amp; agents</title>
<style>{CSS}</style></head>
<body>
<nav id="sidenav"><div class="kicker">{esc(PROJECT_LABEL)}</div>{nav_html}
<div class="kicker" style="margin-top:1rem">press <b>P</b> for presenter mode</div></nav>
<div id="lightbox"></div><div id="counter"></div>
<main>

<section class="slide" id="s0">
<h1>Simulating startup trajectories &amp; multi-agent decision-making — validation review</h1>
<p class="kicker">{esc(PROJECT_LABEL)} &middot; {build_date} &middot; branch
<code>{esc(git_branch)}</code> @ <code>{esc(git_hash)}</code></p>
<p><strong>All numbers VERIFIED/EXECUTED from data on disk; nothing synthetic.</strong></p>
<div class="grades">
  <div class="grade g{sim_grade}"><b>{sim_grade}</b>Simulator — {esc(sim_phrase.lower())}</div>
  <div class="grade g{ag_grade}"><b>{ag_grade}</b>Agents — {esc(ag_phrase.lower())}</div>
</div>
</section>

<section class="slide" id="s1">
<h2>1 · Problem and system</h2>
<p><strong>The simulator</strong> is a monthly SaaS-startup environment: revenue, cash,
churn, pricing, macro shocks; an episode ends at bankruptcy or the time limit.</p>
<p><strong>The agent layer</strong> is a boardroom of rule-based C-suite proposals,
steered by an LLM "oracle brief" through a fixed ActionModifier, with episodic memory
retrieved from past runs.</p>
<p><strong>The two claims defended</strong> (validation_plan.md §1):</p>
<ul>
<li><strong>(S)</strong> {md_inline(claim_S)}.</li>
<li><strong>(P)</strong> {md_inline(claim_P)}.</li>
</ul>
<details><summary>Pipeline (existing text diagram from system_audit.md)</summary>
<div class="sourcequote"><pre>{esc(pipeline)}</pre></div></details>
</section>

<section class="slide" id="s2">
<h2>2 · The dataset</h2>
<p class="callout"><strong>EDGAR is the calibration and validation benchmark;
all agent experience data is simulator-generated.</strong></p>
{figure_block("f1_panel_trajectories")}
{figure_block("f2_edgar_benchmark_bands")}
<details open><summary>Dataset card (data/DATASET_CARD.md, frozen)</summary>
{md_to_html(card_md)}
</details>
</section>

<section class="slide" id="s3">
<h2>3 · Is the simulator realistic?</h2>
<p class="callout"><strong>Verdict framing: a controlled comparative testbed,
not a forecasting model.</strong></p>
{figure_block("f3_growth_distribution_sim_vs_edgar")}
{figure_block("f4_growth_deceleration")}
<h3>Environment scorecard (validation/results/environment_scorecard.csv)</h3>
{scorecard_table(env_sc, "envtable", env_cols, "result")}
</section>

<section class="slide" id="s4">
<h2>4 · Do the agents work?</h2>
<p class="callout">Every agent-value statement on this page is
<span class="siminternal">simulator-internal</span> — a model-based counterfactual,
not a real-world effect.</p>
{figure_block("f5_policy_baselines", simulator_internal=True)}
{figure_block("f6_oracle_paired_gain", simulator_internal=True)}
{figure_block("f7_post_shock_r40_recovery", simulator_internal=True)}
<h3>Agent scorecard (validation/results/agent_scorecard.csv)</h3>
{scorecard_table(agent_sc_sorted, "agenttable", agent_cols, "interpretation")}
</section>

<section class="slide" id="s5">
<h2>5 · What does not work, and why</h2>
{figure_block("f8_backtest_retrodiction")}
<h3>Revenue drawdowns: simulator vs reality (report §5b, E6 — exploratory)</h3>
{md_to_html(e6_table)}
<h3>Memory ablation <span class="siminternal">simulator-internal</span></h3>
<p>Episodic retrieval adds <strong>+${mem_mean}k</strong> final MRR
(95% CI [+${mem_lo}k, +${mem_hi}k], p={mem_p}) — <strong>{mem_share}%</strong> of the
oracle layer's gain over the boardroom. The brief mechanism, not memory, carries the value.</p>
{figure_block("f10_memory_ablation", simulator_internal=True)}
<h3>Observational check against real companies</h3>
<p>{md_inline(c2_par)}</p>
<div class="callout"><strong>Diagnosis (from the C1 scorecard row):</strong>
{md_inline(diagnosis)}</div>
<div class="donotclaim"><h3>Do not claim</h3><p>{md_inline(do_not_claim)}</p></div>
</section>

<section class="slide" id="s6">
<h2>6 · Defensible claims (report §8, verbatim)</h2>
<div class="bigclaims">
<p><strong>Simulator.</strong> {md_inline(sim_claim)}</p>
<p><strong>Agents.</strong> {md_inline(ag_claim)}</p>
</div>
</section>

<section class="slide" id="s7">
<h2>7 · Next step — autonomous agent workflow</h2>
<p><code>run_simulation</code> already runs 120-month episodes unattended end-to-end
(environment, boardroom, oracle briefs, memory writes) with no human in the loop.</p>
<p>Post-review work: expose an auto-run mode in the product frontend — launch a policy,
stream the monthly trace live, with an optional human-in-the-loop override on the
composed action before each step.</p>
</section>

<section class="appendix" id="appendix">
<h2>Appendix</h2>
<h3>The eleven questions (report §7)</h3>
{qa_html}
<h3>Claim audit (validation/results/claim_audit.csv)</h3>
<p>{n_repro} claims REPRODUCED exactly; {n_other} not carried forward:</p>
<div class="disclaimer"><ul>{retracted_html}</ul></div>
<h3>Additional figures</h3>
{figure_block("f9_robustness_grid")}
{figure_block("f11_action_ladders")}
<h3>Reproduction (validation/README.md)</h3>
<pre>{esc(repro)}</pre>
</section>
</main>
<script>{JS}</script>
</body></html>
"""

# ---------------------------------------------------------------- assertions
def visible_text(html: str) -> str:
    t = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    t = re.sub(r"<script.*?</script>", " ", t, flags=re.S)
    t = re.sub(r"<svg.*?</svg>", " ", t, flags=re.S)
    return t


def check_provenance(html: str) -> list[str]:
    text = re.sub(r"<[^>]+>", " ", visible_text(html))
    text = html_mod.unescape(text)
    orphans = []
    for tok in NUM_RE.findall(text):
        if _norm(tok) not in ALLOWED:
            orphans.append(tok)
    return sorted(set(orphans))


def check_banned(html: str) -> list[str]:
    t = visible_text(html)
    # remove the explicitly allowed blocks: the Do-not-claim box, disclaimer
    # spans (figure caveats, retracted-claim list) and verbatim source quotes
    t = re.sub(r'<div class="donotclaim">.*?</div>', " ", t, flags=re.S)
    t = re.sub(r'<div class="(?:disclaimer|sourcequote)">.*?</div>', " ", t, flags=re.S)
    t = re.sub(r'<span class="disclaimer">.*?</span>', " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    bad = []
    for pat, label in [(r"oracle_v4", "oracle_v4"), (r"\breward\b", "reward"),
                       (r"screenshot", "screenshot")]:
        if re.search(pat, t, re.I):
            bad.append(label)
    return bad


orphans = check_provenance(page)
banned = check_banned(page)
if orphans:
    raise SystemExit(f"BUILD FAIL - numeric tokens without a data-source origin: {orphans}")
if banned:
    raise SystemExit(f"BUILD FAIL - banned claim tokens outside allowed boxes: {banned}")

(OUT / "index.html").write_text(page, encoding="utf-8")
size_mb = (OUT / "index.html").stat().st_size / 1e6
if size_mb >= 15:
    raise SystemExit(f"BUILD FAIL - page is {size_mb:.1f} MB (limit 15)")

(OUT / "README.md").write_text(f"""# Review presentation site

**Open:** double-click `index.html` - a single self-contained file (no server,
no internet, no build step at view time). Size {size_mb:.1f} MB.

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
"{PROJECT_LABEL}" label), that banned claim tokens (oracle_v4 / reward /
screenshot) appear only inside the "Do not claim" box, disclaimer-marked text
(figure caveats, the retracted-claims list) or verbatim source quotes (the
pipeline diagram), and that the file is under 15 MB.
""", encoding="utf-8")

print(f"OK  index.html written ({size_mb:.2f} MB)")
print("OK  numeric-provenance assertion passed (0 orphan tokens)")
print("OK  banned-claims assertion passed")
