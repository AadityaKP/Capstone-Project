// Shared founder-facing components. Layout/visual grammar reuses the existing
// styles.css system; new pieces (risk chips, provenance chips, focus bar,
// evidence cards, timeline, staged progress) extend it per spec §19.

import React, { useState } from "react";
import {
  AlertTriangle, ArrowDownRight, ArrowUpRight, CheckCircle2, ChevronDown,
  ChevronRight, Circle, FlaskConical, Info, Minus, ShieldAlert, Sparkles
} from "lucide-react";
// (CheckCircle2 and Circle remain for ProgressStages.)
import {
  money, moneyExact, pct, signedPct, signedPp, pctOfMrr, monthsLabel, deriveCac
} from "./derive.js";
import {
  RISK, OUTCOME, FOCUS_LABELS, CHANNEL_COPY, DOMAIN_META, CAUSAL_STRESS,
  refreshReasonCopy, briefSourceCopy, scaleWord,
  guardBullets, rewriteMemory, SIMULATED_PREFIX, causalEvidenceCopy
} from "./copy.js";
import { confidenceSentence } from "./founderView.js";
import {
  expectedLine, predictionSentences, scoreLine, briefFreshness, actionSummary
} from "./loopView.js";

// ---- small primitives ----

export function RiskChip({ level, large = false }) {
  const meta = RISK[level] || RISK.MEDIUM;
  return (
    <span className={`risk-chip ${meta.tone} ${large ? "large" : ""}`}>
      {meta.tone === "amber" || meta.tone === "red" ? <AlertTriangle size={large ? 16 : 13} /> : <ShieldAlert size={large ? 16 : 13} />}
      {meta.label}
    </span>
  );
}

export function ProvChip({ kind, date }) {
  const copy = {
    provided: date ? `You provided · ${date}` : "You provided",
    estimated: "Estimated by the system",
    derived: "Derived",
    simulated: "Simulated"
  };
  return <span className={`prov-chip ${kind}`}>{copy[kind] || kind}</span>;
}

export function DeltaArrow({ value, goodWhenDown = false, format = signedPct }) {
  if (value == null || Number.isNaN(value) || Math.abs(value) < 0.05) {
    return <em className="delta-inline flat"><Minus size={13} /> flat</em>;
  }
  const up = value > 0;
  const good = goodWhenDown ? !up : up;
  return (
    <em className={`delta-inline ${good ? "good" : "bad"}`}>
      {up ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />} {format(value)}
    </em>
  );
}

export function Banner({ tone = "info", icon = null, children, actions = null }) {
  return (
    <div className={`banner ${tone}`}>
      {icon || <Info size={17} />}
      <div className="banner-body">{children}</div>
      {actions && <div className="banner-actions">{actions}</div>}
    </div>
  );
}

export function SimulatedTag() {
  return (
    <span className="sim-tag"><FlaskConical size={12} /> {SIMULATED_PREFIX}</span>
  );
}

export function DemoBadge() {
  return <span className="demo-badge">Sample company — data is illustrative</span>;
}

// One collapsed section: a panel whose body renders only once opened.
export function Expandable({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <article className={`panel expandable ${open ? "open" : ""}`}>
      <button className="expand-head" type="button" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={17} /> : <ChevronRight size={17} />}
        <h3>{title}</h3>
      </button>
      {open && <div className="expand-body">{children}</div>}
    </article>
  );
}

// ---- KPI card (§9) ----

export function KpiCard({ label, value, sub, delta, band = null, hint = null }) {
  return (
    <article className={`kpi-card founder ${band || ""}`} title={hint || undefined}>
      <span className="kpi-label">{label}</span>
      <strong className="kpi-value">{value}</strong>
      <span className="kpi-sub">{delta != null ? delta : (sub || " ")}</span>
    </article>
  );
}

// ---- plan assembly (§10.2) ----
// Turns the boardroom's final_action + trace into the four founder cards.

export function buildPlanCards(analysis, month) {
  if (!analysis?.trace?.final_action) return [];
  const t = analysis.trace;
  const fa = t.final_action;
  const mrr = month?.values?.mrr || null;
  const weights = t.applied_weights || {};
  const churn = month?.values?.churnMonthly;

  const topWeightKey = Object.keys(weights).sort((a, b) => weights[b] - weights[a])[0];
  const starDomain = topWeightKey === "innovation" ? "product" : topWeightKey === "growth" ? "marketing" : null;

  const narratives = analysis.narratives || {};

  const cards = [];

  // Product / R&D
  {
    const amount = fa.product?.r_and_d_spend ?? 0;
    const word = scaleWord(t.rd_spend_change_pct);
    cards.push({
      domain: "product",
      title: DOMAIN_META.product.title,
      headline: amount > 0 ? `Invest ≈${money(amount)} in product this month` : "Hold product spend",
      amount, share: pctOfMrr(amount, mrr),
      rationale: narratives.CPO || (
        churn != null && churn >= 4
          ? `Your churn (${pct(churn)}/mo) is the board's top concern — retention improves when product investment rises.`
          : "Steady product investment keeps retention compounding."
      ),
      chain: [
        "Base suggestion from your churn level (the product advisor's rule of thumb)",
        word ? `Strategic adjustment: product spend ${word} given the board's read of your risk` : null,
        "Kept above the board's minimum product investment"
      ].filter(Boolean),
      isAction: amount > 0,
      starred: starDomain === "product"
    });
  }

  // Marketing
  {
    const amount = fa.marketing?.spend ?? 0;
    const channel = CHANNEL_COPY[fa.marketing?.channel] || CHANNEL_COPY.ppc;
    const reported = month?.values?.marketingSpend;
    const dirWord = reported != null && Math.abs(amount - reported) / Math.max(reported, 1) > 0.1
      ? (amount > reported ? "up from" : "down from")
      : null;
    const word = scaleWord(t.marketing_spend_change_pct);
    cards.push({
      domain: "marketing",
      title: DOMAIN_META.marketing.title,
      headline: `Spend ≈${money(amount)} on ${channel}`,
      amount, share: pctOfMrr(amount, mrr),
      sub: dirWord && reported != null ? `${dirWord} the ≈${money(reported)} you reported` : null,
      rationale: narratives.CMO || (
        word && word.startsWith("scaled back")
          ? "Acquisition is dialled down while risk is elevated — cash discipline, not channel failure."
          : "Acquisition spend sized to your growth efficiency."
      ),
      chain: [
        "Base suggestion from your growth efficiency (LTV vs. acquisition cost)",
        word ? `Strategic adjustment: marketing ${word} given the board's read of your risk` : null,
        "Kept above the board's minimum presence spend"
      ].filter(Boolean),
      isAction: amount > 0,
      starred: starDomain === "marketing"
    });
  }

  // Hiring — engine hires are salary slots (business_logic.SALARY_SLOT_USD),
  // translated to a monthly payroll figure per §5.4.
  {
    const hires = fa.hiring?.hires ?? 0;
    const cappedByRisk = (t.pre_modifier_action?.hiring?.hires ?? 0) > 0 && hires === 0;
    cards.push({
      domain: "hiring",
      title: DOMAIN_META.hiring.title,
      headline: hires > 0 ? `Room to add ≈${money(hires * 8000)}/mo of payroll` : "Wait on hiring",
      amount: null, share: null,
      sub: hires > 0 ? `roughly ${hires === 1 ? "one hire" : `${hires} hires`} at typical salaries` : null,
      rationale: narratives.CFO || (
        hires > 0
          ? "Runway is long enough to grow the team."
          : cappedByRisk
            ? "Holding hiring while risk is elevated."
            : "Revisit when runway comfortably exceeds two years."
      ),
      chain: [
        "The finance advisor gates hiring on runway and growth efficiency",
        cappedByRisk ? "Strategic adjustment: hiring paused at the board's risk level" : null
      ].filter(Boolean),
      isAction: hires > 0,
      starred: false
    });
  }

  // Pricing — effectively hold vs. consider ≈+5% (§3 J9).
  {
    const change = fa.pricing?.price_change_pct ?? 0;
    cards.push({
      domain: "pricing",
      title: DOMAIN_META.pricing.title,
      headline: change > 0.001 ? `Consider a ≈${Math.round(change * 100)}% price increase` : "Hold pricing",
      amount: null, share: null,
      rationale: narratives.CFO_PRICING || (
        change > 0.001
          ? "Your growth efficiency is below the healthy line — a modest increase can restore it."
          : "No pricing pressure this month; stability signals value to customers."
      ),
      chain: [
        change > 0.001
          ? "The finance advisor suggests ≈+5% only when lifetime value is under 3× acquisition cost"
          : "Pricing holds unless growth efficiency drops below the healthy line"
      ],
      isAction: change > 0.001,
      starred: false
    });
  }

  return cards;
}

// The accept toggle that used to sit here is gone: the founder says what they
// did on the Close form, once, and that answer is the decision History reads.
// A pre-commit toggle produced a second decision per domain and inflated the
// "accepted n of m" denominator.
export function PlanCard({ card, compact = false }) {
  const [open, setOpen] = useState(false);
  return (
    <article className={`plan-card ${card.starred ? "starred" : ""} ${compact ? "compact" : ""}`}>
      <div className="plan-head">
        <span className="plan-domain">{card.title}</span>
        {card.starred && <span className="priority-pill">Priority</span>}
      </div>
      <strong className="plan-headline">{card.headline}</strong>
      {card.sub && <span className="plan-meta">{card.sub}</span>}
      {!compact && <p className="plan-rationale">{card.rationale}</p>}
      {!compact && (
        <div className="plan-actions">
          <button className="link-button" type="button" onClick={() => setOpen(!open)}>
            {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />} Why this number?
          </button>
        </div>
      )}
      {!compact && open && (
        <ol className="plan-chain">
          {card.share != null && <li>≈{Math.round(card.share)}% of your monthly revenue</li>}
          {card.chain.map((step) => <li key={step}>{step}</li>)}
        </ol>
      )}
    </article>
  );
}

// ---- focus mix bar (§10.1 L3) ----

export function FocusBar({ weights }) {
  if (!weights) return null;
  const order = ["innovation", "growth", "efficiency", "macro"];
  return (
    <div className="focus-bar-wrap">
      <span className="focus-caption">The board's focus this month</span>
      <div className="focus-bar" role="img" aria-label="Board focus mix">
        {order.map((k) => (
          <span key={k} className={`focus-seg ${k}`} style={{ flexGrow: Math.max(weights[k] || 0, 0.02) }}>
            {(weights[k] || 0) > 0.14 ? FOCUS_LABELS[k] : ""}
          </span>
        ))}
      </div>
    </div>
  );
}

// ---- evidence (§10.1 L4, §15.3) ----

export function EvidenceList({ analysis }) {
  const memories = analysis?.trace?.retrieved_memories || [];
  const graphLines = causalEvidenceCopy(analysis?.trace?.graph_summary);
  if (!memories.length && !graphLines.length) {
    return <p className="empty-copy">No similar simulated situations yet — evidence builds up as analyses run.</p>;
  }
  return (
    <div className="evidence-list">
      <SimulatedTag />
      {memories.slice(0, 3).map((m, i) => {
        const { sentence, outcome } = rewriteMemory(m);
        return (
          <div className="evidence-card" key={i}>
            <span className={`outcome-dot ${outcome.tone}`}>{outcome.glyph}</span>
            <p>{sentence}</p>
          </div>
        );
      })}
      {graphLines.map((line) => (
        <div className={`evidence-card ${line.kind}`} key={line.kind}>
          <span className={`outcome-dot ${line.kind === "observed" ? "blue" : "grey"}`}>
            {line.kind === "observed" ? "◆" : "◇"}
          </span>
          <p>{line.text}</p>
        </div>
      ))}
    </div>
  );
}

// ---- confidence & freshness strip (§10.1 L6) ----

// Confidence and the assumption count used to be two independent facts printed
// side by side, so the strip could read "High confidence · 6 estimated inputs".
// They are now one sentence in which the count caps the band, computed by
// founder_view on the server; confidenceSentence is the fallback for analyses
// stored before display blocks existed.
// How many of the analysis's inputs were guessed. The server reports what it
// actually assumed (trace.assumed_fields); the client-side count is only the
// fallback for analyses stored before that field existed.
export function estimatedInputCount(analysis, month, company) {
  const assumed = analysis?.trace?.assumed_fields;
  if (assumed) return assumed.length;
  const cac = month?.values ? deriveCac(month.values) : { source: "estimated" };
  return (cac.source === "estimated" ? 1 : 0) + (company?.maturity ? 0 : 1) + 1;
}

// The one confidence sentence, verbatim from the server's display block (the
// assumption count caps the band there); never shortened or recomposed here.
export function confidenceLine(analysis, month, company) {
  if (!analysis) return null;
  return analysis.display?.confidence?.sentence
    || analysis.trace?.display?.confidence?.sentence
    || confidenceSentence(analysis.brief?.confidence, estimatedInputCount(analysis, month, company));
}

export function ConfidenceStrip({ analysis, month, company = null }) {
  if (!analysis) return null;
  const sentence = confidenceLine(analysis, month, company);
  const reason = refreshReasonCopy(analysis.trace?.refresh_reason || analysis.reason);
  const reuse = briefSourceCopy(analysis.trace?.brief_source);
  return (
    <div className="confidence-strip">
      <span>{sentence}</span>
      <span className="dot-sep">·</span>
      <span>{reason}</span>
      {reuse && (<><span className="dot-sep">·</span><span>{reuse}</span></>)}
      {month && (<><span className="dot-sep">·</span><span>numbers from {new Date(month.enteredAt).toLocaleDateString("en-US", { month: "short", day: "numeric" })}</span></>)}
    </div>
  );
}

// ---- guarded LLM bullets (§15.4) ----

export function RiskBullets({ brief, knownNumbers }) {
  const risks = guardBullets(brief?.key_risks, knownNumbers);
  const opps = guardBullets(brief?.key_opportunities, knownNumbers);
  if (!risks.length && !opps.length) return null;
  return (
    <div className="bullets-grid">
      {risks.length > 0 && (
        <div>
          <span className="bullets-title">Watch-outs</span>
          <ul>{risks.map((r) => <li key={r}>{r}</li>)}</ul>
        </div>
      )}
      {opps.length > 0 && (
        <div>
          <span className="bullets-title">Working in your favor</span>
          <ul>{opps.map((o) => <li key={o}>{o}</li>)}</ul>
        </div>
      )}
    </div>
  );
}

// ---- staged progress (§7 O5 / §17.6) ----

export function ProgressStages({ stage, narrativesOn = false }) {
  const stages = [
    "Reading your numbers",
    "Your advisory board is deliberating (≈ half a minute)",
    narrativesOn ? "Your advisors are writing their reasoning (~1 min)" : "Writing up recommendations"
  ];
  return (
    <ol className="stage-list">
      {stages.map((label, i) => (
        <li key={label} className={i < stage ? "done" : i === stage ? "active" : "pending"}>
          {i < stage ? <CheckCircle2 size={17} /> : i === stage ? <Sparkles size={17} className="spin-slow" /> : <Circle size={17} />}
          <span>{label}</span>
        </li>
      ))}
    </ol>
  );
}

// ---- mini line chart (History, §18) ----

export function MiniLine({ points, label, goodWhenDown = false, format = (v) => v }) {
  if (!points || points.length < 3) return null;
  const w = 180, h = 44, pad = 4;
  const min = Math.min(...points), max = Math.max(...points);
  const span = max - min || 1;
  const coords = points.map((p, i) => {
    const x = pad + (i / (points.length - 1)) * (w - pad * 2);
    const y = h - pad - ((p - min) / span) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const rising = points[points.length - 1] >= points[0];
  const good = goodWhenDown ? !rising : rising;
  return (
    <div className="mini-line">
      <span className="mini-line-label">{label}</span>
      <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${label} trend`}>
        <polyline points={coords.join(" ")} className={good ? "good" : "bad"} />
        <circle
          cx={coords[coords.length - 1].split(",")[0]}
          cy={coords[coords.length - 1].split(",")[1]}
          r="2.6" className={good ? "good" : "bad"}
        />
      </svg>
      <span className="mini-line-value">{format(points[points.length - 1])}</span>
    </div>
  );
}

// ---- the OEFA strip (plan section 5.1) ----
//
// Observed / Decided / Expected / Changed, one component used everywhere a
// month appears: each Plan column, the timeline, and retroactively the Advice
// detail page. It is the Module 6 explainability payload in a form that
// screenshots, and it is built from the server's numbers only.

const TREND_WORDS = { INCREASING: "rising", FLAT: "flat", DECREASING: "falling" };

// A single-month analysis (from /api/advise or an older stored one) has the
// Observe/Execute/Expected half of a cycle month and no Feedback: the strip
// still renders, and says the last beat is waiting on the real month.
export function monthFromAnalysis(analysis) {
  if (!analysis?.trace) return null;
  const t = analysis.trace;
  return {
    month_index: 1,
    projection: false,
    observe: {
      memory_count: t.memory_count ?? (t.retrieved_memories || []).length,
      memories: t.retrieved_memories || [],
      trend: null,
      graph: { stress_node: t.causal_stress_node || t.graph_summary?.stress_node || null, enabled: t.graph_store_enabled ?? null }
    },
    execute: {
      action: t.final_action,
      brief: analysis.brief,
      llm_ok: analysis.llm_ok !== false,
      brief_source: t.brief_source,
      refresh_reason: t.refresh_reason,
      expected_delta: t.expected_delta || null,
      proposals: t.proposals || []
    },
    feedback: null,
    adapt: null
  };
}

function Beat({ label, children, tone = "" }) {
  return (
    <div className={`oefa-beat ${tone}`}>
      <span className="oefa-beat-label">{label}</span>
      <ul className="oefa-lines">{children}</ul>
    </div>
  );
}

// The Observed beat's sentences, shared with the Evidence section on Why this
// plan so the two say the same thing about what the board recalled.
export function observedLines(observe) {
  if (!observe) return [];
  const stress = observe.graph?.stress_node ? CAUSAL_STRESS[observe.graph.stress_node] : null;
  const trend = observe.trend?.mrr_trend ? TREND_WORDS[observe.trend.mrr_trend] : null;
  const lines = [
    { key: "memory", text: observe.memory_count ? `${observe.memory_count} similar past month${observe.memory_count === 1 ? "" : "s"} recalled` : "no similar past months yet" }
  ];
  if (trend) lines.push({ key: "trend", text: `revenue trend ${trend}` });
  if (stress) lines.push({ key: "stress", text: `the board's read: ${stress}` });
  if (observe.graph?.enabled === false) lines.push({ key: "graph", text: "causal evidence graph off", muted: true });
  return lines;
}

// Weight moves in the founder's words: the server reports {key, from, to, delta}.
function weightMoveLine(move) {
  const label = FOCUS_LABELS[move.key] || move.key;
  return `${label} focus ${move.delta > 0 ? "up" : "down"} to ${Math.round(move.to * 100)}%`;
}

export function OefaStrip({ month, closed = null, defaultOpen = false, title = null }) {
  const [open, setOpen] = useState(defaultOpen);
  if (!month) return null;
  const { observe, execute, feedback, adapt } = month;
  const fresh = briefFreshness(execute?.brief_source);
  const expected = execute?.expected_delta;
  const adaptations = adapt?.adaptations || [];
  const weightMoves = adapt?.weight_moves || [];
  // The server also narrates weight moves in what_changed; with the moves
  // rendered in their own words those lines would be a second copy.
  const whatChanged = (adapt?.what_changed || []).filter((line) => !(weightMoves.length && / weight /.test(line)));
  // Closed by the founder: the actual numbers replace the simulated ones.
  const result = closed?.result || null;
  const changeError = result ? result.prediction_error : feedback?.prediction_error;
  const changeBefore = observe?.state_before;
  const changeAfter = result ? result.actual_state : feedback?.state_after;
  const changedLabel = result ? "Changed (your numbers)" : changeError ? "Changed (in simulation)" : "Changed";

  return (
    <div className={`oefa-strip ${open ? "open" : ""}`}>
      <button className="oefa-toggle" type="button" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span>{title ? `${title} · ` : ""}Observed · Decided · Expected · Changed</span>
      </button>
      {open && (
        <div className="oefa-body">
          <Beat label="Observed">
            {observedLines(observe).map((l) => <li key={l.key} className={l.muted ? "muted" : ""}>{l.text}</li>)}
          </Beat>
          <Beat label="Decided">
            {actionSummary(execute?.action).map((line) => <li key={line}>{line}</li>)}
            <li className="muted">{refreshReasonCopy(execute?.refresh_reason)} · brief {fresh.label}</li>
            {execute?.llm_ok === false && <li className="muted">strategist unreachable — built-in rules</li>}
          </Beat>
          <Beat label="Expected">
            {expected ? <li>{expectedLine(expected)}</li> : <li className="muted">no numeric prediction on this analysis</li>}
            {expected && <li className="muted"><SimulatedTag /></li>}
          </Beat>
          <Beat label={changedLabel} tone={result ? "actual" : ""}>
            {changeError && !result && (
              <li className="muted">model consistency check, not accuracy</li>
            )}
            {changeError ? (
              <>
                {predictionSentences({ before: changeBefore, actual: changeAfter, expected, error: changeError, basis: result ? "actual" : "simulated" })
                  .map((s) => <li key={s.key} className={`pe-line ${s.tone}`}>{s.text}</li>)}
                <li className="muted">{scoreLine(changeError)}</li>
              </>
            ) : (
              <li className="muted">waiting on your real numbers — close the month to score this plan</li>
            )}
            {whatChanged.slice(0, 3).map((line) => <li key={line}>{line}</li>)}
            {weightMoves.map((m) => <li key={m.key}>{weightMoveLine(m)}</li>)}
            {adaptations.map((a) => <li key={a.agent}><strong>{a.agent}:</strong> {a.sentence}</li>)}
            {feedback && !result && (
              <li className="muted">
                {feedback.evidence_written
                  ? "this simulated month was written back as simulated evidence, kept apart from anything real"
                  : "not written back as evidence (causal graph off)"}
              </li>
            )}
            {result && (
              <li className="muted">{result.evidence?.reason}</li>
            )}
          </Beat>
        </div>
      )}
    </div>
  );
}

// ---- outcome badge (History maturation, §14) ----

export function OutcomeBadge({ outcome }) {
  const meta = OUTCOME[outcome];
  if (!meta) return null;
  return (
    <span className={`outcome-badge ${meta.tone}`}>
      {meta.glyph} 6 months later: {meta.label} <em>(what happened next, not credit)</em>
    </span>
  );
}
