// Shared founder-facing components. Layout/visual grammar reuses the existing
// styles.css system; new pieces (risk chips, provenance chips, focus bar,
// evidence cards, timeline, staged progress) extend it per spec §19.

import React, { useState } from "react";
import {
  AlertTriangle, ArrowDownRight, ArrowUpRight, CheckCircle2, ChevronDown,
  ChevronRight, Circle, FlaskConical, Info, Minus, ShieldAlert, Sparkles
} from "lucide-react";
import {
  money, moneyExact, pct, signedPct, signedPp, pctOfMrr, monthsLabel
} from "./derive.js";
import {
  RISK, OUTCOME, FOCUS_LABELS, CHANNEL_COPY, DOMAIN_META,
  confidenceBand, refreshReasonCopy, briefSourceCopy, scaleWord,
  guardBullets, rewriteMemory, SIMULATED_PREFIX, causalEvidenceCopy
} from "./copy.js";

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
      starred: starDomain === "marketing"
    });
  }

  // Hiring — engine hires are $8k payroll slots, translated per §5.4.
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
      starred: false
    });
  }

  return cards;
}

export function PlanCard({ card, compact = false, decisionState = null, onDecide = null }) {
  const [open, setOpen] = useState(false);
  return (
    <article className={`plan-card ${card.starred ? "starred" : ""} ${compact ? "compact" : ""}`}>
      <div className="plan-head">
        <span className="plan-domain">{card.title}</span>
        {card.starred && <span className="priority-pill">Priority</span>}
      </div>
      <strong className="plan-headline">{card.headline}</strong>
      <span className="plan-meta">
        {card.share != null && <>≈{Math.round(card.share)}% of MRR</>}
        {card.sub && <> · {card.sub}</>}
      </span>
      {!compact && <p className="plan-rationale">{card.rationale}</p>}
      {!compact && (
        <div className="plan-actions">
          <button className="link-button" type="button" onClick={() => setOpen(!open)}>
            {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />} Why this number?
          </button>
          {onDecide && (
            <button
              className={`accept-button ${decisionState === "accepted" ? "on" : ""}`}
              type="button"
              onClick={() => onDecide(card, decisionState === "accepted" ? "suggested" : "accepted")}
            >
              {decisionState === "accepted" ? <CheckCircle2 size={15} /> : <Circle size={15} />}
              {decisionState === "accepted" ? "Doing this" : "I'm doing this"}
            </button>
          )}
        </div>
      )}
      {!compact && open && (
        <ol className="plan-chain">
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
  const graphSentence = causalEvidenceCopy(analysis?.trace?.graph_summary);
  if (!memories.length && !graphSentence) {
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
      {graphSentence && (
        <div className="evidence-card">
          <span className="outcome-dot blue">◆</span>
          <p>{graphSentence}</p>
        </div>
      )}
    </div>
  );
}

// ---- confidence & freshness strip (§10.1 L6) ----

export function ConfidenceStrip({ analysis, month, estimatedCount = 0 }) {
  if (!analysis) return null;
  const band = confidenceBand(analysis.brief?.confidence);
  const reason = refreshReasonCopy(analysis.trace?.refresh_reason || analysis.reason);
  const reuse = briefSourceCopy(analysis.trace?.brief_source);
  return (
    <div className="confidence-strip">
      <span><strong>{band}</strong> confidence</span>
      <span className="dot-sep">·</span>
      <span>{reason}</span>
      {reuse && (<><span className="dot-sep">·</span><span>{reuse}</span></>)}
      {month && (<><span className="dot-sep">·</span><span>numbers from {new Date(month.enteredAt).toLocaleDateString("en-US", { month: "short", day: "numeric" })}</span></>)}
      {estimatedCount > 0 && (<><span className="dot-sep">·</span><span>{estimatedCount} estimated input{estimatedCount > 1 ? "s" : ""}</span></>)}
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
