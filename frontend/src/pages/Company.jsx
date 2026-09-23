// S11 My company — the data honesty center (spec §8, §15.1): every value with
// its provenance, plus the monthly update ritual (spec §13) as pre-filled diff
// editing with an instant what-changed payoff.

import React, { useMemo, useState } from "react";
import { ChevronRight, LoaderCircle, PencilLine } from "lucide-react";
import {
  useStore, latestMonth, latestCycle, feedbackForCycleMonth, uid
} from "../store.jsx";
import {
  CROWDEDNESS, MATURITY, deriveCac, deriveLtv,
  money, moneyExact, pct, signedPct, signedPp, dateLabel, monthName
} from "../derive.js";
import { runwayLabel } from "../founderView.js";
import { ProvChip, Banner, Notice, buildPlanCards } from "../components.jsx";
import { pickNotice } from "../notice.js";
import { submitCycleFeedback } from "../api.js";
import { useCycleRun } from "../cycleRun.jsx";
import { DONE_STATES, DONE_TO_DECISION } from "../loopView.js";

// A value the founder gave carries no marker; only the exceptions do.
function Row({ label, value, chip = "provided" }) {
  return (
    <div className="ledger-row">
      <span className="ledger-label">{label}</span>
      <strong className="ledger-value">{value}</strong>
      <ProvChip kind={chip} />
    </div>
  );
}

export function CompanyView({ navigate }) {
  const { state } = useStore();
  const company = state.company;
  const month = latestMonth(state);
  if (!company || !month) return null;
  const v = month.values;
  const entered = dateLabel(month.enteredAt);
  const cac = deriveCac(v);
  const ltv = deriveLtv(v);
  const crowd = CROWDEDNESS.find((c) => c.id === company.crowdedness);
  const maturity = MATURITY.find((m) => m.id === company.maturity);
  const ageNow = company.ageMonths + (month.index || 0);

  return (
    <section className="content-stack">
      <article className="panel">
        <div className="panel-title-row">
          <h3>{company.name}</h3>
          <button className="primary-button small" type="button" onClick={() => navigate("/update")}>
            <PencilLine size={14} /> Update my numbers
          </button>
        </div>
        {company.whatYouSell && <p className="subtle">{company.whatYouSell}</p>}
        <p className="subtle ledger-source">From your {monthName(month.enteredAt)} close ({entered}). Values without a marker are yours as you entered them.</p>

        <div className="ledger">
          <span className="ledger-section">Money</span>
          <Row label="Monthly recurring revenue" value={moneyExact(v.mrr)} />
          <Row label="Cash in the bank" value={moneyExact(v.cash)} />
          <Row label="Total monthly costs" value={moneyExact(v.costs)} />
          <Row label="Cash lasts" value={runwayLabel(v)} chip="derived" />

          <span className="ledger-section">Customers</span>
          <Row label="Average price" value={`$${v.price}/user/mo`} />
          <Row label="Monthly churn" value={`${pct(v.churnMonthly)}/mo`} />
          {v.newCustomers != null && <Row label="New customers last month" value={v.newCustomers} />}
          {v.marketingSpend != null && <Row label="Marketing spend last month" value={moneyExact(v.marketingSpend)} />}
          <Row
            label="Customer acquisition cost"
            value={cac.value ? money(cac.value) : "unknown"}
            chip={cac.source}
          />
          <Row label="Customer lifetime value" value={ltv ? money(ltv) : "—"} chip="derived" />

          <span className="ledger-section">Company & market</span>
          <Row label="Company age" value={`${ageNow} months`} />
          <Row label="Market crowdedness" value={crowd?.label || "—"} />
          <Row label="Product maturity" value={maturity?.label || "Not set"} chip={maturity ? "provided" : "estimated"} />
          {company.headcountReal && <Row label="Team size" value={`${company.headcountReal} people`} />}
          <Row label="Market conditions (rates, confidence)" value="Typical conditions assumed" chip="estimated" />
        </div>
      </article>

      <p className="subtle">
        Every value above feeds the board's analysis exactly as labelled; estimated values are the system's assumptions, not measurements.
      </p>
    </section>
  );
}

const UPDATE_FIELDS = [
  { key: "mrr", label: "Monthly recurring revenue", prefix: "$" },
  { key: "cash", label: "Cash in the bank", prefix: "$" },
  { key: "costs", label: "Total monthly costs", prefix: "$" },
  { key: "churnMonthly", label: "Monthly churn", suffix: "%/mo" },
  { key: "newCustomers", label: "New customers last month", optional: true },
  { key: "marketingSpend", label: "Marketing spend last month", prefix: "$", optional: true },
  { key: "price", label: "Average price", prefix: "$", optional: true }
];

// The HITL close (plan section 6.1), merged into the update ritual rather than
// added beside it: one screen, one set of numbers, one submit. Above the
// number grid the founder says what happened to each of last month's actions
// (did / partly / didn't, plus a note); the numbers follow; the server then
// scores the prediction, writes evidence only for what was actually done, and
// the next cycle starts from these real numbers.
function ClosableActions({ cards, done, notes, onDone, onNote }) {
  return (
    <div className="close-month">
      <h4>Last month the board asked for these — what happened?</h4>
      <ul className="close-list">
        {cards.map((c) => (
          <li key={c.domain} className="close-item">
            <div className="close-item-head">
              <span className="plan-domain">{c.title}</span>
              <strong>{c.headline}</strong>
            </div>
            <div className="did-toggle" role="group" aria-label={`${c.title}: what happened`}>
              {DONE_STATES.map((s) => (
                <button
                  key={s.id} type="button"
                  className={`did-option ${done[c.domain] === s.id ? `on ${s.id}` : ""}`}
                  onClick={() => onDone(c.domain, s.id)}
                >
                  {s.label}
                </button>
              ))}
            </div>
            <input
              type="text" className="close-note" placeholder="Optional note — e.g. did $4k instead"
              value={notes[c.domain] || ""}
              onChange={(e) => onNote(c.domain, e.target.value)}
            />
            {done[c.domain] === "didnt" && (
              <span className="close-hint">You didn't do this, so this month won't count as evidence about it.</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function UpdateRitual({ navigate }) {
  const { state, dispatch } = useStore();
  const { requestStart } = useCycleRun();
  const last = latestMonth(state);
  const [values, setValues] = useState(() => ({ ...last?.values }));
  const [done, setDone] = useState({});
  const [notes, setNotes] = useState({});
  const [closing, setClosing] = useState(false);
  const [closeError, setCloseError] = useState(null);

  // The plan to close: the latest cycle, if it was made on the month being
  // closed and hasn't been closed already.
  const cycle = latestCycle(state);
  const closable = !state.demo && cycle && last && cycle.monthId === last.id
    && (cycle.months || []).length > 0 && !feedbackForCycleMonth(cycle, 1);
  const cycleAnalysis = closable ? state.analyses.find((a) => a.cycleId === cycle.id) : null;
  const actionCards = useMemo(
    () => (closable && cycleAnalysis ? buildPlanCards(cycleAnalysis, last).filter((c) => c.isAction) : []),
    [closable, cycleAnalysis, last]
  );

  if (!last) { navigate("/"); return null; }

  const diffs = useMemo(() => {
    const out = [];
    if (values.mrr !== last.values.mrr && values.mrr > 0 && last.values.mrr > 0) {
      out.push(`MRR ${signedPct(((values.mrr - last.values.mrr) / last.values.mrr) * 100)}`);
    }
    if (values.churnMonthly !== last.values.churnMonthly && values.churnMonthly != null) {
      out.push(`churn ${signedPp(values.churnMonthly - last.values.churnMonthly)}`);
    }
    if (values.cash !== last.values.cash && values.cash > 0) {
      out.push(`cash ${money(values.cash - last.values.cash)}`);
    }
    return out;
  }, [values, last]);

  const numbersValid = values.mrr > 0 && values.cash > 0 && values.costs > 0 && values.churnMonthly != null && values.churnMonthly >= 0;
  const actionsAnswered = actionCards.every((c) => done[c.domain]);
  const valid = numbersValid && actionsAnswered;

  async function submit() {
    if (!valid || state.demo || closing) return;
    const newMonth = {
      id: uid("m"),
      index: (last.index || 0) + 1,
      enteredAt: new Date().toISOString(),
      values: { ...last.values, ...values },
      decisions: []
    };
    dispatch({ type: "ADD_MONTH", month: newMonth });

    if (closable && actionCards.length) {
      setClosing(true);
      setCloseError(null);
      // The founder's answers become the planned month's decisions, so
      // History shows them with the same glyphs it always used.
      for (const c of actionCards) {
        dispatch({
          type: "SET_DECISION",
          monthId: last.id,
          decision: {
            id: uid("d"), domain: c.domain, text: c.headline,
            state: DONE_TO_DECISION[done[c.domain]], note: notes[c.domain] || null
          }
        });
      }
      const perAction = actionCards.map((c) => ({
        action_key: c.domain, done: done[c.domain], note: notes[c.domain] || null
      }));
      const actuals = {
        mrr: newMonth.values.mrr, cash: newMonth.values.cash,
        churn: newMonth.values.churnMonthly, costs: newMonth.values.costs ?? null
      };
      const r = await submitCycleFeedback(cycle.id, { monthIndex: 1, perAction, actuals });
      dispatch({
        type: "SET_CYCLE_FEEDBACK",
        cycleId: cycle.id, monthIndex: 1,
        submitted: { per_action: perAction, actuals },
        result: r.ok ? r.data : null,
        error: r.ok ? null : (r.offline ? "engine unreachable" : r.error)
      });
      setClosing(false);
      if (!r.ok) {
        // The numbers are saved either way; the founder is told the close did
        // not reach the board rather than shown a plan that pretends it did.
        setCloseError(r.offline
          ? "Your numbers are saved, but the engine couldn't be reached to score last month's plan. The next plan will start from your numbers without that score."
          : r.error);
      }
    }
    // The next cycle is asked for, not started here: the provider's effect
    // runs start() on the render that already holds the new month and the
    // feedback result. `await start()` from this handler would build the
    // cycle from the closure's stale state — without the month just closed
    // and without the track record.
    requestStart();
    navigate("/home");
  }

  return (
    <section className="content-stack narrow-col">
      <article className="panel">
        <h3>{closable && actionCards.length ? "Close the month" : "Update your numbers"}</h3>
        <p className="subtle">Pre-filled with last month ({dateLabel(last.enteredAt)}) — edit what changed. ~2 minutes.</p>
        <Notice notice={pickNotice({ demo: state.demo }).notice} />
        {closable && actionCards.length > 0 && (
          <ClosableActions
            cards={actionCards} done={done} notes={notes}
            onDone={(domain, value) => setDone({ ...done, [domain]: value })}
            onNote={(domain, value) => setNotes({ ...notes, [domain]: value })}
          />
        )}
        {closeError && <Banner tone="warn">{closeError}</Banner>}
        <div className="update-grid">
          {UPDATE_FIELDS.map((f) => (
            <label className="ffield" key={f.key}>
              <span className="ffield-label">{f.label}{f.optional ? " (optional)" : ""}</span>
              <span className="num-input">
                {f.prefix && <em>{f.prefix}</em>}
                <input
                  type="number" inputMode="decimal" step="any"
                  value={values[f.key] ?? ""}
                  onChange={(e) => setValues({ ...values, [f.key]: e.target.value === "" ? null : Number(e.target.value) })}
                />
                {f.suffix && <em>{f.suffix}</em>}
              </span>
            </label>
          ))}
        </div>
        {diffs.length > 0 && (
          <div className="diff-row">
            {diffs.map((d) => <span className="diff-pill" key={d}>{d}</span>)}
          </div>
        )}
        {actionCards.length > 0 && !actionsAnswered && (
          <p className="subtle">Say what happened to each of the board's actions above to continue.</p>
        )}
        <div className="wizard-foot inline">
          <button className="secondary-button" type="button" onClick={() => navigate("/company")}>Cancel</button>
          <button className="primary-button" type="button" disabled={!valid || state.demo || closing} onClick={submit}>
            {closing ? <><LoaderCircle size={15} className="spin" /> Scoring last month…</>
              : <>{closable && actionCards.length ? "Close the month & plan again" : "Save & plan"} <ChevronRight size={16} /></>}
          </button>
        </div>
      </article>
    </section>
  );
}
