// S11 My company — the data honesty center (spec §8, §15.1): every value with
// its provenance, plus the monthly update ritual (spec §13) as pre-filled diff
// editing with an instant what-changed payoff.

import React, { useMemo, useState } from "react";
import { ChevronRight, PencilLine } from "lucide-react";
import { useStore, latestMonth, uid } from "../store.jsx";
import {
  CROWDEDNESS, MATURITY, deriveCac, deriveLtv,
  money, moneyExact, pct, signedPct, signedPp, dateLabel, monthsLabel
} from "../derive.js";
import { runwayLabel } from "../founderView.js";
import { ProvChip, Banner } from "../components.jsx";

function Row({ label, value, chip, chipDate }) {
  return (
    <div className="ledger-row">
      <span className="ledger-label">{label}</span>
      <strong className="ledger-value">{value}</strong>
      <ProvChip kind={chip} date={chipDate} />
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

        <div className="ledger">
          <span className="ledger-section">Money</span>
          <Row label="Monthly recurring revenue" value={moneyExact(v.mrr)} chip="provided" chipDate={entered} />
          <Row label="Cash in the bank" value={moneyExact(v.cash)} chip="provided" chipDate={entered} />
          <Row label="Total monthly costs" value={moneyExact(v.costs)} chip="provided" chipDate={entered} />
          <Row label="Cash lasts" value={runwayLabel(v)} chip="derived" />

          <span className="ledger-section">Customers</span>
          <Row label="Average price" value={`$${v.price}/user/mo`} chip="provided" chipDate={entered} />
          <Row label="Monthly churn" value={`${pct(v.churnMonthly)}/mo`} chip="provided" chipDate={entered} />
          {v.newCustomers != null && <Row label="New customers last month" value={v.newCustomers} chip="provided" chipDate={entered} />}
          {v.marketingSpend != null && <Row label="Marketing spend last month" value={moneyExact(v.marketingSpend)} chip="provided" chipDate={entered} />}
          <Row
            label="Customer acquisition cost"
            value={cac.value ? money(cac.value) : "unknown"}
            chip={cac.source}
          />
          <Row label="Customer lifetime value" value={ltv ? money(ltv) : "—"} chip="derived" />

          <span className="ledger-section">Company & market</span>
          <Row label="Company age" value={`${ageNow} months`} chip="provided" />
          <Row label="Market crowdedness" value={crowd?.label || "—"} chip="provided" />
          <Row label="Product maturity" value={maturity?.label || "Not set"} chip={maturity ? "provided" : "estimated"} />
          {company.headcountReal && <Row label="Team size" value={`${company.headcountReal} people`} chip="provided" />}
          <Row label="Market conditions (rates, confidence)" value="Typical conditions assumed" chip="estimated" />
        </div>
      </article>

      <Banner tone="info">
        Every value above feeds the board's analysis exactly as labelled — nothing else is
        collected or observed. "Estimated" values are the system's assumptions, not measurements.
      </Banner>
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

export function UpdateRitual({ navigate }) {
  const { state, dispatch } = useStore();
  const last = latestMonth(state);
  const [values, setValues] = useState(() => ({ ...last?.values }));

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

  const valid = values.mrr > 0 && values.cash > 0 && values.costs > 0 && values.churnMonthly != null && values.churnMonthly >= 0;

  function submit() {
    if (!valid || state.demo) return;
    dispatch({
      type: "ADD_MONTH",
      month: {
        id: uid("m"),
        index: (last.index || 0) + 1,
        enteredAt: new Date().toISOString(),
        values: { ...last.values, ...values },
        decisions: []
      }
    });
    navigate("/analyzing");
  }

  return (
    <section className="content-stack narrow-col">
      <article className="panel">
        <h3>Update your numbers</h3>
        <p className="subtle">Pre-filled with last month ({dateLabel(last.enteredAt)}) — edit what changed. ~2 minutes.</p>
        {state.demo && <Banner tone="info">Sample company — updates are disabled here. Start your own company from the welcome screen.</Banner>}
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
        <div className="wizard-foot inline">
          <button className="secondary-button" type="button" onClick={() => navigate("/company")}>Cancel</button>
          <button className="primary-button" type="button" disabled={!valid || state.demo} onClick={submit}>
            Save & re-analyse <ChevronRight size={16} />
          </button>
        </div>
      </article>
    </section>
  );
}
