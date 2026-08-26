// S3 Onboarding — three grouped steps with live feedback (spec §7 O2–O4).
// Minimum viable input: 8 fields (§6.2). Everything else is optional enrichment.

import React, { useMemo, useState } from "react";
import { Building2, ChevronDown, ChevronRight, CircleDollarSign, Target } from "lucide-react";
import { useStore, uid } from "../store.jsx";
import {
  CROWDEDNESS, MATURITY, annualToMonthlyChurn, deriveCac,
  deriveLtv, money, pct
} from "../derive.js";
import { runwayPhrase, efficiency } from "../founderView.js";

const STEPS = [
  { id: "company", label: "Your company", icon: Building2 },
  { id: "money", label: "Money", icon: CircleDollarSign },
  { id: "customers", label: "Customers", icon: Target }
];

function Field({ label, help, error, children }) {
  return (
    <label className={`ffield ${error ? "has-error" : ""}`}>
      <span className="ffield-label">{label}</span>
      {children}
      {help && !error && <span className="ffield-help">{help}</span>}
      {error && <span className="ffield-error">{error}</span>}
    </label>
  );
}

function NumInput({ value, onChange, prefix, suffix, placeholder }) {
  return (
    <span className="num-input">
      {prefix && <em>{prefix}</em>}
      <input
        type="number" inputMode="decimal" step="any" placeholder={placeholder || ""}
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
      />
      {suffix && <em>{suffix}</em>}
    </span>
  );
}

export default function Onboarding({ navigate }) {
  const { state, dispatch } = useStore();
  const draft = state.onboardingDraft || {};
  const [step, setStep] = useState(draft._step || 0);
  const [annualMode, setAnnualMode] = useState(false);
  const [enrichOpen, setEnrichOpen] = useState(false);
  const [touched, setTouched] = useState({});

  const set = (patch) => dispatch({ type: "SAVE_DRAFT", draft: { ...patch, _step: step } });

  const runway = useMemo(
    () => runwayPhrase({ cash: draft.cash, costs: draft.costs, mrr: draft.mrr }),
    [draft]
  );

  const errors = {
    name: !draft.name?.trim() ? "Give your company a name" : null,
    ageMonths: draft.ageMonths == null || draft.ageMonths < 0 ? "How many months old is the company?" : draft.ageMonths > 600 ? "That's more than 50 years — check the number" : null,
    crowdedness: !draft.crowdedness ? "Pick the closest option" : null,
    mrr: draft.mrr == null || draft.mrr <= 0 ? "Monthly recurring revenue is required" : null,
    cash: draft.cash == null || draft.cash <= 0 ? "Cash in the bank is required" : null,
    costs: draft.costs == null || draft.costs <= 0 ? "Total monthly costs are required" : null,
    price: draft.price == null || draft.price <= 0 ? "Average price per customer is required" : null,
    churnMonthly: draft.churnMonthly == null || draft.churnMonthly < 0 ? "Monthly churn is required"
      : draft.churnMonthly > 30 ? "Above 30%/month is extreme — double-check monthly vs. annual" : null
  };
  const softErrors = new Set(["costs", "churnMonthly"]);
  const stepFields = [
    ["name", "ageMonths", "crowdedness"],
    ["mrr", "cash", "costs"],
    ["price", "churnMonthly"]
  ];
  const stepBlocked = stepFields[step].some((f) => errors[f] && !(softErrors.has(f) && draft[f] != null && draft[f] > 0));

  const cac = deriveCac(draft);
  const ltv = deriveLtv(draft);

  function next() {
    setTouched(Object.fromEntries(stepFields[step].map((f) => [f, true])));
    if (stepBlocked) return;
    if (step < 2) { setStep(step + 1); set({ _step: step + 1 }); return; }
    finish();
  }

  function finish() {
    const company = {
      id: uid("co"),
      name: draft.name.trim(),
      whatYouSell: draft.whatYouSell || "",
      ageMonths: draft.ageMonths,
      crowdedness: draft.crowdedness,
      maturity: draft.maturity || null,
      headcountReal: draft.headcountReal || null,
      createdAt: new Date().toISOString()
    };
    const month = {
      id: uid("m"), index: 0, enteredAt: new Date().toISOString(),
      values: {
        mrr: draft.mrr, cash: draft.cash, costs: draft.costs, price: draft.price,
        churnMonthly: draft.churnMonthly,
        newCustomers: draft.newCustomers ?? null,
        marketingSpend: draft.marketingSpend ?? null,
        cacDirect: draft.cacDirect ?? null,
        churnEnt: draft.churnEnt ?? null, churnSmb: draft.churnSmb ?? null, churnB2c: draft.churnB2c ?? null
      },
      decisions: []
    };
    dispatch({ type: "CREATE_COMPANY", company, month });
    navigate("/analyzing");
  }

  const StepIcon = STEPS[step].icon;

  return (
    <section className="wizard">
      <header className="wizard-head">
        <span className="wizard-step"><StepIcon size={17} /> Step {step + 1} of 3 · {STEPS[step].label}</span>
        <div className="wizard-dots">{STEPS.map((s, i) => <i key={s.id} className={i <= step ? "on" : ""} />)}</div>
      </header>

      {step === 0 && (
        <div className="wizard-body">
          <h2>Your company</h2>
          <p className="wizard-sub">Who you are and the market you're in.</p>
          <Field label="Company name" error={touched.name && errors.name}>
            <input type="text" value={draft.name ?? ""} onChange={(e) => set({ name: e.target.value })} placeholder="Acme Analytics" />
          </Field>
          <Field label="What you sell (optional)" help="One line, just for your own screens.">
            <input type="text" value={draft.whatYouSell ?? ""} onChange={(e) => set({ whatYouSell: e.target.value })} placeholder="Usage analytics for e-commerce teams" />
          </Field>
          <Field label="Company age" help="Months since first paying customers — retention behaves differently as companies age." error={touched.ageMonths && errors.ageMonths}>
            <NumInput value={draft.ageMonths} onChange={(v) => set({ ageMonths: v })} suffix="months" />
          </Field>
          <Field label="How crowded is your market?" help="How many direct competitors do customers compare you to?" error={touched.crowdedness && errors.crowdedness}>
            <div className="choice-row">
              {CROWDEDNESS.map((c) => (
                <button key={c.id} type="button" className={`choice ${draft.crowdedness === c.id ? "on" : ""}`} onClick={() => set({ crowdedness: c.id })}>
                  {c.label}
                </button>
              ))}
            </div>
          </Field>
        </div>
      )}

      {step === 1 && (
        <div className="wizard-body">
          <h2>Your money</h2>
          <p className="wizard-sub">We use these to compute runway and size this month's plan.</p>
          <Field label="Monthly recurring revenue" error={touched.mrr && errors.mrr}>
            <NumInput value={draft.mrr} onChange={(v) => set({ mrr: v })} prefix="$" suffix="/month" />
          </Field>
          <Field label="Cash in the bank" error={touched.cash && errors.cash}>
            <NumInput value={draft.cash} onChange={(v) => set({ cash: v })} prefix="$" />
          </Field>
          <Field label="Total monthly costs" help="Payroll + tools + rent + marketing + everything. If it is just you and a few subscriptions, say so — the number is used exactly as you enter it." error={touched.costs && errors.costs}>
            <NumInput value={draft.costs} onChange={(v) => set({ costs: v })} prefix="$" suffix="/month" />
          </Field>
          <Field label="Marketing spend last month (optional)" help="Unlocks acquisition-cost analysis together with new customers.">
            <NumInput value={draft.marketingSpend} onChange={(v) => set({ marketingSpend: v })} prefix="$" />
          </Field>
          {draft.cash > 0 && draft.costs > 0 && draft.mrr != null && (
            <div className="live-note">
              <strong>{runway}</strong>
              <span> — assuming revenue and costs stay flat.</span>
            </div>
          )}
        </div>
      )}

      {step === 2 && (
        <div className="wizard-body">
          <h2>Your customers</h2>
          <p className="wizard-sub">Pricing and retention — the advisors' two favorite numbers.</p>
          <Field label="Average price per customer" help="Per month. If you have several plans, use total MRR ÷ paying customers." error={touched.price && errors.price}>
            <NumInput value={draft.price} onChange={(v) => set({ price: v })} prefix="$" suffix="/user/mo" />
          </Field>
          <Field
            label={`Customer churn (${annualMode ? "annual" : "monthly"})`}
            help={annualMode ? "We'll convert to monthly for you." : "Monthly, not annual — the toggle converts if you only know the annual figure."}
            error={touched.churnMonthly && errors.churnMonthly}
          >
            <div className="churn-row">
              <NumInput
                value={annualMode ? draft.churnAnnualRaw : draft.churnMonthly}
                onChange={(v) => annualMode
                  ? set({ churnAnnualRaw: v, churnMonthly: v == null ? null : Number(annualToMonthlyChurn(v).toFixed(2)) })
                  : set({ churnMonthly: v, churnAnnualRaw: null })}
                suffix={`% /${annualMode ? "yr" : "mo"}`}
              />
              <button type="button" className="link-button" onClick={() => setAnnualMode(!annualMode)}>
                I know the {annualMode ? "monthly" : "annual"} figure
              </button>
            </div>
            {annualMode && draft.churnMonthly != null && (
              <span className="ffield-help">= {pct(draft.churnMonthly)} per month</span>
            )}
          </Field>
          <Field label="New customers last month (optional)" help="With marketing spend, this computes your acquisition cost.">
            <NumInput value={draft.newCustomers} onChange={(v) => set({ newCustomers: v })} suffix="customers" />
          </Field>
          <Field label="Product maturity (optional)" help="Your own judgment — used as a rough proxy, always labelled estimated.">
            <div className="choice-row">
              {MATURITY.map((m) => (
                <button key={m.id} type="button" className={`choice ${draft.maturity === m.id ? "on" : ""}`} onClick={() => set({ maturity: m.id })}>
                  {m.label}
                </button>
              ))}
            </div>
          </Field>

          {(cac.value || ltv) && (
            <div className="live-note">
              <strong>{efficiency(ltv, cac.value, draft.newCustomers).label}</strong>
              <span> — {efficiency(ltv, cac.value, draft.newCustomers).detail}</span>
            </div>
          )}

          <div className="enough-banner">
            <strong>That's enough for your first analysis.</strong>
            <button className="link-button" type="button" onClick={() => setEnrichOpen(!enrichOpen)}>
              {enrichOpen ? <ChevronDown size={15} /> : <ChevronRight size={15} />} Add detail first (optional)
            </button>
          </div>
          {enrichOpen && (
            <div className="enrich-grid">
              <Field label="Acquisition cost, if you track it" help="Overrides the derived figure.">
                <NumInput value={draft.cacDirect} onChange={(v) => set({ cacDirect: v })} prefix="$" />
              </Field>
              <Field label="Team size (people)" help="For your screens — plan math uses your costs, not headcount.">
                <NumInput value={draft.headcountReal} onChange={(v) => set({ headcountReal: v })} suffix="people" />
              </Field>
              <Field label="Enterprise churn %/mo (if known)">
                <NumInput value={draft.churnEnt} onChange={(v) => set({ churnEnt: v })} suffix="%" />
              </Field>
              <Field label="SMB churn %/mo (if known)">
                <NumInput value={draft.churnSmb} onChange={(v) => set({ churnSmb: v })} suffix="%" />
              </Field>
              <Field label="Consumer churn %/mo (if known)">
                <NumInput value={draft.churnB2c} onChange={(v) => set({ churnB2c: v })} suffix="%" />
              </Field>
            </div>
          )}
        </div>
      )}

      <footer className="wizard-foot">
        {step > 0
          ? <button className="secondary-button" type="button" onClick={() => setStep(step - 1)}>Back</button>
          : <button className="secondary-button" type="button" onClick={() => navigate("/")}>Cancel</button>}
        <button className="primary-button" type="button" onClick={next}>
          {step < 2 ? "Continue" : "Run my first analysis"} <ChevronRight size={16} />
        </button>
      </footer>
    </section>
  );
}
