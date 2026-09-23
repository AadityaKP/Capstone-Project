// Local-first application store. Company, monthly snapshots, analyses,
// cycles and decisions persist in localStorage (state-ownership decision (a)
// in docs/oefa_loop_decisions.md: the browser is the founder's record; cycles
// carry their own actuals on the server). Sample mode (spec §7 S1) runs
// entirely in memory and never touches the founder's stored data.

import React, { createContext, useContext, useEffect, useMemo, useReducer } from "react";
import { SAMPLE } from "./sample.js";

const STORAGE_KEY = "ssom_founder_v1";

const EMPTY = {
  demo: false,
  company: null,
  months: [],
  analyses: [],
  cycles: [],
  settings: { narratives: false },
  onboardingDraft: null
};

function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return EMPTY;
    const parsed = JSON.parse(raw);
    return { ...EMPTY, ...parsed, demo: false };
  } catch {
    return EMPTY;
  }
}

function persist(state) {
  if (state.demo) return;
  const { demo, ...rest } = state;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(rest));
  } catch {
    // storage full/blocked — the session still works in memory
  }
}

function reducer(state, action) {
  switch (action.type) {
    case "ENTER_DEMO":
      return { ...JSON.parse(JSON.stringify(SAMPLE)), onboardingDraft: null };
    case "EXIT_DEMO":
      return load();
    case "IMPORT_STATE":
      // A seeded workspace (experiments/seed_demo_company.py → /api/demo/bootstrap).
      // Replaces everything; never in demo mode.
      return { ...EMPTY, ...action.state, demo: false, onboardingDraft: null };
    case "SAVE_DRAFT":
      return { ...state, onboardingDraft: { ...state.onboardingDraft, ...action.draft } };
    case "CREATE_COMPANY": {
      return {
        ...state,
        onboardingDraft: null,
        company: action.company,
        months: [action.month],
        analyses: [],
        cycles: []
      };
    }
    case "ADD_MONTH":
      return { ...state, months: [...state.months, action.month] };
    case "ADD_ANALYSIS": {
      // Idempotent for cycle-born analyses: the promotion effect can run twice
      // under React.StrictMode before the store updates. Analyses without a
      // cycleId (the pre-loop /api/advise path) are appended as before.
      const a = action.analysis;
      if (a.cycleId && state.analyses.some((x) => x.cycleId === a.cycleId && (x.monthIndex || 1) === (a.monthIndex || 1))) {
        return state;
      }
      return { ...state, analyses: [...state.analyses, a] };
    }
    case "ADD_CYCLE":
      if ((state.cycles || []).some((c) => c.id === action.cycle.id)) return state;
      return { ...state, cycles: [...state.cycles, action.cycle] };
    case "UPDATE_CYCLE":
      return {
        ...state,
        cycles: state.cycles.map((c) => (c.id === action.cycle.id ? { ...c, ...action.cycle } : c))
      };
    case "SET_CYCLE_FEEDBACK":
      return {
        ...state,
        cycles: state.cycles.map((c) =>
          c.id !== action.cycleId
            ? c
            : {
                ...c,
                feedback: [
                  ...(c.feedback || []).filter((f) => f.monthIndex !== action.monthIndex),
                  { monthIndex: action.monthIndex, submitted: action.submitted, result: action.result, error: action.error || null, closedAt: new Date().toISOString() }
                ]
              }
        )
      };
    case "SET_DECISION": {
      const months = state.months.map((m) =>
        m.id !== action.monthId
          ? m
          : {
              ...m,
              decisions: (m.decisions || []).some((d) => d.id === action.decision.id)
                ? m.decisions.map((d) => (d.id === action.decision.id ? { ...d, ...action.decision } : d))
                : [...(m.decisions || []), action.decision]
            }
      );
      return { ...state, months };
    }
    case "SET_SETTING":
      return { ...state, settings: { ...state.settings, [action.key]: action.value } };
    case "RESET_ALL":
      localStorage.removeItem(STORAGE_KEY);
      return { ...EMPTY };
    default:
      return state;
  }
}

const StoreContext = createContext(null);

export function StoreProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, undefined, load);
  useEffect(() => persist(state), [state]);
  const value = useMemo(() => ({ state, dispatch }), [state]);
  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useStore() {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error("useStore outside provider");
  return ctx;
}

// ---- selectors ----

export function latestMonth(state) {
  return state.months.length ? state.months[state.months.length - 1] : null;
}

export function previousMonth(state) {
  return state.months.length > 1 ? state.months[state.months.length - 2] : null;
}

export function latestAnalysis(state) {
  return state.analyses.length ? state.analyses[state.analyses.length - 1] : null;
}

export function analysisForMonth(state, monthId) {
  return [...state.analyses].reverse().find((a) => a.monthId === monthId) || null;
}

export function monthById(state, id) {
  return state.months.find((m) => m.id === id) || null;
}

export function latestCycle(state) {
  const cycles = state.cycles || [];
  return cycles.length ? cycles[cycles.length - 1] : null;
}

export function cycleForMonth(state, monthId) {
  return [...(state.cycles || [])].reverse().find((c) => c.monthId === monthId) || null;
}

export function cycleById(state, id) {
  return (state.cycles || []).find((c) => c.id === id) || null;
}

// The most recent close-the-month that produced a result: the prediction
// error Home leads with, and the track record the next cycle starts from.
export function latestClosedFeedback(state) {
  for (const cycle of [...(state.cycles || [])].reverse()) {
    const closed = (cycle.feedback || []).filter((f) => f.result);
    if (closed.length) return { cycle, feedback: closed[closed.length - 1] };
  }
  return null;
}

export function feedbackForCycleMonth(cycle, monthIndex) {
  return (cycle?.feedback || []).find((f) => f.monthIndex === monthIndex && f.result) || null;
}

// A cycle's month 1 is a full board analysis of the founder's current
// numbers — the same thing /api/advise produced — so it becomes an analysis
// record and every existing surface (Home, Advice detail, History) keeps
// working unchanged.
export function analysisFromCycle(cycle) {
  const month = (cycle.months || [])[0];
  if (!month) return null;
  return {
    id: uid("a"),
    cycleId: cycle.id,
    monthIndex: 1,
    monthId: cycle.monthId,
    createdAt: new Date().toISOString(),
    source: cycle.source === "sample" ? "sample" : "cycle",
    llm_ok: month.execute.llm_ok !== false,
    reason: month.execute.refresh_reason,
    brief: month.execute.brief,
    trace: month.execute.trace,
    display: month.execute.display || null,
    narratives: null
  };
}

export function uid(prefix) {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}
