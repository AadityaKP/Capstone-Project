// Local-first application store. Company, monthly snapshots, analyses and
// decisions persist in localStorage (the client-side stand-in for the spec's
// G2 company-months store until the backend exists). Sample mode (spec §7 S1)
// runs entirely in memory and never touches the founder's stored data.

import React, { createContext, useContext, useEffect, useMemo, useReducer } from "react";
import { SAMPLE } from "./sample.js";

const STORAGE_KEY = "ssom_founder_v1";

const EMPTY = {
  demo: false,
  company: null,
  months: [],
  analyses: [],
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
    case "SAVE_DRAFT":
      return { ...state, onboardingDraft: { ...state.onboardingDraft, ...action.draft } };
    case "CREATE_COMPANY": {
      return {
        ...state,
        onboardingDraft: null,
        company: action.company,
        months: [action.month],
        analyses: []
      };
    }
    case "ADD_MONTH":
      return { ...state, months: [...state.months, action.month] };
    case "ADD_ANALYSIS":
      return { ...state, analyses: [...state.analyses, action.analysis] };
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

export function uid(prefix) {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}
