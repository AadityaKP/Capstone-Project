// The cycle's lifecycle at app level (docs/ui_simplification_plan.md Phase A).
//
// Starting a cycle, polling it while it runs and promoting its first month to
// an analysis used to live in two pages (Analyzing owned the start, the Plan
// page owned the poll). A cycle started from the close-the-month step would
// then only land if the founder happened to be on the Plan page. This
// provider owns all three so the cycle completes whichever page is open.
//
// `start()` reads the store from the provider's current render — never from a
// closure a caller captured — and `requestStart()` defers the call to the
// first render after the caller's own dispatches have been applied, so a
// cycle asked for by Close is built from the month it just added and the
// track record it just stored.

import React, {
  createContext, useCallback, useContext, useEffect, useRef, useState
} from "react";
import {
  useStore, latestMonth, latestCycle, latestClosedFeedback, analysisFromCycle
} from "./store.jsx";
import { startCycle, getCycle } from "./api.js";

export const CYCLE_HORIZON = 4;
const POLL_MS = 2500;

const CycleRunContext = createContext(null);

export function CycleRunProvider({ children }) {
  const { state, dispatch } = useStore();
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState(null);
  const [pollError, setPollError] = useState(null);
  const [elapsed, setElapsed] = useState(0);

  // The current store, readable from a stable callback.
  const stateRef = useRef(state);
  stateRef.current = state;

  // In-flight guard held in a ref: React.StrictMode runs mount effects twice
  // in development, and a state-held flag would let both invocations through
  // before the first one's setState landed.
  const startingRef = useRef(false);

  const start = useCallback(async () => {
    if (startingRef.current) return { ok: false, error: "A cycle is already starting." };
    const s = stateRef.current;
    const month = latestMonth(s);
    if (!s.company || !month || s.demo) return { ok: false, error: "Nothing to plan from." };

    startingRef.current = true;
    setStarting(true);
    setStartError(null);

    // The last close-the-month's record: the board starts this cycle already
    // knowing how its previous prediction went (oefa_loop_plan.md 6.2 item 5).
    const closed = latestClosedFeedback(s);
    const previousTrackRecord = closed?.feedback?.result?.track_record || null;

    const result = await startCycle(
      s.company,
      { ...month, history: s.months.slice(0, -1).map((m) => m.values) },
      { horizon: CYCLE_HORIZON, previousTrackRecord }
    );

    startingRef.current = false;
    setStarting(false);
    if (!result.ok) {
      setStartError(result);
      return { ok: false, offline: !!result.offline, error: result.error };
    }
    const cycle = result.data;
    dispatch({
      type: "ADD_CYCLE",
      cycle: {
        id: cycle.id,
        monthId: month.id,
        createdAt: new Date().toISOString(),
        source: "api",
        status: cycle.status,
        horizon: cycle.horizon_months || CYCLE_HORIZON,
        months: cycle.months || [],
        summary: cycle.summary || null,
        meta: cycle.meta || null,
        feedback: [],
        startedFromTrackRecord: !!previousTrackRecord
      }
    });
    return { ok: true, offline: false, error: null };
  }, [dispatch]);

  // Deferred start. The flag lives in a ref (cleared synchronously before
  // `start()` runs, so a doubled effect cannot start two cycles); the counter
  // exists only to schedule the render in which the effect below fires.
  const pendingRef = useRef(false);
  const [pendingTick, setPendingTick] = useState(0);
  const requestStart = useCallback(() => {
    pendingRef.current = true;
    setPendingTick((n) => n + 1);
  }, []);
  useEffect(() => {
    if (!pendingRef.current) return;
    pendingRef.current = false;
    start();
  }, [pendingTick, start]);

  const cycle = latestCycle(state);
  const demo = !!state.demo;
  const running = !!cycle && !demo && ["queued", "running"].includes(cycle.status);
  const monthsLanded = (cycle?.months || []).length;

  // Poll while the cycle runs. Keyed on id and status rather than the cycle
  // object, or every UPDATE_CYCLE would tear the interval down and restart it.
  useEffect(() => {
    if (!running) { setPollError(null); return undefined; }
    let alive = true;
    const id = cycle.id;
    async function poll() {
      const r = await getCycle(id);
      if (!alive) return;
      if (!r.ok) {
        setPollError(r.offline
          ? "Lost contact with the engine — it keeps running; this page will catch up when it's back."
          : r.error);
        return;
      }
      setPollError(null);
      const c = r.data;
      dispatch({
        type: "UPDATE_CYCLE",
        cycle: { id: c.id, status: c.status, months: c.months || [], summary: c.summary || null, meta: c.meta || null, error: c.error || null }
      });
    }
    poll();
    const timer = setInterval(poll, POLL_MS);
    return () => { alive = false; clearInterval(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cycle?.id, cycle?.status, running, demo]);

  // Elapsed time from the cycle's own createdAt, so it survives a reload.
  useEffect(() => {
    if (!running) { setElapsed(0); return undefined; }
    const startedAt = new Date(cycle.createdAt).getTime() || Date.now();
    const tick = () => setElapsed(Math.max(0, Math.floor((Date.now() - startedAt) / 1000)));
    tick();
    const timer = setInterval(tick, 1000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cycle?.id, running]);

  // Month 1 of a cycle is a full board analysis of the founder's current
  // numbers, so it becomes an analysis record the moment it exists. The guard
  // here reads the render it ran in; the reducer's own guard (store.jsx) is
  // what actually prevents a duplicate under a doubled effect.
  useEffect(() => {
    if (!cycle || demo || !monthsLanded) return;
    if (state.analyses.some((a) => a.cycleId === cycle.id)) return;
    const analysis = analysisFromCycle(cycle);
    if (analysis) dispatch({ type: "ADD_ANALYSIS", analysis });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cycle?.id, monthsLanded, demo]);

  const value = { start, requestStart, starting, startError, pollError, elapsed };
  return <CycleRunContext.Provider value={value}>{children}</CycleRunContext.Provider>;
}

export function useCycleRun() {
  const ctx = useContext(CycleRunContext);
  if (!ctx) throw new Error("useCycleRun outside provider");
  return ctx;
}
