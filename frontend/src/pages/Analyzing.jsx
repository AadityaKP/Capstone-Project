// S4 Analysis in progress — honest staged progress and the §17.6 failure path.
// The one thing this UI cannot do locally is the analysis itself; when the
// engine API is unreachable the founder gets a truthful card, never a
// fabricated result.

import React, { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, LoaderCircle } from "lucide-react";
import { useStore, latestMonth, uid } from "../store.jsx";
import { advise } from "../api.js";
import { ProgressStages, Banner } from "../components.jsx";
import { eventTrigger } from "../derive.js";

export default function Analyzing({ navigate }) {
  const { state, dispatch } = useStore();
  const [stage, setStage] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [failed, setFailed] = useState(null);
  const runningRef = useRef(false);
  const timersRef = useRef([]);

  const month = latestMonth(state);
  const narrativesOn = !!state.settings.narratives;

  const clearTimers = () => {
    timersRef.current.forEach((t) => { clearTimeout(t); clearInterval(t); });
    timersRef.current = [];
  };

  const run = useCallback(async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    setFailed(null);
    setStage(0);
    setElapsed(0);

    const t0 = Date.now();
    const tick = setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 1000);
    const s1 = setTimeout(() => setStage(1), 1200);
    timersRef.current.push(tick, s1);

    const lastAnalyzedMonth = state.analyses.length
      ? state.months.find((m) => m.id === state.analyses[state.analyses.length - 1].monthId)
      : null;
    const clientReason = state.analyses.length === 0
      ? "initial"
      : eventTrigger(month, lastAnalyzedMonth) || "cadence";

    const result = await advise(state.company, {
      ...month,
      history: state.months.slice(0, -1).map((m) => m.values)
    });

    runningRef.current = false;
    if (!result.ok) {
      clearTimers();
      setFailed(result);
      return;
    }
    setStage(2);
    const a = result.data.analysis || result.data;
    const done = setTimeout(() => {
      clearTimers();
      dispatch({
        type: "ADD_ANALYSIS",
        analysis: {
          id: uid("a"),
          monthId: month.id,
          createdAt: new Date().toISOString(),
          source: "api",
          llm_ok: a.llm_ok !== false,
          reason: a.trace?.refresh_reason || clientReason,
          brief: a.brief,
          trace: a.trace,
          narratives: a.narratives || null
        }
      });
      navigate("/advice");
    }, 700);
    timersRef.current.push(done);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.company, month, state.analyses, state.months, dispatch, navigate]);

  useEffect(() => {
    if (state.demo) { navigate("/advice"); return undefined; }
    if (!state.company || !month) { navigate("/"); return undefined; }
    run();
    return clearTimers;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (failed) {
    return (
      <section className="empty-state">
        <AlertTriangle size={40} className="warn-icon" />
        <h2>The analysis service couldn't be reached</h2>
        <p className="narrow">
          Your numbers are saved. The advisory board runs on the engine service
          (<code>/api/advise</code>), which isn't responding — start the backend and retry,
          or continue and analyse later. Nothing is made up in the meantime.
        </p>
        <div className="welcome-actions">
          <button className="primary-button" type="button" onClick={run}>Retry analysis</button>
          <button className="secondary-button" type="button" onClick={() => navigate("/home")}>
            Continue without analysis
          </button>
        </div>
        <Banner tone="info">
          Want to see what a finished analysis looks like meanwhile? Open the sample
          company from the welcome screen — it's clearly labelled and never mixes with your data.
        </Banner>
      </section>
    );
  }

  return (
    <section className="empty-state">
      <LoaderCircle size={40} className="spin" />
      <h2>Your advisory board is deliberating…</h2>
      <ProgressStages stage={stage} narrativesOn={narrativesOn} />
      <p className="elapsed">elapsed {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}</p>
      <p className="narrow subtle">You can leave this page; we'll keep your seat.</p>
    </section>
  );
}
