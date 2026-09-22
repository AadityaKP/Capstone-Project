// S4 Analysis in progress — starts the board's cycle and hands off.
//
// The cycle runs on the server (POST /api/cycles → 202) and its months land
// one by one on the Plan page, which polls for them. That is what makes
// "you can leave this page; we'll keep your seat" true: the run no longer
// lives in a page-local effect that navigating away would kill.
//
// When the engine API is unreachable the founder gets a truthful card, never
// a fabricated result.

import React, { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, LoaderCircle } from "lucide-react";
import { useStore, latestMonth, latestClosedFeedback } from "../store.jsx";
import { startCycle } from "../api.js";
import { ProgressStages, Banner } from "../components.jsx";

export const CYCLE_HORIZON = 4;

export default function Analyzing({ navigate }) {
  const { state, dispatch } = useStore();
  const [failed, setFailed] = useState(null);
  const runningRef = useRef(false);

  const month = latestMonth(state);
  const narrativesOn = !!state.settings.narratives;

  const run = useCallback(async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    setFailed(null);

    // The last close-the-month's record: the board starts this cycle already
    // knowing how its previous prediction went (plan section 6.2 item 5).
    const closed = latestClosedFeedback(state);
    const previousTrackRecord = closed?.feedback?.result?.track_record || null;

    const result = await startCycle(
      state.company,
      { ...month, history: state.months.slice(0, -1).map((m) => m.values) },
      { horizon: CYCLE_HORIZON, previousTrackRecord }
    );

    runningRef.current = false;
    if (!result.ok) {
      setFailed(result);
      return;
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
    navigate("/plan");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.company, month, state.months, state.cycles, dispatch, navigate]);

  useEffect(() => {
    if (state.demo) { navigate("/plan"); return; }
    if (!state.company || !month) { navigate("/"); return; }
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (failed) {
    return (
      <section className="empty-state">
        <AlertTriangle size={40} className="warn-icon" />
        <h2>The analysis service couldn't be reached</h2>
        <p className="narrow">
          Your numbers are saved. The advisory board runs on the engine service
          (<code>/api/cycles</code>), which isn't responding — start the backend and retry,
          or continue and plan later. Nothing is made up in the meantime.
        </p>
        {failed.error && !failed.offline && <p className="narrow subtle">{failed.error}</p>}
        <div className="welcome-actions">
          <button className="primary-button" type="button" onClick={run}>Retry</button>
          <button className="secondary-button" type="button" onClick={() => navigate("/home")}>
            Continue without a plan
          </button>
        </div>
        <Banner tone="info">
          Want to see what a finished plan looks like meanwhile? Open the sample
          company from the welcome screen — it's clearly labelled and never mixes with your data.
        </Banner>
      </section>
    );
  }

  return (
    <section className="empty-state">
      <LoaderCircle size={40} className="spin" />
      <h2>Convening your advisory board…</h2>
      <ProgressStages stage={0} narrativesOn={narrativesOn} />
      <p className="narrow subtle">
        The board deliberates one month at a time and each month appears on the Plan page as
        it lands. You can leave this page; the cycle keeps running on the engine.
      </p>
    </section>
  );
}
