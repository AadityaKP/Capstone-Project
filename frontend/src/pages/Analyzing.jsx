// S4 Analysis in progress — starts the board's cycle and hands off.
//
// The cycle runs on the server (POST /api/cycles → 202) and its months land
// one by one; the app-level CycleRunProvider polls for them, so "you can
// leave this page; we'll keep your seat" is true regardless of which page
// the founder is on.
//
// When the engine API is unreachable the founder gets a truthful card, never
// a fabricated result.

import React, { useEffect, useRef } from "react";
import { AlertTriangle, LoaderCircle } from "lucide-react";
import { useStore, latestMonth } from "../store.jsx";
import { useCycleRun } from "../cycleRun.jsx";
import { ProgressStages, Banner } from "../components.jsx";

export { CYCLE_HORIZON } from "../cycleRun.jsx";

export default function Analyzing({ navigate }) {
  const { state } = useStore();
  const { start, startError } = useCycleRun();
  const month = latestMonth(state);
  const narrativesOn = !!state.settings.narratives;
  const ranRef = useRef(false);

  async function run() {
    const result = await start();
    if (result.ok) navigate("/plan");
  }

  useEffect(() => {
    if (state.demo) { navigate("/plan"); return; }
    if (!state.company || !month) { navigate("/"); return; }
    if (ranRef.current) return;
    ranRef.current = true;
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (startError) {
    return (
      <section className="empty-state">
        <AlertTriangle size={40} className="warn-icon" />
        <h2>The analysis service couldn't be reached</h2>
        <p className="narrow">
          Your numbers are saved. The advisory board runs on the engine service
          (<code>/api/cycles</code>), which isn't responding — start the backend and retry,
          or continue and plan later. Nothing is made up in the meantime.
        </p>
        {startError.error && !startError.offline && <p className="narrow subtle">{startError.error}</p>}
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
