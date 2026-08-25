// S12 Settings — boring by design (spec §8): narrative toggle, service status,
// sample-mode exit, data reset. No accounts in MVP (spec gap G10).

import React, { useEffect, useState } from "react";
import { CheckCircle2, CircleOff, FlaskConical, Trash2 } from "lucide-react";
import { useStore } from "../store.jsx";
import { health } from "../api.js";
import { Banner } from "../components.jsx";

export default function Settings({ navigate }) {
  const { state, dispatch } = useStore();
  const [apiUp, setApiUp] = useState(null);
  const [confirmReset, setConfirmReset] = useState(false);

  useEffect(() => {
    let alive = true;
    health().then((r) => { if (alive) setApiUp(r.ok); });
    return () => { alive = false; };
  }, []);

  return (
    <section className="content-stack narrow-col">
      <article className="panel">
        <h3>Advice</h3>
        <label className="toggle-row">
          <input
            type="checkbox"
            checked={!!state.settings.narratives}
            onChange={(e) => dispatch({ type: "SET_SETTING", key: "narratives", value: e.target.checked })}
          />
          <span>
            <strong>Richer explanations from each advisor</strong>
            <em>Each recommendation carries its advisor's own two-sentence reasoning. Analysis takes about a minute longer.</em>
          </span>
        </label>
      </article>

      <article className="panel">
        <h3>Analysis service</h3>
        <p className="status-line">
          {apiUp == null ? "Checking…" : apiUp
            ? (<><CheckCircle2 size={16} className="ok-icon" /> Connected — analyses run against the engine.</>)
            : (<><CircleOff size={16} className="warn-icon" /> Not reachable. Data entry and history work; analyses need the engine service at <code>/api</code>.</>)}
        </p>
      </article>

      {state.demo && (
        <article className="panel">
          <h3>Sample company</h3>
          <p className="subtle"><FlaskConical size={14} /> You're exploring illustrative data. Leaving returns you to your own workspace.</p>
          <button className="secondary-button" type="button" onClick={() => { dispatch({ type: "EXIT_DEMO" }); navigate("/"); }}>
            Leave sample company
          </button>
        </article>
      )}

      {!state.demo && state.company && (
        <article className="panel danger-panel">
          <h3>Your data</h3>
          <p className="subtle">Everything lives in this browser. Deleting removes your company, months and analyses permanently.</p>
          {!confirmReset ? (
            <button className="secondary-button" type="button" onClick={() => setConfirmReset(true)}>
              <Trash2 size={15} /> Delete all my data
            </button>
          ) : (
            <Banner tone="warn" actions={
              <>
                <button className="secondary-button small" type="button" onClick={() => setConfirmReset(false)}>Keep it</button>
                <button className="danger-button small" type="button" onClick={() => { dispatch({ type: "RESET_ALL" }); navigate("/"); }}>
                  Delete permanently
                </button>
              </>
            }>
              This removes {state.company.name} and {state.months.length} month{state.months.length === 1 ? "" : "s"} of history. There is no undo.
            </Banner>
          )}
        </article>
      )}

      <p className="subtle center">
        Advice is decision support from a calibrated simulation — not financial advice, and
        not a forecast of your company.
      </p>
    </section>
  );
}
