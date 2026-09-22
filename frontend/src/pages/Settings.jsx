// S12 Settings — boring by design (spec §8): narrative toggle, service status,
// sample-mode exit, data reset. No accounts in MVP (spec gap G10).

import React, { useEffect, useState } from "react";
import { CheckCircle2, CircleOff, FlaskConical, Trash2 } from "lucide-react";
import { useStore } from "../store.jsx";
import { health } from "../api.js";
import { Banner } from "../components.jsx";

// One capability line: what it is, whether it is on, and the server's own
// reason. The reason is shown verbatim because the failure modes behind these
// are silent inside the engine and paraphrasing them would hide the detail.
function Capability({ on, label, reason }) {
  return (
    <li className={`cap-row ${on ? "on" : "off"}`}>
      {on ? <CheckCircle2 size={15} className="ok-icon" /> : <CircleOff size={15} className="warn-icon" />}
      <span>
        <strong>{label}</strong>
        <em>{reason}</em>
      </span>
    </li>
  );
}

export default function Settings({ navigate }) {
  const { state, dispatch } = useStore();
  const [apiUp, setApiUp] = useState(null);
  const [loop, setLoop] = useState(null);
  const [confirmReset, setConfirmReset] = useState(false);

  useEffect(() => {
    let alive = true;
    health().then((r) => {
      if (!alive) return;
      setApiUp(r.ok);
      setLoop(r.ok ? r.data?.loop || null : null);
    });
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
        {/* The learning loop's three capabilities, each stated rather than
            assumed. A plan that claims the board learns while the graph is
            unreachable is the one thing this product must never show. */}
        {loop && (
          <>
            <p className="subtle cap-intro">
              What the board can actually do right now
              {loop.advisor_mode && <> · advisor mode <code>{loop.advisor_mode}</code></>}
            </p>
            <ul className="cap-list">
              <Capability
                on={loop.llm_reachable}
                label="Strategist (language model)"
                reason={loop.llm_reachable
                  ? "Fresh briefs and proposals each month."
                  : "Unreachable — plans come from the board's built-in rules and say so."}
              />
              <Capability
                on={loop.memory_store_enabled}
                label="Memory, scoped to your company"
                reason={loop.memory_store_enabled
                  ? "Each analysis can read what earlier ones learned about this company."
                  : "Off — nothing is remembered between analyses."}
              />
              <Capability
                on={loop.graph_store_enabled}
                label="Causal evidence graph"
                reason={loop.graph_store_enabled
                  ? "What happens after each plan is written back as evidence."
                  : `Off — ${loop.graph_store_reason}. The board still advises and remembers, but what happens next is not written back as evidence.`}
              />
            </ul>
          </>
        )}
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
