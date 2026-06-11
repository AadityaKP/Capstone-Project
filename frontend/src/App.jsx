import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity, BarChart3, Bot, BrainCircuit, Building2, ChevronRight,
  CircleDollarSign, Clock3, Database, GitBranch, LayoutDashboard,
  LineChart, LoaderCircle, Play, Save, ShieldCheck, SlidersHorizontal,
  Sparkles, Target, TrendingUp, Users, Zap
} from "lucide-react";
import "./styles.css";

const navItems = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "setup", label: "Company setup", icon: SlidersHorizontal },
  { id: "oracle", label: "Oracle & memory", icon: Bot },
  { id: "shocks", label: "Shock tracker", icon: Zap },
  { id: "evaluation", label: "Evaluation", icon: BarChart3 }
];

const defaultScenario = {
  company_name: "Demo SaaS", startup_stage: "Series A", business_model: "B2B SaaS",
  market_segment: "SMB", initial_headcount: 1, initial_mrr: 50000,
  initial_cash: 1000000, average_price: 50, valuation_multiple: 10, cac: 50,
  ltv: 7000, churn_smb: 0.03, churn_enterprise: 0.01, churn_b2c: 0.05,
  interest_rate: 3, consumer_confidence: 100, competitors: 5, unemployment: 4,
  product_quality: 0.1, innovation_factor: 1, max_months: 120
};

const fieldGroups = [
  { title: "Company profile", icon: Building2, fields: [
    ["company_name", "Company name", "text"], ["startup_stage", "Startup stage", "text"],
    ["business_model", "Business model", "text"], ["market_segment", "Market segment", "text"],
    ["initial_headcount", "Initial headcount", "number"]
  ]},
  { title: "Financial KPIs", icon: CircleDollarSign, fields: [
    ["initial_mrr", "Monthly recurring revenue", "number"], ["initial_cash", "Cash balance", "number"],
    ["average_price", "Average price", "number"], ["valuation_multiple", "Valuation multiple", "number"]
  ]},
  { title: "Customer metrics", icon: Target, fields: [
    ["cac", "CAC", "number"], ["ltv", "LTV", "number"], ["churn_smb", "SMB churn", "number"],
    ["churn_enterprise", "Enterprise churn", "number"], ["churn_b2c", "B2C churn", "number"]
  ]},
  { title: "Market context", icon: Activity, fields: [
    ["interest_rate", "Interest rate", "number"], ["consumer_confidence", "Consumer confidence", "number"],
    ["competitors", "Competitors", "number"], ["unemployment", "Unemployment", "number"],
    ["product_quality", "Product quality", "number"], ["innovation_factor", "Innovation factor", "number"]
  ]}
];

async function api(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [health, setHealth] = useState(null);
  const [config, setConfig] = useState({ policies: ["boardroom"], default_policy: "boardroom" });
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);
  const [scenario, setScenario] = useState(defaultScenario);
  const [scenarioId, setScenarioId] = useState(null);
  const [policy, setPolicy] = useState("boardroom");
  const [episodes, setEpisodes] = useState(1);
  const [seedStart, setSeedStart] = useState(0);
  const [oracleFrequency, setOracleFrequency] = useState(5);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");

  const refresh = useCallback(async (runId = selectedRun?.id) => {
    const [healthData, configData, runsData] = await Promise.all([
      api("/api/health"), api("/api/config"), api("/api/runs")
    ]);
    setHealth(healthData);
    setConfig(configData);
    setRuns(runsData);
    if (runId) {
      const detail = await api(`/api/runs/${runId}?include_trace=true`);
      setSelectedRun(detail);
      setBusy(["queued", "running"].includes(detail.status));
      return detail;
    }
    const latest = runsData.find((run) => run.status === "completed");
    if (latest) {
      const detail = await api(`/api/runs/${latest.id}?include_trace=true`);
      setSelectedRun(detail);
    }
    return null;
  }, [selectedRun?.id]);

  useEffect(() => {
    refresh().catch((error) => setNotice(error.message));
  }, []);

  useEffect(() => {
    if (!selectedRun || !["queued", "running"].includes(selectedRun.status)) return undefined;
    const timer = window.setInterval(() => {
      refresh(selectedRun.id).catch((error) => setNotice(error.message));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [selectedRun?.id, selectedRun?.status, refresh]);

  async function saveScenario() {
    setBusy(true);
    setNotice("");
    try {
      const saved = await api("/api/scenarios", {
        method: "POST",
        body: JSON.stringify({ name: scenario.company_name, config: scenario })
      });
      setScenarioId(saved.id);
      setNotice(`Scenario "${saved.name}" saved`);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function launchRun() {
    setBusy(true);
    setNotice("");
    try {
      let activeScenarioId = scenarioId;
      if (!activeScenarioId) {
        const saved = await api("/api/scenarios", {
          method: "POST",
          body: JSON.stringify({ name: scenario.company_name, config: scenario })
        });
        activeScenarioId = saved.id;
        setScenarioId(saved.id);
      }
      const run = await api("/api/runs", {
        method: "POST",
        body: JSON.stringify({
          scenario_id: activeScenarioId, policy, episodes: Number(episodes),
          seed_start: Number(seedStart), oracle_frequency: Number(oracleFrequency)
        })
      });
      setSelectedRun(run);
      setActivePage("dashboard");
      setNotice("Simulation queued");
    } catch (error) {
      setBusy(false);
      setNotice(error.message);
    }
  }

  const pageTitle = navItems.find((item) => item.id === activePage)?.label;
  const context = {
    health, config, runs, selectedRun, scenario, setScenario, policy, setPolicy,
    episodes, setEpisodes, seedStart, setSeedStart, oracleFrequency, setOracleFrequency,
    busy, notice, saveScenario, launchRun, selectRun: async (id) => {
      setBusy(true);
      try { await refresh(id); } catch (error) { setNotice(error.message); }
      finally { setBusy(false); }
    }
  };

  return (
    <main className="app-shell">
      <Sidebar activePage={activePage} onNavigate={setActivePage} run={selectedRun} />
      <section className="workspace">
        <Topbar pageTitle={pageTitle} {...context} />
        <div className="page-frame">
          {notice && <div className="notice">{notice}</div>}
          {activePage === "dashboard" && <Dashboard {...context} />}
          {activePage === "setup" && <CompanySetup {...context} />}
          {activePage === "oracle" && <OracleMemory {...context} />}
          {activePage === "shocks" && <ShockTracker {...context} />}
          {activePage === "evaluation" && <Evaluation {...context} />}
        </div>
      </section>
    </main>
  );
}

function Sidebar({ activePage, onNavigate, run }) {
  return (
    <aside className="sidebar">
      <div className="brand-lockup"><div className="brand-mark"><BrainCircuit size={25} /></div>
        <div><strong>Startup Society</strong><span>of Minds</span></div>
      </div>
      <nav className="nav-list" aria-label="Primary">
        {navItems.map(({ id, label, icon: Icon }) => (
          <button key={id} className={`nav-item ${activePage === id ? "active" : ""}`}
            type="button" onClick={() => onNavigate(id)}>
            <Icon size={18} /><span>{label}</span><ChevronRight size={16} />
          </button>
        ))}
      </nav>
      <div className="sidebar-status">
        <span>Simulation status</span>
        <strong>{run?.status || "System ready"}</strong>
        <small>{run ? `${run.policy} - ${run.episodes} episode(s)` : "No run selected"}</small>
      </div>
    </aside>
  );
}

function Topbar({ pageTitle, policy, setPolicy, config, busy, launchRun, health }) {
  return (
    <header className="topbar">
      <div><p>{pageTitle}</p><h1>Startup Society of Minds</h1></div>
      <div className="topbar-actions">
        <select value={policy} onChange={(event) => setPolicy(event.target.value)} aria-label="Policy">
          {config.policies.map((item) => <option key={item}>{item}</option>)}
        </select>
        <span className={`status-pill ${health ? "success" : "neutral"}`}>
          {health ? "API connected" : "Connecting"}
        </span>
        <button className="primary-button" type="button" onClick={launchRun} disabled={busy}>
          {busy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />} Run simulation
        </button>
      </div>
    </header>
  );
}

function Dashboard({ selectedRun }) {
  if (!selectedRun) return <EmptyState />;
  if (selectedRun.status !== "completed") return <RunProgress run={selectedRun} />;
  const summary = selectedRun.summary || {};
  const kpis = [
    ["Final MRR", money(summary.final_mrr), "green"],
    ["Shock recovery time", months(summary.recovery_time_months), "blue"],
    ["Survival rate", percent(summary.survival_rate), "green"],
    ["Post-shock Rule-40", decimal(summary.post_shock_rule_40), "amber"],
    ["LTV / CAC ratio", `${decimal(summary.final_ltv_cac)}x`, "green"]
  ];
  return (
    <section className="content-stack">
      <div className="section-heading"><div><h2>Dashboard</h2><p>Persisted results from {selectedRun.policy}</p></div></div>
      <div className="kpi-grid">
        {kpis.map(([label, value, tone]) => <article className="kpi-card" key={label}>
          <strong className={`metric-value ${tone}`}>{value}</strong><span>{label}</span>
          <em className={`delta-pill ${tone}`}>Simulation result</em>
        </article>)}
      </div>
      <div className="two-column"><TrajectoryChart trace={selectedRun.monthly_trace || []} />
        <AgentStatus run={selectedRun} /></div>
    </section>
  );
}

function TrajectoryChart({ trace }) {
  const rows = trace.filter((row) => row.episode === 0);
  const maxMrr = Math.max(...rows.map((row) => row.mrr || 0), 1);
  const maxMonth = Math.max(...rows.map((row) => row.month || 0), 1);
  const points = rows.map((row) => `${40 + (row.month / maxMonth) * 550},${220 - (row.mrr / maxMrr) * 180}`).join(" ");
  return (
    <article className="panel">
      <div className="panel-title"><LineChart size={18} /><div><h3>MRR over time</h3>
        <p>Episode 1 trajectory with detected shocks</p></div></div>
      <div className="chart-box"><svg viewBox="0 0 620 260" role="img" aria-label="MRR trajectory">
        <path className="grid-line" d="M40 40 H590 M40 95 H590 M40 150 H590 M40 205 H590" />
        <polyline className="oracle-line" points={points} />
        {rows.filter((row) => row.shock_label !== "NO_SHOCK").map((row) => {
          const x = 40 + (row.month / maxMonth) * 550;
          return <line className="shock-line" key={row.month} x1={x} x2={x} y1="34" y2="220" />;
        })}
      </svg></div>
      <div className="legend-row"><span><i className="legend oracle" />MRR</span>
        <span><i className="legend shock" />Shock event</span></div>
    </article>
  );
}

function AgentStatus({ run }) {
  const agents = [["CFO", "Preserve runway and efficiency", "violet"],
    ["CMO", "Optimize customer acquisition", "teal"], ["CPO", "Improve product and retention", "blue"]];
  return <article className="panel"><div className="panel-title"><Users size={18} /><div>
    <h3>Boardroom agents</h3><p>Policy participants for this run</p></div></div>
    <div className="agent-list">{agents.map(([role, action, color]) => <div className="agent-row" key={role}>
      <span className={`role-badge ${color}`}>{role}</span><p>{action}</p><strong>Active</strong></div>)}</div>
    <div className="soft-note">Policy: {run.policy} - Oracle cadence: {run.oracle_frequency} months</div>
  </article>;
}

function CompanySetup(props) {
  const { scenario, setScenario, episodes, setEpisodes, seedStart, setSeedStart,
    oracleFrequency, setOracleFrequency, saveScenario, busy } = props;
  const update = (key, type, value) => setScenario((current) => ({
    ...current, [key]: type === "number" ? Number(value) : value
  }));
  return <section className="content-stack">
    <div className="section-heading"><div><h2>Company setup</h2>
      <p>These values initialize the real simulation environment</p></div>
      <button className="secondary-button" type="button" onClick={saveScenario} disabled={busy}>
        <Save size={16} /> Save scenario</button></div>
    <div className="form-grid">{fieldGroups.map(({ title, icon: Icon, fields }) =>
      <article className="panel setup-panel" key={title}><div className="panel-title"><Icon size={18} /><h3>{title}</h3></div>
        <div className="field-grid">{fields.map(([key, label, type]) => <label key={key}>
          <span>{label}</span><input type={type} step={type === "number" ? "any" : undefined}
            value={scenario[key]} onChange={(event) => update(key, type, event.target.value)} /></label>)}</div>
      </article>)}
      <article className="panel setup-panel wide"><div className="panel-title"><Clock3 size={18} /><h3>Run configuration</h3></div>
        <div className="field-grid"><NumberField label="Max months" value={scenario.max_months}
          onChange={(value) => update("max_months", "number", value)} />
          <NumberField label="Episodes" value={episodes} onChange={setEpisodes} />
          <NumberField label="Oracle frequency" value={oracleFrequency} onChange={setOracleFrequency} />
          <NumberField label="Random seed" value={seedStart} onChange={setSeedStart} /></div>
      </article>
    </div>
  </section>;
}

function NumberField({ label, value, onChange }) {
  return <label><span>{label}</span><input type="number" value={value}
    onChange={(event) => onChange(Number(event.target.value))} /></label>;
}

function OracleMemory({ selectedRun }) {
  const summary = selectedRun?.summary || {};
  const brief = summary.latest_brief;
  const memories = summary.memories || [];
  return <section className="content-stack"><div className="section-heading"><div><h2>Oracle & memory</h2>
    <p>Live recommendation and ChromaDB retrieval output</p></div></div>
    <div className="two-column wide-left"><article className="panel oracle-brief">
      <div className="panel-title"><Sparkles size={18} /><div><h3>Latest Oracle brief</h3>
        <p>{brief ? "Generated by the selected Oracle policy" : "No Oracle brief for this run"}</p></div></div>
      <div className="brief-body"><strong>{brief?.growth_outlook || "Boardroom policy active"}</strong>
        <p>{brief?.rationale || brief?.summary || "Choose an Oracle policy and run a simulation to generate a strategic brief."}</p></div>
      {brief && <div className="modifier-grid"><span>Risk: {brief.risk_level}</span>
        <span>Efficiency: {brief.efficiency_pressure}</span><span>Confidence: {decimal(brief.confidence)}</span></div>}
    </article><article className="panel"><div className="panel-title"><Database size={18} /><div>
      <h3>Retrieved memories</h3><p>{memories.length} persisted matches</p></div></div>
      <div className="memory-list">{memories.length ? memories.map((memory, index) =>
        <div className="memory-card" key={`${memory.document}-${index}`}><strong>Memory {index + 1}</strong>
          <span>Similarity {decimal(memory.similarity_score)}</span><p>{memory.document}</p></div>) :
        <p className="empty-copy">No memories were retrieved for this run.</p>}</div></article></div>
  </section>;
}

function ShockTracker({ selectedRun }) {
  const shocks = selectedRun?.summary?.shock_events || [];
  return <section className="content-stack"><div className="section-heading"><div><h2>Shock tracker</h2>
    <p>Events emitted by the simulation engine</p></div></div>
    <div className="shock-grid">{shocks.length ? shocks.map((shock, index) =>
      <article className="shock-card" key={`${shock.episode}-${shock.month}-${index}`}>
        <span className="event-dot red" /><strong>{shock.type}</strong><p>Month {shock.month}</p>
        <div><em>Episode {shock.episode + 1}</em><span>{money(shock.mrr)} MRR</span></div>
      </article>) : <p className="empty-copy">No shock events are available for this run.</p>}</div>
    <article className="panel graph-panel"><div className="panel-title"><GitBranch size={18} /><div>
      <h3>Causal chain</h3><p>Observed shock to response flow</p></div></div>
      <div className="graph-flow"><span>Shock detected</span><ChevronRight size={18} />
        <span>Boardroom decision</span><ChevronRight size={18} /><span>Environment step</span>
        <ChevronRight size={18} /><span>Outcome persisted</span></div></article>
  </section>;
}

function Evaluation({ runs, selectRun, selectedRun, busy }) {
  return <section className="content-stack"><div className="section-heading"><div><h2>Evaluation</h2>
    <p>Completed and active simulation runs</p></div></div>
    <article className="panel table-panel"><table><thead><tr><th>Policy</th><th>Status</th>
      <th>Final MRR</th><th>Survival</th><th>Created</th><th /></tr></thead>
      <tbody>{runs.map((run) => <tr key={run.id} className={selectedRun?.id === run.id ? "selected-row" : ""}>
        <td>{run.policy}</td><td><span className={`run-status ${run.status}`}>{run.status}</span></td>
        <td>{money(run.summary?.final_mrr)}</td><td>{percent(run.summary?.survival_rate)}</td>
        <td>{new Date(run.created_at).toLocaleString()}</td><td><button className="icon-button" type="button"
          title="Open run" disabled={busy} onClick={() => selectRun(run.id)}><ChevronRight size={17} /></button></td>
      </tr>)}</tbody></table></article>
    <div className="two-column"><MiniMetric title="Results persisted in SQLite" icon={Database} />
      <MiniMetric title="Simulation engine connected" icon={TrendingUp} /></div>
  </section>;
}

function MiniMetric({ title, icon: Icon }) {
  return <article className="panel mini-metric"><div className="panel-title"><Icon size={18} /><h3>{title}</h3></div>
    <div className="system-check"><ShieldCheck size={34} /><strong>Operational</strong></div></article>;
}

function EmptyState() {
  return <section className="empty-state"><BrainCircuit size={42} /><h2>Ready to simulate</h2>
    <p>Configure the company, choose a policy, and run the integrated simulation.</p></section>;
}

function RunProgress({ run }) {
  return <section className="empty-state"><LoaderCircle className="spin" size={42} /><h2>Simulation {run.status}</h2>
    <p>The backend is executing {run.episodes} episode(s) with the {run.policy} policy.</p></section>;
}

function money(value) {
  if (value == null) return "-";
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 2 }).format(value);
}
function percent(value) { return value == null ? "-" : `${Number(value).toFixed(1)}%`; }
function decimal(value) { return value == null ? "-" : Number(value).toFixed(2); }
function months(value) { return value == null ? "N/A" : `${Number(value).toFixed(1)}mo`; }

export default App;
createRoot(document.getElementById("root")).render(<React.StrictMode><App /></React.StrictMode>);
