import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  Bot,
  BrainCircuit,
  Building2,
  ChevronRight,
  CircleDollarSign,
  Clock3,
  Database,
  GitBranch,
  LayoutDashboard,
  LineChart,
  Play,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
  Users,
  Zap
} from "lucide-react";
import "./styles.css";

const navItems = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "setup", label: "Company setup", icon: SlidersHorizontal },
  { id: "oracle", label: "Oracle & memory", icon: Bot },
  { id: "shocks", label: "Shock tracker", icon: Zap },
  { id: "evaluation", label: "Evaluation", icon: BarChart3 }
];

const kpis = [
  { label: "Final MRR", value: "$2.35M", delta: "+69% vs baseline", tone: "green" },
  { label: "Shock recovery time", value: "4.61mo", delta: "-48% vs baseline", tone: "blue" },
  { label: "Survival rate", value: "100%", delta: "vs 97.3% baseline", tone: "green" },
  { label: "Post-shock Rule-40", value: "-36.7", delta: "+8.4pp improvement", tone: "amber" },
  { label: "LTV / CAC ratio", value: "5.6x", delta: "Healthy", tone: "green" }
];

const agents = [
  { role: "CFO", action: "Preserve runway & Rule-of-40", color: "violet" },
  { role: "CMO", action: "Grow user acquisition via PPC", color: "teal" },
  { role: "CPO", action: "Reduce churn via R&D investment", color: "blue" }
];

const memories = [
  {
    title: "Rate hike response",
    meta: "Month 24 - Similarity 0.91 - Positive outcome",
    text: "Lowered paid acquisition, preserved burn multiple, and recovered MRR trend within five months."
  },
  {
    title: "Competitor surge",
    meta: "Month 48 - Similarity 0.87 - Mixed outcome",
    text: "Brand spend alone lagged. Best recovery came after product quality and enterprise pricing moved together."
  },
  {
    title: "Demand recession",
    meta: "Month 72 - Similarity 0.82 - Positive outcome",
    text: "Short-term CAC reduction plus retention work stabilized cash before returning to growth mode."
  }
];

const shocks = [
  { type: "competitor_surge", month: 24, severity: "High", status: "Recovered", tone: "red" },
  { type: "rate_hike", month: 48, severity: "Medium", status: "Stabilized", tone: "amber" },
  { type: "recession", month: 72, severity: "High", status: "Monitoring", tone: "blue" }
];

const policies = [
  { policy: "oracle_v4_causal", mrr: "$2.35M", recovery: "4.61mo", survival: "100%", confidence: "0.84" },
  { policy: "oracle_v4", mrr: "$2.04M", recovery: "6.72mo", survival: "98.1%", confidence: "0.76" },
  { policy: "boardroom baseline", mrr: "$1.39M", recovery: "8.88mo", survival: "97.3%", confidence: "0.62" },
  { policy: "random policy", mrr: "$0.72M", recovery: "11.4mo", survival: "81.5%", confidence: "0.28" }
];

function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const pageTitle = useMemo(
    () => navItems.find((item) => item.id === activePage)?.label ?? "Dashboard",
    [activePage]
  );

  return (
    <main className="app-shell">
      <Sidebar activePage={activePage} onNavigate={setActivePage} />
      <section className="workspace">
        <Topbar pageTitle={pageTitle} />
        <div className="page-frame">
          {activePage === "dashboard" && <Dashboard />}
          {activePage === "setup" && <CompanySetup />}
          {activePage === "oracle" && <OracleMemory />}
          {activePage === "shocks" && <ShockTracker />}
          {activePage === "evaluation" && <Evaluation />}
        </div>
      </section>
    </main>
  );
}

function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="brand-lockup">
        <div className="brand-mark">
          <BrainCircuit size={25} />
        </div>
        <div>
          <strong>Startup Society</strong>
          <span>of Minds</span>
        </div>
      </div>

      <nav className="nav-list" aria-label="Primary">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={`nav-item ${activePage === item.id ? "active" : ""}`}
              type="button"
              onClick={() => onNavigate(item.id)}
            >
              <Icon size={18} />
              <span>{item.label}</span>
              {activePage === item.id && <ChevronRight size={16} />}
            </button>
          );
        })}
      </nav>

      <div className="sidebar-status">
        <span>Simulation status</span>
        <strong>System ready</strong>
        <small>oracle_v4_causal - every 5 months</small>
      </div>
    </aside>
  );
}

function Topbar({ pageTitle }) {
  return (
    <header className="topbar">
      <div>
        <p>{pageTitle}</p>
        <h1>Startup Society of Minds</h1>
      </div>
      <div className="topbar-actions">
        <span className="status-pill neutral">oracle_v4_causal</span>
        <span className="status-pill success">System ready</span>
        <button className="primary-button" type="button">
          <Play size={16} />
          Run new simulation
        </button>
      </div>
    </header>
  );
}

function Dashboard() {
  return (
    <section className="content-stack">
      <div className="section-heading">
        <div>
          <h2>Dashboard</h2>
          <p>Live simulation overview</p>
        </div>
      </div>

      <div className="kpi-grid">
        {kpis.map((kpi) => (
          <article className="kpi-card" key={kpi.label}>
            <strong className={`metric-value ${kpi.tone}`}>{kpi.value}</strong>
            <span>{kpi.label}</span>
            <em className={`delta-pill ${kpi.tone}`}>{kpi.delta}</em>
          </article>
        ))}
      </div>

      <div className="two-column">
        <ChartPanel />
        <AgentStatus />
      </div>
    </section>
  );
}

function ChartPanel() {
  return (
    <article className="panel">
      <div className="panel-title">
        <LineChart size={18} />
        <div>
          <h3>MRR Over Time</h3>
          <p>120-month simulation - Shocks at months 24, 48, 72</p>
        </div>
      </div>
      <div className="chart-box" aria-label="MRR trajectory chart placeholder">
        <svg viewBox="0 0 620 260" role="img">
          <path className="grid-line" d="M40 40 H590 M40 95 H590 M40 150 H590 M40 205 H590" />
          <path className="baseline-line" d="M48 208 C140 190 210 172 282 146 S432 118 585 96" />
          <path className="oracle-line" d="M48 218 C126 196 188 162 250 140 S370 74 585 48" />
          <g className="shock-markers">
            <line x1="172" y1="34" x2="172" y2="220" />
            <line x1="318" y1="34" x2="318" y2="220" />
            <line x1="465" y1="34" x2="465" y2="220" />
          </g>
        </svg>
      </div>
      <div className="legend-row">
        <span><i className="legend oracle" />oracle_v4_causal</span>
        <span><i className="legend baseline" />boardroom baseline</span>
      </div>
    </article>
  );
}

function AgentStatus() {
  return (
    <article className="panel">
      <div className="panel-title">
        <Users size={18} />
        <div>
          <h3>Current agent status</h3>
          <p>C-suite modifiers generated from the Oracle brief</p>
        </div>
      </div>
      <div className="agent-list">
        {agents.map((agent) => (
          <div className="agent-row" key={agent.role}>
            <span className={`role-badge ${agent.color}`}>{agent.role}</span>
            <p>{agent.action}</p>
            <strong>Active</strong>
          </div>
        ))}
      </div>
      <div className="soft-note">
        Oracle mode: oracle_v4_causal - Frequency: every 5 months - Confidence: 0.84
      </div>
    </article>
  );
}

function CompanySetup() {
  return (
    <section className="content-stack">
      <div className="section-heading">
        <div>
          <h2>Company Setup</h2>
          <p>Inputs used by the startup environment and action adapter</p>
        </div>
        <button className="secondary-button" type="button">
          <Settings2 size={16} />
          Save scenario
        </button>
      </div>

      <div className="form-grid">
        <SetupGroup title="Company profile" icon={Building2} fields={["Startup stage", "Business model", "Market segment", "Initial headcount"]} />
        <SetupGroup title="Financial KPIs" icon={CircleDollarSign} fields={["Monthly recurring revenue", "Cash balance", "Average price", "Valuation multiple"]} />
        <SetupGroup title="Customer metrics" icon={Target} fields={["CAC", "LTV", "SMB churn", "Enterprise churn"]} />
        <SetupGroup title="Market context" icon={Activity} fields={["Interest rate", "Consumer confidence", "Competitors", "Unemployment"]} />
        <SetupGroup title="Simulation config" icon={Clock3} fields={["Max months", "Oracle frequency", "Shock schedule", "Random seed"]} wide />
      </div>
    </section>
  );
}

function SetupGroup({ title, icon: Icon, fields, wide = false }) {
  return (
    <article className={`panel setup-panel ${wide ? "wide" : ""}`}>
      <div className="panel-title">
        <Icon size={18} />
        <h3>{title}</h3>
      </div>
      <div className="field-grid">
        {fields.map((field, index) => (
          <label key={field}>
            <span>{field}</span>
            <input defaultValue={index % 2 === 0 ? "Auto-filled" : "Editable"} />
          </label>
        ))}
      </div>
    </article>
  );
}

function OracleMemory() {
  return (
    <section className="content-stack">
      <div className="section-heading">
        <div>
          <h2>Oracle & Memory</h2>
          <p>Boardroom recommendation, ChromaDB retrieval, and causal evidence</p>
        </div>
      </div>

      <div className="two-column wide-left">
        <article className="panel oracle-brief">
          <div className="panel-title">
            <Sparkles size={18} />
            <div>
              <h3>Oracle brief</h3>
              <p>Recommended action bundle for the current month</p>
            </div>
          </div>
          <div className="brief-body">
            <strong>Preserve runway while recovering growth.</strong>
            <p>
              Reduce paid acquisition intensity during the shock window, increase product R&D to protect retention,
              and keep pricing flat until confidence recovers.
            </p>
          </div>
          <div className="modifier-grid">
            <span>CFO: spend guardrail 0.72</span>
            <span>CMO: PPC weight 0.48</span>
            <span>CPO: R&D weight 0.81</span>
          </div>
        </article>

        <article className="panel">
          <div className="panel-title">
            <Database size={18} />
            <div>
              <h3>Retrieved memories</h3>
              <p>Top 3 ChromaDB matches</p>
            </div>
          </div>
          <div className="memory-list">
            {memories.map((memory) => (
              <div className="memory-card" key={memory.title}>
                <strong>{memory.title}</strong>
                <span>{memory.meta}</span>
                <p>{memory.text}</p>
              </div>
            ))}
          </div>
        </article>
      </div>
    </section>
  );
}

function ShockTracker() {
  return (
    <section className="content-stack">
      <div className="section-heading">
        <div>
          <h2>Shock Tracker</h2>
          <p>Macro and competitive events with Neo4j causal graph summaries</p>
        </div>
      </div>

      <div className="shock-grid">
        {shocks.map((shock) => (
          <article className="shock-card" key={shock.type}>
            <span className={`event-dot ${shock.tone}`} />
            <strong>{shock.type}</strong>
            <p>Month {shock.month}</p>
            <div>
              <em>{shock.severity}</em>
              <span>{shock.status}</span>
            </div>
          </article>
        ))}
      </div>

      <article className="panel graph-panel">
        <div className="panel-title">
          <GitBranch size={18} />
          <div>
            <h3>Neo4j causal graph summary</h3>
            <p>Shock to action to outcome chain</p>
          </div>
        </div>
        <div className="graph-flow">
          <span>Rate hike</span>
          <ChevronRight size={18} />
          <span>Lower CAC spend</span>
          <ChevronRight size={18} />
          <span>Preserved cash</span>
          <ChevronRight size={18} />
          <span>Recovered MRR</span>
        </div>
      </article>
    </section>
  );
}

function Evaluation() {
  return (
    <section className="content-stack">
      <div className="section-heading">
        <div>
          <h2>Evaluation</h2>
          <p>Policy comparison table and experiment chart placeholders</p>
        </div>
      </div>

      <article className="panel table-panel">
        <table>
          <thead>
            <tr>
              <th>Policy</th>
              <th>Final MRR</th>
              <th>Recovery</th>
              <th>Survival</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {policies.map((row) => (
              <tr key={row.policy}>
                <td>{row.policy}</td>
                <td>{row.mrr}</td>
                <td>{row.recovery}</td>
                <td>{row.survival}</td>
                <td>{row.confidence}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </article>

      <div className="two-column">
        <MiniMetric title="Global reward over time" icon={TrendingUp} tone="green" />
        <MiniMetric title="Recovery distribution" icon={TrendingDown} tone="amber" />
      </div>
    </section>
  );
}

function MiniMetric({ title, icon: Icon, tone }) {
  return (
    <article className="panel mini-metric">
      <div className="panel-title">
        <Icon size={18} />
        <h3>{title}</h3>
      </div>
      <div className={`mini-chart ${tone}`} />
      <div className="soft-note">
        <ShieldCheck size={15} />
        Significant improvement across prioritized thesis runs
      </div>
    </article>
  );
}

export default App;

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
