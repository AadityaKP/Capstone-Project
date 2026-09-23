// Founder-facing app shell (docs/founder_frontend_spec.md §8, §16).
// Hash-based routing keeps URLs shareable without adding a dependency;
// research-facing controls (policies, seeds, episodes, oracle modes) are
// deliberately absent from every founder surface (spec §19).

import React, { useCallback, useEffect, useState } from "react";
import {
  BrainCircuit, Building2, ChevronRight,
  History as HistoryIcon, LayoutDashboard, PencilLine,
  Settings as SettingsIcon
} from "lucide-react";
import "./styles.css";

import { StoreProvider, useStore, latestMonth } from "./store.jsx";
import { CycleRunProvider } from "./cycleRun.jsx";
import { daysSince } from "./derive.js";
import { DemoBadge } from "./components.jsx";

import Welcome from "./pages/Welcome.jsx";
import Onboarding from "./pages/Onboarding.jsx";
import Analyzing from "./pages/Analyzing.jsx";
import Home from "./pages/Home.jsx";
import Advice from "./pages/Advice.jsx";
import History from "./pages/History.jsx";
import { CompanyView, UpdateRitual } from "./pages/Company.jsx";
import Settings from "./pages/Settings.jsx";

// Four items (docs/ui_simplification_plan.md §3). The Plan page folded into
// This month; /advice/:id is "Why this plan", reached from the plan itself
// or from a History entry. Routes keep their old names (D5) so deep links
// and the seeded demo keep working; only the labels changed.
const NAV = [
  { id: "home", path: "/home", label: "This month", icon: LayoutDashboard },
  { id: "history", path: "/history", label: "History", icon: HistoryIcon },
  { id: "company", path: "/company", label: "My company", icon: Building2 },
  { id: "settings", path: "/settings", label: "Settings", icon: SettingsIcon }
];

const TITLES = {
  home: "This month", advice: "Why this plan", history: "History", company: "My company",
  settings: "Settings", update: "Close the month", onboarding: "Set up your company",
  analyzing: "Analysis", welcome: "Welcome"
};

function useHashRoute() {
  const read = () => (window.location.hash.replace(/^#/, "") || "/");
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const onChange = () => setRoute(read());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const navigate = useCallback((path) => {
    if (`#${path}` === window.location.hash) return;
    window.location.hash = path;
    window.scrollTo(0, 0);
  }, []);
  return [route, navigate];
}

function parseRoute(route) {
  const parts = route.split("/").filter(Boolean);
  if (parts.length === 0) return { page: "welcome", params: {} };
  if (parts[0] === "advice" && parts[1]) return { page: "advice", params: { id: parts[1] } };
  if (parts[0] === "advice" || parts[0] === "plan") return { page: "home", params: {} };
  // #/update/fill opens the Close form with the estimated numbers expanded.
  if (parts[0] === "update" && parts[1] === "fill") return { page: "update", params: { fill: true } };
  return { page: parts[0], params: {} };
}

export function Shell() {
  const { state } = useStore();
  const [route, navigate] = useHashRoute();
  const { page, params } = parseRoute(route);
  const hasCompany = !!state.company;

  // Route guard: no company → welcome/onboarding.
  useEffect(() => {
    if (!hasCompany && !["welcome", "onboarding"].includes(page)) navigate("/");
    if (hasCompany && page === "welcome") navigate("/home");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasCompany, page]);

  const bare = ["welcome", "onboarding", "analyzing"].includes(page) || !hasCompany;
  const month = latestMonth(state);
  const closeDue = month ? (daysSince(month.enteredAt) ?? 0) > 35 : false;

  const pages = {
    welcome: <Welcome navigate={navigate} />,
    onboarding: <Onboarding navigate={navigate} />,
    analyzing: <Analyzing navigate={navigate} />,
    home: <Home navigate={navigate} />,
    // Keyed on the analysis so expander state does not carry over between
    // one analysis and the next.
    advice: <Advice key={params.id || "latest"} navigate={navigate} params={params} />,
    history: <History navigate={navigate} />,
    company: <CompanyView navigate={navigate} />,
    update: <UpdateRitual navigate={navigate} params={params} />,
    settings: <Settings navigate={navigate} />
  };
  const content = pages[page] || pages.home;

  if (bare) {
    return (
      <main className="bare-shell">
        {state.demo && <div className="demo-strip"><DemoBadge /></div>}
        {content}
      </main>
    );
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark"><BrainCircuit size={25} /></div>
          <div>
            <strong>{state.company?.name || "Startup Society"}</strong>
            <span>AI advisory board</span>
          </div>
        </div>
        <nav className="nav-list" aria-label="Primary">
          {NAV.map(({ id, path, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${page === id ? "active" : ""}`}
              type="button"
              onClick={() => navigate(path)}
            >
              <Icon size={18} />
              <span>{label}</span>
              {page === id && <ChevronRight size={16} />}
            </button>
          ))}
        </nav>
      </aside>
      <section className="workspace">
        <header className="topbar">
          <div>
            <p>{TITLES[page] || TITLES.home}</p>
            <h1>{state.company?.name}</h1>
          </div>
          <div className="topbar-actions">
            {state.demo && <DemoBadge />}
            {/* Primary on age only: "not current" is true from the moment a
                founder closes until month 1 lands, and "failed" wants Re-run,
                not Close. Both belong to the notice slot on This month. */}
            {!state.demo && page !== "update" && (
              <button
                className={`${closeDue ? "primary-button" : "secondary-button"} small`}
                type="button"
                onClick={() => navigate("/update")}
              >
                <PencilLine size={14} /> Close the month
              </button>
            )}
          </div>
        </header>
        <div className="page-frame">{content}</div>
      </section>
    </main>
  );
}

export default function App() {
  return (
    <StoreProvider>
      <CycleRunProvider>
        <Shell />
      </CycleRunProvider>
    </StoreProvider>
  );
}
