// Founder-facing app shell (docs/founder_frontend_spec.md §8, §16).
// Hash-based routing keeps URLs shareable without adding a dependency;
// research-facing controls (policies, seeds, episodes, oracle modes) are
// deliberately absent from every founder surface (spec §19).

import React, { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BrainCircuit, Building2, ChevronRight, History as HistoryIcon,
  LayoutDashboard, PencilLine, Settings as SettingsIcon, Sparkles
} from "lucide-react";
import "./styles.css";

import { StoreProvider, useStore, latestMonth } from "./store.jsx";
import { dateLabel } from "./derive.js";
import { DemoBadge } from "./components.jsx";

import Welcome from "./pages/Welcome.jsx";
import Onboarding from "./pages/Onboarding.jsx";
import Analyzing from "./pages/Analyzing.jsx";
import Home from "./pages/Home.jsx";
import Advice from "./pages/Advice.jsx";
import History from "./pages/History.jsx";
import { CompanyView, UpdateRitual } from "./pages/Company.jsx";
import Settings from "./pages/Settings.jsx";

const NAV = [
  { id: "home", path: "/home", label: "Home", icon: LayoutDashboard },
  { id: "advice", path: "/advice", label: "Advice", icon: Sparkles },
  { id: "history", path: "/history", label: "History", icon: HistoryIcon },
  { id: "company", path: "/company", label: "My company", icon: Building2 },
  { id: "settings", path: "/settings", label: "Settings", icon: SettingsIcon }
];

const TITLES = {
  home: "Home", advice: "Advice", history: "History", company: "My company",
  settings: "Settings", update: "Update numbers", onboarding: "Set up your company",
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
  return { page: parts[0], params: {} };
}

function Shell() {
  const { state } = useStore();
  const [route, navigate] = useHashRoute();
  const { page, params } = parseRoute(route);
  const hasCompany = !!state.company;

  // Route guards: no company → welcome/onboarding only.
  useEffect(() => {
    if (!hasCompany && !["welcome", "onboarding"].includes(page)) navigate("/");
    if (hasCompany && page === "welcome") navigate("/home");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasCompany, page]);

  const bare = ["welcome", "onboarding", "analyzing"].includes(page) || !hasCompany;
  const month = latestMonth(state);

  const pages = {
    welcome: <Welcome navigate={navigate} />,
    onboarding: <Onboarding navigate={navigate} />,
    analyzing: <Analyzing navigate={navigate} />,
    home: <Home navigate={navigate} />,
    advice: <Advice navigate={navigate} params={params} />,
    history: <History navigate={navigate} />,
    company: <CompanyView navigate={navigate} />,
    update: <UpdateRitual navigate={navigate} />,
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
        <div className="sidebar-status">
          <span>Advisor</span>
          <strong>{state.demo ? "Sample company" : "Ready"}</strong>
          <small>{month ? `numbers from ${dateLabel(month.enteredAt)}` : "no data yet"}</small>
        </div>
      </aside>
      <section className="workspace">
        <header className="topbar">
          <div>
            <p>{TITLES[page] || "Home"}</p>
            <h1>{state.company?.name}</h1>
          </div>
          <div className="topbar-actions">
            {state.demo && <DemoBadge />}
            {!state.demo && page !== "update" && (
              <button className="secondary-button small" type="button" onClick={() => navigate("/update")}>
                <PencilLine size={14} /> Update numbers
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
      <Shell />
    </StoreProvider>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
