// The honesty checklist (docs/ui_simplification_plan.md Phase E.4, decision D3).
//
// Each honesty rule in docs/ui_components.md keeps one visible surface; these
// tests pin the surfaces that a refactor could silently drop. They render the
// real shell with a fixture state, never a mocked page, so a rule that moves
// to a different component still passes and one that disappears fails.
//
// This file must stay under frontend/test/: the Python scan in
// tests/test_founder_contract.py reads every file under frontend/src for
// engine vocabulary, and the cycle fixtures here would trip it.

import React from "react";
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { render, cleanup, fireEvent, act } from "@testing-library/react";

import { StoreProvider } from "../src/store.jsx";
import { CycleRunProvider } from "../src/cycleRun.jsx";
import { Shell } from "../src/App.jsx";
import { SAMPLE } from "../src/sample.js";

function clone(x) { return JSON.parse(JSON.stringify(x)); }

// Sample-mode state, optionally edited before render.
function sampleState(edit = null) {
  const s = clone(SAMPLE);
  if (edit) edit(s);
  return s;
}

// The founder's own workspace: the sample with the demo flag off.
function ownState(edit = null) {
  return sampleState((s) => { s.demo = false; if (edit) edit(s); });
}

function renderAt(route, state) {
  window.location.hash = route;
  return render(
    <StoreProvider initialState={state}>
      <CycleRunProvider>
        <Shell />
      </CycleRunProvider>
    </StoreProvider>
  );
}

// Every collapsed section on the page, opened.
function expandAll(container) {
  for (let pass = 0; pass < 4; pass += 1) {
    const heads = [...container.querySelectorAll(".expandable:not(.open) .expand-head, .oefa-strip:not(.open) .oefa-toggle")];
    if (!heads.length) break;
    heads.forEach((h) => act(() => { fireEvent.click(h); }));
  }
}

beforeEach(() => { window.localStorage.clear(); });
afterEach(() => { cleanup(); window.location.hash = ""; });

const ROUTES = ["/home", "/advice/a3", "/history", "/company", "/settings", "/update"];

describe("rule 1 — a plan made without the strategist says so, once", () => {
  it("This month shows one rules-only notice when the cycle ran on built-in rules", () => {
    const state = ownState((s) => {
      const c = s.cycles[s.cycles.length - 1];
      c.summary.llm_ok_months = 0;
      c.months.forEach((m) => { m.execute.llm_ok = false; });
      s.analyses[s.analyses.length - 1].llm_ok = false;
    });
    const { container } = renderAt("/home", state);
    const banners = container.querySelectorAll(".banner");
    expect(banners.length).toBe(1);
    expect(banners[0].textContent).toMatch(/built-in rules/);
  });

  it("names the months when only some of them ran on built-in rules", () => {
    const state = ownState((s) => {
      const c = s.cycles[s.cycles.length - 1];
      c.months[2].execute.llm_ok = false;
      c.months[3].execute.llm_ok = false;
    });
    const { container } = renderAt("/home", state);
    const banners = container.querySelectorAll(".banner");
    expect(banners.length).toBe(1);
    expect(banners[0].textContent).toMatch(/Months 3–4/);
  });

  it("Why this plan shows the rules-only notice for an analysis with llm_ok false", () => {
    const state = ownState((s) => {
      s.analyses[s.analyses.length - 1].llm_ok = false;
      const c = s.cycles[s.cycles.length - 1];
      c.months.forEach((m) => { m.execute.llm_ok = false; });
    });
    const { container } = renderAt("/advice/a3", state);
    const banners = container.querySelectorAll(".banner");
    expect(banners.length).toBe(1);
    expect(banners[0].textContent).toMatch(/built-in rules/);
  });
});

describe("rule 2 — markers only on values the founder did not give", () => {
  it("My company marks estimated values and never the founder's own", () => {
    const state = ownState((s) => { s.company.maturity = null; });
    const { container } = renderAt("/company", state);
    expect(container.querySelectorAll(".prov-chip.estimated").length).toBeGreaterThanOrEqual(2);
    expect(container.querySelectorAll(".prov-chip.provided").length).toBe(0);
    expect(container.textContent).toMatch(/From your .* close/);
  });
});

describe("rule 7 — the sample company is labelled on every route", () => {
  for (const route of ROUTES) {
    it(`shows the badge at ${route}`, () => {
      const { container } = renderAt(route, sampleState());
      expect(container.querySelectorAll(".demo-badge").length).toBeGreaterThanOrEqual(1);
    });
  }
  it("disables the Close form in sample mode with one notice", () => {
    const { container } = renderAt("/update", sampleState());
    expect(container.querySelectorAll(".banner").length).toBe(1);
    const submit = [...container.querySelectorAll("button")].find((b) => /Save & plan|Close the month & plan/.test(b.textContent));
    expect(submit.disabled).toBe(true);
  });
});

describe("the one-notice rule", () => {
  it("This month never shows more than one banner", () => {
    for (const state of [ownState(), sampleState(), ownState((s) => {
      // stale plan (numbers newer than the cycle) and rules-only at once
      s.months.push({ ...clone(s.months[2]), id: "m4", index: 3 });
      s.analyses[s.analyses.length - 1].llm_ok = false;
    })]) {
      const { container, unmount } = renderAt("/home", state);
      expect(container.querySelectorAll(".banner").length).toBeLessThanOrEqual(1);
      unmount();
    }
  });

  it("Why this plan shows at most one banner, archived and rules-only together", () => {
    const state = ownState((s) => { s.analyses[1].llm_ok = false; s.cycles[0].months.forEach((m) => { m.execute.llm_ok = false; }); });
    const { container } = renderAt("/advice/a2", state);
    expect(container.querySelectorAll(".banner").length).toBe(1);
    expect(container.textContent).toMatch(/Archived analysis/);
  });

  it("This month opens with no expander open and one caveat", () => {
    const { container } = renderAt("/home", ownState());
    expect(container.querySelectorAll(".expandable.open").length).toBe(0);
    expect(container.querySelectorAll(".wi-caveat").length).toBe(1);
  });
});

describe("rule 6 — the numbers we guessed are asked for at the moment of choice", () => {
  it("Close groups the optional numbers under 'numbers we estimated' when the board guessed one, open from the deep link", () => {
    const state = ownState((s) => {
      s.analyses[s.analyses.length - 1].trace.assumed_fields = [
        { field: "Acquisition cost", value: "$50", why: "not supplied", correctable: true },
        { field: "Unemployment", value: "4.0%", why: "typical conditions", correctable: false }
      ];
    });
    const { container } = renderAt("/update/fill", state);
    expect(container.querySelector(".fill-group.open")).not.toBeNull();
    expect(container.textContent).toMatch(/Numbers we estimated/);
    expect(container.textContent).toMatch(/Acquisition cost/);
    expect(container.querySelectorAll(".update-grid .ffield").length).toBe(7);
  });

  it("Close keeps every number inline when nothing was guessed", () => {
    const { container } = renderAt("/update", ownState());
    expect(container.querySelector(".fill-group")).toBeNull();
    expect(container.querySelectorAll(".update-grid .ffield").length).toBe(7);
  });
});

describe("rule 4 — no simulated-versus-simulated sentence outside the trace", () => {
  it("'the simulation did' appears only inside How the board got here", () => {
    const { container } = renderAt("/advice/a3", ownState());
    expandAll(container);
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    const hits = [];
    while (walker.nextNode()) {
      if (/the simulation did/.test(walker.currentNode.textContent)) hits.push(walker.currentNode);
    }
    expect(hits.length).toBeGreaterThan(0);
    for (const node of hits) {
      expect(node.parentElement.closest(".trace-months")).not.toBeNull();
    }
  });

  it("This month never says 'the simulation did'", () => {
    const { container } = renderAt("/home", ownState());
    expandAll(container);
    expect(container.textContent).not.toMatch(/the simulation did/);
  });

  it("the simulated marker appears only inside Evidence", () => {
    const { container } = renderAt("/advice/a3", ownState());
    expandAll(container);
    const tags = [...container.querySelectorAll(".sim-tag")];
    expect(tags.length).toBeGreaterThan(0);
    for (const tag of tags) {
      const section = tag.closest(".expandable");
      expect(section?.querySelector(".expand-head")?.textContent).toMatch(/Evidence/);
    }
  });
});
