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
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, cleanup, fireEvent, act, waitFor } from "@testing-library/react";

import { StoreProvider } from "../src/store.jsx";
import { CycleRunProvider } from "../src/cycleRun.jsx";
import { Shell } from "../src/App.jsx";
import { SAMPLE } from "../src/sample.js";
import { predictionSentences, scoreLine } from "../src/loopView.js";

// The engine is never reached from here: every fetch fails as if the API
// were down, unless a test stubs it otherwise.
const offline = () => Promise.reject(new TypeError("Failed to fetch"));

// A state in which the latest cycle is the founder's current month and is
// still deliberating (nothing landed yet), or has failed.
function withLatestCycle(edit) {
  return ownState((s) => {
    const c = s.cycles[s.cycles.length - 1];
    c.source = "api";
    edit(c, s);
  });
}

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

beforeEach(() => { window.localStorage.clear(); vi.stubGlobal("fetch", vi.fn(offline)); });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); window.location.hash = ""; });

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

describe("rule 1 — every failure state gets its one notice, with the action", () => {
  it("start failed: one notice with Retry only, nothing invented", async () => {
    const state = ownState((s) => { s.cycles = []; s.analyses = []; });
    const { container } = renderAt("/home", state);
    const run = [...container.querySelectorAll("button")].find((b) => /Run the plan/.test(b.textContent));
    await act(async () => { fireEvent.click(run); });
    await waitFor(() => expect(container.querySelectorAll(".banner").length).toBe(1));
    const banner = container.querySelector(".banner");
    expect(banner.textContent).toMatch(/couldn't be reached/);
    expect(banner.textContent).toMatch(/Retry/);
    expect(banner.textContent).not.toMatch(/Continue without a plan/);
    expect(JSON.parse(window.localStorage.getItem("ssom_founder_v1")).cycles.length).toBe(0);
  });

  it("cycle failed: one notice with the engine's reason and Re-run", () => {
    const state = withLatestCycle((c) => { c.status = "failed"; c.error = "engine restarted"; c.months = []; });
    const { container } = renderAt("/home", state);
    const banners = container.querySelectorAll(".banner");
    expect(banners.length).toBe(1);
    expect(banners[0].textContent).toMatch(/failed on the engine: engine restarted/);
    expect(banners[0].textContent).toMatch(/Re-run/);
  });

  it("lost contact: one notice while a running cycle cannot be polled", async () => {
    const state = withLatestCycle((c) => { c.status = "running"; c.months = []; });
    const { container } = renderAt("/home", state);
    await waitFor(() => expect(container.querySelectorAll(".banner").length).toBe(1));
    expect(container.querySelector(".banner").textContent).toMatch(/Lost contact/);
  });

  it("archived analysis: one notice with Current plan", () => {
    const { container } = renderAt("/advice/a2", ownState());
    const banners = container.querySelectorAll(".banner");
    expect(banners.length).toBe(1);
    expect(banners[0].textContent).toMatch(/Archived analysis/);
    expect(banners[0].textContent).toMatch(/Current plan/);
  });

  it("stale numbers, rules-only and a failed cycle together: the failure wins, the rest is one line", () => {
    const state = ownState((s) => {
      // The founder closed September; the cycle planned on it failed before
      // any month landed, its summary says no month had the strategist; the
      // plan on screen is still August's.
      s.months.push({ ...clone(s.months[2]), id: "m4", index: 3 });
      const c = clone(s.cycles[1]);
      c.id = "c3"; c.monthId = "m4"; c.status = "failed"; c.error = "engine restarted";
      c.months = [];
      c.summary = { ...c.summary, llm_ok_months: 0, months_completed: 0 };
      s.cycles.push(c);
    });
    const { container } = renderAt("/home", state);
    const banners = container.querySelectorAll(".banner");
    expect(banners.length).toBe(1);
    expect(banners[0].textContent).toMatch(/failed on the engine/);
    const note = container.querySelector(".plan-note").textContent;
    expect(note).toMatch(/previous numbers/);
    expect(note).toMatch(/built-in rules/);
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
  const guessedCac = (s) => {
    s.analyses[s.analyses.length - 1].trace.assumed_fields = [
      { field: "Acquisition cost", value: "$50", why: "not supplied", correctable: true },
      { field: "Churn split", value: "one blended rate", why: "not supplied", correctable: true },
      { field: "Unemployment", value: "4.0%", why: "typical conditions", correctable: false }
    ];
  };

  it("Close groups only the fields that answer a guess, open from the deep link; price stays inline", () => {
    const { container } = renderAt("/update/fill", ownState(guessedCac));
    const group = container.querySelector(".fill-group.open");
    expect(group).not.toBeNull();
    expect(group.textContent).toMatch(/Acquisition cost/);
    expect(group.textContent).not.toMatch(/Churn split/);
    const grouped = [...group.querySelectorAll(".ffield-label")].map((l) => l.textContent);
    expect(grouped).toEqual(["New customers last month (optional)", "Marketing spend last month (optional)"]);
    const inline = [...container.querySelectorAll(".update-grid .ffield-label")].map((l) => l.textContent);
    expect(inline).toContain("Average price (optional)");
    expect(container.querySelectorAll(".update-grid .ffield").length).toBe(7);
  });

  it("Close keeps every number inline when nothing was guessed", () => {
    const { container } = renderAt("/update", ownState());
    expect(container.querySelector(".fill-group")).toBeNull();
    expect(container.querySelectorAll(".update-grid .ffield").length).toBe(7);
  });

  it("Why offers 'Fill these in' only for a guess Close can take, and never in sample mode", () => {
    const open = (container) => {
      const head = [...container.querySelectorAll(".expand-head")].find((h) => /Assumptions/.test(h.textContent));
      act(() => { fireEvent.click(head); });
    };
    const a = renderAt("/advice/a3", ownState(guessedCac));
    open(a.container);
    expect(a.container.textContent).toMatch(/Fill these in/);
    a.unmount();
    const b = renderAt("/advice/a3", ownState((s) => {
      s.analyses[2].trace.assumed_fields = [{ field: "Churn split", value: "one blended rate", why: "not supplied", correctable: true }];
    }));
    open(b.container);
    expect(b.container.textContent).not.toMatch(/Fill these in/);
    b.unmount();
    const c = renderAt("/advice/a3", sampleState(guessedCac));
    open(c.container);
    expect(c.container.textContent).not.toMatch(/Fill these in/);
  });
});

describe("the close form", () => {
  it("cannot be submitted twice while the feedback is being scored", async () => {
    // The feedback POST never resolves within the test; the second click
    // must not add a second month.
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
    const { container } = renderAt("/update", ownState());
    for (const b of container.querySelectorAll(".did-option")) {
      if (/Did it/.test(b.textContent)) act(() => { fireEvent.click(b); });
    }
    const submit = [...container.querySelectorAll("button")].find((b) => /Close the month & plan/.test(b.textContent));
    expect(submit.disabled).toBe(false);
    await act(async () => { fireEvent.click(submit); fireEvent.click(submit); });
    await waitFor(() => expect(container.textContent).toMatch(/Scoring last month/));
    expect(JSON.parse(window.localStorage.getItem("ssom_founder_v1")).months.length).toBe(4);
    expect(container.querySelector("fieldset.close-fields").disabled).toBe(true);
    expect(window.fetch).toHaveBeenCalledTimes(1);
  });
});

describe("History", () => {
  it("gives didn't its own mark and spells the line out", () => {
    const { container } = renderAt("/history", ownState());
    const lines = [...container.querySelectorAll(".timeline-decisions")].map((d) => d.textContent);
    expect(lines).toContain("Did 1 · partly 1 · didn't 1 of 3 actions");
    const heads = container.querySelectorAll(".timeline-head");
    act(() => { fireEvent.click(heads[1]); });
    expect(container.querySelector(".decision-line.declined").textContent).toMatch(/^✕/);
    expect(container.querySelector(".decision-line.accepted").textContent).toMatch(/^✓/);
  });

  it("opens the entry This month's Details link points at", () => {
    const { container } = renderAt("/history/m2", ownState());
    const detail = container.querySelector(".timeline-detail");
    expect(detail).not.toBeNull();
    expect(detail.textContent).toMatch(/How the plan held up/);
  });
});

describe("the score line and the sentences agree", () => {
  it("writes a sentence for every scored KPI, runway included", () => {
    const error = {
      mrr_pct: { expected: 4, realized: 2, within_tolerance: false, sign_agrees: true },
      cash_pct: { expected: -3, realized: -2, within_tolerance: true, sign_agrees: true },
      churn_pp: { expected: -0.2, realized: -0.1, within_tolerance: true, sign_agrees: true },
      runway_months: { expected: -0.6, realized: -1.4, within_tolerance: false, sign_agrees: true },
      summary: { kpis_scored: 4, within_tolerance: 2, sign_agrees: 4 }
    };
    const lines = predictionSentences({
      before: { mrr: 30000, cash: 300000 }, actual: { mrr: 30600, cash: 294000 },
      expected: { mrr_pct: 4 }, error
    });
    expect(lines.length).toBe(4);
    expect(lines[3].text).toMatch(/Cash lasts: the board expected -0.6 mo, it moved -1.4 mo/);
    expect(scoreLine(error)).toMatch(/4 of 4/);
  });
});

describe("This month asks for engine time only when it would change something", () => {
  it("offers no Re-run on a fresh plan on the current numbers", () => {
    const { container } = renderAt("/home", ownState());
    expect(container.querySelector(".plan-section-actions").textContent).not.toMatch(/Re-run|Plan again/);
  });

  it("offers Plan again when the numbers moved on", () => {
    const state = ownState((s) => { s.months.push({ ...clone(s.months[2]), id: "m4", index: 3 }); });
    const { container } = renderAt("/home", state);
    expect(container.querySelector(".plan-section-actions").textContent).toMatch(/Plan again/);
  });

  it("links each later month straight to its strip on Why this plan", () => {
    const { container } = renderAt("/home", ownState());
    const head = [...container.querySelectorAll(".expand-head")].find((h) => /Next 3 months/.test(h.textContent));
    act(() => { fireEvent.click(head); });
    const links = container.querySelectorAll(".next-month-link");
    expect(links.length).toBe(3);
    act(() => { fireEvent.click(links[1]); });
    expect(window.location.hash).toBe("#/advice/a3/m3");
  });

  it("a deep link opens that month's strip under How the board got here", () => {
    const { container } = renderAt("/advice/a3/m3", ownState());
    const strips = container.querySelectorAll(".trace-months .oefa-strip");
    expect(strips.length).toBe(4);
    expect([...strips].map((s) => s.classList.contains("open"))).toEqual([false, false, true, false]);
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
