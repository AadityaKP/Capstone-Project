// The one-notice rule (docs/ui_simplification_plan.md Phase E).
//
// A page has one notice slot. `pickNotice` returns at most one notice for it,
// chosen in a fixed priority, plus the inline sentences for every condition
// that lost the slot, so nothing true goes unsaid — it just does not get a
// second banner. Pages render the winner with <Banner> and the inline lines
// in the section they belong to.
//
// Priority: engine unreachable / cycle failed / start failed (with the action)
// · sample company (only on Close, where the form is disabled) · rules-only
// (the whole plan, or "Months 3–4 used built-in rules") · archived analysis
// (Why only).

// Which horizon months ran without the strategist, as "3–4" / "2, 4".
export function rulesOnlyMonths(months) {
  const idx = (months || [])
    .map((m, i) => (m?.execute?.llm_ok === false ? i + 1 : null))
    .filter((i) => i != null);
  if (!idx.length) return null;
  if (idx.length === (months || []).length) return "all";
  const contiguous = idx.every((v, i) => i === 0 || v === idx[i - 1] + 1);
  return contiguous && idx.length > 1 ? `${idx[0]}–${idx[idx.length - 1]}` : idx.join(", ");
}

export const RULES_ONLY_TEXT =
  "The AI strategist couldn't be reached for this plan. It comes from the board's " +
  "built-in rules — still grounded in your numbers, just without the strategist's read.";

// ctx (every field optional):
//   startError   { offline, error }      a cycle could not be started
//   failed       { error }               the current cycle failed on the engine
//   pollError    string                  lost contact with a running cycle
//   demo         boolean                 sample company; only Close asks for it
//   rulesOnly    "all" | "3–4" | null    which months ran without the strategist
//   archived     { monthName }           an archived analysis, Why only
//   actions      { retry, rerun, current }  callbacks the page offers
export function pickNotice(ctx) {
  const candidates = [];
  const a = ctx.actions || {};

  if (ctx.startError) {
    candidates.push({
      kind: "start-failed", tone: "warn",
      text: "The analysis service couldn't be reached, so no plan was started. Your numbers are "
        + "saved; nothing is made up in the meantime."
        + (ctx.startError.error && !ctx.startError.offline ? ` (${ctx.startError.error})` : ""),
      // Retry only: the founder is already on This month, so "continue
      // without a plan" would be a button that does nothing.
      actions: a.retry ? [{ label: "Retry", primary: true, onClick: a.retry }] : []
    });
  }
  if (ctx.failed) {
    candidates.push({
      kind: "cycle-failed", tone: "warn",
      text: `The cycle failed on the engine: ${ctx.failed.error || "unknown error"}. Nothing here is made up — re-run it from your numbers.`,
      actions: a.rerun ? [{ label: "Re-run", primary: true, onClick: a.rerun }] : []
    });
  }
  if (ctx.pollError) {
    candidates.push({ kind: "engine-unreachable", tone: "warn", text: ctx.pollError, actions: [] });
  }
  if (ctx.demo) {
    candidates.push({
      kind: "sample", tone: "info",
      text: "Sample company — updates are disabled here. Start your own company from the welcome screen.",
      actions: []
    });
  }
  if (ctx.rulesOnly) {
    candidates.push({
      kind: "rules-only", tone: "warn",
      text: ctx.rulesOnly === "all"
        ? RULES_ONLY_TEXT
        : `Months ${ctx.rulesOnly} of this plan used the board's built-in rules; the strategist couldn't be reached for them.`,
      actions: []
    });
  }
  if (ctx.archived) {
    candidates.push({
      kind: "archived", tone: "info",
      text: `Archived analysis from ${ctx.archived.monthName} — shown as it was.`,
      actions: a.current ? [{ label: "Current plan", onClick: a.current }] : []
    });
  }

  const notice = candidates[0] || null;
  // What lost the slot, as one sentence each, for the section it belongs to.
  const inline = candidates.slice(1).map((c) => ({
    kind: c.kind,
    text: c.kind === "rules-only"
      ? (ctx.rulesOnly === "all"
        ? "This plan came from the board's built-in rules; the strategist was unreachable."
        : `Months ${ctx.rulesOnly} used the board's built-in rules.`)
      : c.kind === "archived"
        ? `Archived analysis from ${ctx.archived.monthName}, shown as it was.`
        : c.text
  }));
  return { notice, inline };
}
