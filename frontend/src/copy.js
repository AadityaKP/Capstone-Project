// Enum → founder copy. The single permitted mapping from engine vocabulary to
// the screen (docs/founder_frontend_spec.md §26). Nothing elsewhere may
// translate engine terms.

export const RISK = {
  LOW: { label: "Low risk", tone: "green", glyph: "●" },
  MEDIUM: { label: "Moderate risk", tone: "blue", glyph: "●" },
  HIGH: { label: "Elevated risk", tone: "amber", glyph: "⚠" },
  CRITICAL: { label: "Critical risk", tone: "red", glyph: "⚠" }
};

export const OUTLOOK = {
  ACCELERATING: "growth is accelerating",
  STABLE: "growth is steady",
  DECLINING: "growth is slowing",
  COLLAPSING: "growth is falling sharply"
};

export const OUTCOME = {
  GROWTH: { label: "growth", glyph: "▲", tone: "green" },
  STAGNATION: { label: "flat", glyph: "▬", tone: "amber" },
  DECLINE: { label: "decline", glyph: "▼", tone: "red" }
};

// Board focus mix labels (§26): weights → founder words.
export const FOCUS_LABELS = {
  innovation: "Product",
  growth: "Growth",
  efficiency: "Efficiency",
  macro: "Market"
};

// Confidence scalar → qualitative band (§15.1): never a percentage.
export function confidenceBand(confidence) {
  if (confidence == null) return "Moderate";
  if (confidence < 0.4) return "Low";
  if (confidence <= 0.7) return "Moderate";
  return "High";
}

// refresh_reason / client-side trigger → plain copy (§15.1 rule 4).
export function refreshReasonCopy(reason) {
  switch (reason) {
    case "initial": return "your first analysis";
    case "cadence": return "scheduled monthly review";
    case "runway": return "re-analysed early: runway fell below 12 months";
    case "mrr_drop": return "re-analysed early: revenue dropped more than 15%";
    case "churn_jump": return "re-analysed early: churn rose 1.5 points or more";
    case "event": return "re-analysed early: a significant change in your numbers";
    default: return "monthly review";
  }
}

export function briefSourceCopy(source) {
  if (source === "cache_hit" || source === "reuse") return "consistent with your last analysis";
  return null;
}

// L1 position sentence, assembled from enums only — never free LLM text (§10.1).
export function positionSentence(brief) {
  if (!brief) return "Run an analysis to see where you stand.";
  const outlook = OUTLOOK[brief.growth_outlook] || OUTLOOK.STABLE;
  const focusBits = {
    innovation: "protect retention and product momentum",
    efficiency: "tighten spending and protect runway",
    growth: "lean into acquisition",
    macro: "play defensively while conditions are rough"
  };
  const topFocus = brief._topFocus || "innovation";
  switch (brief.risk_level) {
    case "CRITICAL":
      return `Serious pressure on the business — ${outlook}. This month: ${focusBits[topFocus]}.`;
    case "HIGH":
      return `Things need attention — ${outlook}. This month: ${focusBits[topFocus]}.`;
    case "LOW":
      return `You're in good shape — ${outlook}. This month: ${focusBits[topFocus]}.`;
    default:
      return `Steady but watchful — ${outlook}. This month: ${focusBits[topFocus]}.`;
  }
}

// Modifier percentage → verbal scale (§10.1 L3). Numbers appear once, in amounts.
export function scaleWord(changePct) {
  if (changePct == null) return null;
  if (changePct <= -40) return "scaled back sharply";
  if (changePct <= -15) return "scaled back";
  if (changePct <= -5) return "trimmed slightly";
  if (changePct >= 40) return "increased sharply";
  if (changePct >= 15) return "increased";
  if (changePct >= 5) return "nudged up";
  return null;
}

// LLM free-text guardrail (§15.4): clamp length, cap at 3, drop bullets whose
// numeric claims match nothing in the founder's numbers.
export function guardBullets(bullets, knownNumbers) {
  if (!Array.isArray(bullets)) return [];
  const nums = (knownNumbers || []).filter((n) => n != null && Number.isFinite(n));
  return bullets
    .filter((b) => typeof b === "string" && b.trim().length > 0)
    .slice(0, 3)
    .map((b) => (b.length > 140 ? `${b.slice(0, 137)}…` : b))
    .filter((b) => {
      const claims = (b.match(/\$?\d[\d,]*(\.\d+)?%?/g) || []).map((m) =>
        parseFloat(m.replace(/[$,%]/g, ""))
      );
      if (claims.length === 0) return true;
      return claims.every((c) =>
        nums.some((n) => {
          const scale = Math.max(Math.abs(n), 1);
          return Math.abs(n - c) / scale < 0.2 || (c <= 100 && Math.abs(c - n) < 3);
        })
      );
    });
}

// Simulated-memory documents → founder sentences (§15.3). Strips scores; keeps
// tier phrases and the realized outcome.
const PHASE_WORDS = { SEED: "early revenue", EARLY: "growing", GROWTH: "scaling", SCALE: "at scale" };
const CHURN_WORDS = { LOW: "low churn", MEDIUM: "moderate churn", HIGH: "high churn", CRITICAL: "severe churn" };
const INNOV_WORDS = { HEALTHY: "product momentum healthy", DECLINING: "product losing momentum", DEGRADED: "product momentum badly degraded" };

export function rewriteMemory(memory) {
  const doc = memory?.document || "";
  const outcome = OUTCOME[memory?.metadata?.realized_outcome] || OUTCOME.STAGNATION;
  const phase = PHASE_WORDS[(doc.match(/Phase:\s*(\w+)/) || [])[1]] || "a similar stage";
  const churn = CHURN_WORDS[(doc.match(/Churn:\s*(\w+)/) || [])[1]] || null;
  const innov = INNOV_WORDS[(doc.match(/Innovation:\s*(\w+)/) || [])[1]] || null;
  const traits = [phase, churn, innov].filter(Boolean).join(", ");
  const ending =
    outcome.label === "growth" ? "grew over the following 6 months" :
    outcome.label === "decline" ? "declined over the following 6 months" :
    "stayed roughly flat over the following 6 months";
  return { sentence: `A simulated company at ${traits} ${ending}.`, outcome };
}

// Expected outcome headline (§26).
export function expectedOutcomeCopy(expected) {
  const o = OUTCOME[expected];
  if (!o) return null;
  return `In simulation, the next 6–12 months most often looked like: ${o.label}.`;
}

export const SIMULATED_PREFIX = "From simulations, not real companies";

export const DOMAIN_META = {
  product: { title: "Product & retention", weightKey: "innovation" },
  marketing: { title: "Marketing & growth", weightKey: "growth" },
  hiring: { title: "Hiring", weightKey: "efficiency" },
  pricing: { title: "Pricing", weightKey: "efficiency" }
};

export const CHANNEL_COPY = { ppc: "performance channels", brand: "brand building" };
