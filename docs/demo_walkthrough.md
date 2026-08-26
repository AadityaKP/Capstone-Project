# Demo walkthrough — no Ollama, no Neo4j, Review 2 physics

This branch (`demo-offline-review2`) runs the founder product with neither the
model server nor the graph database, on the **review2** simulation profile —
the exact engine configuration of the Review 2 thesis experiments (`oracle_v3`,
repo `chroma_db`, unscaled research floors, salary-slot burn, scheduled
shocks). The recordings below were captured from that stack; the numbers on
every screen are the research engine's own answers for this dataset.
`SIM_PROFILE=review2` is the default, so nothing needs to be set.

## Start it

```
.\start.ps1 -Prod -NoOllama
```

Then open **http://localhost:5173** (or 8000 in `-Prod`) and click
**Get started**.

`GET /api/health` reports `"demo_fixtures": true` when the offline path is
active. Set `FOUNDER_DEMO_FIXTURES=0` to force the live stack back on.

---

## What is actually happening

Nothing here is invented, and nothing pretends a model spoke when one did not.
`/api/advise` answers in two tiers:

| | When | What the founder sees |
|---|---|---|
| **Replay** | The inputs match a recording in `backend/demo_fixtures/` | Exactly what the full stack returned when `record_demo_fixtures.py` captured it, strategist brief and all. `llm_ok: true`, because it was. |
| **Offline board** | Anything else | A real analysis run right now with no Oracle — the heuristic C-suite, the risk modifier and the cash-safety resolver, all driven by the founder's own numbers. `llm_ok: false`, and the UI says the strategist could not be reached. |

`/api/whatif` never needed either service: the projection is pure CPU and has
always run offline.

Every recording is stamped with the `SIM_PROFILE` that produced it, and only
replays under that profile — a server switched to `founder` falls through to
the offline board rather than handing back review2 answers as founder ones.

Re-capture the recordings any time, with Ollama up (the review2 profile runs
`oracle_v3`, so Neo4j is not needed):

```
venv\Scripts\python.exe record_demo_fixtures.py
```

It refuses to write a recording whose `llm_ok` is false, so a capture taken
while Ollama was down cannot silently become the demo.

---

## The dataset

### Onboarding

**Step 1 — Your company**

| Field | Value |
|---|---|
| Company name | `Kettle Analytics` |
| What you sell | `Stock forecasting for independent coffee roasters` |
| Company age | `15` months |
| How crowded is your market? | **A few rivals** |

**Step 2 — Your money**

| Field | Value |
|---|---|
| Monthly recurring revenue | `11000` |
| Cash in the bank | `95000` |
| Total monthly costs | `15500` |
| Marketing spend last month | `2600` |

**Step 3 — Your customers**

| Field | Value |
|---|---|
| Average price per customer | `55` |
| Customer churn (monthly) | `3.4` |
| New customers last month | `20` |
| Product maturity | **Solid** |
| *Add detail →* Team size | `2` |

### Next month — "Update my numbers"

| Field | Value | vs. month 1 |
|---|---|---|
| MRR | `12800` | +16.4% |
| Cash | `92000` | −$3,000 |
| Total monthly costs | `16400` | +$900 |
| Marketing spend | `3000` | +$400 |
| Average price | `56` | +$1 |
| Churn | `2.9` | −0.5pp |
| New customers | `24` | +4 |

The values must match exactly, to the digit, or the request falls through to the
offline board — the same screens, but `llm_ok: false` and a banner saying so.

---

## What each screen shows

**Home, month 1** — Low risk · `Cash lasts 21 mo` · `Revenue $11k` ·
`Customers lost 1 in 29` · `Winning customers: Healthy` ($130 to win, $1,618
back).

**Advice, month 1** — the strategist's own watch-outs, then four cards:
product ≈$20k (Priority, ≈182% of MRR), marketing ≈$29k on brand building
(≈259% of MRR), wait on hiring, hold pricing. The spend dwarfing the company's
revenue is the Review 2 research boardroom speaking: its floors are calibrated
for a $50k-MRR company and are deliberately not rescaled on this profile.

**Projection, month 1** — the plan reaches $366k revenue with $1.76M cash;
doing nothing reaches $55k with $286k. All 50 runs survive on both arms, so
every line stays solid to the horizon.

**Home, month 2** — `Cash lasts 26 mo (+4.4 mo)` · `Revenue $13k (+16.4%)` ·
`Customers lost 1 in 34 (−0.5pp)`, and *What changed* naming both moves.

**Advice, month 2** — the same shape grown with the company: product ≈$20k
(≈156% of MRR), marketing ≈$28k on brand, hiring still waits — the board sizes
spend against the new cash pile, not against headcount.

**History** — two entries with a visible diff:
`MRR $13k (+16.4%) · churn 2.9% (−0.5pp) · cash lasts 26 mo` above
`MRR $11k · churn 3.4% · cash lasts 21 mo`.

**Shock button** — a competitor surge at month 6 costs the plan 6.0% of
terminal revenue (doing nothing loses 3.8%); revenue never dips below its
pre-shock level on either arm, and the table says `no drop` rather than
inventing a recovery time.

---

## Two questions worth expecting

**"Why is it telling an $11k company to spend $49k a month?"** Because this
demo runs the Review 2 research engine unmodified, and that is genuinely what
it concludes: its marketing response curve pays out in absolute dollars
regardless of company size, so at this scale heavy spend is the winning move
inside the model — the projection shows it turning $95k of cash into $1.76M.
The founder-calibrated profile (`SIM_PROFILE=founder`) is the version of this
product that rescales the board to the company; this branch exists to show the
research engine itself through the product's screens.

**"Why is confidence Low when the analysis looks confident?"** Seven of the
engine's inputs — interest rate, consumer confidence, unemployment, valuation
multiple, innovation factor, lifetime value, and on this profile monthly costs
(the research engine charges its own $8,000-per-person convention rather than
the founder's figure) — were never measured for this company. The confidence
band is capped by that count, so "High confidence" beside seven estimates is
not a sentence the product can produce.
