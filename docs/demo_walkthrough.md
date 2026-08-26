# Demo walkthrough — no Ollama, no Neo4j

This branch (`demo-offline`) runs the founder product with neither the model
server nor the graph database. Everything below works on a laptop with nothing
but Python and Node.

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

Re-capture the recordings any time, with Ollama and Neo4j up:

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
`Customers lost 1 in 29` · `Winning customers: Healthy`.

**Advice, month 1** — the strategist's own watch-outs, then four cards:
product ≈$4.0k, marketing ≈$3.0k on performance channels, hold hiring, consider
a ≈1% price rise.

**Projection, month 1** — the plan reaches $28k revenue with $17k cash left and
runs out in 3 of 50 runs; doing nothing reaches $26k with $62k and survives all
50. That partial failure is deliberate: it is what makes the line go dashed and
puts a death marker on the chart.

**Home, month 2** — `Cash lasts 26 mo (+4.4 mo)` · `Revenue $13k (+16.4%)` ·
`Customers lost 1 in 34 (−0.5pp)`, and *What changed* naming both moves.

**Advice, month 2** — the plan grows with the company, and hiring becomes a real
action: *Room to add ≈$8.0k/mo of payroll — roughly one hire*.

**History** — two entries with a visible diff:
`MRR $13k (+16.4%) · churn 2.9% (−0.5pp) · cash lasts 26 mo` above
`MRR $11k · churn 3.4% · cash lasts 21 mo`.

**Shock button** — a competitor surge at month 6 costs the plan about 10% of
revenue.

---

## Two questions worth expecting

**"Why does the plan end with less cash than doing nothing?"** Because it does,
and that is the trade-off the projection exists to show. The board's plan buys
about $2k/month more revenue by month 12 and spends roughly $45k of cash to get
there. Whether that is worth it is the founder's call, which is the point.

**"Why is confidence Low when the analysis looks confident?"** Six of the
engine's inputs — interest rate, consumer confidence, unemployment, valuation
multiple, innovation factor, lifetime value — were never measured for this
company. The confidence band is capped by that count, so "High confidence"
beside six estimates is not a sentence the product can produce.
