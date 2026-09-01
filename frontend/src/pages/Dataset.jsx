// Dataset (EDGAR) tab — the raw panel, nothing else.
//
// As-ingested dollar figures straight from the panel files in data/ via
// /api/review/panel: no derived columns, no charts. Ticker filter, sort by
// quarter, server-side pagination. The header counts come from
// /api/review/meta, which computes them from the panel at request time.

import React, { useEffect, useState } from "react";
import { AlertTriangle, ChevronLeft, ChevronRight, Database } from "lucide-react";
import { reviewMeta, reviewPanel } from "../api.js";

const PAGE_SIZE = 25;

const COLUMN_LABELS = {
  ticker: "Ticker",
  fiscal_period: "Quarter",
  revenue: "Revenue",
  sm_expense: "S&M",
  rnd_expense: "R&D",
  ga_expense: "G&A",
  cost_of_revenue: "Cost of revenue",
  cash_and_investments: "Cash (+STI)",
  operating_cash_flow: "Operating CF"
};

function cell(column, value) {
  if (value == null) return "—";
  if (column === "ticker" || column === "fiscal_period") return value;
  return `$${Math.round(value).toLocaleString("en-US")}`;
}

export default function Dataset() {
  const [meta, setMeta] = useState(null);
  const [page, setPage] = useState(null);
  const [error, setError] = useState(null);
  const [ticker, setTicker] = useState("");
  const [order, setOrder] = useState("asc");
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    let alive = true;
    reviewMeta().then((res) => {
      if (alive && res.ok) setMeta(res.data);
    });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    let alive = true;
    reviewPanel({ ticker: ticker || null, offset, limit: PAGE_SIZE, order }).then((res) => {
      if (!alive) return;
      if (res.ok) { setPage(res.data); setError(null); }
      else setError(res.error || "Panel unavailable");
    });
    return () => { alive = false; };
  }, [ticker, order, offset]);

  // The demo focuses on 5 companies; the header counts describe exactly what
  // is on screen, with the full-panel size stated so the subset is disclosed.
  const ds = meta?.dataset;
  const sub = ds?.demo_subset;
  const headerLine = ds && sub
    ? `${sub.n_companies} companies · ${sub.n_complete_quarters.toLocaleString("en-US")} complete quarters · SEC EDGAR XBRL · ${sub.quarter_range[0]}–${sub.quarter_range[1]} · demo subset of the ${ds.n_companies}-company panel`
    : "";

  const total = page?.total ?? 0;
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + PAGE_SIZE, total);

  return (
    <section className="content-stack">
      <article className="panel rv-panel">
        <div className="panel-title-row">
          <h3><Database size={16} /> Dataset (EDGAR)</h3>
          {headerLine && <span className="rv-header-line">{headerLine}</span>}
        </div>

        {error && <p className="wi-error"><AlertTriangle size={15} /> {error}</p>}

        <div className="rv-controls">
          <label className="rv-field">
            <span>Company</span>
            <select
              value={ticker}
              onChange={(e) => { setTicker(e.target.value); setOffset(0); }}
            >
              <option value="">All companies</option>
              {(page?.tickers || []).map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
          <label className="rv-field">
            <span>Sort by quarter</span>
            <select
              value={order}
              onChange={(e) => { setOrder(e.target.value); setOffset(0); }}
            >
              <option value="asc">Oldest first</option>
              <option value="desc">Newest first</option>
            </select>
          </label>
          <div className="rv-pager">
            <button
              type="button" className="secondary-button small"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
            >
              <ChevronLeft size={14} /> Prev
            </button>
            <span>{from}–{to} of {total.toLocaleString("en-US")}</span>
            <button
              type="button" className="secondary-button small"
              disabled={offset + PAGE_SIZE >= total}
              onClick={() => setOffset(offset + PAGE_SIZE)}
            >
              Next <ChevronRight size={14} />
            </button>
          </div>
        </div>

        <div className="wi-table-wrap">
          <table className="wi-table rv-panel-table">
            <thead>
              <tr>
                {(page?.columns || []).map((c) => <th key={c}>{COLUMN_LABELS[c] || c}</th>)}
              </tr>
            </thead>
            <tbody>
              {(page?.rows || []).map((row, i) => (
                <tr key={`${row.ticker}-${row.fiscal_period}-${i}`}>
                  {page.columns.map((c) => <td key={c}>{cell(c, row[c])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  );
}
