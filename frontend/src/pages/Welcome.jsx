// S1 Welcome — honest framing before any input (spec §7 O1).

import React, { useState } from "react";
import { BrainCircuit, ChevronDown, ChevronRight, FlaskConical } from "lucide-react";
import { useStore } from "../store.jsx";

export default function Welcome({ navigate }) {
  const { dispatch } = useStore();
  const [how, setHow] = useState(false);

  return (
    <section className="welcome">
      <div className="welcome-mark"><BrainCircuit size={44} /></div>
      <h1>Meet your AI advisory board</h1>
      <p className="welcome-lede">
        Three AI advisors — finance, growth, product — plus a strategist analyse your
        numbers and give you a monthly plan. Their experience comes from thousands of
        <strong> simulated startup scenarios, not real company data</strong> — so treat
        advice as a structured second opinion, not a prophecy.
      </p>
      <div className="welcome-actions">
        <button className="primary-button" type="button" onClick={() => navigate("/onboarding")}>
          Get started <ChevronRight size={16} />
        </button>
        <button
          className="secondary-button"
          type="button"
          onClick={() => { dispatch({ type: "ENTER_DEMO" }); navigate("/home"); }}
        >
          <FlaskConical size={15} /> Explore a sample company
        </button>
      </div>
      <button className="link-button center" type="button" onClick={() => setHow(!how)}>
        {how ? <ChevronDown size={15} /> : <ChevronRight size={15} />} How it works
      </button>
      {how && (
        <div className="how-grid">
          <div className="how-card">
            <strong>1 · Describe your company</strong>
            <p>About eight numbers you already know — revenue, cash, costs, price, churn. Around five minutes, once.</p>
          </div>
          <div className="how-card">
            <strong>2 · Get analysed advice</strong>
            <p>The board reads your position, names the biggest risk, and recommends this month's budget across marketing, product, hiring and pricing — with reasons.</p>
          </div>
          <div className="how-card">
            <strong>3 · Update monthly</strong>
            <p>Two minutes a month keeps advice current. Trends appear from your second update, and your own history becomes evidence over time.</p>
          </div>
        </div>
      )}
    </section>
  );
}
