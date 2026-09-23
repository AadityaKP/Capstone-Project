// Browser entry point. App.jsx exports the shell so the honesty checklist in
// test/ can render it without a DOM root; this file is the only place that
// mounts it.

import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
