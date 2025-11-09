// smartvision/frontend/src/index.js
// Entry point of the SmartVision React app

import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

// Get the root element from public/index.html
const container = document.getElementById("root");

// Create a React root and render the App component inside it
if (!container) {
  throw new Error('Root element with id "root" not found in public/index.html');
}
const root = createRoot(container);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
