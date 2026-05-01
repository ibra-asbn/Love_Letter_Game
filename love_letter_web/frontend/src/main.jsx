import React from "react";
import { createRoot } from "react-dom/client";
import { api } from "./api";
import { disposePalaceOstTimers } from "./palaceAudio";
import { MainMenu } from "./MainMenu";
import { PalaceMenu } from "./PalaceMenu";
import { SettingsPage } from "./SettingsPage";
import { CardsPreview, RulesPage } from "./StaticPages";
import { StitchRoyalDecorPreview } from "./StitchRoyalDecorPreview";
import { TutorialPage } from "./TutorialPage";
import { pushView, readCurrentView } from "./navigation";
import "./styles.css";

if (import.meta.hot) {
  import.meta.hot.dispose(disposePalaceOstTimers);
}

function App() {
  const [view, setView] = React.useState(readCurrentView);
  const [rules, setRules] = React.useState([]);

  React.useEffect(() => {
    const syncView = () => setView(readCurrentView());
    window.addEventListener("popstate", syncView);
    window.addEventListener("palace:viewchange", syncView);
    return () => {
      window.removeEventListener("popstate", syncView);
      window.removeEventListener("palace:viewchange", syncView);
    };
  }, []);

  React.useEffect(() => {
    api("/api/rules")
      .then((payload) => setRules(payload.rules || []))
      .catch(() => setRules([]));
  }, []);

  function navigate(viewName) {
    pushView(viewName);
  }

  function withPalaceMenu(page) {
    return (
      <>
        <PalaceMenu onNavigate={navigate} />
        {page}
      </>
    );
  }

  if (view === "cards-preview") {
    return withPalaceMenu(<CardsPreview />);
  }
  if (view === "rules") {
    return withPalaceMenu(<RulesPage rules={rules} />);
  }
  if (view === "settings") {
    return withPalaceMenu(<SettingsPage onNavigate={navigate} />);
  }
  if (view === "tutorial") {
    return withPalaceMenu(<TutorialPage onNavigate={navigate} />);
  }
  if (view === "game" || view === "stitch-decor") {
    return <StitchRoyalDecorPreview rules={rules} onNavigate={navigate} />;
  }

  return <MainMenu onNavigate={navigate} />;
}

createRoot(document.getElementById("root")).render(<App />);
