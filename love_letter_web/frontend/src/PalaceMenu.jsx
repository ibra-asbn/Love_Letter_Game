import React from "react";
import { Menu } from "lucide-react";
import { pushView } from "./navigation";

const PALACE_MENU_ITEMS = [
  { label: "Menu principal", view: null },
  { label: "Jouer", view: "game" },
  { label: "Tutoriel", view: "tutorial" },
  { label: "Cartes", view: "cards-preview" },
  { label: "Règles", view: "rules" },
  { label: "Paramètres", view: "settings" },
];

export function PalaceMenu({
  variant = "global",
  onNavigate,
  onNewGame,
  busy = false,
}) {
  const [open, setOpen] = React.useState(false);

  function navigate(view) {
    setOpen(false);
    if (onNavigate) {
      onNavigate(view);
      return;
    }
    pushView(view);
  }

  function startNewGame() {
    setOpen(false);
    onNewGame?.();
  }

  return (
    <div className={`palace-menu ${variant === "scene" ? "is-scene" : "is-global"}`}>
      <button
        className="stitch-scene-menu-button palace-menu-button"
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        aria-label="Ouvrir le menu du palais"
      >
        <Menu size={18} />
        Menu
      </button>
      {open ? (
        <nav className="stitch-scene-menu palace-menu-panel" aria-label="Menu du palais">
          {onNewGame ? (
            <button type="button" onClick={startNewGame} disabled={busy}>
              Nouvelle partie
            </button>
          ) : null}
          {PALACE_MENU_ITEMS.map((item) => (
            <button key={item.label} type="button" onClick={() => navigate(item.view)}>
              {item.label}
            </button>
          ))}
        </nav>
      ) : null}
    </div>
  );
}
