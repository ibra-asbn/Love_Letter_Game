import React from "react";
import { cardArtById, cardBackArt } from "./assets/cards";

export function CardsPreview() {
  return (
    <main className="cards-preview-screen">
      <section className="cards-preview-panel" aria-label="Cartes Love Letter decoupees">
        <h1>Cartes du Palais</h1>
        <div className="cards-preview-grid">
          <figure className="cards-preview-card cards-preview-card-back">
            <img src={cardBackArt.image} alt={cardBackArt.gameName} />
            <figcaption>
              <strong>{cardBackArt.gameName}</strong>
              <span>{cardBackArt.artName}</span>
            </figcaption>
          </figure>
          {Object.entries(cardArtById).map(([id, card]) => (
            <figure className="cards-preview-card" key={id}>
              <img src={card.image} alt={`${card.gameName} - ${card.artName}`} />
              <figcaption>
                <strong>{id} - {card.gameName}</strong>
                <span>{card.artName}</span>
              </figcaption>
            </figure>
          ))}
        </div>
      </section>
    </main>
  );
}

export function RulesPage({ rules = [] }) {
  return (
    <main className="rules-page-screen">
      <section className="rules-page-panel" aria-label="Règles de Love Letter">
        <div className="rules-scroll-roll rules-scroll-roll-top" aria-hidden="true" />
        <div className="rules-scroll-content">
          <div className="panel-kicker">Le Palais du Sultan</div>
          <h1>Règles du jeu</h1>
          <div className="rules-page-list">
            {(rules.length ? rules : ["Chargement des règles..."]).map((rule, index) => (
              <p key={index}>{rule}</p>
            ))}
          </div>
        </div>
        <div className="rules-scroll-roll rules-scroll-roll-bottom" aria-hidden="true" />
      </section>
    </main>
  );
}
