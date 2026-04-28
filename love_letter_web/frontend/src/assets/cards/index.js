import espionne from "./00_espionne.png";
import garde from "./01_garde.png";
import qadi from "./02_qadi.png";
import emir from "./03_emir.png";
import hajib from "./04_hajib.png";
import wali from "./05_wali.png";
import vizir from "./06_vizir.png";
import sultan from "./07_sultan.png";
import sultane from "./08_sultane.png";
import amira from "./09_amira.png";
import cardBack from "./card_back.png";

export const cardArtById = {
  0: {
    image: espionne,
    gameName: "Espionne",
    artName: "Espionne",
  },
  1: {
    image: garde,
    gameName: "Garde",
    artName: "Garde",
  },
  2: {
    image: qadi,
    gameName: "Qadi",
    artName: "Qadi",
  },
  3: {
    image: emir,
    gameName: "Emir",
    artName: "Emir",
  },
  4: {
    image: hajib,
    gameName: "Hajib",
    artName: "Hajib",
  },
  5: {
    image: wali,
    gameName: "Wali",
    artName: "Wali",
  },
  6: {
    image: vizir,
    gameName: "Vizir",
    artName: "Vizir",
  },
  7: {
    image: sultan,
    gameName: "Sultan",
    artName: "Sultan",
  },
  8: {
    image: sultane,
    gameName: "Sultane",
    artName: "Sultane",
  },
  9: {
    image: amira,
    gameName: "Amira",
    artName: "Amira",
  },
};

export const cardArtList = Object.values(cardArtById);

export const cardBackArt = {
  image: cardBack,
  gameName: "Dos de carte",
  artName: "Motif du Palais",
};
