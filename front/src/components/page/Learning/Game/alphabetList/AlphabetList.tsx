import React, { useState } from "react";
import { AlphabetButton, Container } from "./AlphabetList.style";

const Alphabet = [
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "Q",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "X",
  "Y",
  "Z",
];

interface AlphabetListProps {
  handleSetSrc: (index: number) => void;
}

const AlphabetList = ({ handleSetSrc }: AlphabetListProps) => {
  const [curIndex, setIcurIndex] = useState(0);
  return (
    <Container>
      {Alphabet.map((alpha, index) => {
        return (
          <AlphabetButton
            key={`${alpha} ${index}`}
            onClick={() => {
              handleSetSrc(index);
              setIcurIndex(index);
            }}
            className={curIndex === index ? "target" : "non-target"}
          >
            {alpha}
          </AlphabetButton>
        );
      })}
    </Container>
  );
};

export default AlphabetList;
