import React from "react";
import { Card, CardImg } from "./AlphabetCard.style";

interface AlphabetCardProps {
  src: string;
  alt: string;
}

const AlphabetCard = ({ src, alt }: AlphabetCardProps) => {
  return (
    <Card>
      <CardImg src={src} alt={alt} />
    </Card>
  );
};

export default AlphabetCard;
