import React from "react";
import { Score, MAX_COUNT } from "./index";
import { AnswerImg } from "./index.style";
import Modal from "../../Modal";

const ModalStyle = {
  width: "1000px",
  height: "900px",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
};

interface SolveProps {
  modal: boolean;
  closeModal(): void;
  answer: boolean;
  finish: boolean;
  score: Score;
  nextQuiz(): void;
  MoveRecord(): void;
}

function SolveModal({
  modal,
  closeModal,
  answer,
  finish,
  score,
  nextQuiz,
  MoveRecord,
}: SolveProps) {
  return (
    <Modal
      visible={modal}
      //   closeModal={closeModal}
      style={ModalStyle as React.CSSProperties}
    >
      {answer ? (
        <>
          <h1>정답입니다!!!</h1>
          <AnswerImg
            src={`${process.env.PUBLIC_URL}/quizgamepic/answer.png`}
          ></AnswerImg>
        </>
      ) : (
        <>
          <h1>틀렸네?~~</h1>
          <AnswerImg
            src={`${process.env.PUBLIC_URL}/quizgamepic/wrong.jpg`}
          ></AnswerImg>
        </>
      )}
      <h1>{`${score.ans}/${MAX_COUNT}`}</h1>
      {finish ? (
        <div>
          <button onClick={MoveRecord}>순위 등록하러가기</button>
        </div>
      ) : (
        <div>
          <button onClick={nextQuiz}>다음 문제 풀기</button>
          <button onClick={closeModal}>포.기.하.기</button>
        </div>
      )}
      <h2>{`남은 문제 : ${MAX_COUNT - score.cur}`}</h2>
    </Modal>
  );
}

export default SolveModal;