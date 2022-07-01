import React, { useState, useEffect } from "react";
import { learningCopyRights } from "../../copyRights/copyRights";
import Footer from "../../Footer";
import {
  LearningContainer,
  SearchContainer,
  ResultContainer,
  CardContainer,
  VideoContainer,
  SearchBar,
  SearchButton,
  H1,
} from "./index.style";
import CardTemplate from "../Learning/LearningTemplate/CardTemplate";

import { searchCopyRights } from "../../copyRights/copyRights";
import ReactTooltip from "react-tooltip";
import { imgSrc } from "./wordData";

import * as Api from "../../../api";

interface VideoDataProps {
  _id: string;
  english: string;
  handVideo: string;
  mouthVideo?: string;
}

const Search = () => {
  const [searchWord, setSearchWord] = useState("");
  const [find, setFind] = useState(false);

  const [searchedImage, setSearchedImage] = useState({
    src: "",
    alt: "",
  });

  const [videos, setVideos] = useState<VideoDataProps[]>([]);
  const [videoSrc, setVideoSrc] = useState<string[]>([]);

  const [isEmpty, setIsEmpty] = useState(true);
  const [isFirst, setIsFirst] = useState(true);

  const handleOnChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchWord(e.target.value);
  };

  const handleOnClick = () => {
    setIsFirst(false);
    //이미지 찾기
    const tempImg = imgSrc.filter((img, index) => {
      if (searchWord === img.name) return img;
    });

    console.log(tempImg);

    if (tempImg.length > 0) {
      setSearchedImage({
        src: tempImg[0].src,
        alt: tempImg[0].alt,
      });
      setIsEmpty(false);
    } else {
      console.log("hey");
      setSearchedImage({
        src: "",
        alt: "",
      });
      setIsEmpty(true);
    }

    const temp = imgSrc.filter((data) => {
      return data.name === searchWord;
    });

    const mapped = temp.map((data) => data.alt);
    //비디오 찾기
    const temp2 = videos
      .filter((data) => {
        if (data.english === mapped[0]) {
          return data;
        }
      })
      .map((data) => {
        return data.handVideo;
      });

    if (temp2.length > 0) {
      setVideoSrc(temp2);
    } else {
      setVideoSrc([]);
    }
  };

  const getVideos = async () => {
    const res = await Api.get("hands");
    setVideos(res.data.slice(26));
  };

  useEffect(() => {
    setFind(true);
  }, [searchWord]);

  useEffect(() => {
    console.log("video", videoSrc);
  }, [videoSrc]);

  useEffect(() => {
    getVideos();
  }, []);

  return (
    <LearningContainer>
      <SearchContainer>
        <SearchBar
          type="text"
          placeholder="여기서 입력!"
          onChange={handleOnChange}
        />

        <SearchButton
          onClick={handleOnClick}
          data-tip="main-search"
          data-for="main-search"
        >
          검색!
        </SearchButton>
        <ReactTooltip id="main-search" place="bottom">
          <video autoPlay width="400" muted loop>
            <source
              src="http://sldict.korean.go.kr/multimedia/multimedia_files/convert/20160325/251010/MOV000262948_700X466.mp4"
              type="video/mp4"
            />
          </video>
          <p style={{ textAlign: "right" }}>출처: 국립국어원</p>
        </ReactTooltip>
      </SearchContainer>

      <ResultContainer>
        {isEmpty && isFirst && <H1>공부했던 것을 찾아볼까요?</H1>}
        {isEmpty && !isFirst && <H1>검색 결과가 없습니다!</H1>}
        {!isEmpty && find && videoSrc.length >= 1 && (
          <>
            <CardContainer>
              <CardTemplate
                src={searchedImage.src}
                alt={searchedImage.alt}
                style={{ width: "300px" }}
              />
            </CardContainer>
            <VideoContainer>
              <video
                autoPlay
                controls
                width="300"
                muted
                loop
                style={{ borderRadius: "5px" }}
                key={videoSrc[0]}
              >
                <source src={videoSrc[0]} type="video/mp4" />
              </video>
            </VideoContainer>
          </>
        )}
      </ResultContainer>
      <Footer
        aLinks={searchCopyRights.aLinks}
        contents={searchCopyRights.contents}
      />
    </LearningContainer>
  );
};

export default Search;