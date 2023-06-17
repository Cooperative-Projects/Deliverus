import { Link, Route, useNavigate } from "react-router-dom";
import React, { useContext, useEffect, useState } from "react";
import { UserContext } from "../store/UserContext";
import RestaurantList, {
  RestaurantCard,
  storeInfo,
} from "../partials/restaurantList/RestaurantList";
import RecruitingPartyList from "../restaurant/RecruitingPartyList";
import RecommendationList from "../recommendation/RecommendationList";
import styles from "./MainContents.module.css";
import { API } from "../../utils/config";
import * as status from "../../utils/status";
import {
  Box,
  Button,
  Icon,
  IconButton,
  Typography,
  useMediaQuery,
} from "@mui/material";
import Stack from "@mui/material/Stack";
import { useQuery } from "@tanstack/react-query";
import LocationOnIcon from "@mui/icons-material/LocationOnOutlined";
import KeyboardArrowRightIcon from "@mui/icons-material/KeyboardArrowRight";

const MainContents = () => {
  const context = useContext(UserContext);
  const { userState, handleChangeUserPos, handleLogOut } = context;
  const { username, userPosAddr, userPos } = userState;

  const isMobile = useMediaQuery("(max-width:750px");

  // 딥러닝 기반 AI가 추천해주는 Top 5 음식
  const [recommendList, setRecommendList] = useState(null);

  // 가게 정보 리스트(state로 관리)
  const [restInfoList, setRestInfoList] = useState(null);

  // 인접 파티방 정보 리스트
  const [recruitingPartyList, setRecruitingPartyList] = useState(null);
  const navigate = useNavigate();

  // Restaurant List로 이동
  const navToRestaurantList = () => {
    navigate(`/restaurant/list`, {
      state: {
        restInfoList: restInfoList,
        category: "all",
      },
    });
  };

  const navToPartyList = () => {
    navigate(`/party/list`, {
      state: {
        recruitingPartyList: recruitingPartyList,
      },
    });
  };

  // // react-query 예제 테스트
  const { isLoading, error, queryData } = useQuery(
    ["paryList"],
    () => {
      fetch(`${API.PARTY_ALL}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({
          latitude: userPos.lat,
          longitude: userPos.lng,
        }),
      })
        .then((respones) => {
          status.handlePartyResponse(respones.status);
          return respones.json();
        })
        .then((data) => {
          console.log("Respones Query Data from PARTY LIST API : ", data);
          setRecruitingPartyList(data);
          return data;
        })
        .catch((error) => {
          // 로그인 만료 에러인 경우 로그아웃 실행
          if (error.name === "LoginExpirationError") {
            console.log(`${error.name} : ${error.message}`);
          }
          console.log(`${error.name} : ${error.message}`);
          return error;
        });
    },
    {
      refetchOnWindowFocus: true,
      refetchInterval: 5000,
      refetchIntervalInBackground: true,
      retry: 0,
    }
  );

  useEffect(() => {
    // 처음 화면이 띄워졌을 때 모든 인접 파티방 리스트를 받아옵니다.
    fetch(`${API.PARTY_ALL}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      credentials: "include",
      body: JSON.stringify({
        latitude: userPos.lat,
        longitude: userPos.lng,
      }),
    })
      .then((respones) => {
        status.handlePartyResponse(respones.status);
        return respones.json();
      })
      .then((data) => {
        console.log("Respones Data from PARTY LIST API : ", data);
        setRecruitingPartyList(data);
      })
      .catch((error) => {
        // 로그인 만료 에러인 경우 로그아웃 실행
        if (error.name === "LoginExpirationError") {
          console.log(`${error.name} : ${error.message}`);
          handleLogOut();
        }
        console.log(`${error.name} : ${error.message}`);
      });

    // 처음 화면이 띄워졌을 때 모든 가게 리스트를 받아옵니다.
    fetch(`${API.RESTAURANT_LIST}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      credentials: "include",
      body: JSON.stringify({
        latitude: userPos.lat,
        longitude: userPos.lng,
      }),
    })
      .then((respones) => {
        status.handleRestaurantResponse(respones.status);
        return respones.json();
      })
      .then((data) => {
        console.log("Respones Data from Restaurant LIST API : ", data);
        setRestInfoList(data);
      })
      .catch((error) => {
        // 로그인 만료 에러인 경우 로그아웃 실행
        if (error.name === "LoginExpirationError") {
          console.log(`${error.name} : ${error.message}`);
          handleLogOut();
        }
        console.log(`${error.name} : ${error.message}`);
      });

    // AI 추천 카테고리를 서버로부터 가져옵니다.
    fetch(`${API.AI_RECOMMEND}`, {
      headers: {
        "Content-Type": "application/json",
      },
      credentials: "include",
    })
      .then((respones) => {
        status.handleRecommendResponse(respones.status);
        return respones.json();
      })
      .then((data) => {
        console.log("Respones Data from RECOMMEND API : ", data);
        setRecommendList([
          data.top1,
          data.top2,
          data.top3,
          data.top4,
          data.top5,
        ]);
      })
      .catch((error) => {
        // 로그인 만료 에러인 경우 로그아웃 실행
        if (error.name === "LoginExpirationError") {
          handleLogOut();
        }
        console.log(`GET RECOMMEND API -> ${error.name} : ${error.message}`);
      });
  }, []);

  const renderedPosAddr = userPosAddr.split(" ");
  const prePos = renderedPosAddr.filter((val, idx) => idx <= 1);
  const postPos = renderedPosAddr.filter((val, idx) => idx > 1);

  const renderedComponent = isMobile ? (
    // 모바일 컴포넌트
    <div className={styles.mainContents_body}>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-start",
          width: "100%",
          marginTop: "12px",
          marginBottom: "8px",
        }}
      >
        <IconButton
          sx={{ padding: 0, paddingRight: "4px" }}
          color="primary"
          onClick={handleChangeUserPos}
        >
          <LocationOnIcon />
        </IconButton>
        <Typography variant="body1" component="body1">
          {`${prePos[0]} ${prePos[1]}`}&nbsp;
        </Typography>
        <Typography variant="body2" component="body2" color="text.secondary">
          {`${postPos[0]} ${postPos[1]}`}
        </Typography>
      </Box>
      {recommendList && (
        <RecommendationList list={recommendList} restInfoList={restInfoList} />
      )}
      <div>
        <div className={styles.mainContents_subTitle}>
          <Typography variant="h2">
            👥 내 근처 딜리버스
          </Typography>

          <IconButton
            onClick={navToPartyList}
            sx={{}}
            color="primary"
            aria-label="more"
          >
            <KeyboardArrowRightIcon />
          </IconButton>
        </div>
        {recruitingPartyList ? (
          <RecruitingPartyList partyList={recruitingPartyList} />
        ) : (
          <Box
            sx={{
              backgroundColor: "info.main",
              textAlign: "center",
              paddingY: "10vh",
              borderRadius: 3,
            }}
          >
            <Typography>주변에 모집 중인 딜리버스가 없어요...</Typography>
          </Box>
        )}
        <div className={styles.mainContents_subTitle}>
          <Typography variant="h2">
            🏪 내 근처 가게 리스트
          </Typography>
          <IconButton
            onClick={navToRestaurantList}
            color="primary"
            aria-label="more"
          >
            <KeyboardArrowRightIcon />
          </IconButton>
        </div>
      </div>
      <Stack>
        {restInfoList &&
          restInfoList.map((item, idx) => {
            if (idx < 5) {
              return (
                <RestaurantCard
                  name={item.name}
                  rating={item.rating}
                  id={item.restaurant_id}
                  category={item.category}
                  intro={item.intro}
                  deliveryFee={item.deliveryFee}
                  minOrderPrice={item.minOrderPrice}
                  key={idx}
                />
              );
            }
          })}
      </Stack>
    </div>
  ) : (
    // PC 컴포넌트
    <div className={styles.mainContents_body}>
      <Box sx={{ display: "flex", justifyContent: "flex-start" }}>
        <IconButton sx={{}} color="primary" aria-label="more">
          <LocationOnIcon />
        </IconButton>
        <h4>{userPosAddr}</h4>
        <Button onClick={handleChangeUserPos} sx={{ ml: 1.5 }}>
          위치 바꾸기
        </Button>
      </Box>
      {recommendList && (
        <RecommendationList list={recommendList} restInfoList={restInfoList} />
      )}
      <div>
        <div className={styles.mainContents_subTitle}>
          <Typography variant="h2">
            👥 내 근처에서 모집중인 딜리버스
          </Typography>
          <h4 className={styles.show_more} onClick={navToPartyList}>
            더보기
          </h4>
        </div>
        {recruitingPartyList ? (
          <RecruitingPartyList partyList={recruitingPartyList} />
        ) : (
          <Box
            sx={{
              backgroundColor: "info.main",
              textAlign: "center",
              paddingY: "10vh",
              borderRadius: 3,
            }}
          >
            <Typography>주변에 모집 중인 딜리버스가 없어요...</Typography>
          </Box>
        )}
        <div className={styles.mainContents_subTitle}>
          <Typography variant="h2">
            🏪 내 근처에 있는 가게 리스트
          </Typography>
          <h4 className={styles.show_more} onClick={navToRestaurantList}>
            더보기
          </h4>
        </div>
      </div>
      <Stack>
        {restInfoList &&
          restInfoList.map((item, idx) => {
            if (idx < 5) {
              return (
                <RestaurantCard
                  name={item.name}
                  rating={item.rating}
                  id={item.restaurant_id}
                  category={item.category}
                  intro={item.intro}
                  deliveryFee={item.deliveryFee}
                  minOrderPrice={item.minOrderPrice}
                  key={idx}
                />
              );
            }
          })}
      </Stack>
    </div>
  );

  return renderedComponent;
};

export default MainContents;
