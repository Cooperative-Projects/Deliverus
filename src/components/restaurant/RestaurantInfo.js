import React, { Fragment, useContext, useEffect, useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Image from "mui-image";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import RecruitingPartyList from "./RecruitingPartyList";
import MenuCard from "./MenuCard";
import { API } from "../../utils/config";
import * as status from "../../utils/status";
import { UserContext } from "../store/UserContext";
import { Link, useParams } from "react-router-dom";
import { Divider, Rating, useMediaQuery } from "@mui/material";
import Grid from "@mui/material/Grid";

// 가게 조회 화면 컴포넌트입니다.
// prop으로 보여주고자 하는 가게 ID을 받습니다.
const RestaurantInfo = () => {
  const { userState, handleLogOut } = useContext(UserContext);
  const { userPos } = userState;
  const { id } = useParams();

  const isMobile = useMediaQuery("(max-width:750px)");

  // 가게 정보를 담은 변수
  const [restaurant, setRestaurant] = useState({
    address: "불러오는 중..",
    category: "불러오는 중..",
    intro: "불러오는 중..",
    latitude: 0,
    longitude: 0,
    menu: {
      menu: [
        {
          menuName: "",
          price: 0,
        },
      ],
    },
    name: "",
    phoneNumber: "불러오는 중..",
    rating: 0,
  });

  // 해당 가게에 대해서 모집 중인 딜리버스 파티방 정보를 담은 변수
  const [recruitingPartyList, setRecruitingPartyList] = useState(null);

  useEffect(() => {
    const data = { restaurantId: id };

    // 처음 페이지에 들어갈 때, 가게의 ID를 가지고 서버로부터 가게 정보 받기
    fetch(`${API.RESTAURANT_INFORMATION}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      credentials: "include",
      body: JSON.stringify(data),
    })
      .then((respones) => {
        status.handleRestaurantResponse(respones.status);
        return respones.json();
      })
      .then((data) => {
        //console.log("Respones Data from Restaurant Info API : ", data);
        setRestaurant(data);
      })
      .catch((error) => {
        // 로그인 만료 에러인 경우 로그아웃 실행
        if (error.name === "LoginExpirationError") {
          handleLogOut();
        }
        // 요청한 것에 대한 데이터가 없을 때 에러 처리
        else if (error.name === "NoDataError") {
          alert("error.message");
        }
        console.log(`${error.name} : ${error.message}`);
      });

    // 처음 페이지에 들어갈 때, 해당 가게에 대해 모집중인 파티방 리스트 가져오기
    fetch(`${API.PARTY_RESTAURANT}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      credentials: "include",
      body: JSON.stringify({
        id: id,
        latitude: userPos.lat,
        longitude: userPos.lng,
      }),
    })
      .then((respones) => {
        status.handlePartyResponse(respones.status);
        return respones.json();
      })
      .then((data) => {
        //console.log("Respones Data from PARTY RESTAURANT API : ", data);
        setRecruitingPartyList(data);
      })
      .catch((error) => {
        // 로그인 만료 에러인 경우 로그아웃 실행
        if (error.name === "LoginExpirationError") {
          handleLogOut();
        }
        console.log(`PARTY RESTAURANT API -> ${error.name} : ${error.message}`);
      });
  }, []);

  // 이미지를 src/images 디렉토리에서 가져옵니다.
  // 향후 이미지를 카테고리 별로 S3에 저장해서 모듈화 해야겠습니다.
  let image = null;
  if (!restaurant.name) {
    image = require(`../../images/deliveryIcon/delivery.ico`);
  } else {
    try {
      const category = restaurant.category.replace("/", ",");
      const name = restaurant.name;

      image = require(`../../images/${category}/${name}.png`);
    } catch (e) {
      console.log(e);
      image = require(`../../images/deliveryIcon/delivery.ico`);
    }
  }

  const LabelBox = (props) => {
    const label = props.label;
    const detail = props.detail;
    return (
      <Grid
        container
        sx={{
          alignItems: "center",
          paddingY: "2px",
        }}
      >
        <Grid item xs={4}>
          {label}
        </Grid>
        <Grid item xs={8}>
          <Typography
            component="body2"
            variant="body2"
            sx={{
              color: "text.secondary",
            }}
          >
            {detail}
          </Typography>
        </Grid>
      </Grid>
    );
  };

  // 식당의 정보가 담긴 컴포넌트입니다.
  // 첫번째 항의 컴포넌트가 모바일 컴포넌트입니다.
  const restaurantDescript = isMobile ? (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <Image src={image} width="100%" fit="contain" duration={1000} />
      <Typography component="h4" variant="h4" sx={{ mt: 2 }}>
        {restaurant.name}
      </Typography>
      <Box
        sx={{
          display: "flex",
          flexDirection: "row",
          alignItems: "end",
          my: 1,
        }}
      >
        <Rating
          value={restaurant.rating}
          precision={0.5}
          readOnly
          sx={{ mr: 1 }}
          size="large"
        />
        <Typography
          sx={{ color: "dimgray" }}
        >{`${restaurant.rating} / 5.0`}</Typography>
      </Box>
      <Divider sx={{ width: "100%", my: 1 }}>
        <Typography sx={{ color: "text.secondary" }}>가게 소개</Typography>
      </Divider>
      <Typography component="h6" variant="h6">
        {restaurant.intro}
      </Typography>
      <Divider sx={{ width: "100%", my: 1 }}>
        <Typography sx={{ color: "text.secondary" }}>배달 정보</Typography>
      </Divider>
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          width: "100%",
        }}
      >
        <LabelBox
          label={"배달비"}
          detail={
            restaurant.deliveryFee
              ? restaurant.deliveryFee.toLocaleString() + " 원"
              : 0 + "원"
          }
        />
        <LabelBox
          label={"최소배달비용"}
          detail={
            restaurant.minOrderPrice
              ? restaurant.minOrderPrice.toLocaleString() + " 원"
              : 0 + "원"
          }
        />
        <LabelBox label={"전화번호"} detail={restaurant.phoneNumber} />
      </Box>
    </Box>
  ) : (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <Image src={image} width="80%" fit="contain" duration={1000} />
      <Typography component="h4" variant="h4" sx={{ mt: 2 }}>
        {restaurant.name}
      </Typography>
      <Box
        sx={{
          display: "flex",
          flexDirection: "row",
          alignItems: "end",
          my: 1,
        }}
      >
        <Rating
          value={restaurant.rating}
          precision={0.5}
          readOnly
          sx={{ mr: 1 }}
          size="large"
        />
        <Typography
          sx={{ color: "dimgray" }}
        >{`${restaurant.rating} / 5.0`}</Typography>
      </Box>
      <Divider sx={{ width: "100%", my: 1 }}>
        <Typography variant="body1" sx={{ color: "text.secondary" }}>
          가게 소개
        </Typography>
      </Divider>
      <Typography component="h6" variant="h6">
        {restaurant.intro}
      </Typography>
      <Divider sx={{ width: "100%", my: 1 }}>
        <Typography sx={{ color: "text.secondary" }}>배달 정보</Typography>
      </Divider>
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          width: "100%",
        }}
      >
        <LabelBox
          label={"배달비"}
          detail={
            restaurant.deliveryFee
              ? restaurant.deliveryFee.toLocaleString() + " 원"
              : 0 + "원"
          }
        />
        <LabelBox
          label={"최소배달비용"}
          detail={
            restaurant.minOrderPrice
              ? restaurant.minOrderPrice.toLocaleString() + " 원"
              : 0 + "원"
          }
        />
        <LabelBox label={"전화번호"} detail={restaurant.phoneNumber} />
      </Box>
    </Box>
  );

  // 가게 메뉴 정보 받아오기
  const MenuList = restaurant.menu.menu;
  const restaurantMenu = (
    <Fragment>
      <Typography component="h6" variant="h6" sx={{ mb: 1 }}>
        메뉴
      </Typography>
      <Stack spacing={3}>
        {MenuList.map((item, index) => {
          return <MenuCard key={index} menu={item} />;
        })}
      </Stack>
    </Fragment>
  );

  return (
    <>
      <Box
        component="main"
        sx={{
          my: 6,
          mx: "auto",
          px: 4,
          pb: 5,
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-start",
          maxWidth: "800px",
        }}
      >
        {restaurantDescript}
        <Box sx={{ my: 1 }} />
        {recruitingPartyList && (
          <RecruitingPartyList partyList={recruitingPartyList} />
        )}
        <Link
          to="/party/creation"
          state={{ restaurantInfo: restaurant, resId: id }}
        >
          <Button
            fullWidth
            sx={{ border: 1, my: 2, backgroundColor: "#f5f3f3" }}
          >
            내가 딜리버스 모집하기
          </Button>
        </Link>
        {restaurantMenu}
      </Box>
    </>
  );
};

export default RestaurantInfo;
