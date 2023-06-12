import React, { Fragment, useContext, useState } from "react";
import PositionSettingMap from "./PositionSettingMap";
import { Box, Button, Snackbar } from "@mui/material";
import Typography from "@mui/material/Typography";
import Stepper from "@mui/material/Stepper";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import { UserContext } from "../store/UserContext";

function MyPositionSetting() {
  const context = useContext(UserContext);
  const { userPos } = context.userState;
  const { handleInitUserPos } = context;

  // 처음 보여줄 Kakao map의 중심좌표
  // userState의 userPos가 null이 아니라면 그 값으로 초기화
  const initLatLng =
    userPos === null
      ? {
          lat: 37.57600923748876,
          lng: 126.9012721298886,
        }
      : {
          lat: userPos.lat,
          lng: userPos.lng,
        };

  // 사용자가 위치를 설정했냐 확인하는 변수
  const [state, setState] = useState(false);

  // 카카오지도에서 선택한 좌표
  const [myPos, setMyPos] = useState(null);

  // Kakao map에서 선택한 좌표의 주소
  const [myAddr, setMyAddr] = useState("");

  // kakao map에서 위치를 클릭할 시 호출되는 함수
  const handleClickPosEvent = (position, detailAddr) => {
    setMyPos(position);
    setState(true);
    setMyAddr(detailAddr);
  };

  // finish버튼을 클릭시 Context에 저장된 user의 pos를 최신화한다.
  const submitUserPos = () => {
    handleInitUserPos(myPos, myAddr);
  };

  return (
    <Box
      sx={{
        m: 2,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        border: 1,
        borderRadius: "16px",
        py: 2,
      }}
    >
      <Typography component="h1" variant="h5" sx={{ my: 3 }}>
        Deliverus를 시작할 위치 설정!
      </Typography>
      <Box
        sx={{
          width: "80%",
          height: "45vh",
          minWidth: "md",
          maxWidth: "md",
          border: 1,
        }}
      >
        <PositionSettingMap
          propFunction={handleClickPosEvent}
          initLatLng={initLatLng}
        />
      </Box>
      <Box sx={{ width: "100%", mt: 3 }}>
        <Stepper alternativeLabel activeStep={1}>
          <Step completed={state}>
            <StepLabel>지도를 클릭하세요!</StepLabel>
          </Step>
        </Stepper>
      </Box>
      <Button
        size="medium"
        disabled={!state}
        onClick={submitUserPos}
        sx={{ px: 10, mt: 3, border: 1, backgroundColor: "#f3f5f5" }}
      >
        Finish
      </Button>
    </Box>
  );
}

export default MyPositionSetting;
