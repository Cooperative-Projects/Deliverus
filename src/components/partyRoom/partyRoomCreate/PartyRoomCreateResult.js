import Typography from "@mui/material/Typography";
import KakaoMapStore from "../../restaurant/KakaoMapStore";
import React, { Fragment } from "react";
import Grid from "@mui/material/Grid";
import MenuCard from "../../restaurant/MenuCard";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import {Box, Divider} from "@mui/material";

// 파티방을 만들 때 설정 사항들을 최종적으로 보여주는 컴퍼넌트입니다.
function PartyRoomCrateResult(props) {
    return (
        <Fragment>
            <Box sx={{width: "95%", margin: "auto"}}>
                <Typography variant="h2" my={2}>
                    🏫파티방 정보
                </Typography>
                <Typography variant="body1" mb={1}>
                    방명 : {props.partyInfo.partyName}
                </Typography>
                <Typography variant="body1" mb={1}>
                    정원 : {props.partyInfo.memberNum}명
                </Typography>
                <Typography variant="body1" color="primary">
                    파티방 유지 시간 : 🕓 {props.partyInfo.life}분!
                </Typography>
                <Divider sx={{border: 1, my: 4}}/>
                <Typography variant="h2" mb={2}>
                    🏠가게 정보
                </Typography>
                <Typography variant="body1" mb={1}>
                    가게명 : {props.partyInfo.restaurantName}
                </Typography>
                <Typography variant="body1" mb={1}>
                    최소 주문 금액 : {props.partyInfo.minOrderPrice.toLocaleString()}원
                </Typography>
                <Typography variant="body1" color="primary">
                    배달비 : {props.partyInfo.deliveryFee.toLocaleString()}원
                </Typography>
                <Divider sx={{border: 1, my: 4}}/>
                <Typography  variant="h2" mb={2}>
                    🚩픽업 장소
                </Typography>
            </Box>
            <Box sx={{width: "80%", height: "40vh", border: 1}}>
                <KakaoMapStore
                    lat={props.partyInfo.latitude}
                    lng={props.partyInfo.longitude}
                />
            </Box>
            <Box sx={{width: "95%", margin: "auto"}}>
                <Typography  variant="body1" sx={{ my: 2}}>
                    도로명 주소 : {props.partyInfo.pickUpAddress}
                </Typography>
                <Typography  variant="body1" sx={{ mb: 2}}>
                    {props.detailPos && `상세 주소 : ${props.detailPos}`}
                </Typography>
                <Divider sx={{border: 1,  my: 4}}/>
                <Typography variant="h2" mb={2}>
                    🍽️내가 선택한 메뉴
                </Typography>
            </Box>
            <Stack spacing={3} sx={{width: "95%", mb: 5}}>
                {props.menuList.map((item, index) => {
                    if (props.countList[index] > 0) {
                        return (<MenuCard key={index} menu={item} countNum={props.countList[index]}/>);
                    }
                })}
            </Stack>
        </Fragment>
    );
}

export default PartyRoomCrateResult;
