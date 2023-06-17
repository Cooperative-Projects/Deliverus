import { Box, ButtonBase } from "@mui/material";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import React, { useContext } from "react";
import { UserContext } from "../store/UserContext";
import LetterAvatar from "../ui/LetterAvatar";
import { Grid } from "@mui/material";

// 딜리버스 만료 시간이 몇 분 남았는지 반환하는 함수
const getRemainTime = (expireTime) => {
  const expireDate = new Date(expireTime);
  const diff = expireDate - new Date().getTime();
  const remainMinutes = Math.floor(diff / 60000);

  return remainMinutes;
};

// 모듈화를 진행한 컴포넌트입니다.
// 하나의 파티방의 정보를 담은 컴포넌트입니다,
function RecruitingPartyCard(props) {
  const context = useContext(UserContext);
  const { userState } = context;
  const { username } = userState;

  const recruitPartyInfo = props.partyInfo;
  const remainMinutes = getRemainTime(recruitPartyInfo.expireTime);

  return (
    <ButtonBase
      onClick={(e) => {
        setTimeout(() => {
          props.propFunction(recruitPartyInfo, e);
        }, 200);
      }}
    >
      <Stack
        direction="row"
        sx={{
          alignItems: "center",
          padding: 2,
          marginX: "auto",
          borderRadius: 2,
          width: "100%",
          boxShadow: "0px 3px 5px rgba(0, 0, 0, 0.2)",
          backgroundColor: "#fff",
        }}
      >
        <LetterAvatar name={props.partyInfo.host} />
        <Grid container spacing={0.5} sx={{ paddingLeft: 1 }}>
          <Grid item xs={8} container justifyContent="flex-start">
            <Typography variant="body1" noWrap>
              {recruitPartyInfo.partyName}
            </Typography>
          </Grid>
          <Grid item xs={4} container justifyContent="flex-end">
            <Typography variant="body2" color="text.secondary">
              {recruitPartyInfo.currentMemberNum}/{recruitPartyInfo.memberNum}
            </Typography>
          </Grid>
          <Grid item xs={8} container justifyContent="flex-start">
            <Typography variant="body2" color="text.secondary">
              {recruitPartyInfo.restaurantName}
            </Typography>
          </Grid>
          <Grid item xs={4} container justifyContent="flex-end">
            <Typography variant="body2" color="text.secondary">
              {(Math.round(recruitPartyInfo.distance)).toLocaleString()}m
            </Typography>
          </Grid>
          <Grid item xs={8} container justifyContent="flex-start">
            <Typography variant="body2" color="text.secondary">
              배달 팁: {recruitPartyInfo.deliveryFee.toLocaleString()}원
            </Typography>
          </Grid>
          <Grid item xs={4} container justifyContent="flex-end">
            <Typography variant="body2" color="text.secondary" noWrap>
              {remainMinutes >= 0
                ? remainMinutes + "분 후 만료"
                : "만료됐어요..."}
            </Typography>
          </Grid>
        </Grid>
      </Stack>
    </ButtonBase>
    // <Card variant="outlined" sx={{display: "flex", p: 1, m: 1.5, border: "none", boxShadow: "0px 3px 5px rgba(0, 0, 0, 0.2)"}}>
    //     <CardContent sx={{my: "auto", px: 0, pl: 1}}>
    //         <LetterAvatar name={props.partyInfo.host}/>
    //     </CardContent>
    //     <Box sx={{display: 'flex', flexDirection: 'row', justifyContent: 'space-between'}}>
    //         <CardContent sx={{width: "90%"}}>
    //             <Typography fontSize='1.5rem' variant="h5" component="div" sx={{ mb: 1.5}}>
    //                 {recruitPartyInfo.partyName}
    //             </Typography>
    //             <Typography fontSize='1.2rem' variant="h5" component="div" sx={{color: "#9e9e9e"}}>
    //                 {recruitPartyInfo.restaurantName}
    //             </Typography>
    //             <Box sx={{display: "flex", flexDirection: "row", mt: 1}}>
    //                 <Typography fontSize='1.0rem' variant="body2" sx={{pr: 2}}>
    //                     배달비 : {recruitPartyInfo.deliveryFee}원
    //                 </Typography>
    //                 <Typography fontSize='1.0rem' variant="body2">
    //                     파티방 만료 시간 : {recruitPartyInfo.expireTime}
    //                 </Typography>
    //             </Box>
    //         </CardContent>
    //         <CardActions align="center" sx={{flexDirection: "column", width : "100px", my: "auto", ml: 1, px:0}}>
    //             <Typography fontSize='0.7rem' variant="h5" component="div" sx={{mb: 0.5}} style={{fontSize: "16px"}}>
    //                 {recruitPartyInfo.currentMemberNum}/{recruitPartyInfo.memberNum}
    //             </Typography>
    //             <Typography variant="body2" style={{fontSize: "16px"}}>
    //                 {Math.round(recruitPartyInfo.distance)}m
    //             </Typography>
    //             <Button size="small" onClick={(e) => {props.propFunction(recruitPartyInfo, e)}} style={{fontSize: "16px"}}>
    //                 참여하기</Button>
    //         </CardActions>
    //     </Box>
    // </Card>
  );
}

export default RecruitingPartyCard;
