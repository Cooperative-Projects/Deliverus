import Box from "@mui/material/Box";
import React, {useContext, useEffect, useState} from "react";
import MenuSelecting from "../partyRoomCreate/MenuSelecting";
import Button from "@mui/material/Button";
import {useLocation, useNavigate} from "react-router-dom";
import {UserContext} from "../../store/UserContext";
import * as status from "../../../utils/status";
import { API } from "../../../utils/config";
import MuiAlert from "@mui/material/Alert";
import Snackbar from "@mui/material/Snackbar";

const Alert = React.forwardRef(function Alert(props, ref) {
    return <MuiAlert elevation={6} ref={ref} variant="filled" {...props} />;
});

// 참가자 입장에서 가게의 메뉴를 선택하는 컴포넌트입니다
function PersonalMenuSelecting() {
    const context = useContext(UserContext);
    const {userState, handleLogOut} = context;
    const {username} = userState;

    const navigate = useNavigate();

    // 가게 ID 정보를 받아오기
    const location = useLocation();
    const restaurantId = location.state.resId;
    const partyId = location.state.partyId;

    // 서버로부터 가게 정보를 받을 변수
    const [restaurant, setRestaurant] = useState({
        menu: {
            menu: [
                {
                    "menuName": "string",
                    "price": 0
                }
            ]
        },
        name: "",
    });

    // 각 메뉴에 대한 수량을 담은 리스트
    const [countList, setCountList] = useState([0]);

    // 경고창 띄우기 위한 변수
    const [open, setOpen] = useState(false);

    // 경고창의 message에 대한 변수
    const [alertMessage, setAlertMessage] = useState("");

    // alert창 종류
    const [alertType, setAlertType] = useState("error");

    // 경고창을 닫는 함수
    const handleClose = () => {
        setOpen(false);
        if (alertType === "success"){
            navigate("/myPage/0");
        }
        else {
            //에러 시 메인페이지로 이동
            navigate("/");
        }
    };

    // 가게의 ID를 가지고 서버로부터 가게 정보 받기
    useEffect(() => {
        const data = {restaurantId: restaurantId};
        fetch(`${API.RESTAURANT_INFORMATION}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            credentials: "include",
            body: JSON.stringify(data)
        })
            .then((respones) => {
                status.handleRestaurantResponse(respones.status);
                return respones.json();
            })
            .then((data) => {
                //console.log("Respones Data from Restaurant Info API : ", data);
                setRestaurant(data);
                setCountList(new Array(data.menu.menu.length).fill(0));
            })
            .catch((error) => {
                // 로그인 만료 에러인 경우 로그아웃 실행
                if (error.name === "LoginExpirationError") {
                    console.log(`${error.name} : ${error.message}`);
                }
                //에러 시 메인페이지로 이동
                navigate("/");
            });
    }, []);

    const handleNext = () => {
        // 사용자가 선택한 메뉴에 대한 정보 담기
        const orderList = [];
        restaurant.menu.menu.map((item, index) => {
            if(countList[index] > 0){
                orderList.push({
                    menuName : item.menuName,
                    price : item.price,
                    num : countList[index]
                })
            }
        })

        // 최종적으로 서버에게 보낼 데이터 형태
        const data = {
            partyId : partyId,
            nickname : username,
            order : orderList
        };
        fetch(`${API.PARTY_MEMBER}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            credentials: "include",
            body: JSON.stringify(data)
        })
            .then((respones) => {
                status.handlePartyResponse(respones.status);
                return respones.text();
            })
            .then((data) => {
                //console.log("Respones Data from Party Member API : ", data);
                // MyPage에서 나의 파티방 페이지로 이동
                setAlertType("success");
                setAlertMessage("파티방에 입장 완료하였습니다!")
                setOpen(true);
            })
            .catch((error) => {
                // 로그인 만료 에러인 경우 로그아웃 실행
                if (error.name === "LoginExpirationError") {
                    handleLogOut();
                }
                else if (error.name === "DuplicateJoinError") {
                    setAlertType("error");
                    setAlertMessage("이미 딜리버스 중입니다!")
                    setOpen(true);
                }
                else {
                    setAlertType("error");
                    setAlertMessage("파티방 입장이 거부되었습니다");
                    setOpen(true);
                }
                console.log(`${error.name} : ${error.message}`);
            });
    };

    return (<Box sx={{display: "flex", flexDirection: "column", alignItems: "center",
        maxWidth: 'lg', mx: 'auto', py: "20px"}}>
        <MenuSelecting countList={countList} setCountList={setCountList} menuList={restaurant.menu.menu}/>
        <Button type="submit"
                onClick={handleNext}
                disabled={!countList.some(element => element > 0)}>
            🚩 Deliverus 파티방 입장하기
        </Button>
        <Snackbar open={open} autoHideDuration={3000} onClose={handleClose}
                  anchorOrigin={{vertical: "top", horizontal : "center"}}>
            <Alert onClose={handleClose} severity={alertType} sx={{ width: '100%' }}>
                {alertMessage}
            </Alert>
        </Snackbar>
    </Box>);
}

export default PersonalMenuSelecting;
