import * as React from 'react';
import PropTypes from 'prop-types';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Box from '@mui/material/Box';
import MyPartyRoom from "./MyPartyRoom";
import Chat from "./chatting/Chat";
import {Fragment, useContext, useEffect, useState} from "react";
import {API} from "../../utils/config";
import * as status from "../../utils/status";
import {UserContext} from "../store/UserContext";
import {useNavigate, useParams} from "react-router-dom";
import CircularProgress from "@mui/material/CircularProgress";
import useMediaQuery from "@mui/material/useMediaQuery";
import Backdrop from '@mui/material/Backdrop';
import MuiAlert from "@mui/material/Alert";
import Snackbar from "@mui/material/Snackbar";

const Alert = React.forwardRef(function Alert(props, ref) {
    return <MuiAlert elevation={6} ref={ref} variant="filled" {...props} />;
});

function TabPanel(props) {
    const {children, value, index, ...other} = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`vertical-tabpanel-${index}`}
            aria-labelledby={`vertical-tab-${index}`}
            style={{width: "100%"}}
            {...other}
        >
            {value === index && (
                children
            )}
        </div>
    );
}

function MobileTabPanel(props) {
    const {children, value, index, ...other} = props;
    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            {...other}
        >
            {value === index && (
                children
            )}
        </div>
    )
}

TabPanel.propTypes = {
    children: PropTypes.node,
    index: PropTypes.number.isRequired,
    value: PropTypes.number.isRequired,
};

function a11yProps(index) {
    return {
        id: `vertical-tab-${index}`,
        'aria-controls': `vertical-tabpanel-${index}`,
    };
}

export default function MyPage() {
    const context = useContext(UserContext);
    const {userState, handleLogOut} = context;
    const {username} = userState;

    const isMobile = useMediaQuery("(max-width: 750px)");

    // key값을 통해 페이지에서 어떤 컴포넌트를 보여줄 지 결정합니다.
    const {key} = useParams();

    const navigate = useNavigate();

    // Tab value
    const [value, setValue] = useState(Number(key));

    // 내가 속해 있는 파티방 ID를 가지고 있는 변수
    const [myPartyId, setMyPartyId] = useState(-1);

    const handleChange = (event, newValue) => {
        setValue(newValue);
    };

    // 경고창 띄우기 위한 변수
    const [open, setOpen] = useState(false);

    // 경고창의 message에 대한 변수
    const [alertMessage, setAlertMessage] = useState("");

    // alert창 종류
    const [alertType, setAlertType] = useState("error");

    // 경고창을 닫는 함수
    const handleClose = () => {
        setOpen(false);
        navigate("/");
    };

    // 맨 처음에 username을 가지고 사용자가 속해있는 파티방의 ID를 GET 합니다.
    useEffect(() => {
        fetch(`${API.PARTY_ID}?name=${username}`, {
            headers: {
                "Content-Type": "application/json",
            },
            credentials: "include",
        })
            .then((respones) => {
                status.handlePartyResponse(respones.status);
                return respones.text();
            })
            .then((data) => {
                //console.log("Respones Data from PARTY ID API : ", data);
                // 사용자가 속해 있는 파티방이 있는 경우
                if (Number(data) !== -1) {
                    setMyPartyId(Number(data));
                }
                // 사용자가 속해있는 파티방이 없는 경우 main화면으로 이동
                else {
                    setAlertType("error");
                    setAlertMessage("참여중인 파티방이 없습니다");
                    setOpen(true);
                }
            })
            .catch((error) => {
                // 로그인 만료 에러인 경우 로그아웃 실행
                if (error.name === "LoginExpirationError") {
                    handleLogOut();
                }
                console.log(`PARTY ID API -> ${error.name} : ${error.message}`);
            });
    }, []);

    return (isMobile ? <Fragment>
                <Tabs
                    value={value}
                    onChange={handleChange}
                    centered
                    sx={{bgcolor : '#fafafa'}}
                >
                    <Tab label="내 파티방"/>
                    <Tab label="내 채팅방"/>
                </Tabs>
                <MobileTabPanel value={value} index={0}>
                    <MyPartyRoom/>
                </MobileTabPanel>
                <MobileTabPanel value={value} index={1}>
                    <Chat/>
                </MobileTabPanel>
                <Snackbar open={open} autoHideDuration={3000}
                          anchorOrigin={{vertical: "top", horizontal : "center"}}>
                    <Alert onClose={handleClose} severity={alertType} sx={{ width: '100%' }}>
                        {alertMessage}
                    </Alert>
                </Snackbar>
            </Fragment> :
            <Box sx={{flexGrow: 1, bgcolor: 'background.paper', display: 'flex', height: "100%"}}>
                {myPartyId !== -1 ? (<Fragment>
                    <Tabs
                        orientation="vertical"
                        variant="scrollable"
                        value={value}
                        onChange={handleChange}
                        sx={{borderRight: 1, borderColor: 'divider', width: "130px"}}
                    >
                        <Tab label="내 파티방" {...a11yProps(0)} />
                        <Tab label="내 채팅방" {...a11yProps(1)} />
                    </Tabs>
                    <TabPanel value={value} index={0}>
                        <MyPartyRoom/>
                    </TabPanel>
                    <TabPanel value={value} index={1}>
                        <Chat/>
                    </TabPanel>
                </Fragment>) : (<Backdrop
                    sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
                    open={true}
                >
                    <CircularProgress color="inherit" />
                </Backdrop>)}
                <Snackbar open={open} autoHideDuration={3000} onClose={handleClose}
                          anchorOrigin={{vertical: "top", horizontal : "center"}}>
                    <Alert onClose={handleClose} severity={alertType} sx={{ width: '100%' }}>
                        {alertMessage}
                    </Alert>
                </Snackbar>
            </Box>
    );
}