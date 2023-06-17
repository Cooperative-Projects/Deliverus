import {
    Divider, Grid, List, ListItem,
    ListItemText, TextField, Button, Box
} from "@mui/material";
import {Fragment, useContext, useEffect, useState} from "react";
import ChatLog from "./ChatLog";
import * as StompJS from '@stomp/stompjs';
import * as SockJS from 'sockjs-client';
import {UserContext} from "../../store/UserContext";
import {API, BASE_CHAT_URL} from "../../../utils/config";
import * as status from "../../../utils/status";
import {useNavigate} from "react-router-dom";

let client = null;
let subscription;

export function mySocketFactory() {
    return new SockJS(BASE_CHAT_URL);
}

function Chat() {
    const {userState, handleLogOut} = useContext(UserContext);
    const {username} = userState;
    const [message, setMessage] = useState("");
    const [chatLog, setChatLog] = useState([]);

    const navigate = useNavigate();

    // 내가 속해 있는 파티방 ID를 가지고 있는 변수
    const [myPartyId, setMyPartyId] = useState(-1);

    // 메세지 전송 시 호출되는 함수
    const handleSendMessage = (e) => {
        e.preventDefault();

        if (message.length > 0) {
            const publishData = {
                sender: username,
                channelId: myPartyId,
                chat: message
            }

            client.publish({
                destination: '/pub/chat',
                body: JSON.stringify(publishData),
                headers: {}
            });

            setMessage("");
        }
    };

    // 메세지 입력 시 호출 되는 함수
    const handleKeyPress = (e) => {
        setMessage(e.target.value);
    };

    // 구독한 채널로부터 메세지가 왔을 때 호출되는 callback 함수
    const callback = (message) => {
        const data = JSON.parse(message.body);

        setChatLog(prevState => [...prevState, {
            sender: data.sender, time: data.time,
            chat: data.chat, type: data.type
        }]);
    }

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
                    setMyPartyId(data);
                }
                // 사용자가 속해있는 파티방이 없는 경우 main화면으로 이동
                else {
                    alert("속해 있는 파티방이 없습니다ㅠ");
                    navigate("/");
                }
            })
            .catch((error) => {
                // 로그인 만료 에러인 경우 로그아웃 실행
                if (error.name === "LoginExpirationError") {
                    handleLogOut();
                }
                console.log(`PARTY ID API -> ${error.name} : ${error.message}`);
            });

        return () => {
            // client가 생성되어 있다면 deactivate하기
            if (client) {
                client.deactivate();
            }
        }
    }, []);

    // 파티방 ID를 이용해서 구독을 하고, chatlog를 가져옵니다.
    useEffect(() => {
        if (myPartyId !== -1) {
            client = new StompJS.Client({
                brokerURL: BASE_CHAT_URL,
                connectHeaders: {
                    sender: username,
                    channelId: Number(myPartyId)
                },
                debug: function (str) {
                    //console.log(str);
                },
                heartbeatIncoming: 4000,
                heartbeatOutgoing: 4000,
            });

            // Fallback code
            if (typeof WebSocket !== 'function') {
                // For SockJS you need to set a factory that creates a new SockJS instance
                // to be used for each (re)connect
                client.webSocketFactory = mySocketFactory();
            }

            // Fallback code
            if (typeof WebSocket !== 'function') {
                // For SockJS you need to set a factory that creates a new SockJS instance
                // to be used for each (re)connect
                client.webSocketFactory = function () {
                    // Note that the URL is different from the WebSocket URL
                    return new SockJS('http://localhost:15674/stomp');
                };
            }

            client.onConnect = function (frame) {
                //console.log(frame);
                // Do something, all subscribes must be done is this callback
                // This is needed because this will be executed after a (re)connect

                if(Array.isArray(chatLog) && chatLog.length === 0) {
                    // connect하고 chatLog 불러오기
                    fetch(`${API.CHAT_MESSAGE}?name=${username}&id=${myPartyId}`, {
                        headers: {
                            "Content-Type": "application/json",
                        },
                        credentials: "include",
                    })
                        .then((respones) => {
                            status.handleChatResponse(respones.status);
                            return respones.json();
                        })
                        .then((data) => {
                            //console.log("Respones Data from CHAT MESSAGE API : ", data);
                            if (Array.isArray(data) && data.length === 0) {
                                data.push({
                                    sender: username, time: "00:00",
                                    chat: `${username}님이 입장하셨습니다`, type: 0
                                });
                            }
                            setChatLog(prevState => [...prevState].concat(data));

                        })
                        .catch((error) => {
                            // 로그인 만료 에러인 경우 로그아웃 실행
                            if (error.name === "LoginExpirationError") {
                                handleLogOut();
                            }
                            console.log(`CHAT MESSAGE API -> ${error.name} : ${error.message}`);
                        });
                }

                subscription = client.subscribe(`/sub/chat/${myPartyId}`, callback, {});
                console.log("subscribed!");
            };

            client.onStompError = function (frame) {
                // Will be invoked in case of error encountered at Broker
                // Bad login/passcode typically will cause an error
                // Complaint brokers will set `message` header with a brief message. Body may contain details.
                // Compliant brokers will terminate the connection after any error
                console.log('Broker reported error: ' + frame.headers['message']);
                console.log('Additional details: ' + frame.body);
            };

            client.activate();
        }
    }, [myPartyId]);


    return (
        <Fragment>
            <Box sx={{margin: "auto", bgcolor: "#ffebee", maxWidth: "md"}}>
                <ChatLog list={chatLog} name={username}/>
                <Divider sx={{border: 2}}/>
                <Box component="form" sx={{display: "flex", alignItem: "row", margin: "auto", padding: 3}}>
                    <TextField
                        fullWidth
                        label="Message"
                        value={message}
                        onChange={handleKeyPress}
                    />
                    <Button type="submit" variant="contained" color="primary" sx={{ml: 2}}
                            onClick={handleSendMessage}>
                        Send
                    </Button>
                </Box>
            </Box>
        </Fragment>
    );
}

export default Chat;