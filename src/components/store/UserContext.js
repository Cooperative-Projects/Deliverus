import { createContext, useEffect, useState } from "react";
import { API } from "../../utils/config";
import { useNavigate } from "react-router-dom";

// 컨텍스트 생성
export const UserContext = createContext();

// 컨텍스트 Provider 정의
const UserContextProvider = (props) => {
  const navigate = useNavigate();

  /** 유저 정보 State */
  const [userState, setUserState] = useState(() => {
    const userInfo = sessionStorage.getItem("userInfo");
    if (userInfo) {
      // 세션 스토리지에 유저 정보가 존재하는 경우
      const newUserState = JSON.parse(userInfo);
      return newUserState;
    } else {
      const initInfo = {
        username: "",
        isLoggedIn: false,
        userPos: null,
        userPosAddr : "",
      };
      sessionStorage.setItem("userInfo", JSON.stringify(initInfo));
      return initInfo;
    }
  });

  //   useEffect(() => {
  //     const userInfo = sessionStorage.getItem("userInfo");
  //     if (!userInfo) {
  //       // 세션 스토리지에 유저 정보가 없는 경우
  //       sessionStorage.setItem("userInfo", JSON.stringify(userState));
  //     } else {
  //       const newUserState = JSON.parse(userInfo);
  //       setUserState(newUserState);
  //     }
  //   }, []);

  useEffect(() => {
    const newUserState = userState;
    sessionStorage.setItem("userInfo", JSON.stringify(newUserState));
  }, [userState]);

  /**
   * 로그아웃 여부를 부모 컴포넌트에서만 관리하도록 하는 함수
   * 로그아웃할 때, 서버에게 User의 Nickname을 전달해줌으로 로그아웃 구현
   */
  const handleLogOut = () => {
    //console.log("user nickname logout...: ", userState.username);
    try {
      fetch(`${API.LOGOUT}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
      }).then((respones) => {
        navigate("/");
        setUserState({ ...userState, username: "", isLoggedIn: false, userPos : null , userPosAddr : ""});
      });
    } catch (error) {
      console.log("Error in Logout : ", error);
    }
  };

  /** TODO: 사용자의 행동에 의한 로그아웃 함수와 서버 세션 만료에 의한 로그아웃을 분리 */

  /**
   *  로그인 여부를 부모 컴포넌트에서만 관리하도록 하는 함수
   *  로그인할 때 서버로부터 받은 User의 Nickname으로 state 변경
   */
  const handleLogIn = (_userNickname) => {
    setUserState({ ...userState, username: _userNickname, isLoggedIn: true });
  };

  // User의 Position을 인자로 들어오는 값으로 변경하는 함수
  const handleInitUserPos = (_userPos, _userPosAddr) => {
    setUserState({...userState, userPos : _userPos, userPosAddr : _userPosAddr});
  }

  // User의 Position값을 변경할 수 있도록 User의 Position을 초기화 해주는 함수
  const handleChangeUserPos = () => {
    setUserState({...userState, userPosAddr: ""});
  }

  // 초기 값 설정
  return (
    <UserContext.Provider value={{ userState, handleLogOut, handleLogIn, handleInitUserPos, handleChangeUserPos }}>
      {props.children}
    </UserContext.Provider>
  );
};

export default UserContextProvider;
