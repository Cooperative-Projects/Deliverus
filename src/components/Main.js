import MainContents from "./mainContents/MainContents";
import SignPage from "./partials/SignPage";
import { useContext} from "react";
import { UserContext } from "./store/UserContext";
import MyPositionSetting from "./postionSetting/MyPositionSetting";

const Main = () => {
  const context = useContext(UserContext);
  const { userPosAddr, isLoggedIn } = context.userState;


  if (isLoggedIn) {
    if (userPosAddr === "") {
      return <MyPositionSetting userLatLng={null} />;
    } else {
      return <MainContents/>;
    }
  } else {
    return <SignPage />;
  }
};

export default Main;
