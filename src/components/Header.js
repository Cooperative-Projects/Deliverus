import {
  AppBar,
  Box,
  Toolbar,
  Typography,
  Button,
  ThemeProvider,
  createTheme,
  useMediaQuery,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListSubheader,
  IconButton,
} from "@mui/material";
import HomeIcon from "@mui/icons-material/Home";
import GroupAddIcon from "@mui/icons-material/GroupAdd";
import AccountBoxIcon from "@mui/icons-material/AccountBox";
import LogoutIcon from "@mui/icons-material/Logout";
import MenuIcon from "@mui/icons-material/Menu";
import React, { useContext } from "react";
import { UserContext } from "./store/UserContext";
import TitleTab from "./headerComponents/TitleTab";
import LinkTab from "./headerComponents/LinkTab";
import { useState } from "react";
import DrawerLinkTab from "./headerComponents/DrawerLinkTab";

const Header = () => {
  const { userState, handleLogOut } = useContext(UserContext);
  const { isLoggedIn } = userState;
  const { userPosAddr } = userState;
  const matches = useMediaQuery("(min-width:750px)");
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  const theme = createTheme({
    palette: {
      primary: {
        main: "#ff0700",
      },
    },
  });

  // 모바일 환경에서 Drawer를 열고 닫는 함수
  const toggleDrawer = () => {
    setIsDrawerOpen((curr) => !curr);
  };

  // 모바일 환경에서 Drawer에 들어가는 Component의 정보
  const drawerTab = [
    {
      icon: <HomeIcon />,
      component: <DrawerLinkTab to="/" text="홈" />,
    },
    {
      icon: <GroupAddIcon />,
      component: <DrawerLinkTab to="/restaurant/list" text="방 만들기" />,
    },
    {
      icon: <AccountBoxIcon />,
      component: <DrawerLinkTab to="/myPage/0" text="마이 페이지" />,
    },
    {
      icon: <LogoutIcon />,
      component: (
        <Typography
          onClick={() => {
            handleLogOut();
          }}
          sx={{
            fontWeight: 100,
            fontFamily: "var(--font-family-DoHyeon)",
          }}
        >
          {"로그아웃"}
        </Typography>
      ),
    },
  ];

  // 모바일 환경에서 Drawer에 들어가는 Box와 그 산하 List
  const list = (userPosAddr) => (
    <Box sx={{ width: 180 }} role="presentation">
      <List
        aria-labelledby="nested-list-subheader"
        subheader={
          <ListSubheader
            color="primary"
            component="div"
            id="nested-list-subheader"
          >
            Deliverus Menu
          </ListSubheader>
        }
      >
        {drawerTab.map((tab, index) => (
          <ListItem onClick={toggleDrawer} key={index} disableGutters>
            {(tab.component.props.children === "로그아웃" || userPosAddr) && (
              <ListItemButton>
                <ListItemIcon>{tab.icon}</ListItemIcon>
                {tab.component}
              </ListItemButton>
            )}
          </ListItem>
        ))}
      </List>
    </Box>
  );

  // 실질적으로 렌더링 되는 컴포넌트입니다.
  // 미디어 쿼리의 값에 따라 렌더링 되는 컴포넌트가 변경됩니다.
  const renderedComponent = matches ? (
    <header>
      <ThemeProvider theme={theme}>
        <Box sx={{ flexGrow: 1 }}>
          <AppBar position="static" sx={{ backgroundColor: "#FFFFFF" }}>
            <Toolbar>
              <Typography
                variant="h5"
                component="div"
                sx={{
                  flexGrow: 1,
                  fontWeight: 100,
                  fontFamily: "var(--font-family-DoHyeon)",
                }}
              >
                <div style={{ display: "flex" }}>
                  <TitleTab />
                  <div style={{ margin: "0 0 0 100px" }}>
                    {isLoggedIn && userPosAddr && (
                      <>
                        <LinkTab to="/" name="홈" />
                        <LinkTab to="/restaurant/list" name="방 만들기" />
                        <LinkTab to="/myPage/0" name="마이페이지" />
                      </>
                    )}
                  </div>
                </div>
              </Typography>
              {isLoggedIn && (
                <Button
                  color="primary"
                  onClick={() => handleLogOut()}
                  sx={{
                    marginRight: "20px",
                    border: 1,
                    fontFamily: "inherit",
                  }}
                >
                  로그아웃
                </Button>
              )}
            </Toolbar>
          </AppBar>
        </Box>
      </ThemeProvider>
    </header>
  ) : (
    <header>
      <ThemeProvider theme={theme}>
        <Box sx={{ flexGrow: 1 }}>
          <AppBar position="static" sx={{ backgroundColor: "#FFFFFF" }}>
            <Toolbar>
              <Typography
                variant="h5"
                component="div"
                sx={{
                  flexGrow: 1,
                  fontWeight: 100,
                  fontFamily: "var(--font-family-DoHyeon)",
                }}
              >
                <TitleTab />
              </Typography>
              {isLoggedIn && (
                <React.Fragment>
                  <IconButton
                    color="primary"
                    onClick={toggleDrawer}
                    aria-label="menu"
                  >
                    <MenuIcon />
                  </IconButton>
                  <Drawer
                    anchor="right"
                    open={isDrawerOpen}
                    onClose={toggleDrawer}
                  >
                    {list(userPosAddr)}
                  </Drawer>
                </React.Fragment>
              )}
            </Toolbar>
          </AppBar>
        </Box>
      </ThemeProvider>
    </header>
  );

  return renderedComponent;
};

export default Header;
