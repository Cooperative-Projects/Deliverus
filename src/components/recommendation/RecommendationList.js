import * as React from "react";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import LooksOneOutlinedIcon from "@mui/icons-material/LooksOneOutlined";
import LooksTwoOutlinedIcon from "@mui/icons-material/LooksTwoOutlined";
import Looks3OutlinedIcon from "@mui/icons-material/Looks3Outlined";
import Looks4OutlinedIcon from "@mui/icons-material/Looks4Outlined";
import Looks5OutlinedIcon from "@mui/icons-material/Looks5Outlined";
import Image from "mui-image";

import { Typography } from "@mui/material";
// import Card from "@mui/material/Card";
import { ScrollCard } from "./ScrollCard";
import useMediaQuery from "@mui/material/useMediaQuery";
import { Fragment } from "react";
import ListSubheader from "@mui/material/ListSubheader";
import { useNavigate } from "react-router-dom";
import { ScrollMenu, VisibilityContext } from "react-horizontal-scrolling-menu";
import "react-horizontal-scrolling-menu/dist/styles.css";
import useDrag from "./useDrag.ts";

function onWheel(apiObj, ev) {
  const isThouchpad = Math.abs(ev.deltaX) !== 0 || Math.abs(ev.deltaY) < 15;

  if (isThouchpad) {
    ev.stopPropagation();
    return;
  }

  if (ev.deltaY < 0) {
    apiObj.scrollNext();
  } else if (ev.deltaY > 0) {
    apiObj.scrollPrev();
  }
}

export default function RecommendationList(props) {
  const iconList = [
    <LooksOneOutlinedIcon />,
    <LooksTwoOutlinedIcon />,
    <Looks3OutlinedIcon />,
    <Looks4OutlinedIcon />,
    <Looks5OutlinedIcon />,
  ];
  const isMobile = useMediaQuery("(max-width: 1400px)");
  const navigate = useNavigate();

  // 추천 카테고리를 선택했을 때 callback 함수
  const handleClick = (event) => {
    event.preventDefault();
    //console.log(event.target.textContent);
    navigate(`/restaurant/list`, {
      state: {
        restInfoList: props.restInfoList,
        category: event.target.textContent,
      },
    });
  };

  // 슬라이드를 위한 드래그 관련 함수입니다.
  const { dragStart, dragStop, dragMove, dragging } = useDrag();
  const handleDrag =
    ({ scrollContainer }) =>
    (ev) =>
      dragMove(ev, (posDiff) => {
        if (scrollContainer.current) {
          scrollContainer.current.scrollLeft += posDiff;
        }
      });

  const [selected, setSelected] = React.useState("");
  const handleItemClick = (itemId) => () => {
    if (dragging) {
      return false;
    }
    setSelected(selected !== itemId ? itemId : "");
  };

  return (
    <Fragment>
      <Typography paddingBottom={"16px"} variant="h2">
        🚀 AI가 추천해주는 현재 TOP 5 음식
      </Typography>
      <ScrollMenu
        onMouseLeave={dragStop}
        onMouseDown={() => dragStart}
        onMouseUp={() => dragStop}
        onMouseMove={handleDrag}
        onWheel={onWheel}
      >
        {props.list.map((item, idx) => {
          const slashIdx = item.indexOf("/");
          const title =
            slashIdx == -1
              ? item
              : item.substring(0, slashIdx) + item.substring(slashIdx + 1);
          const carouselImg = require(`../../images/carousel/${title}.jpg`);

          return (
            <ScrollCard
              key={idx}
              title={item}
              itemId={idx}
              idx={idx}
              img={carouselImg}
              onClick={handleItemClick(idx)}
              selected={idx === selected}
              restInfoList={props.restInfoList}
            />
          );
        })}
      </ScrollMenu>
    </Fragment>
  );
  // PC 화면인 경우

  // <List
  //   sx={{
  //     width: "100%",
  //     maxWidth: 270,
  //     bgcolor: "background.paper",
  //     borderRadius: "16px",
  //     border: 3,
  //     position: "absolute",
  //     top: "300px",
  //     left: "80vw",
  //   }}
  //   aria-label="contacts"
  //   subheader={<ListSubheader>📈 AI가 추천해주는 Top5</ListSubheader>}
  // >
  //   {props.list.map((item, idx) => {
  //     if (idx <= 5) {
  //       return (
  //         <ListItem onClick={handleClick} key={idx}>
  //           <ListItemButton>
  //             <ListItemIcon>{iconList[idx]}</ListItemIcon>
  //             <ListItemText primary={item} />
  //           </ListItemButton>
  //         </ListItem>
  //       );
  //     }
  //   })}
  // </List>
}
