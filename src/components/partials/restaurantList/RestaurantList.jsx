import {useContext, useEffect, useState} from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography"
import React from 'react'
import styles from './Restaurant.module.css'
import LocationOnIcon from '@mui/icons-material/LocationOn';
import {useLocation, useNavigate} from 'react-router-dom';
import useMediaQuery from "@mui/material/useMediaQuery";
import {ButtonBase, Grid, IconButton} from "@mui/material";
import {UserContext} from "../../store/UserContext";
import {API} from "../../../utils/config";
import * as status from "../../../utils/status";

const restaurantCategories = ["한식", "분식", "치킨", "아시안/양식", "족발/보쌈", "돈까스/일식", "카페/디저트", "찜탕", "패스트푸드", "피자"];

export default function RestaurantList() {
    // 설정한 도로명 주소, 위도/경도 가져오기
    const { userState, handleLogOut} = useContext(UserContext);
    const { userPosAddr, userPos } = userState;

    // 가게 정보 리스트
    const {state} = useLocation();
    const [restInfoList, setRestInfoList] = useState(state ? state.restInfoList : null);

    const [currentCategories, setCurrentCategories] = useState(state ? state.category : "all");

    const handleCategories = (e) => {
        const category = e.target.textContent;
        setCurrentCategories(category);
    }

    useEffect(() => {
        // Header의 방 만들기 버튼을 통해 들어온 경우
        if(state === null || typeof state === "undefined"){
            // 모든 가게 리스트를 받아옵니다.
            fetch(`${API.RESTAURANT_LIST}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                credentials: "include",
                body: JSON.stringify({
                    latitude: userPos.lat,
                    longitude: userPos.lng,
                }),
            })
                .then((respones) => {
                    status.handleRestaurantResponse(respones.status);
                    return respones.json();
                })
                .then((data) => {
                    console.log("Respones Data from Restaurant LIST API : ", data);
                    setRestInfoList(data);
                })
                .catch((error) => {
                    // 로그인 만료 에러인 경우 로그아웃 실행
                    if (error.name === "LoginExpirationError") {
                        console.log(`${error.name} : ${error.message}`);
                        handleLogOut();
                    }
                    console.log(`${error.name} : ${error.message}`);
                });
        }
    })
    return (
        <div className={styles.list_body}>
            <div className={styles.list_all} onClick={e => setCurrentCategories('all')}>전체</div>
            <div className={styles.list_category_wrapper}>
                {restaurantCategories.map((items, idx) => {
                    return (
                        <div key={idx} className={styles.list_category} onClick={handleCategories}>{items}</div>
                    );
                })}
            </div>
            <Box sx={{ display: "flex", justifyContent: "flex-start"}}>
                <IconButton
                    sx={{}}
                    color="primary"
                    aria-label="more"
                >
                    <LocationOnIcon />
                </IconButton>
                <h4>{userPosAddr}</h4>
            </Box>
            <div className={styles.list_card}>
                {restInfoList && restInfoList.map((item, idx) => {
                    if (
                        currentCategories === 'all' ||
                        currentCategories === item.category
                    ) {
                        return (
                            <RestaurantCard
                                name={item.name}
                                rating={item.rating}
                                id={item.restaurant_id}
                                category={item.category}
                                intro={item.intro}
                                deliveryFee={item.deliveryFee}
                                minOrderPrice={item.minOrderPrice}
                                key={idx}
                            />
                        );
                    }
                    return null;
                })}
            </div>
        </div>
    );
}

export function RestaurantCard({name, rating, id, category, intro, deliveryFee, minOrderPrice}) {
    const navigate = useNavigate();
    const isMobile = useMediaQuery("(max-width: 750px)");

    const handleClickStoreInfo = () => {
        navigate(`/restaurant/information/${id}`);
    }
    let image = null;
    if (!name) {
        image = require(`../../../images/delivery-cat.png`);
    } else {
        try {
            const currentCategory = category.replace("/", ",");
            // console.log(`../../../images/${currentCategory}/${name}.png`);
            image = require(`../../../images/${currentCategory}/${name}.png`);
        } catch (e) {
            console.log(e);
            image = require(`../../../images/delivery-cat.png`);
        }
    }

    // 가게 설명 주석입니다. 재활용 할 수 있을 것 같아서 남겨요
    // <Typography variant="body2" color="text.secondary" sx={{
        // overflow: "hidden",
        // textOverflow: "ellipsis",
    //     display: "-webkit-box",
    //     WebkitLineClamp: 2, 
    //     WebkitBoxOrient: "vertical",
    //     textAlign: "start"
    // }}>{intro}</Typography>

    rating = rating % 1 === 0 ? rating + '.0' : rating;
    
    // pc화면
    if (!isMobile) {
        return (
            <ButtonBase sx={{ marginY: 1,}} onClick={
                () => {
                    setTimeout(() => {
                        handleClickStoreInfo()
                    }, 200)
                }
            }>
                <Grid container sx={{
                    alignItems: "center",
                    padding: 2,
                    marginX: "auto",
                    width: "100%",
                    borderRadius: 2,
                    boxShadow: "0px 3px 5px rgba(0, 0, 0, 0.2)",
                    backgroundColor: "#fff"
                }}>
                    <Grid item xs={3}>
                        <img src={image} style={{
                            width: "120px",
                            aspectRatio: "1 / 1",
                            borderRadius: "6px",
                            boxShadow: "0px 3px 5px rgba(0, 0, 0, 0.5)",
                        }} />
                    </Grid>
                    <Grid item xs={9} paddingLeft={"4px"} container spacing={0.5} alignItems={"center"}>
                        <Grid item xs={8}>
                            <Typography variant="h5" noWrap textAlign="start" textOverflow={"ellipsis"}>{name}</Typography>
                        </Grid>
                        <Grid item xs={4}>
                            <Typography variant="body2" sx={{border: 1, borderRadius: 3, width:"100%", textAlign: "center", backgroundColor: "info.main"}}>
                                            ⭐ {rating} / 5.0&nbsp;
                            </Typography>
                        </Grid>
                        <Grid item xs={12} container>
                            <Typography variant="h6" textAlign="start" color="text.secondary" justifySelf={"flex-start"}>
                                {intro}
                            </Typography>
                        </Grid>
                        <Grid item xs={12} container>
                            <Typography variant="body2" color="text.secondary" justifySelf={"flex-start"}>
                                배달비: {deliveryFee.toLocaleString()}원
                            </Typography>
                        </Grid>
                        <Grid item xs={12} container>
                            <Typography variant="body2" color="text.secondary" justifySelf={"flex-start"}>
                                최소 주문: {minOrderPrice.toLocaleString()}원
                            </Typography>
                        </Grid>
                    </Grid>
                </Grid>
            </ButtonBase>
        );
    } else { // 모바일 화면
        return (
            <ButtonBase sx={{marginY: 1,}} onClick={
                () => {
                    setTimeout(() => {
                        handleClickStoreInfo()
                    }, 200)
                }
            }>
                <Grid container sx={{
                    alignItems: "center",
                    padding: 2,
                    marginX: "auto",
                    width: "100%",
                    borderRadius: 2,
                    boxShadow: "0px 3px 5px rgba(0, 0, 0, 0.2)",
                    backgroundColor: "#fff"
                }}>
                    <Grid item xs={4}>
                        <img src={image} style={{
                            width: "80px",
                            aspectRatio: "1 / 1",
                            borderRadius: "6px",
                            boxShadow: "0px 3px 5px rgba(0, 0, 0, 0.5)",
                        }} />
                    </Grid>
                    <Grid item xs={8} paddingLeft={"4px"} container spacing={0.5}>
                        {/* <Stack direction="row"> */}
                            {/* <Stack direction="column" justifyContent="flex-start" spacing={1} minWidth={0}> */}
                        <Grid item xs={6}>
                            <Typography noWrap textAlign="start" textOverflow={"ellipsis"}>{name}</Typography>
                        </Grid>
                        <Grid item xs={6}>
                            <Typography variant="body2" sx={{border: 1, borderRadius: 3, width:"100%", textAlign: "center", backgroundColor: "info.main"}}>
                                            ⭐ {rating} / 5.0&nbsp;
                            </Typography>
                        </Grid>
                        <Grid item xs={6} container>
                            <Typography variant="body2" color="text.secondary" justifySelf={"flex-start"}>
                                배달비: {deliveryFee.toLocaleString()}원
                            </Typography>
                        </Grid>
                        <Grid item xs={6}>
                        
                        </Grid>
                        <Grid item xs={12} container>
                            <Typography variant="body2" color="text.secondary" justifySelf={"flex-start"}>
                                최소 주문: {minOrderPrice.toLocaleString()}원
                            </Typography>
                        </Grid>
                            {/* </Stack> */}
                        
                        {/* </Stack> */}
                    </Grid>
                </Grid>
            </ButtonBase>
        );
    }
} 
