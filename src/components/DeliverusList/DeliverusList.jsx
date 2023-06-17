// 가게 정보에 대한 Card
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import React, {useContext, useState} from "react";
import styles from './DeliverusList.module.css'
import {UserContext} from "../store/UserContext";
import {useLocation} from "react-router-dom";
import LocationOnIcon from "@mui/icons-material/LocationOn";
import RecruitingPartyList from "../restaurant/RecruitingPartyList";
import {IconButton} from "@mui/material";

const restaurantCategories = ["한식", "분식", "치킨", "아시안/양식", "족발/보쌈", "돈까스/일식", "카페/디저트", "찜탕", "패스트푸드", "피자"];

// 해당 가게 주문을 위해 모집 중인 파티방을 보여주는 컴포넌트입니다.
const DeliverusList = () => {
    // 설정한 도로명 주소, 위도/경도 가져오기
    const {userState} = useContext(UserContext);
    const {userPosAddr} = userState;

    // 가게 정보 리스트
    const {state} = useLocation();

    // 전체적인 partyList
    const [recruitingPartyList, setRecruitingPartyList] = useState(state ? state.recruitingPartyList : null)

    // 필터링된 partyList
    const [filteredPartyList, setFilteredPartyList] = useState(recruitingPartyList);

    const [currentCategories, setCurrentCategories] = useState(state ? state.category : "all");

    const handleCategories = (e) => {
        const category = (e.target.textContent === "전체") ? "all" : e.target.textContent;
        setCurrentCategories(category);

        if (recruitingPartyList !== null) {
            const tempPartyList = [];
            recruitingPartyList.map((item, idx) => {
                if (item.category === category) {
                    tempPartyList.push(item);
                }
                return null;
            })
            setFilteredPartyList(tempPartyList);
        }
    }

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
            <Box sx={{display: "flex", justifyContent: "flex-start", alignItems: "center", mt: 3}}>
                <IconButton
                    sx={{px: 0}}
                    color="primary"
                    aria-label="more"
                >
                    <LocationOnIcon/>
                </IconButton>
                <Typography variant="h2">{userPosAddr}</Typography>
            </Box>
            <div className={styles.list_card}>
                {(filteredPartyList && (Array.isArray(filteredPartyList) && filteredPartyList.length !== 0)) ? (
                    <RecruitingPartyList partyList={filteredPartyList}/>
                ) : (
                    <Box
                        sx={{
                            backgroundColor: "info.main",
                            textAlign: "center",
                            paddingY: "10vh",
                            borderRadius: 3,
                        }}
                    >
                        <Typography>{currentCategories === "all" || !currentCategories?
                            "주변에 모집중인 딜리버스가 없어요..." :
                            `주변에 ${currentCategories}을 시키는 딜리버스가 없어요...`}</Typography>
                    </Box>
                )}
            </div>
        </div>
    );
}

export default DeliverusList;
