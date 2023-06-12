import React, {useEffect, useState} from 'react';
const {kakao} = window;

// 위치를 선택할 때 사용되는 카카오맵 컴포넌트입니다.
export default function PositionSettingMap(props) {

    const [kakaoMap, setKakaoMap] = useState();
    const [marker, setMarker] = useState(null);
    const [infowindow, setInfowindow] = useState(null);

    useEffect(() => {
        kakao.maps.load(() => {
            const container = document.getElementById("map"); // 지도를 표시할 div
            const options = {
                center: new kakao.maps.LatLng(props.initLatLng.lat, props.initLatLng.lng), // 지도의 중심좌표
                level: 3 // 지도의 확대 레벨
            };
            const map = new kakao.maps.Map(container, options); // 지도 생성
            // 지도 확대 축소를 제어할 수 있는  줌 컨트롤을 생성합니다
            var zoomControl = new kakao.maps.ZoomControl();
            map.addControl(zoomControl, kakao.maps.ControlPosition.RIGHT);

            // 지도에 표시할 원을 생성합니다. 원의 중심은 가게 위치입니다.
            if (props.radius) {
                var circle = new kakao.maps.Circle({
                    center: new kakao.maps.LatLng(props.resLatLng.lat, props.resLatLng.lng),  // 원의 중심좌표 입니다
                    radius: props.radius, // 미터 단위의 원의 반지름입니다
                    strokeWeight: 5, // 선의 두께입니다
                    strokeColor: '#75B8FA', // 선의 색깔입니다
                    strokeOpacity: 1, // 선의 불투명도 입니다 1에서 0 사이의 값이며 0에 가까울수록 투명합니다
                    strokeStyle: 'dashed', // 선의 스타일 입니다
                    fillColor: '#CFE7FF', // 채우기 색깔입니다
                    fillOpacity: 0.35  // 채우기 불투명도 입니다
                });

                // 지도에 원을 표시합니다
                circle.setMap(map);
            }

            setKakaoMap(map); // 생성된 지도를 상태로 저장
        });
    }, []);

    useEffect(() => {
        if (kakaoMap) {
            const clickHandler = (mouseEvent) => {
                if (marker) {
                    marker.setMap(null);
                }
                if (infowindow) {
                    infowindow.close();
                }

                const latlng = mouseEvent.latLng;
                const newMarker = new kakao.maps.Marker({
                    position: latlng
                });

                // 클릭한 위치에 대한 주소를 표시할 인포윈도우입니다
                const newInfowindow = new kakao.maps.InfoWindow({zindex:1});

                // 좌표를 이용해서 도로명 주소 받아오기
                const geocoder = new kakao.maps.services.Geocoder();
                geocoder.coord2Address(latlng.La, latlng.Ma, function(result, status) {
                    let detailAddr = "";
                    if (status === kakao.maps.services.Status.OK) {
                        // 도로명 주소가 없으면 지번 주소로 설정
                        detailAddr = !!result[0].road_address ? result[0].road_address.address_name : result[0].address.address_name;

                        const myLatlng = {
                            lat : latlng.Ma,
                            lng : latlng.La
                        }
                        // 상위 컴포넌트에게 위치, 도로명 주소 넘겨주기
                        if(props.propFunction) {
                            props.propFunction(myLatlng, detailAddr);
                        }
                    }

                    setMarker(newMarker);
                    newMarker.setMap(kakaoMap);

                    // 인포윈도우에 클릭한 위치에 대한 법정동 상세 주소정보를 표시합니다
                    setInfowindow(newInfowindow);
                    newInfowindow.setContent(detailAddr);
                    newInfowindow.open(kakaoMap, newMarker);
                });
            };
            kakao.maps.event.addListener(kakaoMap, "click", clickHandler);
            return () => {
                kakao.maps.event.removeListener(kakaoMap, "click", clickHandler);
            };
        }
    }, [kakaoMap, marker]);

    return (
        <div id='map' style={{
            width: '100%',
            height: '100%'
        }}></div>
    )

}
