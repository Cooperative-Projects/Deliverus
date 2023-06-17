import React, { useEffect, useState } from 'react';
const { kakao } = window;

export default function KakoMap() {

    const [kakaoMap, setKakaoMap] = useState();
    const [marker, setMarker] = useState(null);

    useEffect(() => {
        kakao.maps.load(() => {
          const container = document.getElementById("map"); // 지도를 표시할 div
          const options = {
            center: new kakao.maps.LatLng(37.57600923748876, 126.9012721298886), // 지도의 중심좌표
            level: 5 // 지도의 확대 레벨
          };
          const map = new kakao.maps.Map(container, options); // 지도 생성
          setKakaoMap(map); // 생성된 지도를 상태로 저장
        });
      }, []);

    //   useEffect(() => {
    //     if (kakaoMap) {
    //       kakao.maps.event.addListener(kakaoMap, "click", function(mouseEvent) {
    //         if (marker) {
    //             marker.setMap(null);
    //           }
    //         let latlng = mouseEvent.latLng;
    //         let newMarker = new kakao.maps.Marker({
    //           position: latlng
    //         });
    //         console.log(newMarker);
    //         setMarker(newMarker);
    //         newMarker.setMap(kakaoMap);
    //       });
    //     }
    //   }, [kakaoMap, marker]);
        
    useEffect(() => {
        if (kakaoMap) {
          const clickHandler = (mouseEvent) => {
            if (marker) {
              marker.setMap(null);
            }
            const latlng = mouseEvent.latLng;
            const newMarker = new kakao.maps.Marker({
              position: latlng
            });
            setMarker(newMarker);
            newMarker.setMap(kakaoMap);
          };
          kakao.maps.event.addListener(kakaoMap, "click", clickHandler);
          return () => {
            kakao.maps.event.removeListener(kakaoMap, "click", clickHandler);
          };
        }
      }, [kakaoMap, marker]);

        return (
        <div id='map' style={{
            width: '1000px',
            height: '500px'
        }}></div>
      )
    
}
