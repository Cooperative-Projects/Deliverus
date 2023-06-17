// API Respond가 정상적일 때 오는 번호
const successNum = [200, 201, 203];

/**
 * @param statusNum Respond의 status 번호
 * @returns {boolean} 오류가 없는 정상적인 API Respond인 경우 true 리턴
 */
const checkStatus = (statusNum) => successNum.includes(statusNum);

/**
 * Register API 관련 Error 처리 함수
 * @param statusNum Respond의 status 번호
 */
const handleRegisterResponse = (statusNum) => {
    if (!checkStatus(statusNum)) {
        const error = new Error();

        switch (statusNum) {
            // 중복된 아이디, 닉네임 입력 에러 처리
            case 406:
                error.message = "중복된 아이디, 닉네임이 존재합니다";
                error.name = "DuplicationError";
                break;

            // 기타 에러 처리
            default:
                error.message = `${statusNum} 오류`;
                error.name = "UnkownError";
        }

        throw error;
    }
};

/**
 * Login API 관련 Error 처리 함수
 * @param statusNum Respond의 status 번호
 */
const handleLogInResponse = (statusNum) => {
    if (!checkStatus(statusNum)) {
        const error = new Error("Error occurred.");

        switch (statusNum) {
            // 로그인 시도 거부 에러 처리
            case 404:
                error.message = "로그인이 거부되었습니다.";
                error.name = "LoginError";
                break;

            // 기타 에러 처리
            default:
                error.message = `${statusNum} 오류`;
                error.name = "UnkownError";
        }

        throw error;
    }
};

/**
 * Restaurant API 관련 Error 처리 함수
 * @param statusNum Respond의 status 번호
 */
const handleRestaurantResponse = (statusNum) => {
    if (!checkStatus(statusNum)) {
        const error = new Error("Error occurred.");

        switch (statusNum) {
            // 요청한 것에 대한 데이터가 없을 때 에러 처리
            case 204:
                error.message = "요청한 데이터가 없습니다.";
                error.name = "NoDataError";
                break;

            // 로그인 시간이 만료된 에러 처리
            case 401:
                error.message = "로그인 시간이 만료되었습니다.";
                error.name = "LoginExpirationError";
                break;

            // 기타 에러 처리
            default:
                error.message = `${statusNum} 오류`;
                error.name = "UnkownError";
        }

        throw error;
    }
};

/**
 * Party API 관련 Error 처리 함수
 * @param statusNum Respond의 status 번호
 */
const handlePartyResponse = (statusNum) => {
    if (!checkStatus(statusNum)) {
        const error = new Error("Error occurred.");

        switch (statusNum) {
            // 데이터가 없는 경우
            case 204:
                error.message = "데이터가 없습니다";
                error.name = "NoDataError";
                break;

            // 로그인 시간이 만료된 에러 처리
            case 401:
                error.message = "로그인 시간이 만료되었습니다.";
                error.name = "LoginExpirationError";
                break;

            // 파티방에 이미 참여했는데 또 참여하려할 때 에러 처리
            case 406:
                error.message = "이미 파티방에 참여하고 있습니다";
                error.name = "DuplicateJoinError";
                break;

            // 기타 에러 처리
            default:
                error.message = `${statusNum} 오류`;
                error.name = "UnkownError";
        }

        throw error;
    }
};

/**
 * AI 추천 API 관련 Error 처리 함수
 * @param statusNum Respond의 status 번호
 */
const handleRecommendResponse = (statusNum) => {
    if (!checkStatus(statusNum)) {
        const error = new Error("Error occurred.");

        switch (statusNum) {
            // 로그인 시간이 만료된 에러 처리
            case 401:
                error.message = "로그인 시간이 만료되었습니다.";
                error.name = "LoginExpirationError";
                break;

            // 기타 에러 처리
            default:
                error.message = `${statusNum} 오류`;
                error.name = "UnkownError";
                break;
        }

        throw error;
    }
};

/**
 * Chat API 관련 Error 처리 함수
 * @param statusNum Respond의 status 번호
 */
const handleChatResponse = (statusNum) => {
    if (!checkStatus(statusNum)) {
        const error = new Error("Error occurred.");

        switch (statusNum) {
            // 로그인 시간이 만료된 에러 처리
            case 401:
                error.message = "로그인 시간이 만료되었습니다.";
                error.name = "LoginExpirationError";
                break;

            case 204:
                error.message = "처음 연결한 경우입니다.";
                break;

            // 기타 에러 처리
            default:
                error.message = `${statusNum} 오류`;
                error.name = "UnkownError";
                break;
        }

        throw error;
    }
};

export {
    handleLogInResponse,
    handleRegisterResponse,
    handleRestaurantResponse,
    handlePartyResponse,
    handleRecommendResponse,
    handleChatResponse
};
