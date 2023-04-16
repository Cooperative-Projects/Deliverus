//import { Link } from "react-router-dom";
import { useState } from "react";
import { API } from "../../utils/config";
import * as status from "../../utils/status";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Link from "@mui/material/Link";
import Paper from "@mui/material/Paper";
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid";
import TextField from "@mui/material/TextField";
import { useDispatch } from "react-redux";
import { logIn } from "../store/sessionSlice";
import Alert from '@mui/material/Alert';
import Collapse from '@mui/material/Collapse';

const LogIn = ({ togglePage }) => {
  const dispatch = useDispatch();

  // 알람창을 위한 변수입니다.
  const [alertOpen, setAlertOpen] = useState(false);

  // open과 이하의 2개 함수는 로딩 모달 관련 함수입니다!!
  const [open, setOpen] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleClose = () => {
    setOpen(false);
  };

  const handleOpen = () => {
    setOpen(true);
  };

  const handleIdInput = (e) => {
    setUsername(e.target.value);
  };

  const handlePwInput = (e) => {
    setPassword(e.target.value);
  };

  const getLogInResult = async (data) => {
    const response = await fetch(`${API.LOGIN}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      credentials: "include",
      body: JSON.stringify(data),
    });
    // respones의 status를 확인해서 상황에 알맞은 Error를 던집니다.
    status.handleLogInResponse(response.status);
    return response.json();
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const loginData = { userid: username, passwd: password };
    handleOpen();
    try {
      const result = await getLogInResult(loginData);
      console.log("Login Success", result);
      // handleLogIn(result.userId);
      dispatch(logIn(result.userId));
    } catch (error) {
      // 아이디가 존재하지 않는 에러
      if (error.name === "NoUserError") {
        alert(error.message);
      }
      // 비밀번호가 틀린 에러
      else if (error.name === "WrongPasswordError") {
        alert(error.message);
      } else {
        // alert(error.message);
        setAlertOpen(true);
      }
      console.log(`${error.name} : ${error.message}`);
    } finally {
      setUsername("");
      setPassword("");
      handleClose();
    }
  };

  return (
    <Grid item xs={12} sm={6} md={5} component={Paper} elevation={6} square>
      <Box
        sx={{
          my: 8,
          mx: 4,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <Typography component="h1" variant="h5">
          로그인
        </Typography>
        <Box component="form" noValidate onSubmit={handleSubmit} sx={{ mt: 1 }}>
          <TextField
            margin="normal"
            required
            fullWidth
            id="id"
            label="Id"
            name="id"
            autoFocus
            value={username}
            onChange={handleIdInput}
          />
          <TextField
            margin="normal"
            required
            fullWidth
            name="password"
            label="Password"
            type="password"
            id="password"
            value={password}
            onChange={handlePwInput}
          />

          <Button
            type="submit"
            fullWidth
            variant="contained"
            sx={{ mt: 3, mb: 2 }}
          >
            Sign In
          </Button>
          <Collapse in={alertOpen}>
            <Alert
                severity="error"
                sx={{ mb: 2 }}
            >
              로그인에 실패하였습니다!
            </Alert>
          </Collapse>
        </Box>
        <Link href="#" onClick={() => togglePage()} variant="body2">
          {"계정이 없나요? 회원가입하러 가기"}
        </Link>
      </Box>
    </Grid>
  );
};

export default LogIn;
