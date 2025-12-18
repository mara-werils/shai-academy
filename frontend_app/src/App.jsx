import React, { useEffect, useState } from "react";
import {
  AppBar,
  Toolbar,
  Box,
  Button,
  Container,
  CssBaseline,
  Grid,
  Paper,
  Tab,
  Tabs,
  TextField,
  Typography,
  Alert,
  Card,
  CardContent,
  CardActions,
  Chip,
  Divider,
  Stack,
  LinearProgress,
  Avatar,
  Skeleton,
} from "@mui/material";
import LoginIcon from "@mui/icons-material/Login";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DownloadIcon from "@mui/icons-material/Download";
import RocketLaunchIcon from "@mui/icons-material/RocketLaunch";
import { login, register, generate, generateVideo } from "./api";

export default function App() {
  const [token, setToken] = useState("");
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [tasks, setTasks] = useState([]);
  const [tab, setTab] = useState(0);
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [authMode, setAuthMode] = useState("login"); // login | register
  const [genType, setGenType] = useState("image"); // 'image' | 'video'

  useEffect(() => {
    const saved = localStorage.getItem("gen_token");
    if (saved) setToken(saved);
  }, []);

  const handleAuth = async () => {
    setError("");
    try {
      let res;
      if (authMode === "register") {
        if (!email || !name || !password) {
          setError("Email, Name and Password are required.");
          return;
        }
        res = await register({ email, name, password });
      } else {
        if (!email || !password) {
          setError("Email and Password are required.");
          return;
        }
        res = await login(email, password);
      }
      setToken(res.access_token);
      localStorage.setItem("gen_token", res.access_token);
      setTab(1);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        "Auth failed. Please check credentials or API availability."
      );
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError("Prompt is required.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const isVideo = genType === "video";
      if (!token) {
        setError("Please sign in first.");
        setLoading(false);
        return;
      }

      const res = isVideo
        ? await generateVideo({ prompt, token })
        : await generate({ prompt, token });

      console.log("Generation response:", res);
      const url = isVideo ? res.video_url : res.image_url;

      if (res && url) {
        const newTask = {
          ...res,
          prompt,
          type: isVideo ? "video" : "image",
          url,
        };
        setTasks((prev) => [newTask, ...prev]);
        setPrompt("");
      } else {
        setError("Invalid response from server. Missing result URL.");
      }
    } catch (err) {
      console.error("Generation error:", err);
      setError(
        err.response?.data?.detail || 
        err.message || 
        "Generation request failed. Check API availability."
      );
    } finally {
      setLoading(false);
    }
  };

  const isAuthed = Boolean(token);
  const latest = tasks[0];

  const quickPrompts = [
    "Cinematic portrait of an astronaut in neon city",
    "Studio photo of a wooden chair on marble floor",
    "Isometric illustration of a smart home dashboard",
    "Moody landscape, misty mountains at sunrise",
  ];

  return (
    <>
      <CssBaseline />
      <AppBar
        position="sticky"
        color="transparent"
        elevation={0}
        sx={{ backdropFilter: "blur(10px)", borderBottom: "1px solid #e2e8f0" }}
      >
        <Toolbar sx={{ display: "flex", gap: 1 }}>
          <RocketLaunchIcon color="primary" />
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            shai.academy
          </Typography>
          <Chip
            label={isAuthed ? "Signed in" : "Guest"}
            color={isAuthed ? "success" : "default"}
            size="small"
          />
          <Button
            color="primary"
            variant="contained"
            size="small"
            startIcon={<RocketLaunchIcon />}
            onClick={() => setTab(1)}
          >
            Launch Studio
          </Button>
        </Toolbar>
      </AppBar>

      {loading && <LinearProgress color="primary" />}

      <Container maxWidth="lg" sx={{ py: 4, display: "grid", gap: 3 }}>
        <Paper
          elevation={0}
          sx={{
            p: 4,
            borderRadius: 4,
            background:
              "linear-gradient(135deg, rgba(59,130,246,0.10), rgba(236,72,153,0.12))",
            border: "1px solid rgba(148,163,184,0.25)",
            display: "grid",
            gap: 2,
          }}
        >
          <Box display="flex" alignItems="center" gap={1}>
            <Chip label="shai.academy" color="primary" size="small" />
            <Chip label="Creative Lab" variant="outlined" size="small" />
          </Box>
          <Typography variant="h4" fontWeight={800} sx={{ letterSpacing: -0.5 }}>
            Craft production-grade visuals with one click.
          </Typography>
          <Typography color="text.secondary" maxWidth="720px">
            Describe the scene, hit Generate, and get a ready-to-ship image. Built for product teams and rapid prototyping.
          </Typography>
          <Stack direction={{ xs: "column", sm: "row" }} spacing={1}>
            <Button
              variant="contained"
              color="primary"
              startIcon={<RocketLaunchIcon />}
              onClick={() => setTab(1)}
              disabled={!isAuthed}
            >
              Start generating
            </Button>
            <Button variant="outlined" onClick={() => setTab(0)}>
              Authenticate
            </Button>
          </Stack>
          <Divider sx={{ borderColor: "rgba(148,163,184,0.4)" }} />
          <Stack direction={{ xs: "column", md: "row" }} spacing={1} flexWrap="wrap" useFlexGap>
            {quickPrompts.map((qp) => (
              <Chip
                key={qp}
                label={qp}
                variant="outlined"
                onClick={() => {
                  setPrompt(qp);
                  setTab(1);
                }}
                sx={{ backgroundColor: "rgba(255,255,255,0.6)" }}
              />
            ))}
          </Stack>
        </Paper>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            {tab === 0 ? (
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  borderRadius: 3,
                  border: "1px solid #e2e8f0",
                  backgroundColor: "white",
                  height: "100%",
                }}
              >
                <Stack spacing={2}>
                  <Typography variant="subtitle1" fontWeight={700}>
                    {authMode === "login" ? "Sign in" : "Register"}
                  </Typography>
                  <Stack direction="row" spacing={1}>
                    <Chip
                      label="Login"
                      color={authMode === "login" ? "primary" : "default"}
                      variant={authMode === "login" ? "filled" : "outlined"}
                      onClick={() => setAuthMode("login")}
                      clickable
                    />
                    <Chip
                      label="Register"
                      color={authMode === "register" ? "primary" : "default"}
                      variant={authMode === "register" ? "filled" : "outlined"}
                      onClick={() => setAuthMode("register")}
                      clickable
                    />
                  </Stack>
                  <TextField
                    label="Email"
                    type="email"
                    fullWidth
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                  {authMode === "register" && (
                    <TextField
                      label="Name"
                      fullWidth
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                    />
                  )}
                  <TextField
                    label="Password"
                    type="password"
                    fullWidth
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                  />
                  <Button
                    variant="contained"
                    startIcon={<LoginIcon />}
                    onClick={handleAuth}
                    size="large"
                  >
                    {authMode === "login" ? "Sign In" : "Register"}
                  </Button>
                </Stack>
              </Paper>
            ) : (
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  borderRadius: 3,
                  border: "1px solid #e2e8f0",
                  backgroundColor: "white",
                  height: "100%",
                }}
              >
                <Stack spacing={2} height="100%">
                  <Box
                    display="flex"
                    alignItems="center"
                    justifyContent="space-between"
                    gap={2}
                  >
                    <Box>
                      <Typography variant="h6" fontWeight={700}>
                        Prompt your vision
                      </Typography>
                      <Typography color="text.secondary" variant="body2">
                        Be descriptive for the best results.
                      </Typography>
                    </Box>
                    <Stack direction="row" spacing={1}>
                      <Chip
                        label="Image"
                        color={genType === "image" ? "primary" : "default"}
                        variant={genType === "image" ? "filled" : "outlined"}
                        onClick={() => setGenType("image")}
                        clickable
                      />
                      <Chip
                        label="Video"
                        color={genType === "video" ? "primary" : "default"}
                        variant={genType === "video" ? "filled" : "outlined"}
                        onClick={() => setGenType("video")}
                        clickable
                      />
                    </Stack>
                  </Box>
                  {error && (
                    <Alert severity="error">{error}</Alert>
                  )}
                  <TextField
                    label="Prompt"
                    multiline
                    minRows={6}
                    fullWidth
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                  />
                  <Button
                    fullWidth
                    size="large"
                    variant="contained"
                    color="primary"
                    startIcon={<CloudUploadIcon />}
                    onClick={handleGenerate}
                    disabled={loading}
                    sx={{ mt: "auto" }}
                  >
                    {loading ? "Generating..." : "Generate"}
                  </Button>
                </Stack>
              </Paper>
            )}
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper
              elevation={0}
              sx={{
                p: 3,
                borderRadius: 3,
                border: "1px solid #e2e8f0",
                backgroundColor: "white",
                height: "100%",
              }}
            >
              <Stack spacing={2}>
                <Box display="flex" alignItems="center" gap={1}>
                  <Avatar sx={{ bgcolor: "#312e81" }}>AI</Avatar>
                  <Box>
                    <Typography fontWeight={700}>Live preview</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Latest generated image
                    </Typography>
                  </Box>
                </Box>
                {latest ? (() => {
                  const url = latest.url || latest.video_url || latest.image_url;
                  const isVideoFile = /\.(webm|mp4)$/i.test(url || "");
                  if (latest.type === "video" && isVideoFile) {
                    return (
                      <Box
                        component="video"
                        src={url}
                        controls
                        sx={{
                          width: "100%",
                          maxWidth: "100%",
                          borderRadius: 2,
                          height: 360,
                          objectFit: "contain",
                          border: "1px solid #e2e8f0",
                          backgroundColor: "#000",
                          display: "block",
                        }}
                      />
                    );
                  }
                  return (
                    <Box
                      component="img"
                      src={url}
                      alt="latest"
                      sx={{
                        width: "100%",
                        maxWidth: "100%",
                        borderRadius: 2,
                        height: 360,
                        objectFit: "contain",
                        border: "1px solid #e2e8f0",
                        display: "block",
                      }}
                    />
                  );
                })() : (
                  <Skeleton
                    variant="rectangular"
                    height={360}
                    sx={{ borderRadius: 2 }}
                  />
                )}
                <Divider />
                {latest ? (
                  <Stack spacing={0.5}>
                    <Typography variant="body2" color="text.secondary">
                      Prompt
                    </Typography>
                    <Typography>{latest.prompt}</Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      Task ID: {latest.task_id}
                    </Typography>
                  </Stack>
                ) : (
                  <Typography color="text.secondary">
                    Generate your first image to see it here.
                  </Typography>
                )}
              </Stack>
            </Paper>
          </Grid>
        </Grid>

        <Paper
          elevation={0}
          sx={{
            p: 3,
            borderRadius: 3,
            border: "1px solid #e2e8f0",
            backgroundColor: "white",
          }}
        >
          <Box
            display="flex"
            alignItems="center"
            justifyContent="space-between"
            mb={2}
          >
            <Typography variant="h6" fontWeight={700}>
              History & Results
            </Typography>
            <Chip
              label={`${tasks.length} item${tasks.length === 1 ? "" : "s"}`}
              size="small"
            />
          </Box>
          {!tasks.length && (
            <Typography color="text.secondary">
              No tasks yet. Generate your first image.
            </Typography>
          )}
          <Grid container spacing={2} sx={{ mt: 1 }}>
            {tasks.map((task, idx) => (
              <Grid item xs={12} md={4} key={idx}>
                <Card
                  variant="outlined"
                  sx={{ borderRadius: 2, borderColor: "#e2e8f0" }}
                >
                  <CardContent>
                    <Stack direction="row" alignItems="center" spacing={1}>
                      <Chip
                        label={task.type === "video" ? "Video" : "Image"}
                        color="primary"
                        size="small"
                      />
                      <Typography variant="body2" color="text.secondary" noWrap>
                        {task.prompt}
                      </Typography>
                    </Stack>
                    {(() => {
                      const url = task.url || task.video_url || task.image_url;
                      const isVideoFile = /\.(webm|mp4)$/i.test(url || "");
                      if (task.type === "video" && isVideoFile) {
                        return (
                          <Box
                            component="video"
                            src={url}
                            controls
                            sx={{
                              width: "100%",
                              maxWidth: "100%",
                              borderRadius: 2,
                              mt: 1.5,
                              height: 180,
                              objectFit: "contain",
                              border: "1px solid #e2e8f0",
                              backgroundColor: "#000",
                              display: "block",
                            }}
                          />
                        );
                      }
                      return (
                        <Box
                          component="img"
                          src={url}
                          alt="result"
                          sx={{
                            width: "100%",
                            maxWidth: "100%",
                            borderRadius: 2,
                            mt: 1.5,
                            height: 180,
                            objectFit: "contain",
                            border: "1px solid #e2e8f0",
                            display: "block",
                          }}
                        />
                      );
                    })()}
                    <Divider sx={{ my: 1.2 }} />
                    <Typography variant="body2" color="success.main">
                      Status: {task.status || "completed"}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {task.task_id}
                    </Typography>
                  </CardContent>
                  <CardActions>
                    <Button
                      size="small"
                      startIcon={<DownloadIcon />}
                      component="a"
                      href={task.url || task.image_url || task.video_url}
                      download={`${task.type || "file"}-${task.task_id}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Download
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      </Container>

      <Box
        component="footer"
        sx={{
          py: 3,
          textAlign: "center",
          color: "text.secondary",
          fontSize: 14,
        }}
      >
        Crafted for product teams • shai.academy
      </Box>
    </>
  );
}

