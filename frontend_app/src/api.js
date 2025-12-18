import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

export const login = async (username, password) => {
  const res = await axios.post(`${API_BASE}/auth/login`, {
    email: username,
    password,
  });
  return res.data;
};

export const register = async ({ email, name, password }) => {
  const res = await axios.post(`${API_BASE}/auth/register`, {
    email,
    name,
    password,
  });
  return res.data;
};

export const generate = async ({ prompt, steps = 20, token }) => {
  const res = await axios.post(
    `${API_BASE}/generate/image`,
    { prompt, steps },
    {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    }
  );
  return res.data;
};

export const generateVideo = async ({ prompt, steps = 20, token }) => {
  const res = await axios.post(
    `${API_BASE}/generate/video`,
    { prompt, steps },
    {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    }
  );
  return res.data;
};

