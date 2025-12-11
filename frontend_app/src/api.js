import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

export const login = async (username, password) => {
  const res = await axios.post(`${API_BASE}/auth/login`, {
    username,
    password,
  });
  return res.data;
};

export const generate = async ({ prompt, token }) => {
  const res = await axios.post(
    `${API_BASE}/generate/image`,
    { prompt },
    {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    }
  );
  return res.data;
};

