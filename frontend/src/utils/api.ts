// src/utils/api.ts
//
// Reusable Axios instance that points to your Python backend
// URL is provided via NEXT_PUBLIC_BACKEND_URL in .env.local.

import axios from 'axios';

/**
 * Axios instance pre‑configured with the backend base URL.
 * All frontend code should import this instead of creating Axios on the fly.
 */
export const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_BACKEND_URL,
});
