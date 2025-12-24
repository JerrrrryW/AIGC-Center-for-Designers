import axios from 'axios';

const envBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const defaultBaseUrl = `${window.location.protocol}//${window.location.hostname}:8000`;

export const apiBaseUrl = (envBaseUrl || defaultBaseUrl).replace(/\/$/, '');

export const api = axios.create({
  baseURL: apiBaseUrl,
});
