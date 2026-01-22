import { WebStorageStateStore } from "oidc-client-ts";
import { darkTheme, lightTheme } from "./theme";

// Debug environment variables
const issuer = import.meta.env.VITE_ISSUER;
const clientId = import.meta.env.VITE_CLIENT_ID;
const clientSecret = import.meta.env.VITE_CLIENT_SECRET;
const baseUrl = import.meta.env.VITE_BASE_URL;

console.log('[Config] Environment variables:', {
  VITE_ISSUER: issuer,
  VITE_CLIENT_ID: clientId,
  VITE_CLIENT_SECRET: clientSecret ? '***' : undefined,
  VITE_BASE_URL: baseUrl
});

export default {
  server: {
    baseUrl: baseUrl || 'https://ai.longstorymedia.com',
    // baseUrl: import.meta.env.VITE_BASE_URL,
    // baseUrl: 'http://192.168.0.71:8000',
  },
  auth: {
    clientId: clientId,
    clientSecret: clientSecret,
    tokenEndpoint: issuer + '/token',
    logoutEndpoint: issuer + '/logout',
    usrmgrBaseUrl: issuer + '/api',
    oidc: {
      authority: issuer,
      client_id: clientId,
      client_secret: clientSecret,
      redirect_uri: window.location.origin + '/callback',
      response_type: 'code',
      scope: 'openid groups profile email offline_access',
      post_logout_redirect_uri: window.location.origin,
      userStore: new WebStorageStateStore({ store: window.sessionStorage })
    }
  },
  theme: {
    light: lightTheme,
    dark: darkTheme
  }
}