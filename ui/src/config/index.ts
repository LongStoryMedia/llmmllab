import { WebStorageStateStore } from "oidc-client-ts";
import { darkTheme, lightTheme } from "./theme";

export default {
  server: {
    baseUrl: import.meta.env.VITE_BASE_URL
  },
  auth: {
    clientId: import.meta.env.VITE_CLIENT_ID,
    clientSecret: import.meta.env.VITE_CLIENT_SECRET,
    tokenEndpoint: import.meta.env.VITE_ISSUER + '/token',
    logoutEndpoint: import.meta.env.VITE_ISSUER + '/logout',
    usrmgrBaseUrl: import.meta.env.VITE_ISSUER + '/api',
    oidc: {
      authority: import.meta.env.VITE_ISSUER,
      client_id: import.meta.env.VITE_CLIENT_ID,
      client_secret: import.meta.env.VITE_CLIENT_SECRET,
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