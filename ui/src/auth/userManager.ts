import { UserManager } from "oidc-client-ts";
import config from "../config";

export const userManager = new UserManager(config.auth.oidc);

export const logoutSession = async () => {
  try {
    await userManager.removeUser();
    // window.location.href = config.auth.oidc.post_logout_redirect_uri;
  } catch (error) {
    console.error('Error during logout:', error);
  }
};

