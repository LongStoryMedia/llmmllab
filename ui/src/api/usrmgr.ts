import { userManager } from "../auth";
import config from "../config";
import { getHeaders, req, getToken } from "./base";
import { LllabUser, NewUserReq, UserInfo } from "./types";

export const getUserInfo = async () => {
  const user = await userManager.getUser();
  if (!user) {
    throw new Error('User not authenticated');
  }

  return req<UserInfo[]>({
    method: 'GET',
    baseUrl: config.auth.usrmgrBaseUrl,
    path: `search?filter=(uid=${user.profile.preferred_username})`,
    headers: getHeaders(getToken(user))
  })
}

export const getAllUserInfo = async () => {
  const user = await userManager.getUser();
  if (!user) {
    throw new Error('User not authenticated');
  }
  return req<UserInfo[]>({
    method: 'GET',
    baseUrl: config.auth.usrmgrBaseUrl,
    path: 'search',
    headers: getHeaders(getToken(user))
  });
}

export const getLllabUsers = async () => {
  const user = await userManager.getUser();
  if (!user) {
    throw new Error('User not authenticated');
  }
  return req<LllabUser[]>({
    method: 'GET',
    baseUrl: config.server.baseUrl,
    path: `users`,
    headers: getHeaders(getToken(user))
  });
}

export const updatePassword = async (oldPassword: string, newPassword: string) => {
  const user = await userManager.getUser();
  if (!user) {
    throw new Error('User not authenticated');
  }

  return req<{ message: string, success: boolean }>({
    method: 'POST',
    baseUrl: config.auth.usrmgrBaseUrl,
    path: 'change-password',
    headers: getHeaders(getToken(user)),
    body: JSON.stringify({
      username: user.profile.preferred_username,
      oldPassword,
      newPassword
    })
  });
}

export const addUser = async (newUser: NewUserReq) => {
  const user = await userManager.getUser();
  if (!user) {
    throw new Error('User not authenticated');
  }

  return req<{ message: string, success: boolean }>({
    method: 'POST',
    baseUrl: config.auth.usrmgrBaseUrl,
    path: 'user',
    headers: getHeaders(getToken(user)),
    body: JSON.stringify(newUser)
  });
}

export const deleteUser = async (username: string) => {
  const user = await userManager.getUser();
  if (!user) {
    throw new Error('User not authenticated');
  }

  return req<{ message: string, success: boolean }>({
    method: 'DELETE',
    baseUrl: config.auth.usrmgrBaseUrl,
    path: 'user',
    headers: getHeaders(getToken(user)),
    body: JSON.stringify({ username })
  });
}