import { userManager } from "../auth";
import config from "../config";
import { getHeaders, req, getToken } from "./base";
export const getUserInfo = async () => {
    const user = await userManager.getUser();
    if (!user) {
        throw new Error('User not authenticated');
    }
    return req({
        method: 'GET',
        baseUrl: config.auth.usrmgrBaseUrl,
        path: `search?filter=(uid=${user.profile.preferred_username})`,
        headers: getHeaders(getToken(user))
    });
};
export const getAllUserInfo = async () => {
    const user = await userManager.getUser();
    if (!user) {
        throw new Error('User not authenticated');
    }
    return req({
        method: 'GET',
        baseUrl: config.auth.usrmgrBaseUrl,
        path: 'search',
        headers: getHeaders(getToken(user))
    });
};
export const getLllabUsers = async () => {
    const user = await userManager.getUser();
    if (!user) {
        throw new Error('User not authenticated');
    }
    return req({
        method: 'GET',
        baseUrl: config.server.baseUrl,
        path: `users`,
        headers: getHeaders(getToken(user))
    });
};
export const updatePassword = async (oldPassword, newPassword) => {
    const user = await userManager.getUser();
    if (!user) {
        throw new Error('User not authenticated');
    }
    return req({
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
};
export const addUser = async (newUser) => {
    const user = await userManager.getUser();
    if (!user) {
        throw new Error('User not authenticated');
    }
    return req({
        method: 'POST',
        baseUrl: config.auth.usrmgrBaseUrl,
        path: 'user',
        headers: getHeaders(getToken(user)),
        body: JSON.stringify(newUser)
    });
};
export const deleteUser = async (username) => {
    const user = await userManager.getUser();
    if (!user) {
        throw new Error('User not authenticated');
    }
    return req({
        method: 'DELETE',
        baseUrl: config.auth.usrmgrBaseUrl,
        path: 'user',
        headers: getHeaders(getToken(user)),
        body: JSON.stringify({ username })
    });
};
