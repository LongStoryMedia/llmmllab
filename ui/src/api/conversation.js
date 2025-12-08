import { getHeaders, req } from "./base";
export const startConversation = async (accessToken) => {
    return req({
        method: 'POST',
        headers: getHeaders(accessToken),
        path: 'chat/conversations'
    });
};
export const getUserConversations = async (accessToken, userId) => {
    return await req({
        method: 'GET',
        headers: getHeaders(accessToken),
        path: `users/${userId}/conversations`
    });
};
export const getManyConversations = async (accessToken) => req({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: 'chat/conversations'
});
export const getOneConversation = async (accessToken, id) => req({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: `chat/conversations/${id}`
});
export const removeConversation = async (accessToken, id) => {
    await req({
        method: 'DELETE',
        headers: getHeaders(accessToken),
        path: `chat/conversations/${id}`
    });
};
export const cancel = async (accessToken) => req({
    method: 'GET',
    path: `chat/cancel`,
    headers: getHeaders(accessToken)
});
