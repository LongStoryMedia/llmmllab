import { Conversation } from "../types/Conversation";
import { getHeaders, req } from "./base";

export const startConversation = async (accessToken: string) => {
  return req<Conversation>({
    method: 'POST',
    headers: getHeaders(accessToken),
    path: 'chat/conversations'
  });
}

export const getUserConversations = async (accessToken: string, userId: string) => {
  return await req<Conversation[]>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: `users/${userId}/conversations`
  });
}

export const getManyConversations = async (accessToken: string) =>
  req<Conversation[]>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: 'chat/conversations'
  });

export const getOneConversation = async (accessToken: string, id: number) =>
  req<Conversation>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: `chat/conversations/${id}`
  });

export const removeConversation = async (accessToken: string, id: number) => {
  await req({
    method: 'DELETE',
    headers: getHeaders(accessToken),
    path: `chat/conversations/${id}`
  });
}

export const cancel = async (accessToken: string) => req({
  method: 'GET',
  path: `chat/cancel`,
  headers: getHeaders(accessToken)
})