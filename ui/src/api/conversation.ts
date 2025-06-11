import { Conversation } from "../types/Conversation";
import { getHeaders, req } from "./base";

export const startConversation = async (accessToken: string, model: string) => {
  const { conversation_id } = await req<{ conversation_id: number }>({
    method: 'POST',
    headers: getHeaders(accessToken),
    body: JSON.stringify({
      model,
      title: ''
    }),
    path: 'api/conversations'
  });

  // Fetch the new conversation details
  return await req<Conversation>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: `api/conversations/${conversation_id}`
  });
}

export const getUserConversations = async (accessToken: string, userId: string) => {
  return await req<Conversation[]>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: `api/conversations/user/${userId}`
  });
}

export const getManyConversations = async (accessToken: string) =>
  req<Conversation[]>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: 'api/conversations'
  });

export const getOneConversation = async (accessToken: string, id: number) =>
  req<Conversation>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: `api/conversations/${id}`
  });

export const updateConversationTitle = async (accessToken: string, id: number, title: string) => {
  await req({
    method: 'PUT',
    headers: getHeaders(accessToken),
    body: JSON.stringify({ title }),
    path: `api/conversations/${id}`
  });
}

export const removeConversation = async (accessToken: string, id: number) => {
  await req({
    method: 'DELETE',
    headers: getHeaders(accessToken),
    path: `api/conversations/${id}`
  });
}