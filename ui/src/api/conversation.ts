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
    path: `api/users/${userId}/conversations`
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

export const removeConversation = async (accessToken: string, id: number) => {
  await req({
    method: 'DELETE',
    headers: getHeaders(accessToken),
    path: `api/conversations/${id}`
  });
}

export const pause = async (accessToken: string, conversationId: number) => req({
  method: 'POST',
  path: `api/conversations/${conversationId}/pause`,
  headers: getHeaders(accessToken)
})

export const resume = async (accessToken: string, conversationId: number) => req({
  method: 'POST',
  path: `api/conversations/${conversationId}/resume`,
  headers: getHeaders(accessToken)
})

export const cancel = async (accessToken: string, conversationId: number) => req({
  method: 'POST',
  path: `api/conversations/${conversationId}/cancel`,
  headers: getHeaders(accessToken)
})