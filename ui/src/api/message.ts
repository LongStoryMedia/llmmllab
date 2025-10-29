import { Message } from "../types/Message";
import { ChatResponse } from "../types/ChatResponse";
import { MessageRoleValues } from "../types/MessageRole";
import { gen, getHeaders, req } from "./base";

export async function* chat(accessToken: string, message: Message, abortSignal?: AbortSignal): AsyncGenerator<ChatResponse> {
  console.log('Sending message to chat API:', message);

  try {
    const generator = gen({
      body: JSON.stringify(message),
      method: 'POST',
      headers: getHeaders(accessToken),
      path: `chat/completions`,
      signal: abortSignal
    });

    for await (const chunk of generator) {
      const chatResponse = chunk as ChatResponse;

      // Log observer messages but continue to yield the full ChatResponse
      if (chatResponse.message?.role === MessageRoleValues.OBSERVER) {
        if (chatResponse.message?.content && Array.isArray(chatResponse.message.content) && chatResponse.message.content.length > 0) {
          console.log('[STATUS]', chatResponse.message.content[0].text);
        }
      }

      // Yield the full ChatResponse directly
      yield chatResponse;

      if (chatResponse.done) {
        break;
      }
    }
  } catch (error) {
    console.error('Chat API error:', error);
    throw error;
  }
};

export const getMessages = async (accessToken: string, conversationId: number) =>
  req<Message[]>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: `chat/conversations/${conversationId}/messages`
  });

export const deleteMessage = async (accessToken: string, conversationId: number, messageId: number) =>
  req<{ status: string; message: string }>({
    method: 'DELETE',
    headers: getHeaders(accessToken),
    path: `chat/conversations/${conversationId}/messages/${messageId}`
  });

export const bulkDeleteMessagesFromTimestamp = async (
  accessToken: string,
  conversationId: number,
  fromTimestamp?: Date | string
) => {
  // Handle both Date objects and ISO string timestamps
  let timestampString = '';
  if (fromTimestamp) {
    if (fromTimestamp instanceof Date) {
      timestampString = fromTimestamp.toISOString();
    } else if (typeof fromTimestamp === 'string') {
      timestampString = fromTimestamp;
    }
  }

  return req<{ status: string; message: string; deleted_count: number }>({
    method: 'DELETE',
    headers: getHeaders(accessToken),
    path: `chat/conversations/${conversationId}/messages/bulk/from-timestamp?from_timestamp=${encodeURIComponent(timestampString)}`
  });
};
