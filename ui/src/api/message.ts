import { Message } from "../types/Message";
import { ChatResponse } from "../types/ChatResponse";
import { MessageRoleValues } from "../types/MessageRole";
import { gen, getHeaders, req } from "./base";

/**
 * Shared streaming wrapper for calling the server streaming endpoints.
 * path should be a relative API path (e.g. 'chat/completions' or 'chat/conversations/{id}/replay')
 */
async function* streamEndpoint(accessToken: string, path: string, body: Message | { message?: Message, timestamp: string }, abortSignal?: AbortSignal): AsyncGenerator<ChatResponse> {
  // Log the request for debugging
  console.log('Streaming to endpoint:', path, body);

  try {
    const generator = gen({
      body: body ? JSON.stringify(body) : undefined,
      method: 'POST',
      headers: getHeaders(accessToken),
      path,
      signal: abortSignal
    });

    for await (const chunk of generator) {
      const chatResponse = chunk as ChatResponse;

      // Standardized observer logging for all streaming endpoints
      if (chatResponse.message?.role === MessageRoleValues.OBSERVER) {
        if (chatResponse.message?.content && Array.isArray(chatResponse.message.content) && chatResponse.message.content.length > 0) {
          console.log('[STATUS]', chatResponse.message.content[0].text);
        }
      }

      yield chatResponse;

      if (chatResponse.done && chatResponse.finish_reason !== 'tool_call') {
        break;
      }
    }
  } catch (error) {
    console.error(`Streaming endpoint ${path} error:`, error);
    throw error;
  }
}

export async function* chat(accessToken: string, message: Message, abortSignal?: AbortSignal): AsyncGenerator<ChatResponse> {
  yield* streamEndpoint(accessToken, `chat/completions`, message, abortSignal);
}

export async function* replay(accessToken: string, conversationId: number, message: Message, abortSignal?: AbortSignal): AsyncGenerator<ChatResponse> {
  const body: { message?: Message, timestamp: string } = {
    timestamp: message?.created_at ? message.created_at.toString() : ''
  };
  if (message) {
    body.message = message;
  }

  yield* streamEndpoint(accessToken, `chat/conversations/${conversationId}/replay`, body, abortSignal);
}

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

export const replayFromTimestamp = async (
  accessToken: string,
  conversationId: number,
  message?: Message,
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

  const body = {
    timestamp: encodeURIComponent(timestampString),
    message: message ? JSON.stringify(message) : undefined
  }

  return req<{ status: string; message: string; deleted_count: number }>({
    method: 'POST',
    headers: getHeaders(accessToken),
    path: `chat/conversations/${conversationId}/replay`,
    body: JSON.stringify(body)
  });
};
