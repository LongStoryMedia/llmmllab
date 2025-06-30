import { ChatMessage } from "../types/ChatMessage";
import { ChatRequest } from "../types/ChatRequest";
import { gen, getHeaders, req } from "./base";

export async function* chat(accessToken: string, message: ChatRequest, abortSignal?: AbortSignal): AsyncGenerator<string> {
  console.log('Sending message to chat API:', message);

  try {
    const generator = gen({
      body: JSON.stringify(message),
      method: 'POST',
      headers: getHeaders(accessToken),
      path: `api/conversations/${message.conversation_id}/messages`,
      signal: abortSignal
    });

    for await (const chunk of generator) {
      if (chunk.message?.content) {
        yield chunk.message.content;
      }

      if (chunk.done) {
        break;
      }
    }
  } catch (error) {
    console.error('Chat API error:', error);
    throw error;
  }
};

export const getMessages = async (accessToken: string, conversationId: number) =>
  req<ChatMessage[]>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: `api/conversations/${conversationId}/messages`
  });

