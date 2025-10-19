import { Message } from "../types/Message";
import { ChatResponse } from "../types/ChatResponse";
import { MessageRoleValues } from "../types/MessageRole";
import { gen, getHeaders, req } from "./base";

export interface ChatChunk {
  content: string;
  thinking?: string;
  channels?: Record<string, unknown>;
  observer_messages?: string[];
  done: boolean;
}

export async function* chat(accessToken: string, message: Message, abortSignal?: AbortSignal): AsyncGenerator<ChatChunk> {
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

      // Extract text content
      let textContent = '';
      if (chatResponse.message?.content && Array.isArray(chatResponse.message.content) && chatResponse.message.content.length > 0) {
        // Filter out OBSERVER role messages (status messages) from main content stream
        if (chatResponse.message.role === MessageRoleValues.OBSERVER) {
          // Log status messages but don't yield them as main content
          console.log('[STATUS]', chatResponse.message.content[0].text);
          // Skip OBSERVER messages from text content
          textContent = '';
        } else {
          // Extract normal content
          textContent = chatResponse.message.content[0].text ?? '';
        }
      } else if (chatResponse.message) {
        // Handle case where content might not be properly formatted
        console.warn('Received improperly formatted message content:', chatResponse.message);

        // If content exists but isn't an array, try to convert it
        if (chatResponse.message.content && !Array.isArray(chatResponse.message.content)) {
          const content = String(chatResponse.message.content);

          // Don't include observer messages as main content
          if (chatResponse.message.role !== MessageRoleValues.OBSERVER) {
            textContent = content;
          } else {
            console.log('[STATUS]', content);
            textContent = '';
          }
        }
      }
      // console.log('chunk:', chatResponse);

      // Yield structured chunk with all ChatResponse fields
      yield {
        content: textContent,
        thinking: chatResponse.thinking,
        channels: chatResponse.channels,
        observer_messages: chatResponse.observer_messages,
        done: chatResponse.done || false
      };

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
