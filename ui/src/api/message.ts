import { Message } from "../types/Message";
import { MessageRoleValues } from "../types/MessageRole";
import { MessageContentTypeValues } from "../types/MessageContentType";
import { gen, getHeaders, req } from "./base";

export async function* chat(accessToken: string, message: Message, abortSignal?: AbortSignal): AsyncGenerator<string> {
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
      // Check if the message and content are properly structured
      if (chunk.message?.content && Array.isArray(chunk.message.content) && chunk.message.content.length > 0) {
        // Filter out OBSERVER role messages (status messages) from main content stream
        if (chunk.message.role === MessageRoleValues.OBSERVER) {
          // Log status messages but don't yield them as main content
          console.log('[STATUS]', chunk.message.content[0].text);
          // TODO: Emit to status channel/event system
          continue;
        }

        // Ensure text exists before yielding it
        yield chunk.message.content[0].text ?? '';
      } else if (chunk.message) {
        // Handle case where content might not be properly formatted
        console.warn('Received improperly formatted message content:', chunk.message);

        // If content exists but isn't an array, try to convert it
        if (chunk.message.content && !Array.isArray(chunk.message.content)) {
          const textContent = String(chunk.message.content);

          // Don't yield observer messages as main content
          if (chunk.message.role !== MessageRoleValues.OBSERVER) {
            yield textContent;
          } else {
            console.log('[STATUS]', textContent);
          }

          // Fix the message content structure for downstream code
          chunk.message.content = [{
            type: MessageContentTypeValues.TEXT,
            text: textContent
          }];
        } else {
          // Default empty string if we can't extract content (and not observer)
          if (chunk.message.role !== MessageRoleValues.OBSERVER) {
            yield '';
          }
        }
      }

      // Ensure the message has conversation_id
      if (chunk.message && !chunk.message.conversation_id) {
        chunk.message.conversation_id = message.conversation_id;
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
  req<Message[]>({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: `chat/conversations/${conversationId}/messages`
  });
