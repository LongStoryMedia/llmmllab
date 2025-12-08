import { MessageRoleValues } from "../types/MessageRole";
import { gen, getHeaders, req } from "./base";
/**
 * Shared streaming wrapper for calling the server streaming endpoints.
 * path should be a relative API path (e.g. 'chat/completions' or 'chat/conversations/{id}/replay')
 */
async function* streamEndpoint(accessToken, path, body, abortSignal) {
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
            const chatResponse = chunk;
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
    }
    catch (error) {
        console.error(`Streaming endpoint ${path} error:`, error);
        throw error;
    }
}
export async function* chat(accessToken, message, abortSignal) {
    yield* streamEndpoint(accessToken, `chat/completions`, message, abortSignal);
}
export async function* replay(accessToken, conversationId, message, abortSignal) {
    const body = {
        timestamp: message?.created_at ? message.created_at.toString() : ''
    };
    if (message) {
        body.message = message;
    }
    yield* streamEndpoint(accessToken, `chat/conversations/${conversationId}/replay`, body, abortSignal);
}
export const getMessages = async (accessToken, conversationId) => req({
    method: 'GET',
    headers: getHeaders(accessToken),
    path: `chat/conversations/${conversationId}/messages`
});
export const deleteMessage = async (accessToken, conversationId, messageId) => req({
    method: 'DELETE',
    headers: getHeaders(accessToken),
    path: `chat/conversations/${conversationId}/messages/${messageId}`
});
