import { MessageContentTypeValues } from "../types/MessageContentType";
import { nanoid } from "nanoid";
/**
 * Convert our Message type to AI SDK UIMessage format
 */
export function convertToUIMessage(message) {
    const parts = [];
    // Convert message content to UI parts
    if (message.content) {
        for (const content of message.content) {
            if (content.type === MessageContentTypeValues.TEXT && content.text) {
                parts.push({
                    type: 'text',
                    text: content.text
                });
            }
            else if (content.type === MessageContentTypeValues.IMAGE && content.url) {
                const fileUiPart = {
                    type: 'file',
                    url: content.url,
                    mediaType: 'image/png' // Default, could be enhanced
                };
                parts.push(fileUiPart);
            }
        }
    }
    return {
        id: message.id?.toString() || nanoid(),
        role: message.role,
        parts
    };
}
/**
 * Convert AI SDK UIMessage back to our Message format
 */
export function convertFromUIMessage(uiMessage) {
    const content = uiMessage.parts.map(part => {
        if (part.type === 'text') {
            return {
                type: MessageContentTypeValues.TEXT,
                text: part.text
            };
        }
        else if (part.type === 'file') {
            return {
                type: MessageContentTypeValues.IMAGE,
                url: part.url
            };
        }
        return {
            type: MessageContentTypeValues.TEXT,
            text: ''
        };
    });
    return {
        id: parseInt(uiMessage.id) || undefined,
        role: uiMessage.role,
        content
    };
}
/**
 * Convert array of Messages to UIMessages
 */
export function convertMessagesToUI(messages) {
    return messages.map(convertToUIMessage);
}
/**
 * Convert array of UIMessages to Messages
 */
export function convertMessagesFromUI(uiMessages) {
    return uiMessages.map(convertFromUIMessage);
}
