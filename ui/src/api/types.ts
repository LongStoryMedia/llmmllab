import { UserConfig } from "../types/UserConfig";
import { Message } from "../types/Message";
import { MessageContentTypeValues } from "../types/MessageContentType";
import { MessageRole } from "../types/MessageRole";
import type { UIMessage, TextUIPart, ReasoningUIPart, ToolUIPart, DynamicToolUIPart, SourceUrlUIPart, SourceDocumentUIPart, FileUIPart, StepStartUIPart } from "ai";
import { nanoid } from "nanoid";

export type UIPart = TextUIPart | ReasoningUIPart | ToolUIPart | DynamicToolUIPart | SourceUrlUIPart | SourceDocumentUIPart | FileUIPart | StepStartUIPart;

export type RequestOptions = {
  method?: 'POST' | 'GET' | 'PUT' | 'DELETE';
  headers?: HeadersInit;
  body?: string;
  path: string;
  signal?: AbortSignal;
  timeout?: number;
  requestKey?: string;
  baseUrl?: string;
};

export type UserAttribute = {
  Name: "uid" | "sn" | "cn" | "mail" | "dn";
  Values: [string];
  ByteValues: [string];
}

export type UserInfo = {
  DN: string,
  Attributes: UserAttribute[];
}

export type NewUserReq = {
  Username: string;
  Password: string;
  CN: string;
  Mail: string;
}

export type LllabUser = {
  id: string;
  username: string;
  config: UserConfig;
  createdAt: string;
}

/**
 * Convert our Message type to AI SDK UIMessage format
 */
export function convertToUIMessage(message: Message): UIMessage {
  const parts: UIPart[] = [];

  // Convert message content to UI parts
  if (message.content) {
    for (const content of message.content) {
      if (content.type === MessageContentTypeValues.TEXT && content.text) {
        parts.push({
          type: 'text',
          text: content.text
        });
      } else if (content.type === MessageContentTypeValues.IMAGE && content.url) {
        const fileUiPart: FileUIPart = {
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
    role: message.role as UIMessage['role'],
    parts
  };
}

/**
 * Convert AI SDK UIMessage back to our Message format
 */
export function convertFromUIMessage(uiMessage: UIMessage): Message {
  const content = uiMessage.parts.map(part => {
    if (part.type === 'text') {
      return {
        type: MessageContentTypeValues.TEXT,
        text: part.text
      };
    } else if (part.type === 'file') {
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
    role: uiMessage.role as MessageRole,
    content
  };
}

/**
 * Convert array of Messages to UIMessages
 */
export function convertMessagesToUI(messages: Message[]): UIMessage[] {
  return messages.map(convertToUIMessage);
}

/**
 * Convert array of UIMessages to Messages
 */
export function convertMessagesFromUI(uiMessages: UIMessage[]): Message[] {
  return uiMessages.map(convertFromUIMessage);
}