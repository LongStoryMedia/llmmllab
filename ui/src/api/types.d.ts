import { UserConfig } from "../types/UserConfig";
import { Message } from "../types/Message";
import type { UIMessage, TextUIPart, ReasoningUIPart, ToolUIPart, DynamicToolUIPart, SourceUrlUIPart, SourceDocumentUIPart, FileUIPart, StepStartUIPart } from "ai";
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
    /**
     * Optional API version override. If not provided, uses the default from config.
     * Use this to target specific API versions for compatibility.
     */
    apiVersion?: string;
};
export type UserAttribute = {
    Name: "uid" | "sn" | "cn" | "mail" | "dn";
    Values: [string];
    ByteValues: [string];
};
export type UserInfo = {
    DN: string;
    Attributes: UserAttribute[];
};
export type NewUserReq = {
    Username: string;
    Password: string;
    CN: string;
    Mail: string;
};
export type LllabUser = {
    id: string;
    username: string;
    config: UserConfig;
    createdAt: string;
};
/**
 * Convert our Message type to AI SDK UIMessage format
 */
export declare function convertToUIMessage(message: Message): UIMessage;
/**
 * Convert AI SDK UIMessage back to our Message format
 */
export declare function convertFromUIMessage(uiMessage: UIMessage): Message;
/**
 * Convert array of Messages to UIMessages
 */
export declare function convertMessagesToUI(messages: Message[]): UIMessage[];
/**
 * Convert array of UIMessages to Messages
 */
export declare function convertMessagesFromUI(uiMessages: UIMessage[]): Message[];
