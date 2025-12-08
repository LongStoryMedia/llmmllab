import { Message } from "../types/Message";
import { ChatResponse } from "../types/ChatResponse";
export declare function chat(accessToken: string, message: Message, abortSignal?: AbortSignal): AsyncGenerator<ChatResponse>;
export declare function replay(accessToken: string, conversationId: number, message: Message, abortSignal?: AbortSignal): AsyncGenerator<ChatResponse>;
export declare const getMessages: (accessToken: string, conversationId: number) => Promise<Message[]>;
export declare const deleteMessage: (accessToken: string, conversationId: number, messageId: number) => Promise<{
    status: string;
    message: string;
}>;
