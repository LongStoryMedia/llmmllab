import { Conversation } from "../types/Conversation";
export declare const startConversation: (accessToken: string) => Promise<Conversation>;
export declare const getUserConversations: (accessToken: string, userId: string) => Promise<Conversation[]>;
export declare const getManyConversations: (accessToken: string) => Promise<Conversation[]>;
export declare const getOneConversation: (accessToken: string, id: number) => Promise<Conversation>;
export declare const removeConversation: (accessToken: string, id: number) => Promise<void>;
export declare const cancel: (accessToken: string) => Promise<unknown>;
