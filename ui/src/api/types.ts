import { UserConfig } from "../types/UserConfig";

export type BodyDeserialized = {
  model: string;
  messages: Array<{
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string;
  }>;
};

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
// export type MessageStatus = 'sending' | 'success' | 'error';

// export interface ChatMessage {
//   role: 'user' | 'assistant' | 'system';
//   content: string;
//   status?: MessageStatus;
//   id?: number;
//   createdAt?: Date;
// }

// export interface ChatUserMessage extends ChatMessage {
//   role: 'user';
//   conversationId?: number;
// }

// export interface ChatAgentMessage extends ChatMessage {
//   role: 'assistant';
// }

// export type ChatRequest = {
//   model: string;
//   messages: ChatUserMessage[];
//   tools?: Record<string, unknown>;
//   format?: string;
//   options?: string[];
//   stream?: boolean;
//   keep_alive?: string;
// }

// export type ChatResponse = {
//   done: boolean;
//   message?: ChatAgentMessage;
//   createdAt: string;
//   model: string;
//   context?: number[];
//   done_reason?: string;
//   total_duration?: number;
//   load_duration?: number;
//   prompt_eval_count?: number;
//   prompt_eval_duration?: number;
//   eval_count?: number;
//   eval_duration?: number;
// };

// export type ModelDetails = {
//   parentModel: string;
//   format: string;
//   family: string;
//   families: string[];
//   parameterSize: number;
//   quantizationLevel: string;
// }

// export type Model = {
//   name: string;
//   modifiedAt: Date;
//   size: number;
//   digest: string;
//   details: ModelDetails;
// }

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