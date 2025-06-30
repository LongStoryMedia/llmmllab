import { UserConfig } from "../types/UserConfig";
import { ChatWebSocketClient } from "./websocket";

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
  socket?: ChatWebSocketClient;
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