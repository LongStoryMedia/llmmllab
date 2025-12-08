import { User } from "oidc-client-ts";
import { RequestOptions } from "./types";
import { ChatResponse } from "../types/ChatResponse";
export declare function getToken(user?: User): string;
export declare function gen(opts: RequestOptions): AsyncGenerator<ChatResponse>;
export declare function req<T>(opts: RequestOptions): Promise<T>;
export declare const getHeaders: (accessToken: string) => {
    Authorization: string;
    'Content-Type': string;
};
