import { User } from "oidc-client-ts";
import config from "../config";
import { RequestOptions } from "./types";
import { logoutSession } from "../auth";
import { ChatResponse } from "../types/ChatResponse";

export function getToken(user?: User): string {
  if (!user) {
    return '';
  }

  const token = user.access_token;
  if (!token) {
    throw new Error('No access token found');
  }

  return token;
}

export async function* gen(opts: RequestOptions): AsyncGenerator<ChatResponse> {
  opts.headers = {
    ...opts.headers,
    'Content-Type': 'application/json'
  }
  opts.method = opts.method || 'GET';
  const response = await fetch(`${config.server.baseUrl}/${opts.path}`, opts);

  // Handle authentication errors
  if (response.status === 401 || response.status === 403) {
    await logoutSession();
    throw new Error('Authentication failed');
  }

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();

  if (!reader) {
    throw new Error('Failed to get reader from response body');
  }

  const decoder = new TextDecoder('utf-8');
  let buffer = ''; // Add buffer to accumulate partial JSON chunks
  let done = false;

  while (!done) {
    if (opts.signal?.aborted) {
      console.log('Request was cancelled');
      break;
    }
    const { done: doneReading, value } = await reader.read();
    done = doneReading;

    if (value) {
      buffer += decoder.decode(value, { stream: !doneReading });

      // Process complete JSON objects from the buffer
      let startIdx = 0;
      let parseIndex;

      // Try to parse as many complete JSON objects as possible from the buffer
      while (startIdx < buffer.length) {
        try {
          parseIndex = findCompleteJsonEnd(buffer, startIdx);
          if (parseIndex === -1) {
            break; // No complete JSON found
          }

          const jsonStr = buffer.substring(startIdx, parseIndex + 1);
          const res = JSON.parse(jsonStr) as ChatResponse;
          yield res;

          // Move start index past this JSON object
          startIdx = parseIndex + 1;
        } catch {
          // If we can't parse a complete JSON object, break and wait for more data
          break;
        }
      }

      // Remove processed JSON from buffer
      if (startIdx > 0) {
        buffer = buffer.substring(startIdx);
      }
    }

    if (done && buffer.trim()) {
      // Try to parse any remaining data in the buffer
      try {
        const res = JSON.parse(buffer) as ChatResponse;
        yield res;
      } catch (e: unknown) {
        if (e instanceof Error) {
          console.error("Error parsing final JSON chunk:", e);
          // Try parsing individual lines as separate JSON objects
          const lines = buffer.split('\n').filter(line => line.trim());
          for (const line of lines) {
            try {
              const res = JSON.parse(line) as ChatResponse;
              yield res;
            } catch {
              // Ignore parsing errors for individual lines
            }
          }
        }
      }
    }
  }

  // Clean up
  reader.releaseLock();
  reader.cancel();
}

// Helper function to find the end of a complete JSON object
function findCompleteJsonEnd(str: string, startIndex: number): number {
  let openBraces = 0;
  let inString = false;
  let escaped = false;

  for (let i = startIndex; i < str.length; i++) {
    const char = str[i];

    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === '\\') {
        escaped = true;
      } else if (char === '"') {
        inString = false;
      }
    } else {
      if (char === '"') {
        inString = true;
      } else if (char === '{') {
        openBraces++;
      } else if (char === '}') {
        openBraces--;
        if (openBraces === 0) {
          // Found complete JSON object
          return i;
        }
      }
    }
  }

  // No complete JSON object found
  return -1;
}

export async function req<T>(opts: RequestOptions): Promise<T> {
  const controller = new AbortController();
  opts.signal = controller.signal;
  opts.baseUrl = opts.baseUrl || config.server.baseUrl;

  // Cancel previous requests with the same key if specified
  if (opts.requestKey && pendingRequests[opts.requestKey]) {
    pendingRequests[opts.requestKey].abort();
  }

  if (opts.requestKey) {
    pendingRequests[opts.requestKey] = controller;
  }

  opts.headers = {
    ...opts.headers,
    'Content-Type': 'application/json'
  }
  opts.method = opts.method || 'GET';

  try {
    if (opts.timeout) {
      setTimeout(() => {
        if (opts.requestKey && pendingRequests[opts.requestKey]) {
          pendingRequests[opts.requestKey].abort();
          delete pendingRequests[opts.requestKey];
        }
      }, opts.timeout);
    }

    const response = await fetch(`${opts.baseUrl}/${opts.path}`, opts);

    // Handle authentication errors
    if (response.status === 401 || response.status === 403) {
      await logoutSession();
      throw new Error('Authentication failed');
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    if (opts.requestKey) {
      delete pendingRequests[opts.requestKey];
    }

    if (opts.method === 'DELETE') {
      return await response.text() as unknown as T;
    }

    return await response.json();
  } catch (error: unknown) {
    if (opts.requestKey) {
      delete pendingRequests[opts.requestKey];
    }

    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        console.log('Request was cancelled');
      } else {
        console.error("Error in request:", error);
      }
    } else {
      console.error("Unknown error in request:", error);
    }

    throw error;
  }
}

export const getHeaders = (accessToken: string) => ({
  Authorization: `Bearer ${accessToken}`,
  'Content-Type': 'application/json'
});

// Track pending requests
const pendingRequests: Record<string, AbortController> = {};