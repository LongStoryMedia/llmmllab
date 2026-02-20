import { ApiKey, ApiKeyResponse, ApiKeyRequest } from '../types';
import { getHeaders, req } from './base';

/**
 * Create a new API key
 */
export async function createApiKey(
  token: string,
  data: ApiKeyRequest
): Promise<ApiKeyResponse> {
  return req<ApiKeyResponse>({
    method: 'POST',
    path: 'api-keys/create',
    headers: getHeaders(token),
    body: JSON.stringify(data)
  });
}

/**
 * List all API keys for the current user
 */
export async function listApiKeys(token: string): Promise<ApiKey[]> {
  return req<ApiKey[]>({
    method: 'GET',
    path: 'api-keys/list',
    headers: getHeaders(token)
  });
}

/**
 * Get information about a specific API key
 */
export async function getApiKeyInfo(token: string, keyId: string): Promise<ApiKey> {
  return req<ApiKey>({
    method: 'GET',
    path: `api-keys/info/${keyId}`,
    headers: getHeaders(token)
  });
}

/**
 * Revoke an API key (disable but keep record)
 */
export async function revokeApiKey(token: string, keyId: string): Promise<{ status: string; message: string }> {
  return req<{ status: string; message: string }>({
    method: 'POST',
    path: 'api-keys/revoke',
    headers: getHeaders(token),
    body: JSON.stringify({ key_id: keyId })
  });
}

/**
 * Delete an API key permanently
 */
export async function deleteApiKey(token: string, keyId: string): Promise<{ status: string; message: string }> {
  return req<{ status: string; message: string }>({
    method: 'POST',
    path: 'api-keys/delete',
    headers: getHeaders(token),
    body: JSON.stringify({ key_id: keyId })
  });
}
