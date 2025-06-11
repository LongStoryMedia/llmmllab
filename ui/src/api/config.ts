import { UserConfig } from '../types/UserConfig';
import { getHeaders, req } from './base';

/**
 * Get the current user configuration
 */
export async function getConfig(token: string): Promise<UserConfig> {
  return req<UserConfig>({
    method: 'GET',
    path: 'api/config',
    headers: getHeaders(token)
  });
}

/**
 * Update the user configuration
 */
export async function updateConfig(token: string, config: UserConfig): Promise<UserConfig> {
  return req<UserConfig>({
    method: 'PUT',
    path: 'api/config',
    headers: getHeaders(token),
    body: JSON.stringify(config)
  });
}

/**
 * Update user's model profile assignments
 * Used to associate profile IDs with specific tasks (e.g., summarization, memory retrieval)
 */
export async function updateModelProfileAssignments(
  token: string,
  assignments: Record<string, string>
): Promise<UserConfig> {
  // First, fetch the current config to ensure we only update the modelProfiles part
  const currentConfig = await getConfig(token);

  // This ensures we don't overwrite other config properties
  return updateConfig(token, {
    ...currentConfig,
    model_profiles: {
      ...currentConfig.model_profiles,  // Preserve existing settings
      ...assignments  // Apply new assignments
    }
  });
}