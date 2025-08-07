import config from "../config";

/**
 * Constructs a versioned API path
 * @param path The API endpoint path without version
 * @param apiVersion Optional custom API version to use (defaults to config)
 * @returns Versioned API path
 */
export function getVersionedPath(path: string, apiVersion?: string): string {
  const version = apiVersion || config.server.apiVersion;
  return `${version}/${path}`;
}

/**
 * Helper function to check if an API version is compatible with the minimum required version
 * This can be used for feature detection in the UI
 * @param requiredVersion The minimum version required
 * @param currentVersion The current version to check (defaults to config version)
 * @returns boolean indicating if the current version meets the requirement
 */
export function isVersionCompatible(requiredVersion: string, currentVersion?: string): boolean {
  const currentVer = currentVersion || config.server.apiVersion;
  
  // Extract numeric parts of version strings (e.g., 'v1' -> 1, 'v2.1' -> 2.1)
  const required = parseFloat(requiredVersion.replace(/^v/, ''));
  const current = parseFloat(currentVer.replace(/^v/, ''));
  
  return current >= required;
}
