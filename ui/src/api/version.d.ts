/**
 * Constructs a versioned API path
 * @param path The API endpoint path without version
 * @param apiVersion Optional custom API version to use (defaults to config)
 * @returns Versioned API path
 */
export declare function getVersionedPath(path: string, apiVersion?: string): string;
/**
 * Helper function to check if an API version is compatible with the minimum required version
 * This can be used for feature detection in the UI
 * @param requiredVersion The minimum version required
 * @param currentVersion The current version to check (defaults to config version)
 * @returns boolean indicating if the current version meets the requirement
 */
export declare function isVersionCompatible(requiredVersion: string, currentVersion?: string): boolean;
