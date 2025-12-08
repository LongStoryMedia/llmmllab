import { UserConfig } from '../types/UserConfig';
/**
 * Get the current user configuration
 */
export declare function getConfig(token: string): Promise<UserConfig>;
/**
 * Update the user configuration
 */
export declare function updateConfig(token: string, config: UserConfig): Promise<UserConfig>;
/**
 * Update user's model profile assignments
 * Used to associate profile IDs with specific tasks (e.g., summarization, memory retrieval)
 */
export declare function updateModelProfileAssignments(token: string, assignments: Record<string, string>): Promise<UserConfig>;
