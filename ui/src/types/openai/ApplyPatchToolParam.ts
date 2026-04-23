

/**
 * Allows the assistant to create, delete, or update files using unified diffs.
 */
export interface ApplyPatchToolParam {
  /**
   * The type of the tool. Always `apply_patch`.
   */
  type: 'apply_patch';
}