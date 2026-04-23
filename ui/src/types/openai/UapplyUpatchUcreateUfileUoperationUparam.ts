

/**
 * Instruction for creating a new file via the apply_patch tool.
 */
export interface ApplyPatchCreateFileOperationParam {
  /**
   * Unified diff content to apply when creating the file.
   */
  diff: string;
  /**
   * Path of the file to create relative to the workspace root.
   */
  path: string;
  /**
   * The operation type. Always `create_file`.
   */
  type: 'create_file';
}