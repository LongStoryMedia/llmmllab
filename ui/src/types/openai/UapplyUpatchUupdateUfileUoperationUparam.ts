

/**
 * Instruction for updating an existing file via the apply_patch tool.
 */
export interface ApplyPatchUpdateFileOperationParam {
  /**
   * Unified diff content to apply to the existing file.
   */
  diff: string;
  /**
   * Path of the file to update relative to the workspace root.
   */
  path: string;
  /**
   * The operation type. Always `update_file`.
   */
  type: 'update_file';
}