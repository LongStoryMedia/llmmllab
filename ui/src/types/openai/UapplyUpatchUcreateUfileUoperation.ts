

/**
 * Instruction describing how to create a file via the apply_patch tool.
 */
export interface ApplyPatchCreateFileOperation {
  /**
   * Diff to apply.
   */
  diff: string;
  /**
   * Path of the file to create.
   */
  path: string;
  /**
   * Create a new file with the provided diff.
   */
  type: 'create_file';
}