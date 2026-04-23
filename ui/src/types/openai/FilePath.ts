

/**
 * A path to a file.

 */
export interface FilePath {
  /**
   * The ID of the file.

   */
  file_id: string;
  /**
   * The index of the file in the list of files.

   */
  index: number;
  /**
   * The type of the file path. Always `file_path`.

   */
  type: 'file_path';
}