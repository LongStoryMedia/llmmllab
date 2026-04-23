

export interface FilePath {
  /**
   * The ID of the file that was generated.
   */
  file_id: string;
}

/**
 * A URL for the file that's generated when the assistant used the `code_interpreter` tool to generate a file.
 */
export interface MessageContentTextAnnotationsFilePathObject {
  end_index: number;
  file_path: FilePath;
  start_index: number;
  /**
   * The text in the message content that needs to be replaced.
   */
  text: string;
  /**
   * Always `file_path`.
   */
  type: 'file_path';
}