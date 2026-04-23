

export interface FilePath {
  /**
   * The ID of the file that was generated.
   */
  file_id?: string;
}

/**
 * A URL for the file that's generated when the assistant used the `code_interpreter` tool to generate a file.
 */
export interface MessageDeltaContentTextAnnotationsFilePathObject {
  end_index?: number;
  file_path?: FilePath;
  /**
   * The index of the annotation in the text content part.
   */
  index: number;
  start_index?: number;
  /**
   * The text in the message content that needs to be replaced.
   */
  text?: string;
  /**
   * Always `file_path`.
   */
  type: 'file_path';
}