

export interface FileCitation {
  /**
   * The ID of the specific File the citation is from.
   */
  file_id: string;
}

/**
 * A citation within the message that points to a specific quote from a specific File associated with the assistant or the message. Generated when the assistant uses the "file_search" tool to search files.
 */
export interface MessageContentTextAnnotationsFileCitationObject {
  end_index: number;
  file_citation: FileCitation;
  start_index: number;
  /**
   * The text in the message content that needs to be replaced.
   */
  text: string;
  /**
   * Always `file_citation`.
   */
  type: 'file_citation';
}