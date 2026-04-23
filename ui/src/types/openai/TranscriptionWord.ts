

export interface TranscriptionWord {
  /**
   * End time of the word in seconds.
   */
  end: number;
  /**
   * Start time of the word in seconds.
   */
  start: number;
  /**
   * The text content of the word.
   */
  word: string;
}