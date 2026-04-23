

/**
 * Action type "find": Searches for a pattern within a loaded page.

 */
export interface WebSearchActionFind {
  /**
   * The pattern or text to search for within the page.

   */
  pattern: string;
  /**
   * The action type.

   */
  type: 'find';
  /**
   * The URL of the page searched for the pattern.

   */
  url: URL;
}