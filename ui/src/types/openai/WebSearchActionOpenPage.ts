

/**
 * Action type "open_page" - Opens a specific URL from search results.

 */
export interface WebSearchActionOpenPage {
  /**
   * The action type.

   */
  type: 'open_page';
  /**
   * The URL opened by the model.

   */
  url: URL;
}