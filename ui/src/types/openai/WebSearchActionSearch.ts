

/**
 * Action type "search" - Performs a web search query.

 */
export interface WebSearchActionSearch {
  /**
   * The search queries.

   */
  queries?: (string)[];
  /**
   * [DEPRECATED] The search query.

   */
  query: string;
  /**
   * The sources used in the search.

   */
  sources?: (SourcesItem)[];
  /**
   * The action type.

   */
  type: 'search';
}