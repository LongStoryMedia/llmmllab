

import { VectorStoreSearchResultItem } from './VectorStoreSearchResultItem';



export interface VectorStoreSearchResultsPage {
  /**
   * The list of search result items.
   */
  data: (VectorStoreSearchResultItem)[];
  /**
   * Indicates if there are more results to fetch.
   */
  has_more: boolean;
  next_page: string | unknown;
  /**
   * The object type, which is always `vector_store.search_results.page`
   */
  object: 'vector_store.search_results.page';
  search_query: (string)[];
}