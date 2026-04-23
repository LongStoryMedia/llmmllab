

/**
 * A citation for a web resource used to generate a model response.
 */
export interface UrlCitationBody {
  /**
   * The index of the last character of the URL citation in the message.
   */
  end_index: number;
  /**
   * The index of the first character of the URL citation in the message.
   */
  start_index: number;
  /**
   * The title of the web resource.
   */
  title: string;
  /**
   * The type of the URL citation. Always `url_citation`.
   */
  type: 'url_citation';
  /**
   * The URL of the web resource.
   */
  url: string;
}