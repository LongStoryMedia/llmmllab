

/**
 * The monetary value in its associated currency.
 */
export interface Amount {
  /**
   * Lowercase ISO-4217 currency e.g. "usd"
   */
  currency?: string;
  /**
   * The numeric value of the cost.
   */
  value?: number;
}

/**
 * The aggregated costs details of the specific time bucket.
 */
export interface CostsResult {
  /**
   * The monetary value in its associated currency.
   */
  amount?: Amount;
  line_item?: string | unknown;
  object: 'organization.costs.result';
  project_id?: string | unknown;
}