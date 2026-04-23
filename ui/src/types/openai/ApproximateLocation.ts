

export interface ApproximateLocation {
  city?: string | unknown;
  country?: string | unknown;
  region?: string | unknown;
  timezone?: string | unknown;
  /**
   * The type of location approximation. Always `approximate`.
   */
  type: 'approximate';
}