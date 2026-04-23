

/**
 * Parameters required to transfer a SIP call to a new destination using the
Realtime API.
 */
export interface RealtimeCallReferRequest {
  /**
   * URI that should appear in the SIP Refer-To header. Supports values like
`tel:+14155550123` or `sip:agent@example.com`.
   */
  target_uri: string;
}