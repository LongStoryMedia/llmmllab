

/**
 * Parameters used to decline an incoming SIP call handled by the Realtime API.
 */
export interface RealtimeCallRejectRequest {
  /**
   * SIP response code to send back to the caller. Defaults to `603` (Decline)
when omitted.
   */
  status_code?: number;
}