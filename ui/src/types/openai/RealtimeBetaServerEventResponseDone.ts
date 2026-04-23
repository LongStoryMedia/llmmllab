

import { RealtimeBetaResponse } from './RealtimeBetaResponse';



/**
 * Returned when a Response is done streaming. Always emitted, no matter the 
final state. The Response object included in the `response.done` event will 
include all output Items in the Response but will omit the raw audio data.

 */
export interface RealtimeBetaServerEventResponseDone {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  response: RealtimeBetaResponse;
  /**
   * The event type, must be `response.done`.
   */
  type: string;
}