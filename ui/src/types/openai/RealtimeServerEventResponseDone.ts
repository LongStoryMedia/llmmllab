

import { RealtimeResponse } from './RealtimeResponse';



/**
 * Returned when a Response is done streaming. Always emitted, no matter the 
final state. The Response object included in the `response.done` event will 
include all output Items in the Response but will omit the raw audio data.

Clients should check the `status` field of the Response to determine if it was successful
(`completed`) or if there was another outcome: `cancelled`, `failed`, or `incomplete`.

A response will contain all output items that were generated during the response, excluding
any audio content.

 */
export interface RealtimeServerEventResponseDone {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  response: RealtimeResponse;
  /**
   * The event type, must be `response.done`.
   */
  type: string;
}