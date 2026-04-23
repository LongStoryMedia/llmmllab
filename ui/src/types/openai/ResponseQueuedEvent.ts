

import { Response } from './Response';



/**
 * Emitted when a response is queued and waiting to be processed.

 */
export interface ResponseQueuedEvent {
  /**
   * The full response object that is queued.
   */
  response: Response;
  /**
   * The sequence number for this event.
   */
  sequence_number: number;
  /**
   * The type of the event. Always 'response.queued'.
   */
  type: 'response.queued';
}