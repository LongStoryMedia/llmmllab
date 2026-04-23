

import { Response } from './Response';



/**
 * An event that is emitted when a response finishes as incomplete.

 */
export interface ResponseIncompleteEvent {
  /**
   * The response that was incomplete.

   */
  response: Response;
  /**
   * The sequence number of this event.
   */
  sequence_number: number;
  /**
   * The type of the event. Always `response.incomplete`.

   */
  type: 'response.incomplete';
}