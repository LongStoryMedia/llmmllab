

import { Response } from './Response';



/**
 * An event that is emitted when a response is created.

 */
export interface ResponseCreatedEvent {
  /**
   * The response that was created.

   */
  response: Response;
  /**
   * The sequence number for this event.
   */
  sequence_number: number;
  /**
   * The type of the event. Always `response.created`.

   */
  type: 'response.created';
}