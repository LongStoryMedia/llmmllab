

import { RealtimeConversationItem } from './RealtimeConversationItem';



/**
 * Returned when an Item is done streaming. Also emitted when a Response is 
interrupted, incomplete, or cancelled.

 */
export interface RealtimeBetaServerEventResponseOutputItemDone {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  item: RealtimeConversationItem;
  /**
   * The index of the output item in the Response.
   */
  output_index: number;
  /**
   * The ID of the Response to which the item belongs.
   */
  response_id: string;
  /**
   * The event type, must be `response.output_item.done`.
   */
  type: string;
}