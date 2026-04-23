

import { RealtimeConversationItem } from './RealtimeConversationItem';



/**
 * Returned when a conversation item is finalized.

The event will include the full content of the Item except for audio data, which can be retrieved separately with a `conversation.item.retrieve` event if needed.

 */
export interface RealtimeServerEventConversationItemDone {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  item: RealtimeConversationItem;
  previous_item_id?: string | unknown;
  /**
   * The event type, must be `conversation.item.done`.
   */
  type: string;
}