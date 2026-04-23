

import { OutputItem } from './OutputItem';



/**
 * Emitted when a new output item is added.
 */
export interface ResponseOutputItemAddedEvent {
  /**
   * The output item that was added.

   */
  item: OutputItem;
  /**
   * The index of the output item that was added.

   */
  output_index: number;
  /**
   * The sequence number of this event.

   */
  sequence_number: number;
  /**
   * The type of the event. Always `response.output_item.added`.

   */
  type: 'response.output_item.added';
}