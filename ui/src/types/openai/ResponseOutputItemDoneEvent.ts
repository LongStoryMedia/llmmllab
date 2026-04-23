

import { OutputItem } from './OutputItem';



/**
 * Emitted when an output item is marked done.
 */
export interface ResponseOutputItemDoneEvent {
  /**
   * The output item that was marked done.

   */
  item: OutputItem;
  /**
   * The index of the output item that was marked done.

   */
  output_index: number;
  /**
   * The sequence number of this event.

   */
  sequence_number: number;
  /**
   * The type of the event. Always `response.output_item.done`.

   */
  type: 'response.output_item.done';
}