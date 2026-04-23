

import { ResponseLogProb } from './ResponseLogProb';



/**
 * Emitted when there is an additional text delta.
 */
export interface ResponseTextDeltaEvent {
  /**
   * The index of the content part that the text delta was added to.

   */
  content_index: number;
  /**
   * The text delta that was added.

   */
  delta: string;
  /**
   * The ID of the output item that the text delta was added to.

   */
  item_id: string;
  /**
   * The log probabilities of the tokens in the delta.

   */
  logprobs: (ResponseLogProb)[];
  /**
   * The index of the output item that the text delta was added to.

   */
  output_index: number;
  /**
   * The sequence number for this event.
   */
  sequence_number: number;
  /**
   * The type of the event. Always `response.output_text.delta`.

   */
  type: 'response.output_text.delta';
}