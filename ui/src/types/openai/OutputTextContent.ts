

import { Annotation } from './Annotation';

import { LogProb } from './LogProb';



/**
 * A text output from the model.
 */
export interface OutputTextContent {
  /**
   * The annotations of the text output.
   */
  annotations: (Annotation)[];
  logprobs?: (LogProb)[];
  /**
   * The text output from the model.
   */
  text: string;
  /**
   * The type of the output text. Always `output_text`.
   */
  type: 'output_text';
}