

import { ChatCompletionRequestMessageContentPartText } from './ChatCompletionRequestMessageContentPartText';



/**
 * Developer-provided instructions that the model should follow, regardless of
messages sent by the user. With o1 models and newer, `developer` messages
replace the previous `system` messages.

 */
export interface ChatCompletionRequestDeveloperMessage {
  /**
   * The contents of the developer message.
   */
  content: string | (ChatCompletionRequestMessageContentPartText)[];
  /**
   * An optional name for the participant. Provides the model information to differentiate between participants of the same role.
   */
  name?: string;
  /**
   * The role of the messages author, in this case `developer`.
   */
  role: 'developer';
}