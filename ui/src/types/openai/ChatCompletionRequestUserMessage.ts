

import { ChatCompletionRequestUserMessageContentPart } from './ChatCompletionRequestUserMessageContentPart';



/**
 * Messages sent by an end user, containing prompts or additional context
information.

 */
export interface ChatCompletionRequestUserMessage {
  /**
   * The contents of the user message.

   */
  content: string | (ChatCompletionRequestUserMessageContentPart)[];
  /**
   * An optional name for the participant. Provides the model information to differentiate between participants of the same role.
   */
  name?: string;
  /**
   * The role of the messages author, in this case `user`.
   */
  role: 'user';
}