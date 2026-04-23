

import { ChatCompletionRequestSystemMessageContentPart } from './ChatCompletionRequestSystemMessageContentPart';



/**
 * Developer-provided instructions that the model should follow, regardless of
messages sent by the user. With o1 models and newer, use `developer` messages
for this purpose instead.

 */
export interface ChatCompletionRequestSystemMessage {
  /**
   * The contents of the system message.
   */
  content: string | (ChatCompletionRequestSystemMessageContentPart)[];
  /**
   * An optional name for the participant. Provides the model information to differentiate between participants of the same role.
   */
  name?: string;
  /**
   * The role of the messages author, in this case `system`.
   */
  role: 'system';
}