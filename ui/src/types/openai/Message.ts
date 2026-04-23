

import { MessageRole } from './MessageRole';

import { MessageStatus } from './MessageStatus';

import { InputTextContent } from './InputTextContent';

import { OutputTextContent } from './OutputTextContent';

import { TextContent } from './TextContent';

import { SummaryTextContent } from './SummaryTextContent';

import { ReasoningTextContent } from './ReasoningTextContent';

import { RefusalContent } from './RefusalContent';

import { InputImageContent } from './InputImageContent';

import { ComputerScreenshotContent } from './ComputerScreenshotContent';

import { InputFileContent } from './InputFileContent';



/**
 * A message to or from the model.
 */
export interface Message {
  /**
   * The content of the message
   */
  content: (InputTextContent | OutputTextContent | TextContent | SummaryTextContent | ReasoningTextContent | RefusalContent | InputImageContent | ComputerScreenshotContent | InputFileContent)[];
  /**
   * The unique ID of the message.
   */
  id: string;
  /**
   * The role of the message. One of `unknown`, `user`, `assistant`, `system`, `critic`, `discriminator`, `developer`, or `tool`.
   */
  role: MessageRole;
  /**
   * The status of item. One of `in_progress`, `completed`, or `incomplete`. Populated when items are returned via API.
   */
  status: MessageStatus;
  /**
   * The type of the message. Always set to `message`.
   */
  type: 'message';
}