

import { TextResponseFormatConfiguration } from './TextResponseFormatConfiguration';

import { Verbosity } from './Verbosity';



/**
 * Configuration options for a text response from the model. Can be plain
text or structured JSON data. Learn more:
- [Text inputs and outputs](https://platform.openai.com/docs/guides/text)
- [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)

 */
export interface ResponseTextParam {
  format?: TextResponseFormatConfiguration;
  verbosity?: Verbosity;
}