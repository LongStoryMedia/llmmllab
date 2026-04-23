

import { Error } from './Error';



/**
 * Occurs when an [error](https://platform.openai.com/docs/guides/error-codes#api-errors) occurs. This can happen due to an internal server error or a timeout.
 */
export interface ErrorEvent {
  data: Error;
  event: 'error';
}