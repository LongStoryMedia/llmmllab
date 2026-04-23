

import { ChatSessionAutomaticThreadTitling } from './ChatSessionAutomaticThreadTitling';

import { ChatSessionFileUpload } from './ChatSessionFileUpload';

import { ChatSessionHistory } from './ChatSessionHistory';



/**
 * ChatKit configuration for the session.
 */
export interface ChatSessionChatkitConfiguration {
  /**
   * Automatic thread titling preferences.
   */
  automatic_thread_titling: ChatSessionAutomaticThreadTitling;
  /**
   * Upload settings for the session.
   */
  file_upload: ChatSessionFileUpload;
  /**
   * History retention configuration.
   */
  history: ChatSessionHistory;
}