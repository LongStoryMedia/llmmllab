

/**
 * Upload permissions and limits applied to the session.
 */
export interface ChatSessionFileUpload {
  /**
   * Indicates if uploads are enabled for the session.
   */
  enabled: boolean;
  max_file_size: number | unknown;
  max_files: number | unknown;
}