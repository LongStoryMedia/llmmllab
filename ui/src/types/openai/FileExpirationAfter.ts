

/**
 * The expiration policy for a file. By default, files with `purpose=batch` expire after 30 days and all other files are persisted until they are manually deleted.
 */
export interface FileExpirationAfter {
  /**
   * Anchor timestamp after which the expiration policy applies. Supported anchors: `created_at`.
   */
  anchor: 'created_at';
  /**
   * The number of seconds after the anchor time that the file will expire. Must be between 3600 (1 hour) and 2592000 (30 days).
   */
  seconds: number;
}