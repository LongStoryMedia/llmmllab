

export interface ContainerFileResource {
  /**
   * Size of the file in bytes.
   */
  bytes: number;
  /**
   * The container this file belongs to.
   */
  container_id: string;
  /**
   * Unix timestamp (in seconds) when the file was created.
   */
  created_at: number;
  /**
   * Unique identifier for the file.
   */
  id: string;
  /**
   * The type of this object (`container.file`).
   */
  object: string;
  /**
   * Path of the file in the container.
   */
  path: string;
  /**
   * Source of the file (e.g., `user`, `assistant`).
   */
  source: string;
}