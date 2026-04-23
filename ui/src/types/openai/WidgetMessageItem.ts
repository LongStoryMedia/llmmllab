

/**
 * Thread item that renders a widget payload.
 */
export interface WidgetMessageItem {
  /**
   * Unix timestamp (in seconds) for when the item was created.
   */
  created_at: number;
  /**
   * Identifier of the thread item.
   */
  id: string;
  /**
   * Type discriminator that is always `chatkit.thread_item`.
   */
  object: 'chatkit.thread_item';
  /**
   * Identifier of the parent thread.
   */
  thread_id: string;
  /**
   * Type discriminator that is always `chatkit.widget`.
   */
  type: 'chatkit.widget';
  /**
   * Serialized widget payload rendered in the UI.
   */
  widget: string;
}