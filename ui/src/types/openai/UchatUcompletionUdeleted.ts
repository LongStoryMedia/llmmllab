

export interface ChatCompletionDeleted {
  /**
   * Whether the chat completion was deleted.
   */
  deleted: boolean;
  /**
   * The ID of the chat completion that was deleted.
   */
  id: string;
  /**
   * The type of object being deleted.
   */
  object: 'chat.completion.deleted';
}