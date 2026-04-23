

export interface EvalJsonlFileIdSource {
  /**
   * The identifier of the file.
   */
  id: string;
  /**
   * The type of jsonl source. Always `file_id`.
   */
  type: 'file_id';
}