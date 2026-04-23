

export interface EvalGraderStringCheck {
  /**
   * The input text. This may include template strings.
   */
  input: string;
  /**
   * The name of the grader.
   */
  name: string;
  /**
   * The string check operation to perform. One of `eq`, `ne`, `like`, or `ilike`.
   */
  operation: 'eq' | 'ne' | 'like' | 'ilike';
  /**
   * The reference text. This may include template strings.
   */
  reference: string;
  /**
   * The object type, which is always `string_check`.
   */
  type: 'string_check';
}