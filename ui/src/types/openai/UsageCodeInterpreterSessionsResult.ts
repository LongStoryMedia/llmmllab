

/**
 * The aggregated code interpreter sessions usage details of the specific time bucket.
 */
export interface UsageCodeInterpreterSessionsResult {
  /**
   * The number of code interpreter sessions.
   */
  num_sessions?: number;
  object: 'organization.usage.code_interpreter_sessions.result';
  project_id?: string | unknown;
}