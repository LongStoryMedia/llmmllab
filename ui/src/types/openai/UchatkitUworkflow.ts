

import { ChatkitWorkflowTracing } from './ChatkitWorkflowTracing';



/**
 * Workflow metadata and state returned for the session.
 */
export interface ChatkitWorkflow {
  /**
   * Identifier of the workflow backing the session.
   */
  id: string;
  state_variables: Record<string, unknown> | unknown;
  /**
   * Tracing settings applied to the workflow.
   */
  tracing: ChatkitWorkflowTracing;
  version: string | unknown;
}