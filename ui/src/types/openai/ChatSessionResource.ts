

import { ChatSessionChatkitConfiguration } from './ChatSessionChatkitConfiguration';

import { ChatSessionRateLimits } from './ChatSessionRateLimits';

import { ChatSessionStatus } from './ChatSessionStatus';

import { ChatkitWorkflow } from './ChatkitWorkflow';



/**
 * Represents a ChatKit session and its resolved configuration.
 */
export interface ChatSessionResource {
  /**
   * Resolved ChatKit feature configuration for the session.
   */
  chatkit_configuration: ChatSessionChatkitConfiguration;
  /**
   * Ephemeral client secret that authenticates session requests.
   */
  client_secret: string;
  /**
   * Unix timestamp (in seconds) for when the session expires.
   */
  expires_at: number;
  /**
   * Identifier for the ChatKit session.
   */
  id: string;
  /**
   * Convenience copy of the per-minute request limit.
   */
  max_requests_per_1_minute: number;
  /**
   * Type discriminator that is always `chatkit.session`.
   */
  object: 'chatkit.session';
  /**
   * Resolved rate limit values.
   */
  rate_limits: ChatSessionRateLimits;
  /**
   * Current lifecycle state of the session.
   */
  status: ChatSessionStatus;
  /**
   * User identifier associated with the session.
   */
  user: string;
  /**
   * Workflow metadata for the session.
   */
  workflow: ChatkitWorkflow;
}