

import { AuditLogActorApiKey } from './AuditLogActorApiKey';

import { AuditLogActorSession } from './AuditLogActorSession';



/**
 * The actor who performed the audit logged action.
 */
export interface AuditLogActor {
  api_key?: AuditLogActorApiKey;
  session?: AuditLogActorSession;
  /**
   * The type of actor. Is either `session` or `api_key`.
   */
  type?: 'session' | 'api_key';
}