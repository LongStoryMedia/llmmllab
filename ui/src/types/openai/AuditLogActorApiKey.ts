

import { AuditLogActorServiceAccount } from './AuditLogActorServiceAccount';

import { AuditLogActorUser } from './AuditLogActorUser';



/**
 * The API Key used to perform the audit logged action.
 */
export interface AuditLogActorApiKey {
  /**
   * The tracking id of the API key.
   */
  id?: string;
  service_account?: AuditLogActorServiceAccount;
  /**
   * The type of API key. Can be either `user` or `service_account`.
   */
  type?: 'user' | 'service_account';
  user?: AuditLogActorUser;
}