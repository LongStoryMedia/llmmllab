

/**
 * The event type.
 */
export type AuditLogEventType = 'api_key.created' | 'api_key.updated' | 'api_key.deleted' | 'certificate.created' | 'certificate.updated' | 'certificate.deleted' | 'certificates.activated' | 'certificates.deactivated' | 'checkpoint.permission.created' | 'checkpoint.permission.deleted' | 'external_key.registered' | 'external_key.removed' | 'group.created' | 'group.updated' | 'group.deleted' | 'invite.sent' | 'invite.accepted' | 'invite.deleted' | 'ip_allowlist.created' | 'ip_allowlist.updated' | 'ip_allowlist.deleted' | 'ip_allowlist.config.activated' | 'ip_allowlist.config.deactivated' | 'login.succeeded' | 'login.failed' | 'logout.succeeded' | 'logout.failed' | 'organization.updated' | 'project.created' | 'project.updated' | 'project.archived' | 'project.deleted' | 'rate_limit.updated' | 'rate_limit.deleted' | 'resource.deleted' | 'tunnel.created' | 'tunnel.updated' | 'tunnel.deleted' | 'role.created' | 'role.updated' | 'role.deleted' | 'role.assignment.created' | 'role.assignment.deleted' | 'scim.enabled' | 'scim.disabled' | 'service_account.created' | 'service_account.updated' | 'service_account.deleted' | 'user.added' | 'user.updated' | 'user.deleted';

/**
 * Constant values for AuditLogEventType
 */
export const AuditLogEventTypeValues = {
  /** api_key.created */
  API_KEY_CREATED: 'api_key.created',
  /** api_key.updated */
  API_KEY_UPDATED: 'api_key.updated',
  /** api_key.deleted */
  API_KEY_DELETED: 'api_key.deleted',
  /** certificate.created */
  CERTIFICATE_CREATED: 'certificate.created',
  /** certificate.updated */
  CERTIFICATE_UPDATED: 'certificate.updated',
  /** certificate.deleted */
  CERTIFICATE_DELETED: 'certificate.deleted',
  /** certificates.activated */
  CERTIFICATES_ACTIVATED: 'certificates.activated',
  /** certificates.deactivated */
  CERTIFICATES_DEACTIVATED: 'certificates.deactivated',
  /** checkpoint.permission.created */
  CHECKPOINT_PERMISSION_CREATED: 'checkpoint.permission.created',
  /** checkpoint.permission.deleted */
  CHECKPOINT_PERMISSION_DELETED: 'checkpoint.permission.deleted',
  /** external_key.registered */
  EXTERNAL_KEY_REGISTERED: 'external_key.registered',
  /** external_key.removed */
  EXTERNAL_KEY_REMOVED: 'external_key.removed',
  /** group.created */
  GROUP_CREATED: 'group.created',
  /** group.updated */
  GROUP_UPDATED: 'group.updated',
  /** group.deleted */
  GROUP_DELETED: 'group.deleted',
  /** invite.sent */
  INVITE_SENT: 'invite.sent',
  /** invite.accepted */
  INVITE_ACCEPTED: 'invite.accepted',
  /** invite.deleted */
  INVITE_DELETED: 'invite.deleted',
  /** ip_allowlist.created */
  IP_ALLOWLIST_CREATED: 'ip_allowlist.created',
  /** ip_allowlist.updated */
  IP_ALLOWLIST_UPDATED: 'ip_allowlist.updated',
  /** ip_allowlist.deleted */
  IP_ALLOWLIST_DELETED: 'ip_allowlist.deleted',
  /** ip_allowlist.config.activated */
  IP_ALLOWLIST_CONFIG_ACTIVATED: 'ip_allowlist.config.activated',
  /** ip_allowlist.config.deactivated */
  IP_ALLOWLIST_CONFIG_DEACTIVATED: 'ip_allowlist.config.deactivated',
  /** login.succeeded */
  LOGIN_SUCCEEDED: 'login.succeeded',
  /** login.failed */
  LOGIN_FAILED: 'login.failed',
  /** logout.succeeded */
  LOGOUT_SUCCEEDED: 'logout.succeeded',
  /** logout.failed */
  LOGOUT_FAILED: 'logout.failed',
  /** organization.updated */
  ORGANIZATION_UPDATED: 'organization.updated',
  /** project.created */
  PROJECT_CREATED: 'project.created',
  /** project.updated */
  PROJECT_UPDATED: 'project.updated',
  /** project.archived */
  PROJECT_ARCHIVED: 'project.archived',
  /** project.deleted */
  PROJECT_DELETED: 'project.deleted',
  /** rate_limit.updated */
  RATE_LIMIT_UPDATED: 'rate_limit.updated',
  /** rate_limit.deleted */
  RATE_LIMIT_DELETED: 'rate_limit.deleted',
  /** resource.deleted */
  RESOURCE_DELETED: 'resource.deleted',
  /** tunnel.created */
  TUNNEL_CREATED: 'tunnel.created',
  /** tunnel.updated */
  TUNNEL_UPDATED: 'tunnel.updated',
  /** tunnel.deleted */
  TUNNEL_DELETED: 'tunnel.deleted',
  /** role.created */
  ROLE_CREATED: 'role.created',
  /** role.updated */
  ROLE_UPDATED: 'role.updated',
  /** role.deleted */
  ROLE_DELETED: 'role.deleted',
  /** role.assignment.created */
  ROLE_ASSIGNMENT_CREATED: 'role.assignment.created',
  /** role.assignment.deleted */
  ROLE_ASSIGNMENT_DELETED: 'role.assignment.deleted',
  /** scim.enabled */
  SCIM_ENABLED: 'scim.enabled',
  /** scim.disabled */
  SCIM_DISABLED: 'scim.disabled',
  /** service_account.created */
  SERVICE_ACCOUNT_CREATED: 'service_account.created',
  /** service_account.updated */
  SERVICE_ACCOUNT_UPDATED: 'service_account.updated',
  /** service_account.deleted */
  SERVICE_ACCOUNT_DELETED: 'service_account.deleted',
  /** user.added */
  USER_ADDED: 'user.added',
  /** user.updated */
  USER_UPDATED: 'user.updated',
  /** user.deleted */
  USER_DELETED: 'user.deleted'
} as const;