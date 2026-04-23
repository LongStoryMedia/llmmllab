

/**
 * Request payload for updating an existing role.
 */
export interface PublicUpdateOrganizationRoleBody {
  /**
   * New description for the role.
   */
  description?: string | unknown;
  /**
   * Updated set of permissions for the role.
   */
  permissions?: (string)[] | unknown;
  /**
   * New name for the role.
   */
  role_name?: string | unknown;
}