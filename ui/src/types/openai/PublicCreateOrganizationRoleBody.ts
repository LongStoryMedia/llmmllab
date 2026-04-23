

/**
 * Request payload for creating a custom role.
 */
export interface PublicCreateOrganizationRoleBody {
  /**
   * Optional description of the role.
   */
  description?: string | unknown;
  /**
   * Permissions to grant to the role.
   */
  permissions: (string)[];
  /**
   * Unique name for the role.
   */
  role_name: string;
}