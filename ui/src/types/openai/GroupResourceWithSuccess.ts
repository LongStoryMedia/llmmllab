

/**
 * Response returned after updating a group.
 */
export interface GroupResourceWithSuccess {
  /**
   * Unix timestamp (in seconds) when the group was created.
   */
  created_at: number;
  /**
   * Identifier for the group.
   */
  id: string;
  /**
   * Whether the group is managed through SCIM and controlled by your identity provider.
   */
  is_scim_managed: boolean;
  /**
   * Updated display name for the group.
   */
  name: string;
}