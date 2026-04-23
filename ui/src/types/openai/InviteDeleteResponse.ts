

export interface InviteDeleteResponse {
  deleted: boolean;
  id: string;
  /**
   * The object type, which is always `organization.invite.deleted`
   */
  object: 'organization.invite.deleted';
}