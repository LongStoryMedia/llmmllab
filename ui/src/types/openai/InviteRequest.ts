

export interface InviteRequest {
  /**
   * Send an email to this address
   */
  email: string;
  /**
   * An array of projects to which membership is granted at the same time the org invite is accepted. If omitted, the user will be invited to the default project for compatibility with legacy behavior.
   */
  projects?: (ProjectsItem)[];
  /**
   * `owner` or `reader`
   */
  role: 'reader' | 'owner';
}