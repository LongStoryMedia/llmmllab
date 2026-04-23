

import { Role } from './Role';

import { User } from './User';



/**
 * Role assignment linking a user to a role.
 */
export interface UserRoleAssignment {
  /**
   * Always `user.role`.
   */
  object: 'user.role';
  role: Role;
  user: User;
}