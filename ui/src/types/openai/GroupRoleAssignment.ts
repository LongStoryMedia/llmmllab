

import { Group } from './Group';

import { Role } from './Role';



/**
 * Role assignment linking a group to a role.
 */
export interface GroupRoleAssignment {
  group: Group;
  /**
   * Always `group.role`.
   */
  object: 'group.role';
  role: Role;
}