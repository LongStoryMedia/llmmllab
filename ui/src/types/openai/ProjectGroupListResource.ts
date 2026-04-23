

import { ProjectGroup } from './ProjectGroup';



/**
 * Paginated list of groups that have access to a project.
 */
export interface ProjectGroupListResource {
  /**
   * Project group memberships returned in the current page.
   */
  data: (ProjectGroup)[];
  /**
   * Whether additional project group memberships are available.
   */
  has_more: boolean;
  /**
   * Cursor to fetch the next page of results, or `null` when there are no more results.
   */
  next: string | unknown;
  /**
   * Always `list`.
   */
  object: 'list';
}