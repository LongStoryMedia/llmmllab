

import { ProjectUser } from './ProjectUser';



export interface ProjectUserListResponse {
  data: (ProjectUser)[];
  first_id: string;
  has_more: boolean;
  last_id: string;
  object: string;
}