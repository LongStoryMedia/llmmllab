

import { ProjectServiceAccount } from './ProjectServiceAccount';



export interface ProjectServiceAccountListResponse {
  data: (ProjectServiceAccount)[];
  first_id: string;
  has_more: boolean;
  last_id: string;
  object: 'list';
}