

import { ProjectApiKey } from './ProjectApiKey';



export interface ProjectApiKeyListResponse {
  data: (ProjectApiKey)[];
  first_id: string;
  has_more: boolean;
  last_id: string;
  object: 'list';
}