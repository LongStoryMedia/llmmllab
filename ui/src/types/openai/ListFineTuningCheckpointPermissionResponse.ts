

import { FineTuningCheckpointPermission } from './FineTuningCheckpointPermission';



export interface ListFineTuningCheckpointPermissionResponse {
  data: (FineTuningCheckpointPermission)[];
  first_id?: string | unknown;
  has_more: boolean;
  last_id?: string | unknown;
  object: 'list';
}