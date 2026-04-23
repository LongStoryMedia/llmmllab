

import { FineTuningJobCheckpoint } from './FineTuningJobCheckpoint';



export interface ListFineTuningJobCheckpointsResponse {
  data: (FineTuningJobCheckpoint)[];
  first_id?: string | unknown;
  has_more: boolean;
  last_id?: string | unknown;
  object: 'list';
}