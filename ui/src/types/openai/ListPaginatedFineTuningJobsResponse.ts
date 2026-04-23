

import { FineTuningJob } from './FineTuningJob';



export interface ListPaginatedFineTuningJobsResponse {
  data: (FineTuningJob)[];
  has_more: boolean;
  object: 'list';
}