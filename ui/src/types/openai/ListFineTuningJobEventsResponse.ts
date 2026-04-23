

import { FineTuningJobEvent } from './FineTuningJobEvent';



export interface ListFineTuningJobEventsResponse {
  data: (FineTuningJobEvent)[];
  has_more: boolean;
  object: 'list';
}