

import { RunStepObject } from './RunStepObject';



export interface ListRunStepsResponse {
  data: (RunStepObject)[];
  first_id: string;
  has_more: boolean;
  last_id: string;
  object: string;
}