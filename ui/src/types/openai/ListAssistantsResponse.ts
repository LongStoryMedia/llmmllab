

import { AssistantObject } from './AssistantObject';



export interface ListAssistantsResponse {
  data: (AssistantObject)[];
  first_id: string;
  has_more: boolean;
  last_id: string;
  object: string;
}