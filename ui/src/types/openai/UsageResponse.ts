

import { UsageTimeBucket } from './UsageTimeBucket';



export interface UsageResponse {
  data: (UsageTimeBucket)[];
  has_more: boolean;
  next_page: string;
  object: 'page';
}