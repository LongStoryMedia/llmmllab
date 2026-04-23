

import { Certificate } from './Certificate';



export interface ListCertificatesResponse {
  data: (Certificate)[];
  first_id?: string;
  has_more: boolean;
  last_id?: string;
  object: 'list';
}