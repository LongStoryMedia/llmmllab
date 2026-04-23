

import { AuditLog } from './AuditLog';



export interface ListAuditLogsResponse {
  data: (AuditLog)[];
  first_id: string;
  has_more: boolean;
  last_id: string;
  object: 'list';
}