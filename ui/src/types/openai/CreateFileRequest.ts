

import { FileExpirationAfter } from './FileExpirationAfter';

import { FilePurpose } from './FilePurpose';



export interface CreateFileRequest {
  expires_after?: FileExpirationAfter;
  /**
   * The File object (not file name) to be uploaded.

   */
  file: string;
  purpose: FilePurpose;
}