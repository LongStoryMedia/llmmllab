

export interface CompleteUploadRequest {
  /**
   * The optional md5 checksum for the file contents to verify if the bytes uploaded matches what you expect.

   */
  md5?: string;
  /**
   * The ordered list of Part IDs.

   */
  part_ids: (string)[];
}