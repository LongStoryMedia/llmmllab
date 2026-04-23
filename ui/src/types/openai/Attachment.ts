

import { AttachmentType } from './AttachmentType';



/**
 * Attachment metadata included on thread items.
 */
export interface Attachment {
  /**
   * Identifier for the attachment.
   */
  id: string;
  /**
   * MIME type of the attachment.
   */
  mime_type: string;
  /**
   * Original display name for the attachment.
   */
  name: string;
  preview_url: string | unknown;
  /**
   * Attachment discriminator.
   */
  type: AttachmentType;
}