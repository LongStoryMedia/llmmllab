

/**
 * Emitted when a partial image is available during image editing streaming.

 */
export interface ImageEditPartialImageEvent {
  /**
   * Base64-encoded partial image data, suitable for rendering as an image.

   */
  b64_json: string;
  /**
   * The background setting for the requested edited image.

   */
  background: 'transparent' | 'opaque' | 'auto';
  /**
   * The Unix timestamp when the event was created.

   */
  created_at: number;
  /**
   * The output format for the requested edited image.

   */
  output_format: 'png' | 'webp' | 'jpeg';
  /**
   * 0-based index for the partial image (streaming).

   */
  partial_image_index: number;
  /**
   * The quality setting for the requested edited image.

   */
  quality: 'low' | 'medium' | 'high' | 'auto';
  /**
   * The size of the requested edited image.

   */
  size: '1024x1024' | '1024x1536' | '1536x1024' | 'auto';
  /**
   * The type of the event. Always `image_edit.partial_image`.

   */
  type: 'image_edit.partial_image';
}