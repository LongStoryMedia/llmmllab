

export interface ImageUrl {
  /**
   * Specifies the detail level of the image. Learn more in the [Vision guide](https://platform.openai.com/docs/guides/vision#low-or-high-fidelity-image-understanding).
   */
  detail?: 'auto' | 'low' | 'high';
  /**
   * Either a URL of the image or the base64 encoded image data.
   */
  url: URL;
}

/**
 * Learn about [image inputs](https://platform.openai.com/docs/guides/vision).

 */
export interface ChatCompletionRequestMessageContentPartImage {
  image_url: ImageUrl;
  /**
   * The type of the content part.
   */
  type: 'image_url';
}