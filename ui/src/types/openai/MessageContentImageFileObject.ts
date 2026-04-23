

export interface ImageFile {
  /**
   * Specifies the detail level of the image if specified by the user. `low` uses fewer tokens, you can opt in to high resolution using `high`.
   */
  detail?: 'auto' | 'low' | 'high';
  /**
   * The [File](https://platform.openai.com/docs/api-reference/files) ID of the image in the message content. Set `purpose="vision"` when uploading the File if you need to later display the file content.
   */
  file_id: string;
}

/**
 * References an image [File](https://platform.openai.com/docs/api-reference/files) in the content of a message.
 */
export interface MessageContentImageFileObject {
  image_file: ImageFile;
  /**
   * Always `image_file`.
   */
  type: 'image_file';
}