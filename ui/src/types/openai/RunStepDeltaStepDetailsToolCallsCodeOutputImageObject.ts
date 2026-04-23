

export interface Image {
  /**
   * The [file](https://platform.openai.com/docs/api-reference/files) ID of the image.
   */
  file_id?: string;
}

export interface RunStepDeltaStepDetailsToolCallsCodeOutputImageObject {
  image?: Image;
  /**
   * The index of the output in the outputs array.
   */
  index: number;
  /**
   * Always `image`.
   */
  type: 'image';
}