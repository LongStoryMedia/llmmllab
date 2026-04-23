

/**
 * The image output from the code interpreter.
 */
export interface CodeInterpreterOutputImage {
  /**
   * The type of the output. Always `image`.
   */
  type: 'image';
  /**
   * The URL of the image output from the code interpreter.
   */
  url: string;
}