

/**
 * A screenshot of a computer.
 */
export interface ComputerScreenshotContent {
  file_id: string | unknown;
  image_url: string | unknown;
  /**
   * Specifies the event type. For a computer screenshot, this property is always set to `computer_screenshot`.
   */
  type: 'computer_screenshot';
}