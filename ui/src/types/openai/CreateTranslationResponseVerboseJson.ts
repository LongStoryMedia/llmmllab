

import { TranscriptionSegment } from './TranscriptionSegment';



export interface CreateTranslationResponseVerboseJson {
  /**
   * The duration of the input audio.
   */
  duration: number;
  /**
   * The language of the output translation (always `english`).
   */
  language: string;
  /**
   * Segments of the translated text and their corresponding details.
   */
  segments?: (TranscriptionSegment)[];
  /**
   * The translated text.
   */
  text: string;
}