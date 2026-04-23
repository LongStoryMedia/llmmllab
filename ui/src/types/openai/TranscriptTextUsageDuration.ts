

/**
 * Usage statistics for models billed by audio input duration.
 */
export interface TranscriptTextUsageDuration {
  /**
   * Duration of the input audio in seconds.
   */
  seconds: number;
  /**
   * The type of the usage object. Always `duration` for this variant.
   */
  type: 'duration';
}