

/**
 * Returned when an input audio transcription segment is identified for an item.
 */
export interface RealtimeServerEventConversationItemInputAudioTranscriptionSegment {
  /**
   * The index of the input audio content part within the item.
   */
  content_index: number;
  /**
   * End time of the segment in seconds.
   */
  end: number;
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * The segment identifier.
   */
  id: string;
  /**
   * The ID of the item containing the input audio content.
   */
  item_id: string;
  /**
   * The detected speaker label for this segment.
   */
  speaker: string;
  /**
   * Start time of the segment in seconds.
   */
  start: number;
  /**
   * The text for this segment.
   */
  text: string;
  /**
   * The event type, must be `conversation.item.input_audio_transcription.segment`.
   */
  type: string;
}