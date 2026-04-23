

/**
 * An action to type in text.

 */
export interface Type {
  /**
   * The text to type.

   */
  text: string;
  /**
   * Specifies the event type. For a type action, this property is 
always set to `type`.

   */
  type: 'type';
}