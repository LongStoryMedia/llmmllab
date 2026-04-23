

/**
 * A mouse move action.

 */
export interface Move {
  /**
   * Specifies the event type. For a move action, this property is 
always set to `move`.

   */
  type: 'move';
  /**
   * The x-coordinate to move to.

   */
  x: number;
  /**
   * The y-coordinate to move to.

   */
  y: number;
}