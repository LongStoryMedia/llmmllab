

import { DragPoint } from './DragPoint';



/**
 * A drag action.

 */
export interface Drag {
  /**
   * An array of coordinates representing the path of the drag action. Coordinates will appear as an array
of objects, eg
```
[
  { x: 100, y: 200 },
  { x: 200, y: 300 }
]
```

   */
  path: (DragPoint)[];
  /**
   * Specifies the event type. For a drag action, this property is 
always set to `drag`.

   */
  type: 'drag';
}