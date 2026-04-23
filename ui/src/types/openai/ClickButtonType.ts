

export type ClickButtonType = 'left' | 'right' | 'wheel' | 'back' | 'forward';

/**
 * Constant values for ClickButtonType
 */
export const ClickButtonTypeValues = {
  /** left */
  LEFT: 'left',
  /** right */
  RIGHT: 'right',
  /** wheel */
  WHEEL: 'wheel',
  /** back */
  BACK: 'back',
  /** forward */
  FORWARD: 'forward'
} as const;