

/**
 * Type of noise reduction. `near_field` is for close-talking microphones such as headphones, `far_field` is for far-field microphones such as laptop or conference room microphones.

 */
export type NoiseReductionType = 'near_field' | 'far_field';

/**
 * Constant values for NoiseReductionType
 */
export const NoiseReductionTypeValues = {
  /** near_field */
  NEAR_FIELD: 'near_field',
  /** far_field */
  FAR_FIELD: 'far_field'
} as const;