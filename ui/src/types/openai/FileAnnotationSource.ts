

/**
 * Attachment source referenced by an annotation.
 */
export interface FileAnnotationSource {
  /**
   * Filename referenced by the annotation.
   */
  filename: string;
  /**
   * Type discriminator that is always `file`.
   */
  type: 'file';
}