

import { UrlAnnotationSource } from './UrlAnnotationSource';



/**
 * Annotation that references a URL.
 */
export interface UrlAnnotation {
  /**
   * URL referenced by the annotation.
   */
  source: UrlAnnotationSource;
  /**
   * Type discriminator that is always `url` for this annotation.
   */
  type: 'url';
}