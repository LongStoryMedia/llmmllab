import { GenerationState } from '../types/GenerationState';
import { ResponseSection } from '../types/ResponseSection';
/**
 * Helper to create a new section based on generation state
 */
export declare function createSection(state: GenerationState, order: number): ResponseSection;
