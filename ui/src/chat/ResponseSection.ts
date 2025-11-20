// Types for managing ordered response sections during streaming

import { GenerationState } from '../types/GenerationState';
import { ToolCall } from '../types/ToolCall';

export interface ResponseSection {
  id: string; // Unique identifier for this section
  order: number; // Order in which this section appeared
  startedAt: number; // Timestamp when section started
  completedAt?: number; // Timestamp when section completed (state changed)
  type: GenerationState; // Type of section based on generation state
  content?: string; // Accumulated content for this section
  toolCalls?: ToolCall[]; // Tool calls for executing sections
}

/**
 * Helper to create a new section based on generation state
 */
export function createSection(
  state: GenerationState,
  order: number
): ResponseSection {
  const section: ResponseSection = {
    id: `${state}-${order}-${Date.now()}`,
    order,
    startedAt: Date.now(),
    type: state
  };

  switch (state) {
    case 'thinking':
      return { ...section, type: 'thinking', content: '' };
    case 'executing':
      return { ...section, type: 'executing', toolCalls: [] };
    case 'analyzing':
      return { ...section, type: 'analyzing', content: '' };
    case 'responding':
    default:
      return { ...section, type: 'responding', content: '' };
  }
}