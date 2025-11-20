// Hook to handle streaming response logic and section management
import { useCallback, useRef, useState } from 'react';
import { ChatResponse } from '../../types/ChatResponse';
import { ResponseSection, createSection } from '../ResponseSection';
import { GenerationState } from '../../types/GenerationState';

interface StreamingState {
  sections: ResponseSection[];
  currentSection: ResponseSection | null;
  observerMessages: string[];
}

export const useStreamHandler = () => {
  const [streamingState, setStreamingState] = useState<StreamingState>({
    sections: [],
    currentSection: null,
    observerMessages: []
  });

  const sectionOrderCounter = useRef(0);

  /**
   * Process a single chunk from the streaming response
   */
  const processChunk = useCallback((chunk: ChatResponse) => {
    const currentState = chunk.state;
    const prevState = chunk.prev_state;

    setStreamingState(prev => {
      // Handle state transition - create new section if state changed
      let sections = [...prev.sections];
      let currentSection = prev.currentSection;

      // Check if we need to start a new section
      const stateChanged = prevState && currentState && prevState !== currentState;

      if (stateChanged || !currentSection) {
        // Complete the previous section if it exists
        if (currentSection) {
          const completedSection = {
            ...currentSection,
            completedAt: Date.now()
          };
          sections = sections.map(s =>
            s.id === currentSection!.id ? completedSection : s
          );
        }

        // Create new section for current state
        if (currentState) {
          currentSection = createSection(currentState, sectionOrderCounter.current++);
          sections = [...sections, currentSection];
        }
      }

      // Update current section with new content
      if (currentSection && currentState) {
        currentSection = updateSectionContent(
          currentSection,
          currentState,
          chunk
        );

        // Update the section in the array
        sections = sections.map(s =>
          (s.id === currentSection!.id ? currentSection : s) as ResponseSection
        );
      }

      // Handle observer messages
      const observerMessages = chunk.observer_messages || prev.observerMessages;

      return {
        sections,
        currentSection,
        observerMessages
      };
    });
  }, []);

  /**
   * Reset streaming state
   */
  const resetStreaming = useCallback(() => {
    setStreamingState({
      sections: [],
      currentSection: null,
      observerMessages: []
    });
    sectionOrderCounter.current = 0;
  }, []);

  /**
   * Get the final combined content from all sections
   */
  const getFinalContent = useCallback(() => {
    return streamingState.sections
      .filter(s => s.type === 'responding')
      .map(s => s.content)
      .join('');
  }, [streamingState.sections]);

  return {
    sections: streamingState.sections,
    currentSection: streamingState.currentSection,
    observerMessages: streamingState.observerMessages,
    processChunk,
    resetStreaming,
    getFinalContent
  };
};

/**
 * Update a section with new content based on the chunk
 */
function updateSectionContent(
  section: ResponseSection,
  state: GenerationState,
  chunk: ChatResponse
): ResponseSection {
  switch (state) {
    case 'thinking':
      if (section.type === 'thinking') {
        const thinkingText = chunk.message?.thoughts?.map(t => t.text).join(' ') || '';
        return {
          ...section,
          content: section.content + thinkingText
        };
      }
      break;

    case 'executing':
      if (section.type === 'executing') {
        const newToolCalls = chunk.message?.tool_calls || [];
        return {
          ...section,
          toolCalls: [...section.toolCalls ?? [], ...newToolCalls]
        };
      }
      break;

    case 'responding':
      if (section.type === 'responding') {
        const textContent = chunk.message?.content?.[0]?.text || '';
        return {
          ...section,
          content: section.content + textContent
        };
      }
      break;

    case 'analyzing':
      if (section.type === 'analyzing') {
        // Handle analysis content if needed
        return section;
      }
      break;
  }

  return section;
}