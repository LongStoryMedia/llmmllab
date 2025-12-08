// Hook to handle streaming response logic and section management
import { useCallback, useRef, useState } from 'react';
import { createSection } from '../utils';
export const useStreamHandler = () => {
    // Keep refs for frequently-updated values to avoid recreating variables
    const sectionsRef = useRef([]);
    const currentSectionRef = useRef(undefined);
    const observerMessagesRef = useRef([]);
    const [tick, setTick] = useState(0); // used to force re-renders when needed
    const sectionOrderCounter = useRef(0);
    /**
     * Process a single chunk from the streaming response
     * Returns the new state immediately for synchronous access
     */
    const processChunk = useCallback((chunk) => {
        const currentState = chunk.state;
        const prevState = chunk.prev_state;
        // alias refs to local vars for easier reading
        let sections = sectionsRef.current;
        let currentSection = currentSectionRef.current;
        // Check if we need to start a new section
        const stateChanged = prevState && currentState && prevState !== currentState;
        if (stateChanged || !currentSection) {
            // Complete the previous section if it exists
            if (currentSection) {
                const completedSection = {
                    ...currentSection,
                    completedAt: Date.now()
                };
                sections = sections.map(s => (s.id === currentSection.id ? completedSection : s));
            }
            // Create new section for current state
            if (currentState) {
                currentSection = createSection(currentState, sectionOrderCounter.current++);
                sections = [...sections, currentSection];
            }
        }
        // Update current section with new content
        if (currentSection && currentState) {
            const updatedSection = updateSectionContent(currentSection, currentState, chunk);
            // Update the section in the array
            sections = sections.map(s => (s.id === currentSection.id ? updatedSection : s));
            currentSection = updatedSection;
        }
        // Handle observer messages
        const observerMessages = chunk.observer_messages || observerMessagesRef.current;
        // Write back to refs
        sectionsRef.current = sections;
        currentSectionRef.current = currentSection;
        observerMessagesRef.current = observerMessages;
        // Trigger a re-render asynchronously so components reading hook values update
        setTick(tick + 1);
        const newState = {
            sections,
            currentSection,
            observerMessages
        };
        // Return the new state synchronously
        return newState;
    }, [tick]);
    /**
     * Reset streaming state
     */
    const resetStreaming = useCallback(() => {
        console.log("🔄 Resetting streaming state");
        sectionsRef.current = [];
        currentSectionRef.current = undefined;
        observerMessagesRef.current = [];
        sectionOrderCounter.current = 0;
        setTick(tick + 1);
    }, [tick]);
    /**
     * Get the final combined content from all sections
     */
    const getFinalContent = useCallback(() => {
        return sectionsRef.current
            .filter((s) => s.type === 'responding')
            .map((s) => s.content)
            .join('');
    }, []);
    return {
        sections: sectionsRef.current,
        currentSection: currentSectionRef.current,
        observerMessages: observerMessagesRef.current,
        processChunk,
        resetStreaming,
        getFinalContent
    };
};
/**
 * Update a section with new content based on the chunk
 */
function updateSectionContent(section, state, chunk) {
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
