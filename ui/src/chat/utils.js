// Types for managing ordered response sections during streaming
/**
 * Helper to create a new section based on generation state
 */
export function createSection(state, order) {
    const section = {
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
