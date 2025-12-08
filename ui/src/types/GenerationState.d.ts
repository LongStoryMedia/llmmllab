/**
 * Current state of the generation process.
 */
export type GenerationState = 'analyzing' | 'thinking' | 'executing' | 'responding';
/**
 * Constant values for GenerationState
 */
export declare const GenerationStateValues: {
    /** analyzing */
    readonly ANALYZING: "analyzing";
    /** thinking */
    readonly THINKING: "thinking";
    /** executing */
    readonly EXECUTING: "executing";
    /** responding */
    readonly RESPONDING: "responding";
};
