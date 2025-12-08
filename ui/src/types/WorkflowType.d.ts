/**
 * Supported workflow types for composer orchestration.

 */
export type WorkflowType = 'general' | 'research' | 'engineering' | 'creative' | 'image_generation' | 'image_refinement' | 'analysis' | 'planning' | 'focused';
/**
 * Constant values for WorkflowType
 */
export declare const WorkflowTypeValues: {
    /** general */
    readonly GENERAL: "general";
    /** research */
    readonly RESEARCH: "research";
    /** engineering */
    readonly ENGINEERING: "engineering";
    /** creative */
    readonly CREATIVE: "creative";
    /** image_generation */
    readonly IMAGE_GENERATION: "image_generation";
    /** image_refinement */
    readonly IMAGE_REFINEMENT: "image_refinement";
    /** analysis */
    readonly ANALYSIS: "analysis";
    /** planning */
    readonly PLANNING: "planning";
    /** focused */
    readonly FOCUSED: "focused";
};
