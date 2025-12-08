/**
 * Format options for engineering responses
 */
export type ResponseFormat = 'detailed_analysis' | 'code_solution' | 'step_by_step_guide' | 'best_practices' | 'troubleshooting';
/**
 * Constant values for ResponseFormat
 */
export declare const ResponseFormatValues: {
    /** detailed_analysis */
    readonly DETAILED_ANALYSIS: "detailed_analysis";
    /** code_solution */
    readonly CODE_SOLUTION: "code_solution";
    /** step_by_step_guide */
    readonly STEP_BY_STEP_GUIDE: "step_by_step_guide";
    /** best_practices */
    readonly BEST_PRACTICES: "best_practices";
    /** troubleshooting */
    readonly TROUBLESHOOTING: "troubleshooting";
};
