/**
 * Strategy for tuning parameters optimally.
 */
export type ParameterTuningStrategy = 'binary_search' | 'exponential_backoff' | 'conservative_increment';
/**
 * Constant values for ParameterTuningStrategy
 */
export declare const ParameterTuningStrategyValues: {
    /** binary_search */
    readonly BINARY_SEARCH: "binary_search";
    /** exponential_backoff */
    readonly EXPONENTIAL_BACKOFF: "exponential_backoff";
    /** conservative_increment */
    readonly CONSERVATIVE_INCREMENT: "conservative_increment";
};
