/**
 * The current processing stage.
 */
export type SocketStageType = 'initializing' | 'retrieving_memories' | 'searching_web' | 'summarizing' | 'generating_image' | 'processing' | 'rendering' | 'interpreting' | 'open';
/**
 * Constant values for SocketStageType
 */
export declare const SocketStageTypeValues: {
    /** initializing */
    readonly INITIALIZING: "initializing";
    /** retrieving_memories */
    readonly RETRIEVING_MEMORIES: "retrieving_memories";
    /** searching_web */
    readonly SEARCHING_WEB: "searching_web";
    /** summarizing */
    readonly SUMMARIZING: "summarizing";
    /** generating_image */
    readonly GENERATING_IMAGE: "generating_image";
    /** processing */
    readonly PROCESSING: "processing";
    /** rendering */
    readonly RENDERING: "rendering";
    /** interpreting */
    readonly INTERPRETING: "interpreting";
    /** open */
    readonly OPEN: "open";
};
