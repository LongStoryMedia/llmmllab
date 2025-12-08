/**
 * Type of content in a message or streaming chunk
 */
export type MessageContentType = 'text' | 'image' | 'tool_call' | 'tool_result' | 'image_generation' | 'audio' | 'video' | 'image_edit' | 'file' | 'thinking' | 'error_content' | 'analysis';
/**
 * Constant values for MessageContentType
 */
export declare const MessageContentTypeValues: {
    /** text */
    readonly TEXT: "text";
    /** image */
    readonly IMAGE: "image";
    /** tool_call */
    readonly TOOL_CALL: "tool_call";
    /** tool_result */
    readonly TOOL_RESULT: "tool_result";
    /** image_generation */
    readonly IMAGE_GENERATION: "image_generation";
    /** audio */
    readonly AUDIO: "audio";
    /** video */
    readonly VIDEO: "video";
    /** image_edit */
    readonly IMAGE_EDIT: "image_edit";
    /** file */
    readonly FILE: "file";
    /** thinking */
    readonly THINKING: "thinking";
    /** error_content */
    readonly ERROR_CONTENT: "error_content";
    /** analysis */
    readonly ANALYSIS: "analysis";
};
