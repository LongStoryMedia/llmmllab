/**
 * Role of a participant in a chat conversation
 */
export type MessageRole = 'user' | 'assistant' | 'system' | 'tool' | 'agent' | 'observer';
/**
 * Constant values for MessageRole
 */
export declare const MessageRoleValues: {
    /** user */
    readonly USER: "user";
    /** assistant */
    readonly ASSISTANT: "assistant";
    /** system */
    readonly SYSTEM: "system";
    /** tool */
    readonly TOOL: "tool";
    /** agent */
    readonly AGENT: "agent";
    /** observer */
    readonly OBSERVER: "observer";
};
