/**
 * Types of capabilities that can be required for tool operations
 */
export type RequiredCapability = 'basic_math' | 'text_processing' | 'information_retrieval' | 'conversation_memory' | 'web_search' | 'summarization' | 'reasoning' | 'general_knowledge' | 'api_integration' | 'async_processing' | 'file_manipulation' | 'data_processing' | 'image_processing' | 'audio_processing' | 'database_access' | 'network_communication' | 'scheduling' | 'temporal_reasoning' | 'personalization';
/**
 * Constant values for RequiredCapability
 */
export declare const RequiredCapabilityValues: {
    /** basic_math */
    readonly BASIC_MATH: "basic_math";
    /** text_processing */
    readonly TEXT_PROCESSING: "text_processing";
    /** information_retrieval */
    readonly INFORMATION_RETRIEVAL: "information_retrieval";
    /** conversation_memory */
    readonly CONVERSATION_MEMORY: "conversation_memory";
    /** web_search */
    readonly WEB_SEARCH: "web_search";
    /** summarization */
    readonly SUMMARIZATION: "summarization";
    /** reasoning */
    readonly REASONING: "reasoning";
    /** general_knowledge */
    readonly GENERAL_KNOWLEDGE: "general_knowledge";
    /** api_integration */
    readonly API_INTEGRATION: "api_integration";
    /** async_processing */
    readonly ASYNC_PROCESSING: "async_processing";
    /** file_manipulation */
    readonly FILE_MANIPULATION: "file_manipulation";
    /** data_processing */
    readonly DATA_PROCESSING: "data_processing";
    /** image_processing */
    readonly IMAGE_PROCESSING: "image_processing";
    /** audio_processing */
    readonly AUDIO_PROCESSING: "audio_processing";
    /** database_access */
    readonly DATABASE_ACCESS: "database_access";
    /** network_communication */
    readonly NETWORK_COMMUNICATION: "network_communication";
    /** scheduling */
    readonly SCHEDULING: "scheduling";
    /** temporal_reasoning */
    readonly TEMPORAL_REASONING: "temporal_reasoning";
    /** personalization */
    readonly PERSONALIZATION: "personalization";
};
