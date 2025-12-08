/**
 * Provider / runtime of the model (e.g., 'llama.cpp', 'hf', 'hugging face', 'openai', 'stable-diffusion.cpp', 'anthropic')
 */
export type ModelProvider = 'llama_cpp' | 'hf' | 'hugging_face' | 'openai' | 'stable_diffusion_cpp' | 'anthropic' | 'other';
/**
 * Constant values for ModelProvider
 */
export declare const ModelProviderValues: {
    /** llama_cpp */
    readonly LLAMA_CPP: "llama_cpp";
    /** hf */
    readonly HF: "hf";
    /** hugging_face */
    readonly HUGGING_FACE: "hugging_face";
    /** openai */
    readonly OPENAI: "openai";
    /** stable_diffusion_cpp */
    readonly STABLE_DIFFUSION_CPP: "stable_diffusion_cpp";
    /** anthropic */
    readonly ANTHROPIC: "anthropic";
    /** other */
    readonly OTHER: "other";
};
