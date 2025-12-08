import React from 'react';
import { UserConfig } from '../types/UserConfig';
import { SummarizationConfig } from '../types/SummarizationConfig';
import { RefinementConfig } from '../types/RefinementConfig';
import { WebSearchConfig } from '../types/WebSearchConfig';
import { ImageGenerationConfig } from '../types/ImageGenerationConfig';
import { ModelProfileConfig } from '../types/ModelProfileConfig';
import { PreferencesConfig } from '../types/PreferencesConfig';
import { MemoryConfig } from '../types/MemoryConfig';
import { CircuitBreakerConfig } from '../types/CircuitBreakerConfig';
import { GPUConfig } from '../types/GpuConfig';
export type ConfigSection = SummarizationConfig | RefinementConfig | WebSearchConfig | ImageGenerationConfig | ModelProfileConfig | PreferencesConfig | MemoryConfig | CircuitBreakerConfig | GPUConfig;
interface ConfigContextType {
    config: UserConfig | null;
    isLoading: boolean;
    error: Error | null;
    fetchConfig: () => Promise<void>;
    updateConfig: (newConfig: UserConfig) => void;
    updatePartialConfig: (section: keyof UserConfig, sectionConfig: ConfigSection) => Promise<boolean>;
}
export declare const ConfigProvider: React.FC<{
    children: React.ReactNode;
}>;
export declare const useConfigContext: () => ConfigContextType;
export {};
