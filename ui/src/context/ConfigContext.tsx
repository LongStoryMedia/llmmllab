import React, { createContext, useContext } from 'react';
import { useConfig } from '../hooks/useConfig';
import { UserConfig } from '../types/UserConfig';
import { SummarizationConfig } from '../types/SummarizationConfig';
import { WebSearchConfig } from '../types/WebSearchConfig';
import { ImageGenerationConfig } from '../types/ImageGenerationConfig';
import { PreferencesConfig } from '../types/PreferencesConfig';
import { MemoryConfig } from '../types/MemoryConfig';


export type ConfigSection = SummarizationConfig | WebSearchConfig | ImageGenerationConfig | PreferencesConfig | MemoryConfig;
interface ConfigContextType {
  config: UserConfig | null;
  isLoading: boolean;
  error: Error | null;
  fetchConfig: () => Promise<void>;
  updateConfig: (newConfig: UserConfig) => void;
  updatePartialConfig: (section: keyof UserConfig, sectionConfig: ConfigSection) => Promise<boolean>;
}

const ConfigContext = createContext<ConfigContextType | null>(null);

export const ConfigProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Use the hook to manage configuration state
  const configState = useConfig();

  return (
    <ConfigContext.Provider value={configState}>
      {children}
    </ConfigContext.Provider>
  );
};

export const useConfigContext = (): ConfigContextType => {
  const context = useContext(ConfigContext);

  if (!context) {
    throw new Error('useConfigContext must be used within a ConfigProvider');
  }

  return context;
};