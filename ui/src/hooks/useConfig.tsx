import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../auth';
import { getConfig, getToken, updateConfig } from '../api';
import { UserConfig } from '../types/UserConfig';

export function useConfig() {
  const [config, setConfig] = useState<UserConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const { user } = useAuth();

  // Fetch user configuration from the API
  const fetchConfig = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const token = getToken(user);
      if (!token) {
        throw new Error('Authentication required');
      }
      
      const data = await getConfig(token);
      setConfig(data);
    } catch (err) {
      console.error('Error fetching configuration:', err);
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  }, [user]);

  // Update full user configuration
  const updateUserConfig = async (newConfig: UserConfig): Promise<boolean> => {
    if (!getToken(user)) {
      setError(new Error('Authentication required'));
      return false;
    }
    
    try {
      await updateConfig(getToken(user), newConfig);
      // Refresh the config after update
      await fetchConfig();
      return true;
    } catch (err) {
      console.error('Error updating configuration:', err);
      setError(err instanceof Error ? err : new Error(String(err)));
      return false;
    }
  };

  // Update a section of the user configuration (e.g., just summarization settings)
  const updatePartialConfig =  async (section: keyof UserConfig, sectionConfig: unknown): Promise<boolean> => {
    if (!config) {
      setError(new Error('No existing configuration to update'));
      return false;
    }
    
    // Create a copy with the updated section
    const updatedConfig = {
      ...config,
      [section]: sectionConfig
    };
    
    return await updateUserConfig(updatedConfig);
  };

  // Load configuration on mount and when user changes
  useEffect(() => {
    if (getToken(user)) {
      fetchConfig();
    }
  }, [user, fetchConfig]);

  return { 
    config, 
    isLoading, 
    error,
    fetchConfig,
    updateConfig: setConfig,
    updatePartialConfig
  };
}