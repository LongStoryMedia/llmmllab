import { useState, useEffect } from 'react';
import { Box, Typography, FormControlLabel, Switch, Slider, Alert, Button } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';
import { WebSearchConfig } from '../../types/WebSearchConfig';

const WebSearchSettings = () => {
  const { config, updatePartialConfig, isLoading } = useConfigContext();
  const [localConfig, setLocalConfig] = useState<WebSearchConfig>({
    enabled: false,
    auto_detect: true,
    max_results: 3,
    include_results: true
  });
  const [saveStatus, setSaveStatus] = useState<{success?: boolean; message: string} | null>(null);

  useEffect(() => {
    // When user config loads, update local state
    if (config?.web_search) {
      setLocalConfig({
        enabled: config.web_search.enabled ?? false,
        auto_detect: config.web_search.auto_detect ?? true,
        max_results: config.web_search.max_results ?? 3,
        include_results: config.web_search.include_results ?? true
      });
    }
  }, [config]);

  const handleToggleEnabled = () => {
    setLocalConfig({
      ...localConfig,
      enabled: !localConfig.enabled
    });
  };

  const handleToggleAutoDetect = () => {
    setLocalConfig({
      ...localConfig,
      auto_detect: !localConfig.auto_detect
    });
  };

  const handleToggleIncludeResults = () => {
    setLocalConfig({
      ...localConfig,
      include_results: !localConfig.include_results
    });
  };

  const handleMaxResultsChange = (_event: Event, newValue: number | number[]) => {
    setLocalConfig({
      ...localConfig,
      max_results: newValue as number
    });
  };

  const handleSave = async () => {
    setSaveStatus(null);
    try {
      const success = await updatePartialConfig('web_search', localConfig);
      
      if (success) {
        setSaveStatus({
          success: true,
          message: 'Web search settings saved successfully!'
        });
      } else {
        setSaveStatus({
          success: false,
          message: 'Failed to save settings.'
        });
      }
    } catch (err) {
      console.error('Error saving web search settings:', err);
      setSaveStatus({
        success: false,
        message: 'An error occurred while saving settings.'
      });
    }
  };

  if (isLoading) {
    return <Box sx={{ padding: 2 }}><Typography>Loading web search settings...</Typography></Box>;
  }

  return (
    <Box sx={{ padding: 2 }}>
      <Typography variant="h6" gutterBottom>
        Web Search Settings
      </Typography>
      
      {saveStatus && (
        <Alert 
          severity={saveStatus.success ? "success" : "error"} 
          sx={{ mb: 2 }}
          onClose={() => setSaveStatus(null)}
        >
          {saveStatus.message}
        </Alert>
      )}
      
      <FormControlLabel
        control={
          <Switch 
            checked={localConfig.enabled} 
            onChange={handleToggleEnabled} 
          />
        }
        label="Enable Web Search"
        sx={{ mb: 2, display: 'block' }}
      />
      
      {localConfig.enabled && (
        <>
          <FormControlLabel
            control={
              <Switch 
                checked={localConfig.auto_detect} 
                onChange={handleToggleAutoDetect} 
              />
            }
            label="Auto-detect when to search"
            sx={{ mb: 2, display: 'block' }}
          />
          
          <FormControlLabel
            control={
              <Switch 
                checked={localConfig.include_results} 
                onChange={handleToggleIncludeResults} 
              />
            }
            label="Include search results in responses"
            sx={{ mb: 2, display: 'block' }}
          />
          
          <Typography id="max-results-slider" gutterBottom>
            Maximum search results: {localConfig.max_results}
          </Typography>
          <Slider
            aria-labelledby="max-results-slider"
            value={localConfig.max_results}
            onChange={handleMaxResultsChange}
            step={1}
            marks
            min={1}
            max={5}
            valueLabelDisplay="auto"
            sx={{ mb: 3 }}
          />
        </>
      )}
      
      <Button 
        variant="contained" 
        color="primary" 
        onClick={handleSave}
      >
        Save Web Search Settings
      </Button>
    </Box>
  );
};

export default WebSearchSettings;