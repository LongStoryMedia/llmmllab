import { useState, useEffect } from 'react';
import { Box, Typography, FormControlLabel, Switch, Slider, Alert, Button } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';

const WebSearchSettings = () => {
  const { config, updatePartialConfig, isLoading } = useConfigContext();
  const [localConfig, setLocalConfig] = useState({
    enabled: false,
    autoDetect: true,
    maxResults: 3,
    includeResults: true
  });
  const [saveStatus, setSaveStatus] = useState<{success?: boolean; message: string} | null>(null);

  useEffect(() => {
    // When user config loads, update local state
    if (config?.web_search) {
      setLocalConfig({
        enabled: config.web_search.enabled ?? false,
        autoDetect: config.web_search.auto_detect ?? true,
        maxResults: config.web_search.max_results ?? 3,
        includeResults: config.web_search.include_results ?? true
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
      autoDetect: !localConfig.autoDetect
    });
  };

  const handleToggleIncludeResults = () => {
    setLocalConfig({
      ...localConfig,
      includeResults: !localConfig.includeResults
    });
  };

  const handleMaxResultsChange = (_event: Event, newValue: number | number[]) => {
    setLocalConfig({
      ...localConfig,
      maxResults: newValue as number
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
                checked={localConfig.autoDetect} 
                onChange={handleToggleAutoDetect} 
              />
            }
            label="Auto-detect when to search"
            sx={{ mb: 2, display: 'block' }}
          />
          
          <FormControlLabel
            control={
              <Switch 
                checked={localConfig.includeResults} 
                onChange={handleToggleIncludeResults} 
              />
            }
            label="Include search results in responses"
            sx={{ mb: 2, display: 'block' }}
          />
          
          <Typography id="max-results-slider" gutterBottom>
            Maximum search results: {localConfig.maxResults}
          </Typography>
          <Slider
            aria-labelledby="max-results-slider"
            value={localConfig.maxResults}
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