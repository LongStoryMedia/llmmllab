import { useState, useEffect } from 'react';
import { Box, TextField, Typography, Button, Switch, FormControlLabel, Slider, Alert } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';
import { MemoryConfig } from '../../types/MemoryConfig';

const RetrievalSettings = () => {
  const { config, updatePartialConfig, isLoading } = useConfigContext();
  const [localConfig, setLocalConfig] = useState<MemoryConfig>({
    enabled: true,
    limit: 5,
    enable_cross_user: false,
    enable_cross_conversation: false,
    similarity_threshold: 0.7,
    always_retrieve: false
  });
  const [saveStatus, setSaveStatus] = useState<{success?: boolean; message: string} | null>(null);

  useEffect(() => {
    // When user config loads, update local state
    if (config?.memory) {
      setLocalConfig({
        enabled: config.memory.enabled ?? true,
        limit: config.memory.limit ?? 5,
        enable_cross_user: config.memory.enable_cross_user ?? false,
        enable_cross_conversation: config.memory.enable_cross_conversation ?? false,
        similarity_threshold: config.memory.similarity_threshold ?? 0.7,
        always_retrieve: config.memory.always_retrieve ?? false
      });
    }
  }, [config]);

  const handleToggleEnabled = () => {
    setLocalConfig({
      ...localConfig,
      enabled: !localConfig.enabled
    });
  };

  const handleToggleAlwaysRetrieve = () => {
    setLocalConfig({
      ...localConfig,
      always_retrieve: !localConfig.always_retrieve
    });
  };

  const handleToggleCrossConversation = () => {
    setLocalConfig({
      ...localConfig,
      enable_cross_conversation: !localConfig.enable_cross_conversation
    });
  };
  const handleToggleCrossUser = () => {
    setLocalConfig({
      ...localConfig,
      enable_cross_user: !localConfig.enable_cross_user
    });
  };

  const handleThresholdChange = (_event: Event, newValue: number | number[]) => {
    setLocalConfig({
      ...localConfig,
      similarity_threshold: newValue as number
    });
  };

  const handleSave = async () => {
    setSaveStatus(null);
    try {
      // Convert camelCase to snake_case when passing to updatePartialConfig
      const snakeCaseConfig = {
        enabled: localConfig.enabled,
        limit: localConfig.limit,
        enable_cross_user: localConfig.enable_cross_user,
        enable_cross_conversation: localConfig.enable_cross_conversation,
        similarity_threshold: localConfig.similarity_threshold,
        always_retrieve: localConfig.always_retrieve
      };
      
      const success = await updatePartialConfig('memory', snakeCaseConfig);
      
      if (success) {
        setSaveStatus({
          success: true,
          message: 'Memory retrieval settings saved successfully!'
        });
      } else {
        setSaveStatus({
          success: false,
          message: 'Failed to save memory retrieval settings.'
        });
      }
    } catch (err) {
      console.error('Error saving memory retrieval settings:', err);
      setSaveStatus({
        success: false,
        message: 'An error occurred while saving settings.'
      });
    }
  };

  if (isLoading) {
    return <Box sx={{ padding: 2 }}><Typography>Loading memory retrieval settings...</Typography></Box>;
  }

  return (
    <Box sx={{ padding: 2 }}>
      <Typography variant="h6" gutterBottom>
        Memory Retrieval Settings
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
        label="Enable Memory Retrieval"
        sx={{ mb: 2, display: 'block' }}
      />
      
      {localConfig.enabled && (
        <>
          <TextField
            label="Retrieval Limit"
            type="number"
            value={localConfig.limit}
            onChange={(e) => setLocalConfig({...localConfig, limit: parseInt(e.target.value) || 5})}
            fullWidth
            margin="normal"
            helperText="Maximum number of memory items to retrieve"
          />
          <FormControlLabel
            control={
              <Switch 
                checked={localConfig.always_retrieve} 
                onChange={handleToggleAlwaysRetrieve} 
              />
            }
            label="Always Attempt Memory Retrieval"
            sx={{ mt: 2, display: 'block' }}
          />
          <FormControlLabel
            control={
              <Switch 
                checked={localConfig.enable_cross_conversation} 
                onChange={handleToggleCrossConversation} 
              />
            }
            label="Enable Cross-Conversation Memory Retrieval"
            sx={{ mt: 2, display: 'block' }}
          />
          <FormControlLabel
            control={
              <Switch 
                checked={localConfig.enable_cross_user} 
                onChange={handleToggleCrossUser} 
              />
            }
            label="Enable Cross-User Memory Retrieval"
            sx={{ mt: 2, display: 'block' }}
          />
          {localConfig.enable_cross_conversation && (
            <Box sx={{ mt: 3, mb: 2 }}>
              <Typography id="similarity-threshold-slider" gutterBottom>
                Similarity Threshold: {localConfig.similarity_threshold.toFixed(2)}
              </Typography>
              <Slider
                value={localConfig.similarity_threshold}
                onChange={handleThresholdChange}
                aria-labelledby="similarity-threshold-slider"
                step={0.05}
                marks
                min={0.3}
                max={1.0}
                valueLabelDisplay="auto"
              />
              <Typography variant="caption" color="text.secondary">
                Higher values require more similar memories (more precise, fewer results)
              </Typography>
            </Box>
          )}
        </>
      )}
      <Button 
        variant="contained" 
        color="primary" 
        sx={{ mt: 2 }} 
        onClick={handleSave}
        disabled={isLoading}
      >
        Save Memory Settings
      </Button>
    </Box>
  );
};

export default RetrievalSettings;