import { useState, useEffect } from 'react';
import { Box, TextField, Typography, Button, Switch, FormControlLabel, Slider, Alert } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';

const RetrievalSettings = () => {
  const { config, updatePartialConfig, isLoading } = useConfigContext();
  const [localConfig, setLocalConfig] = useState({
    enabled: true,
    limit: 5,
    enableCrossUser: false,
    enableCrossConversation: false,
    similarityThreshold: 0.7,
    alwaysRetrieve: false
  });
  const [saveStatus, setSaveStatus] = useState<{success?: boolean; message: string} | null>(null);

  useEffect(() => {
    // When user config loads, update local state
    if (config?.memory) {
      setLocalConfig({
        enabled: config.memory.enabled ?? true,
        limit: config.memory.limit ?? 5,
        enableCrossUser: config.memory.enable_cross_user ?? false,
        enableCrossConversation: config.memory.enable_cross_conversation ?? false,
        similarityThreshold: config.memory.similarity_threshold ?? 0.7,
        alwaysRetrieve: config.memory.always_retrieve ?? false
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
      alwaysRetrieve: !localConfig.alwaysRetrieve
    });
  };

  const handleToggleCrossConversation = () => {
    setLocalConfig({
      ...localConfig,
      enableCrossConversation: !localConfig.enableCrossConversation
    });
  };
  const handleToggleCrossUser = () => {
    setLocalConfig({
      ...localConfig,
      enableCrossUser: !localConfig.enableCrossUser
    });
  };

  const handleThresholdChange = (_event: Event, newValue: number | number[]) => {
    setLocalConfig({
      ...localConfig,
      similarityThreshold: newValue as number
    });
  };

  const handleSave = async () => {
    setSaveStatus(null);
    try {
      const success = await updatePartialConfig('memory', localConfig);
      
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
                checked={localConfig.alwaysRetrieve} 
                onChange={handleToggleAlwaysRetrieve} 
              />
            }
            label="Always Attempt Memory Retrieval"
            sx={{ mt: 2, display: 'block' }}
          />
          <FormControlLabel
            control={
              <Switch 
                checked={localConfig.enableCrossConversation} 
                onChange={handleToggleCrossConversation} 
              />
            }
            label="Enable Cross-Conversation Memory Retrieval"
            sx={{ mt: 2, display: 'block' }}
          />
          <FormControlLabel
            control={
              <Switch 
                checked={localConfig.enableCrossUser} 
                onChange={handleToggleCrossUser} 
              />
            }
            label="Enable Cross-User Memory Retrieval"
            sx={{ mt: 2, display: 'block' }}
          />
          {localConfig.enableCrossConversation && (
            <Box sx={{ mt: 3, mb: 2 }}>
              <Typography id="similarity-threshold-slider" gutterBottom>
                Similarity Threshold: {localConfig.similarityThreshold.toFixed(2)}
              </Typography>
              <Slider
                value={localConfig.similarityThreshold}
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