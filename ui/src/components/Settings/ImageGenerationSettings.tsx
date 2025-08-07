import { useState, useEffect } from 'react';
import { Box, Typography, Button, Switch, FormControlLabel, Alert, TextField, Divider } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';
import { ImageGenerationConfig } from '../../types/ImageGenerationConfig';
import ImageModelSelector from '../ModelSelector/ImageModelSelector';

const ImageGenerationSettings = () => {
  const { config, updatePartialConfig, isLoading } = useConfigContext();
  const [localConfig, setLocalConfig] = useState<ImageGenerationConfig>({
    enabled: false,
    storage_directory: '',
    max_image_size: 1024,
    retention_hours: 720,
    auto_prompt_refinement: true
  });
  const [saveStatus, setSaveStatus] = useState<{ success?: boolean; message: string } | null>(null);

  useEffect(() => {
    // When user config loads, update local state
    if (config?.image_generation) {
      setLocalConfig({
        enabled: config.image_generation.enabled ?? false,
        storage_directory: config.image_generation.storage_directory ?? '',
        max_image_size: config.image_generation.max_image_size ?? 1024,
        retention_hours: config.image_generation.retention_hours ?? 720,
        auto_prompt_refinement: config.image_generation.auto_prompt_refinement ?? true
      });
    }
  }, [config]);

  const handleToggleEnabled = () => {
    setLocalConfig({
      ...localConfig,
      enabled: !localConfig.enabled
    });
  };

  const handleToggleAutoPromptRefinement = () => {
    setLocalConfig({
      ...localConfig,
      auto_prompt_refinement: !localConfig.auto_prompt_refinement
    });
  };

  const handleMaxSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value)) {
      setLocalConfig({
        ...localConfig,
        max_image_size: Math.min(Math.max(value, 128), 4096) // Enforce min/max values
      });
    }
  };

  const handleRetentionHoursChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value)) {
      setLocalConfig({
        ...localConfig,
        retention_hours: Math.min(Math.max(value, 1), 720) // Enforce min/max values
      });
    }
  };

  const handleStorageDirectoryChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setLocalConfig({
      ...localConfig,
      storage_directory: e.target.value
    });
  };

  const handleSave = async () => {
    setSaveStatus(null);
    try {
      const success = await updatePartialConfig('image_generation', localConfig);

      if (success) {
        setSaveStatus({
          success: true,
          message: 'Image generation settings saved successfully!'
        });
      } else {
        setSaveStatus({
          success: false,
          message: 'Failed to save image generation settings.'
        });
      }
    } catch (err) {
      console.error('Error saving image generation settings:', err);
      setSaveStatus({
        success: false,
        message: 'An error occurred while saving settings.'
      });
    }
  };

  if (isLoading) {
    return <Box sx={{ padding: 2 }}><Typography>Loading image generation settings...</Typography></Box>;
  }

  return (
    <Box sx={{ padding: 2 }}>
      <Typography variant="h6" gutterBottom>
        Image Generation Settings
      </Typography>

      {saveStatus && (
        <Alert
          severity={saveStatus.success ? 'success' : 'error'}
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
        label="Enable Image Generation"
        sx={{ mb: 2, display: 'block' }}
      />

      <FormControlLabel
        control={
          <Switch
            checked={localConfig.auto_prompt_refinement}
            onChange={handleToggleAutoPromptRefinement}
            disabled={!localConfig.enabled}
          />
        }
        label="Enable Automatic Prompt Refinement"
        sx={{ mb: 2, display: 'block' }}
      />

      <TextField
        label="Storage Directory"
        value={localConfig.storage_directory}
        onChange={handleStorageDirectoryChange}
        fullWidth
        margin="normal"
        disabled={!localConfig.enabled}
        helperText="Directory where generated images will be stored"
      />

      <TextField
        label="Maximum Image Size"
        type="number"
        value={localConfig.max_image_size}
        onChange={handleMaxSizeChange}
        fullWidth
        margin="normal"
        disabled={!localConfig.enabled}
        inputProps={{ min: 128, max: 4096 }}
        helperText="Maximum size in pixels (128-4096)"
      />

      <TextField
        label="Retention Hours"
        type="number"
        value={localConfig.retention_hours}
        onChange={handleRetentionHoursChange}
        fullWidth
        margin="normal"
        disabled={!localConfig.enabled}
        inputProps={{ min: 1, max: 720 }}
        helperText="How long to keep generated images (1-720 hours)"
      />

      <Button
        variant="contained"
        color="primary"
        sx={{ mt: 2 }}
        onClick={handleSave}
        disabled={isLoading}
      >
        Save Image Generation Settings
      </Button>

      {localConfig.enabled && (
        <>
          <Divider sx={{ my: 4 }} />
          <Typography variant="h6" gutterBottom>
            Active Model Selection
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Select which model to use for image generation. Only models with TextToImage specialization are shown.
          </Typography>
          <ImageModelSelector mode="TextToImage" />
        </>
      )}
    </Box>
  );
};

export default ImageGenerationSettings;
