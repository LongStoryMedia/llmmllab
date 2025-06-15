import { useState, useEffect } from 'react';
import { Box, Typography, Button, Switch, FormControlLabel, Alert } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';
import { RefinementConfig } from '../../types/RefinementConfig';

const RefinementSettings = () => {
  const { config, updatePartialConfig, isLoading } = useConfigContext();
  const [localConfig, setLocalConfig] = useState<RefinementConfig>({
    enable_response_filtering: false,
    enable_response_critique: false
  });
  const [saveStatus, setSaveStatus] = useState<{ success?: boolean; message: string } | null>(null);

  useEffect(() => {
    // When user config loads, update local state
    if (config?.refinement) {
      setLocalConfig({
        enable_response_filtering: config.refinement.enable_response_filtering ?? false,
        enable_response_critique: config.refinement.enable_response_critique ?? false
      });
    }
  }, [config]);

  const handleToggleFiltering = () => {
    setLocalConfig({
      ...localConfig,
      enable_response_filtering: !localConfig.enable_response_filtering
    });
  };

  const handleToggleCritique = () => {
    setLocalConfig({
      ...localConfig,
      enable_response_critique: !localConfig.enable_response_critique
    });
  };

  const handleSave = async () => {
    setSaveStatus(null);
    try {
      // Convert camelCase to snake_case when passing to updatePartialConfig
      const snakeCaseConfig = {
        enable_response_filtering: localConfig.enable_response_filtering,
        enable_response_critique: localConfig.enable_response_critique
      };

      const success = await updatePartialConfig('refinement', snakeCaseConfig);

      if (success) {
        setSaveStatus({
          success: true,
          message: 'Refinement settings saved successfully!'
        });
      } else {
        setSaveStatus({
          success: false,
          message: 'Failed to save refinement settings.'
        });
      }
    } catch (err) {
      console.error('Error saving refinement settings:', err);
      setSaveStatus({
        success: false,
        message: 'An error occurred while saving settings.'
      });
    }
  };

  if (isLoading) {
    return <Box sx={{ padding: 2 }}><Typography>Loading refinement settings...</Typography></Box>;
  }

  return (
    <Box sx={{ padding: 2 }}>
      <Typography variant="h6" gutterBottom>
        Refinement Settings
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
            checked={localConfig.enable_response_filtering}
            onChange={handleToggleFiltering}
          />
        }
        label="Enable Response Filtering"
        sx={{ mb: 2, display: 'block' }}
      />

      <FormControlLabel
        control={
          <Switch
            checked={localConfig.enable_response_critique}
            onChange={handleToggleCritique}
          />
        }
        label="Enable Response Critique"
        sx={{ mb: 2, display: 'block' }}
      />

      <Button
        variant="contained"
        color="primary"
        sx={{ mt: 2 }}
        onClick={handleSave}
        disabled={isLoading}
      >
        Save Refinement Settings
      </Button>
    </Box>
  );
};

export default RefinementSettings;