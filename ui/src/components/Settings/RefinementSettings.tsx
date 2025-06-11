import { useState, useEffect } from 'react';
import { Box, Typography, Button, Switch, FormControlLabel, Alert } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';

const RefinementSettings = () => {
  const { config, updatePartialConfig, isLoading } = useConfigContext();
  const [localConfig, setLocalConfig] = useState({
    enableResponseFiltering: false,
    enableResponseCritique: false
  });
  const [saveStatus, setSaveStatus] = useState<{ success?: boolean; message: string } | null>(null);

  useEffect(() => {
    // When user config loads, update local state
    if (config?.refinement) {
      setLocalConfig({
        enableResponseFiltering: config.refinement.enable_response_filtering ?? false,
        enableResponseCritique: config.refinement.enable_response_critique ?? false
      });
    }
  }, [config]);

  const handleToggleFiltering = () => {
    setLocalConfig({
      ...localConfig,
      enableResponseFiltering: !localConfig.enableResponseFiltering
    });
  };

  const handleToggleCritique = () => {
    setLocalConfig({
      ...localConfig,
      enableResponseCritique: !localConfig.enableResponseCritique
    });
  };

  const handleSave = async () => {
    setSaveStatus(null);
    try {
      const success = await updatePartialConfig('refinement', localConfig);

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
            checked={localConfig.enableResponseFiltering}
            onChange={handleToggleFiltering}
          />
        }
        label="Enable Response Filtering"
        sx={{ mb: 2, display: 'block' }}
      />

      <FormControlLabel
        control={
          <Switch
            checked={localConfig.enableResponseCritique}
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