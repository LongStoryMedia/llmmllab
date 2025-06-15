import { useState, useEffect } from 'react';
import { Box, TextField, Typography, Button, Alert, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';
import { useAuth } from '../../auth';
import { PreferencesConfig } from '../../types/PreferencesConfig';

const ProfileSettings = () => {
  const { user } = useAuth();
  const { config, updatePartialConfig, isLoading } = useConfigContext();
  const [preferences, setPreferences] = useState<PreferencesConfig>({
    font_size: 14,
    language: 'en',
    notifications_on: true
  });
  
  const [saveStatus, setSaveStatus] = useState<{success?: boolean; message: string} | null>(null);

  // Update local state when the user config loads or changes
  useEffect(() => {
    if (config?.preferences) {
      setPreferences({
        font_size: config.preferences.font_size || 14,
        language: config.preferences.language || 'en',
        notifications_on: config.preferences.notifications_on !== false
      });
    }
  }, [config]);

  const handleSave = async () => {
    setSaveStatus(null);
    try {
      // Convert camelCase to snake_case when passing to updatePartialConfig
      const snakeCasePreferences = {
        font_size: preferences.font_size,
        language: preferences.language,
        notifications_on: preferences.notifications_on
      };
      
      const success = await updatePartialConfig('preferences', snakeCasePreferences);
      
      if (success) {
        setSaveStatus({
          success: true,
          message: 'Profile settings saved successfully!'
        });
      } else {
        setSaveStatus({
          success: false,
          message: 'Failed to save settings.'
        });
      }
    } catch (err) {
      setSaveStatus({
        success: false,
        message: `Error: ${err instanceof Error ? err.message : String(err)}`
      });
    }
  };

  if (isLoading) {
    return <Box sx={{ padding: 2 }}><Typography>Loading profile settings...</Typography></Box>;
  }

  return (
    <Box sx={{ padding: 2 }}>
      <Typography variant="h6" gutterBottom>
        Profile Settings
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
      <Typography variant="subtitle1" gutterBottom>
        Account Information
      </Typography>
      <TextField
        label="Name"
        value={user?.profile.name || ''}
        fullWidth
        margin="normal"
        disabled
        helperText="Your account name (managed by authentication provider)"
      />
      <TextField
        label="Email"
        value={user?.profile.email || ''}
        fullWidth
        margin="normal"
        disabled
        helperText="Your account email (managed by authentication provider)"
      />
      <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
        Display Preferences
      </Typography>
      <TextField
        label="Font Size"
        type="number"
        value={preferences.font_size}
        onChange={(e) => setPreferences({...preferences, font_size: parseInt(e.target.value) || 14})}
        fullWidth
        margin="normal"
        helperText="Font size for chat messages (10-24px)"
        slotProps={{
          input: { inputProps: { min: 10, max: 24 } }
        }}
      />
      <FormControl fullWidth margin="normal">
        <InputLabel id="language-select-label">Language</InputLabel>
        <Select
          labelId="language-select-label"
          id="language-select"
          value={preferences.language}
          onChange={(e) => setPreferences({...preferences, language: e.target.value})}
          label="Language"
        >
          <MenuItem value="en">English</MenuItem>
          <MenuItem value="es">Spanish</MenuItem>
          <MenuItem value="fr">French</MenuItem>
          <MenuItem value="de">German</MenuItem>
          <MenuItem value="zh">Chinese</MenuItem>
          <MenuItem value="ja">Japanese</MenuItem>
        </Select>
      </FormControl>
      <Button 
        variant="contained" 
        color="primary" 
        sx={{ mt: 2 }} 
        onClick={handleSave}
      >
        Save Profile Settings
      </Button>
    </Box>
  );
};

export default ProfileSettings;