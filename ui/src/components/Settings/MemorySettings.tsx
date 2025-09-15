import { useState, useEffect } from 'react';
import { Box, TextField, Typography, Button, Switch, FormControlLabel, Slider, Alert, Divider, Dialog, DialogTitle, DialogContent, DialogActions } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import MemoryIcon from '@mui/icons-material/Memory';
import { useConfigContext } from '../../context/ConfigContext';
import { useAuth } from '../../auth';
import { MemoryConfig } from '../../types/MemoryConfig';
import { clearMemory, nuclearClearMemory } from '../../api/resources';

const RetrievalSettings = () => {
  const { config, updatePartialConfig, isLoading } = useConfigContext();
  const { user, isAdmin } = useAuth();
  const [localConfig, setLocalConfig] = useState<MemoryConfig>({
    enabled: true,
    limit: 5,
    enable_cross_user: false,
    enable_cross_conversation: false,
    similarity_threshold: 0.7,
    always_retrieve: false
  });
  const [saveStatus, setSaveStatus] = useState<{ success?: boolean; message: string } | null>(null);
  const [memoryCleanupStatus, setMemoryCleanupStatus] = useState<{ success?: boolean; message: string } | null>(null);
  const [isCleaningMemory, setIsCleaningMemory] = useState(false);
  const [showNuclearDialog, setShowNuclearDialog] = useState(false);

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

  const handleBasicMemoryCleanup = async () => {
    if (!user?.access_token) {
      return;
    }

    setIsCleaningMemory(true);
    setMemoryCleanupStatus(null);

    try {
      const result = await clearMemory(user.access_token, { aggressive: true });
      setMemoryCleanupStatus({
        success: true,
        message: `Memory cleared successfully: ${result.detail}`
      });
    } catch (error) {
      setMemoryCleanupStatus({
        success: false,
        message: `Failed to clear memory: ${error instanceof Error ? error.message : 'Unknown error'}`
      });
    } finally {
      setIsCleaningMemory(false);
    }
  };

  const handleNuclearMemoryCleanup = async () => {
    if (!user?.access_token) {
      return;
    }

    setIsCleaningMemory(true);
    setMemoryCleanupStatus(null);
    setShowNuclearDialog(false);

    try {
      const result = await nuclearClearMemory(user.access_token, undefined, true);
      setMemoryCleanupStatus({
        success: true,
        message: `Nuclear memory cleanup completed: ${result.detail}`
      });
    } catch (error) {
      setMemoryCleanupStatus({
        success: false,
        message: `Nuclear cleanup failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      });
    } finally {
      setIsCleaningMemory(false);
    }
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
            onChange={(e) => setLocalConfig({ ...localConfig, limit: parseInt(e.target.value) || 5 })}
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

      {/* Admin Memory Cleanup Section */}
      {isAdmin && (
        <>
          <Divider sx={{ mt: 4, mb: 3 }} />
          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            <MemoryIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Memory Management (Admin Only)
          </Typography>

          {memoryCleanupStatus && (
            <Alert
              severity={memoryCleanupStatus.success ? "success" : "error"}
              sx={{ mb: 2 }}
              onClose={() => setMemoryCleanupStatus(null)}
            >
              {memoryCleanupStatus.message}
            </Alert>
          )}

          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
            <Button
              variant="outlined"
              color="warning"
              onClick={handleBasicMemoryCleanup}
              disabled={isCleaningMemory}
              startIcon={<DeleteIcon />}
              sx={{ minWidth: '200px' }}
            >
              {isCleaningMemory ? 'Cleaning...' : 'Clear Memory Cache'}
            </Button>

            <Button
              variant="outlined"
              color="error"
              onClick={() => setShowNuclearDialog(true)}
              disabled={isCleaningMemory}
              startIcon={<DeleteIcon />}
              sx={{ minWidth: '200px' }}
            >
              Nuclear Memory Cleanup
            </Button>
          </Box>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            <strong>Clear Memory Cache:</strong> Releases GPU memory and unloads cached models.<br />
            <strong>Nuclear Cleanup:</strong> Force-kills all processes and performs aggressive memory cleanup. Use with caution!
          </Typography>
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

      {/* Nuclear Cleanup Confirmation Dialog */}
      <Dialog
        open={showNuclearDialog}
        onClose={() => setShowNuclearDialog(false)}
        aria-labelledby="nuclear-cleanup-dialog-title"
        aria-describedby="nuclear-cleanup-dialog-description"
      >
        <DialogTitle id="nuclear-cleanup-dialog-title">
          ⚠️ Nuclear Memory Cleanup
        </DialogTitle>
        <DialogContent>
          <Typography id="nuclear-cleanup-dialog-description" sx={{ mb: 2 }}>
            This will forcefully terminate all running processes and perform aggressive memory cleanup.
          </Typography>
          <Alert severity="warning" sx={{ mb: 2 }}>
            <strong>Warning:</strong> This action may interrupt running tasks and cause data loss.
            Only use this if normal memory cleanup fails.
          </Alert>
          <Typography variant="body2" color="text.secondary">
            Are you sure you want to proceed with nuclear memory cleanup?
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowNuclearDialog(false)} color="primary">
            Cancel
          </Button>
          <Button
            onClick={handleNuclearMemoryCleanup}
            color="error"
            variant="contained"
            disabled={isCleaningMemory}
          >
            {isCleaningMemory ? 'Cleaning...' : 'Proceed'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default RetrievalSettings;