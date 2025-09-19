import { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Alert,
  Paper,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
  Chip,
  Stack
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  ExpandMore as ExpandMoreIcon,
  Memory as MemoryIcon
} from '@mui/icons-material';
import { useConfigContext } from '../../context/ConfigContext';
import { useAuth } from '../../auth';
import { GPUConfig } from '../../types/GpuConfig';
import { getDeviceMappings, DeviceMappingsResponse, DeviceInfo } from '../../api/resources';

const GpuSettings = () => {
  const { config, updatePartialConfig, isLoading } = useConfigContext();
  const { user } = useAuth();

  const [localConfig, setLocalConfig] = useState<GPUConfig>({
    no_kv_offload: false,
    main_gpu: -1,
    tensor_split: [],
    tensor_split_devices: [],
    split_mode: 'none',
    offload_kqv: true
  });

  const [devices, setDevices] = useState<Record<string, DeviceInfo>>({});
  const [isLoadingDevices, setIsLoadingDevices] = useState(false);
  const [saveStatus, setSaveStatus] = useState<{ success?: boolean; message: string } | null>(null);
  const [deviceError, setDeviceError] = useState<string | null>(null);

  // Load initial GPU config from gpu_config (now separate from circuit_breaker)
  useEffect(() => {
    if (config?.gpu_config) {
      setLocalConfig(config.gpu_config);
    }
  }, [config]);

  // Load available devices
  const loadDevices = useCallback(async () => {
    if (!user?.access_token) {
      return;
    }

    setIsLoadingDevices(true);
    setDeviceError(null);

    try {
      const response: DeviceMappingsResponse = await getDeviceMappings(user.access_token);
      setDevices(response.devices);
    } catch (error) {
      console.error('Failed to load devices:', error);
      setDeviceError(`Failed to load devices: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsLoadingDevices(false);
    }
  }, [user?.access_token]);

  useEffect(() => {
    loadDevices();
  }, [loadDevices]);

  const handleChange = (field: keyof GPUConfig, value: string | number | boolean | number[] | string[] | undefined) => {
    setLocalConfig(prev => ({ ...prev, [field]: value }));
  };

  const handleTensorSplitChange = (index: number, value: number) => {
    const newTensorSplit = [...(localConfig.tensor_split || [])];
    newTensorSplit[index] = value;
    handleChange('tensor_split', newTensorSplit);
  };

  const handleTensorSplitDeviceChange = (index: number, deviceId: string) => {
    const newDevices = [...(localConfig.tensor_split_devices || [])];
    newDevices[index] = deviceId;
    handleChange('tensor_split_devices', newDevices);
  };

  const addTensorSplitDevice = () => {
    const availableDevices = Object.keys(devices);
    if (availableDevices.length === 0) {
      return; // No devices available
    }

    // Find the first available device not already in use
    const usedDevices = localConfig.tensor_split_devices || [];
    const availableDevice = availableDevices.find(deviceId => !usedDevices.includes(deviceId)) || availableDevices[0];

    const newTensorSplit = [...(localConfig.tensor_split || []), 0.5];
    const newDevices = [...(localConfig.tensor_split_devices || []), availableDevice];

    handleChange('tensor_split', newTensorSplit);
    handleChange('tensor_split_devices', newDevices);
  };

  const removeTensorSplitDevice = (index: number) => {
    const newTensorSplit = [...(localConfig.tensor_split || [])];
    const newDevices = [...(localConfig.tensor_split_devices || [])];

    newTensorSplit.splice(index, 1);
    newDevices.splice(index, 1);

    handleChange('tensor_split', newTensorSplit);
    handleChange('tensor_split_devices', newDevices);
  };

  const handleSave = async () => {
    setSaveStatus(null);

    try {
      // Update the gpu_config directly (now separate from circuit_breaker)
      const success = await updatePartialConfig('gpu_config', localConfig);

      if (success) {
        setSaveStatus({
          success: true,
          message: 'GPU settings saved successfully!'
        });
      } else {
        setSaveStatus({
          success: false,
          message: 'Failed to save GPU settings.'
        });
      }
    } catch (err) {
      console.error('Error saving GPU settings:', err);
      setSaveStatus({
        success: false,
        message: 'An error occurred while saving GPU settings.'
      });
    }
  };

  const getDeviceOptions = () => {
    return Object.entries(devices).map(([key, device]) => ({
      value: key,
      label: `${device.name} (${key})`,
      info: device
    }));
  };

  const getAvailableDevicesForTensorSplit = (currentIndex: number) => {
    const usedDevices = (localConfig.tensor_split_devices || []).filter((_, i) => i !== currentIndex);
    return Object.entries(devices).filter(([key]) => !usedDevices.includes(key));
  };

  const getTensorSplitSum = () => {
    return (localConfig.tensor_split || []).reduce((sum, val) => sum + val, 0);
  };

  const isTensorSplitValid = () => {
    const sum = getTensorSplitSum();
    return Math.abs(sum - 1.0) < 0.01; // Allow small floating point errors
  };

  const getDeviceName = (deviceId: string) => {
    return devices[deviceId]?.name || deviceId;
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 4 }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading GPU settings...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ padding: 2 }}>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
        <MemoryIcon sx={{ mr: 1 }} />
        GPU Configuration
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure GPU memory management, device allocation, and performance optimization settings.
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

      {/* Device Discovery Section */}
      <Accordion sx={{ mb: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">
            Available Devices
            <IconButton
              onClick={(e) => {
                e.stopPropagation();
                loadDevices();
              }}
              size="small"
              sx={{ ml: 1 }}
              disabled={isLoadingDevices}
            >
              {isLoadingDevices ? <CircularProgress size={16} /> : <RefreshIcon />}
            </IconButton>
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          {deviceError ? (
            <Alert severity="error" sx={{ mb: 2 }}>
              {deviceError}
              <Button onClick={loadDevices} sx={{ ml: 2 }} size="small">
                Retry
              </Button>
            </Alert>
          ) : (
            <Grid container spacing={2}>
              {Object.entries(devices).map(([key, device]) => (
                <Grid key={key} size={{ xs: 12, sm: 6, md: 4 }}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="subtitle2">{device.name}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Device ID: {key}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Memory: Available
                    </Typography>
                  </Paper>
                </Grid>
              ))}
              {Object.keys(devices).length === 0 && !isLoadingDevices && (
                <Grid size={12}>
                  <Typography variant="body2" color="text.secondary">
                    No devices found. Click refresh to discover available GPUs and devices.
                  </Typography>
                </Grid>
              )}
            </Grid>
          )}
        </AccordionDetails>
      </Accordion>

      <Grid container spacing={3}>
        {/* Memory Management */}
        <Grid size={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Memory Management
            </Typography>

            <FormControlLabel
              control={
                <Switch
                  checked={localConfig.no_kv_offload || false}
                  onChange={(e) => handleChange('no_kv_offload', e.target.checked)}
                />
              }
              label="Force KV Cache to CPU (saves VRAM)"
              sx={{ mb: 2, display: 'block' }}
            />

            <FormControlLabel
              control={
                <Switch
                  checked={localConfig.offload_kqv || false}
                  onChange={(e) => handleChange('offload_kqv', e.target.checked)}
                />
              }
              label="Offload Key/Query/Value tensors to GPU"
              sx={{ mb: 2, display: 'block' }}
            />
          </Paper>
        </Grid>

        {/* Device Selection */}
        <Grid size={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Device Selection
            </Typography>

            <FormControl fullWidth margin="normal">
              <InputLabel>Main GPU Device</InputLabel>
              <Select
                value={localConfig.main_gpu_device_id || ''}
                onChange={(e) => handleChange('main_gpu_device_id', e.target.value)}
                label="Main GPU Device"
              >
                <MenuItem value="">
                  <em>Auto-select</em>
                </MenuItem>
                {getDeviceOptions().map(device => (
                  <MenuItem key={device.value} value={device.value}>
                    {device.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              label="Main GPU Index"
              type="number"
              value={localConfig.main_gpu ?? -1}
              onChange={(e) => handleChange('main_gpu', parseInt(e.target.value) || -1)}
              fullWidth
              margin="normal"
              helperText="GPU device index (-1 for auto-selection, overridden by device ID above)"
              inputProps={{ min: -1 }}
            />

            <FormControl fullWidth margin="normal">
              <InputLabel>Model Split Mode</InputLabel>
              <Select
                value={localConfig.split_mode || 'none'}
                onChange={(e) => handleChange('split_mode', e.target.value)}
                label="Model Split Mode"
              >
                <MenuItem value="none">None - Single device</MenuItem>
                <MenuItem value="layer">Layer - Split by layers</MenuItem>
                <MenuItem value="row">Row - Split by tensor rows</MenuItem>
              </Select>
            </FormControl>
          </Paper>
        </Grid>

        {/* Tensor Split Configuration */}
        <Grid size={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              Tensor Split Configuration
              <Button
                onClick={addTensorSplitDevice}
                startIcon={<AddIcon />}
                size="small"
                sx={{ ml: 2 }}
                disabled={Object.keys(devices).length === 0 || (localConfig.tensor_split_devices?.length || 0) >= Object.keys(devices).length}
              >
                Add Device
              </Button>
            </Typography>

            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Distribute model computation across multiple GPUs. Values must sum to 1.0.
            </Typography>

            {localConfig.tensor_split && localConfig.tensor_split.length > 0 && (
              <>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" color={isTensorSplitValid() ? 'success.main' : 'error.main'}>
                    Current sum: {getTensorSplitSum().toFixed(3)} {isTensorSplitValid() ? '✓' : '(must equal 1.0)'}
                  </Typography>
                </Box>

                {localConfig.tensor_split.map((split, index) => {
                  const currentDeviceId = localConfig.tensor_split_devices?.[index];
                  const availableDevices = getAvailableDevicesForTensorSplit(index);

                  return (
                    <Box key={index} sx={{ mb: 3, p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                      <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" sx={{ minWidth: 80 }}>
                          Device {index + 1}:
                        </Typography>

                        <FormControl sx={{ minWidth: 200 }}>
                          <InputLabel size="small">Select Device</InputLabel>
                          <Select
                            value={currentDeviceId || ''}
                            onChange={(e) => handleTensorSplitDeviceChange(index, e.target.value)}
                            label="Select Device"
                            size="small"
                          >
                            {availableDevices.map(([deviceId, device]) => (
                              <MenuItem key={deviceId} value={deviceId}>
                                {device.name} ({deviceId})
                              </MenuItem>
                            ))}
                            {currentDeviceId && !availableDevices.find(([id]) => id === currentDeviceId) && (
                              <MenuItem value={currentDeviceId}>
                                {getDeviceName(currentDeviceId)} ({currentDeviceId})
                              </MenuItem>
                            )}
                          </Select>
                        </FormControl>

                        <IconButton
                          onClick={() => removeTensorSplitDevice(index)}
                          size="small"
                          color="error"
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Stack>

                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Typography sx={{ minWidth: 80 }}>
                          Split ratio:
                        </Typography>
                        <Slider
                          value={split}
                          onChange={(_, value) => handleTensorSplitChange(index, value as number)}
                          min={0}
                          max={1}
                          step={0.01}
                          sx={{ mx: 2, flex: 1 }}
                          valueLabelDisplay="auto"
                          valueLabelFormat={(value) => value.toFixed(2)}
                        />
                        <Typography sx={{ minWidth: 60, textAlign: 'center' }}>
                          {split.toFixed(2)}
                        </Typography>
                      </Box>

                      {currentDeviceId && (
                        <Chip
                          label={`${getDeviceName(currentDeviceId)}: ${(split * 100).toFixed(1)}%`}
                          size="small"
                          sx={{ mt: 1 }}
                        />
                      )}
                    </Box>
                  );
                })}
              </>
            )}

            {(!localConfig.tensor_split || localConfig.tensor_split.length === 0) && (
              <Typography variant="body2" color="text.secondary">
                No tensor split configured. Device allocation will be determined automatically based on available resources.
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>

      <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleSave}
          disabled={isLoading}
        >
          Save GPU Settings
        </Button>

        <Button
          variant="outlined"
          onClick={() => setLocalConfig(config?.gpu_config || {
            no_kv_offload: false,
            main_gpu: -1,
            tensor_split: [],
            tensor_split_devices: [],
            split_mode: 'none',
            offload_kqv: true
          })}
          disabled={isLoading}
        >
          Reset Changes
        </Button>
      </Box>
    </Box>
  );
};

export default GpuSettings;