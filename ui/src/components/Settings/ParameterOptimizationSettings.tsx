import { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Alert,
  Paper,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,

  Divider,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Info as InfoIcon,
  Tune as TuneIcon,
  Memory as MemoryIcon,
  Security as SecurityIcon
} from '@mui/icons-material';
import { useConfigContext } from '../../context/ConfigContext';
import { UserConfig } from '../../types/UserConfig';
import { updateConfig } from '../../api';
import { useAuth } from '../../auth';
import { getToken } from '../../api';
import { ParameterOptimizationConfig } from '../../types/ParameterOptimizationConfig';
import { PerformanceParameter } from '../../types/PerformanceParameter';
import { ParameterTuningStrategy, ParameterTuningStrategyValues } from '../../types/ParameterTuningStrategy';
import { CrashPrevention } from '../../types/CrashPrevention';
import { getAllParameterDisplayInfo, createDefaultPerformanceParameter } from '../../utils/parameterUtils';

const OPTIMIZATION_STRATEGIES = [
  { value: ParameterTuningStrategyValues.BINARY_SEARCH, label: 'Binary Search', description: 'Fast, precise optimization for stable systems' },
  { value: ParameterTuningStrategyValues.CONSERVATIVE_INCREMENT, label: 'Conservative Increment', description: 'Gradual increase, safer for large models' },
  { value: ParameterTuningStrategyValues.EXPONENTIAL_BACKOFF, label: 'Exponential Backoff', description: 'Advanced strategy for complex scenarios' }
] as const;

// All parameter configurations now dynamically generated from type-safe utilities
const OPTIMIZATION_PARAMETERS = getAllParameterDisplayInfo();

const OPERATORS = [
  { value: '+', label: 'Add (+)', description: 'Add modifier to parameter value' },
  { value: '-', label: 'Subtract (-)', description: 'Subtract modifier from parameter value' },
  { value: '*', label: 'Multiply (*)', description: 'Multiply parameter value by modifier' },
  { value: '/', label: 'Divide (/)', description: 'Divide parameter value by modifier' }
] as const;

// All default parameter configurations now handled by parameterUtils.ts

// Default performance parameter creation now handled by utility function

const DEFAULT_CRASH_PREVENTION: CrashPrevention = {
  enable_preallocation_test: true,
  memory_buffer_mb: 1024,
  timeout_seconds: 120,
  enable_graceful_degradation: true
};

const DEFAULT_OPTIMIZATION_CONFIG: ParameterOptimizationConfig = {
  enabled: false,
  parameters: [
    createDefaultPerformanceParameter('n_ctx'),
    createDefaultPerformanceParameter('n_batch')
  ],
  crash_prevention: DEFAULT_CRASH_PREVENTION
};

const ParameterOptimizationSettings = () => {
  const { config, isLoading } = useConfigContext();
  const auth = useAuth();
  const [localConfig, setLocalConfig] = useState<ParameterOptimizationConfig>(DEFAULT_OPTIMIZATION_CONFIG);
  const [saveStatus, setSaveStatus] = useState<{ success?: boolean; message: string } | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  // Load current optimization config from user config
  useEffect(() => {
    if (config?.parameter_optimization) {
      const serverConfig = config.parameter_optimization as ParameterOptimizationConfig;
      setLocalConfig({
        enabled: serverConfig.enabled || false,
        parameters: serverConfig.parameters || DEFAULT_OPTIMIZATION_CONFIG.parameters,
        crash_prevention: serverConfig.crash_prevention || DEFAULT_CRASH_PREVENTION
      });
    }
  }, [config]);

  const handleEnabledChange = (enabled: boolean) => {
    setLocalConfig(prev => ({ ...prev, enabled }));
  };

  const handleParameterChange = (index: number, updatedParameter: PerformanceParameter) => {
    setLocalConfig(prev => ({
      ...prev,
      parameters: prev.parameters.map((param, i) => i === index ? updatedParameter : param)
    }));
  };

  const handleAddParameter = () => {
    const newParameter = createDefaultPerformanceParameter('n_ctx');
    setLocalConfig(prev => ({
      ...prev,
      parameters: [...prev.parameters, newParameter]
    }));
  };

  const handleRemoveParameter = (index: number) => {
    setLocalConfig(prev => ({
      ...prev,
      parameters: prev.parameters.filter((_, i) => i !== index)
    }));
  };

  const handleCrashPreventionChange = (field: keyof CrashPrevention, value: boolean | number) => {
    setLocalConfig(prev => ({
      ...prev,
      crash_prevention: {
        ...prev.crash_prevention,
        [field]: value
      }
    }));
  };

  const handleSave = async () => {
    setSaveStatus(null);
    setIsSaving(true);

    try {
      if (!config) {
        setSaveStatus({
          success: false,
          message: 'No configuration available to save.'
        });
        return;
      }

      // Update the user config with new parameter optimization settings
      const updatedConfig = {
        ...config,
        parameter_optimization: localConfig
      };

      const success = await updateConfig(getToken(auth.user), updatedConfig as UserConfig);

      if (success) {
        setSaveStatus({
          success: true,
          message: 'Parameter optimization settings saved successfully!'
        });
      } else {
        setSaveStatus({
          success: false,
          message: 'Failed to save parameter optimization settings.'
        });
      }
    } catch (error) {
      setSaveStatus({
        success: false,
        message: error instanceof Error ? error.message : 'Failed to save settings'
      });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Box>
      <Box display="flex" alignItems="center" gap={1} mb={3}>
        <TuneIcon color="primary" />
        <Typography variant="h5">Parameter Optimization</Typography>
        <Tooltip title="Automatically find optimal LLM parameters for your hardware">
          <IconButton size="small">
            <InfoIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {saveStatus && (
        <Alert
          severity={saveStatus.success ? 'success' : 'error'}
          sx={{ mb: 2 }}
          onClose={() => setSaveStatus(null)}
        >
          {saveStatus.message}
        </Alert>
      )}

      <Box display="flex" flexDirection="column" gap={3}>
        {/* Main Configuration */}
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            <MemoryIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Optimization Configuration
          </Typography>

          <FormControlLabel
            control={
              <Switch
                checked={localConfig.enabled}
                onChange={(e) => handleEnabledChange(e.target.checked)}
              />
            }
            label="Enable Parameter Optimization"
            sx={{ mb: 2 }}
          />

          {localConfig.enabled && (
            <>
              <Divider sx={{ my: 3 }} />

              {/* Performance Parameters */}
              <Typography variant="h6" gutterBottom>
                Performance Parameters
              </Typography>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Configure individual parameter optimization settings:
              </Typography>

              {localConfig.parameters.map((param, index) => (
                <Paper key={index} sx={{ p: 2, mb: 2, bgcolor: 'grey.50' }}>
                  <Box display="flex" flexDirection="column" gap={2}>
                    <Box display="flex" gap={2} alignItems="center">
                      <FormControl sx={{ minWidth: 200 }}>
                        <InputLabel>Parameter</InputLabel>
                        <Select
                          value={param.parameter_name}
                          onChange={(e) => handleParameterChange(index, {
                            ...param,
                            parameter_name: e.target.value as PerformanceParameter['parameter_name']
                          })}
                          label="Parameter"
                        >
                          {OPTIMIZATION_PARAMETERS.map(p => (
                            <MenuItem key={p.value} value={p.value}>
                              <Box>
                                <Typography>{p.label}</Typography>
                                <Typography variant="caption" color="textSecondary">
                                  {p.description}
                                </Typography>
                              </Box>
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>

                      <TextField
                        label="Priority"
                        type="number"
                        value={param.priority}
                        onChange={(e) => handleParameterChange(index, {
                          ...param,
                          priority: parseInt(e.target.value) || 1
                        })}
                        inputProps={{ min: 1, max: 10 }}
                        helperText="Lower = higher priority"
                        sx={{ width: 150 }}
                      />

                      <FormControl sx={{ minWidth: 200 }}>
                        <InputLabel>Strategy</InputLabel>
                        <Select
                          value={param.tuning_strategy}
                          onChange={(e) => handleParameterChange(index, {
                            ...param,
                            tuning_strategy: e.target.value as ParameterTuningStrategy
                          })}
                          label="Strategy"
                        >
                          {OPTIMIZATION_STRATEGIES.map(strategy => (
                            <MenuItem key={strategy.value} value={strategy.value}>
                              <Box>
                                <Typography>{strategy.label}</Typography>
                                <Typography variant="caption" color="textSecondary">
                                  {strategy.description}
                                </Typography>
                              </Box>
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>

                      <Button
                        color="error"
                        onClick={() => handleRemoveParameter(index)}
                        disabled={localConfig.parameters.length <= 1}
                      >
                        Remove
                      </Button>
                    </Box>

                    <Box display="flex" gap={2}>
                      <TextField
                        label="Max Attempts"
                        type="number"
                        value={param.max_search_attempts}
                        onChange={(e) => handleParameterChange(index, {
                          ...param,
                          max_search_attempts: parseInt(e.target.value) || 1
                        })}
                        inputProps={{ min: 1, max: 20 }}
                        sx={{ width: 150 }}
                      />

                      <TextField
                        label="Floor (Min Value)"
                        type="number"
                        value={param.floor}
                        onChange={(e) => handleParameterChange(index, {
                          ...param,
                          floor: parseFloat(e.target.value) || 0
                        })}
                        sx={{ flex: 1 }}
                      />

                      <FormControl sx={{ width: 120 }}>
                        <InputLabel>Operator</InputLabel>
                        <Select
                          value={param.operator}
                          onChange={(e) => handleParameterChange(index, {
                            ...param,
                            operator: e.target.value as PerformanceParameter['operator']
                          })}
                          label="Operator"
                        >
                          {OPERATORS.map(op => (
                            <MenuItem key={op.value} value={op.value}>
                              {op.label}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>

                      <TextField
                        label="Modifier"
                        type="number"
                        value={param.modifier}
                        onChange={(e) => handleParameterChange(index, {
                          ...param,
                          modifier: parseFloat(e.target.value) || 1
                        })}
                        sx={{ width: 150 }}
                      />

                      <TextField
                        label="Max Value"
                        type="number"
                        value={param.max_value}
                        onChange={(e) => handleParameterChange(index, {
                          ...param,
                          max_value: parseFloat(e.target.value) || 1000
                        })}
                        sx={{ flex: 1 }}
                      />
                    </Box>
                  </Box>
                </Paper>
              ))}

              <Button
                variant="outlined"
                onClick={handleAddParameter}
                sx={{ mt: 1 }}
              >
                Add Parameter
              </Button>

              {/* Crash Prevention */}
              <Accordion sx={{ mt: 3 }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="h6">
                    <SecurityIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Crash Prevention Settings
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box display="flex" flexDirection="column" gap={2}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={localConfig.crash_prevention.enable_preallocation_test}
                          onChange={(e) => handleCrashPreventionChange('enable_preallocation_test', e.target.checked)}
                        />
                      }
                      label="Enable Memory Preallocation Test"
                    />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={localConfig.crash_prevention.enable_graceful_degradation}
                          onChange={(e) => handleCrashPreventionChange('enable_graceful_degradation', e.target.checked)}
                        />
                      }
                      label="Enable Graceful Degradation"
                    />
                    <Box display="flex" gap={2}>
                      <TextField
                        label="Memory Buffer (MB)"
                        type="number"
                        value={localConfig.crash_prevention.memory_buffer_mb}
                        onChange={(e) => handleCrashPreventionChange('memory_buffer_mb', parseInt(e.target.value) || 512)}
                        helperText="Memory buffer to prevent system OOM"
                        inputProps={{ min: 512, max: 8192 }}
                        sx={{ flex: 1 }}
                      />
                      <TextField
                        label="Timeout (seconds)"
                        type="number"
                        value={localConfig.crash_prevention.timeout_seconds}
                        onChange={(e) => handleCrashPreventionChange('timeout_seconds', parseInt(e.target.value) || 30)}
                        helperText="Maximum time for initialization"
                        inputProps={{ min: 30, max: 300 }}
                        sx={{ flex: 1 }}
                      />
                    </Box>
                  </Box>
                </AccordionDetails>
              </Accordion>
            </>
          )}
        </Paper>

        {/* Information Panel */}
        <Paper sx={{ p: 3, bgcolor: 'background.default' }}>
          <Typography variant="h6" gutterBottom>
            💡 How Dynamic Parameter Optimization Works
          </Typography>
          <Typography variant="body2" paragraph>
            Each parameter can now be configured with individual optimization strategies:
          </Typography>
          <Box component="ul" sx={{ pl: 2, mt: 1 }}>
            <Typography component="li" variant="body2">
              <strong>Priority:</strong> Lower numbers get optimized first
            </Typography>
            <Typography component="li" variant="body2">
              <strong>Strategy:</strong> Choose binary search, conservative increment, or exponential backoff per parameter
            </Typography>
            <Typography component="li" variant="body2">
              <strong>Floor/Max:</strong> Set individual min/max bounds for each parameter
            </Typography>
            <Typography component="li" variant="body2">
              <strong>Operator/Modifier:</strong> Control how values are adjusted during optimization
            </Typography>
          </Box>
        </Paper>
      </Box>

      <Box display="flex" justifyContent="flex-end" mt={3}>
        <Button
          variant="contained"
          onClick={handleSave}
          disabled={isLoading || isSaving}
          size="large"
        >
          {isSaving ? 'Saving...' : 'Save Parameter Optimization Settings'}
        </Button>
      </Box>
    </Box>
  );
};

export default ParameterOptimizationSettings;