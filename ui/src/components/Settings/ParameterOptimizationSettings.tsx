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
    Chip,
    Stack,
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

const OPTIMIZATION_STRATEGIES = [
    { value: 'binary_search', label: 'Binary Search', description: 'Fast, precise optimization for stable systems' },
    { value: 'conservative_increment', label: 'Conservative Increment', description: 'Gradual increase, safer for large models' },
    { value: 'exponential_backoff', label: 'Exponential Backoff', description: 'Advanced strategy for complex scenarios' }
] as const;

const OPTIMIZATION_PARAMETERS = [
    { value: 'n_ctx', label: 'Context Size (n_ctx)', description: 'Memory window for model attention' },
    { value: 'n_batch', label: 'Batch Size (n_batch)', description: 'Number of tokens processed together' },
    { value: 'n_ubatch', label: 'Micro-batch (n_ubatch)', description: 'Internal batching for efficiency' },
    { value: 'n_gpu_layers', label: 'GPU Layers', description: 'Layers to offload to GPU vs CPU' }
] as const;

type OptimizationParam = 'n_ctx' | 'n_batch' | 'n_ubatch' | 'n_gpu_layers';
type SearchStrategy = 'binary_search' | 'conservative_increment' | 'exponential_backoff';

interface LocalParameterOptimizationConfiguration {
    enabled: boolean;
    optimization_priority: OptimizationParam[];
    parameter_floors: {
        n_ctx?: number;
        n_batch?: number;
        n_ubatch?: number;
        n_gpu_layers?: number;
    };
    search_strategy: SearchStrategy;
    max_search_attempts: number;
    crash_prevention: {
        enable_preallocation_test?: boolean;
        memory_buffer_mb?: number;
        timeout_seconds?: number;
        enable_graceful_degradation?: boolean;
    };
}

const DEFAULT_OPTIMIZATION_CONFIG: LocalParameterOptimizationConfiguration = {
    enabled: false,
    optimization_priority: ['n_ctx', 'n_batch'],
    parameter_floors: {
        n_ctx: 8192,
        n_batch: 64,
        n_ubatch: 64,
        n_gpu_layers: 0
    },
    search_strategy: 'conservative_increment',
    max_search_attempts: 6,
    crash_prevention: {
        enable_preallocation_test: true,
        memory_buffer_mb: 4096,
        timeout_seconds: 120,
        enable_graceful_degradation: true
    }
};

const ParameterOptimizationSettings = () => {
    const { config, isLoading } = useConfigContext();
    const auth = useAuth();
    const [localConfig, setLocalConfig] = useState<LocalParameterOptimizationConfiguration>(DEFAULT_OPTIMIZATION_CONFIG);
    const [saveStatus, setSaveStatus] = useState<{ success?: boolean; message: string } | null>(null);
    const [isSaving, setIsSaving] = useState(false);

    // Load current optimization config from user config
    useEffect(() => {
        if (config?.parameter_optimization) {
            // Convert from server format to local format (handle any type differences)
            const serverConfig = config.parameter_optimization as {
                enabled: boolean;
                optimization_priority: OptimizationParam[] | string[];
                parameter_floors: LocalParameterOptimizationConfiguration['parameter_floors'];
                search_strategy: SearchStrategy;
                max_search_attempts: number;
                crash_prevention: LocalParameterOptimizationConfiguration['crash_prevention'];
            };
            setLocalConfig({
                enabled: serverConfig.enabled || false,
                optimization_priority: Array.isArray(serverConfig.optimization_priority)
                    ? serverConfig.optimization_priority as OptimizationParam[]
                    : ['n_ctx', 'n_batch'],
                parameter_floors: serverConfig.parameter_floors || {
                    n_ctx: 8192,
                    n_batch: 64,
                    n_ubatch: 64,
                    n_gpu_layers: 0
                },
                search_strategy: serverConfig.search_strategy || 'conservative_increment',
                max_search_attempts: serverConfig.max_search_attempts || 6,
                crash_prevention: serverConfig.crash_prevention || {
                    enable_preallocation_test: true,
                    memory_buffer_mb: 4096,
                    timeout_seconds: 120,
                    enable_graceful_degradation: true
                }
            });
        }
    }, [config]);

    const handleChange = <K extends keyof LocalParameterOptimizationConfiguration>(
        field: K,
        value: LocalParameterOptimizationConfiguration[K]
    ) => {
        setLocalConfig(prev => ({ ...prev, [field]: value }));
    };

    const handleFloorChange = (param: string, value: number) => {
        setLocalConfig(prev => ({
            ...prev,
            parameter_floors: {
                ...prev.parameter_floors,
                [param]: value
            }
        }));
    };

    const handleCrashPreventionChange = (field: string, value: boolean | number) => {
        setLocalConfig(prev => ({
            ...prev,
            crash_prevention: {
                ...prev.crash_prevention,
                [field]: value
            }
        }));
    };

    const handlePriorityChange = (priorities: OptimizationParam[]) => {
        setLocalConfig(prev => ({
            ...prev,
            optimization_priority: priorities
        }));
    };

    const togglePriority = (param: string) => {
        const current = localConfig.optimization_priority || [];
        const updated = current.includes(param as OptimizationParam)
            ? current.filter(p => p !== param)
            : [...current, param as OptimizationParam];
        handlePriorityChange(updated);
    };

    const convertLocalToServerConfig = (local: LocalParameterOptimizationConfiguration) => {
        // Convert our properly typed local config to the server's expected format
        return {
            enabled: local.enabled,
            optimization_priority: local.optimization_priority as string[], // Type assertion due to generated type issue
            parameter_floors: local.parameter_floors,
            search_strategy: local.search_strategy,
            max_search_attempts: local.max_search_attempts,
            crash_prevention: local.crash_prevention
        };
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
                parameter_optimization: convertLocalToServerConfig(localConfig)
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
                                onChange={(e) => handleChange('enabled', e.target.checked)}
                            />
                        }
                        label="Enable Parameter Optimization"
                        sx={{ mb: 2 }}
                    />

                    {localConfig.enabled && (
                        <>
                            <Box display="flex" gap={2} mb={3}>
                                <FormControl sx={{ minWidth: 250 }}>
                                    <InputLabel>Optimization Strategy</InputLabel>
                                    <Select
                                        value={localConfig.search_strategy}
                                        onChange={(e) => handleChange('search_strategy', e.target.value as SearchStrategy)}
                                        label="Optimization Strategy"
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

                                <TextField
                                    label="Max Search Attempts"
                                    type="number"
                                    value={localConfig.max_search_attempts}
                                    onChange={(e) => handleChange('max_search_attempts', parseInt(e.target.value))}
                                    inputProps={{ min: 1, max: 20 }}
                                    helperText="Number of optimization attempts per parameter"
                                    sx={{ minWidth: 200 }}
                                />
                            </Box>

                            <Divider sx={{ my: 3 }} />

                            {/* Optimization Priority */}
                            <Typography variant="h6" gutterBottom>
                                Parameter Priority
                            </Typography>
                            <Typography variant="body2" color="textSecondary" gutterBottom>
                                Select which parameters to optimize and their priority order:
                            </Typography>

                            <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mb: 3 }}>
                                {OPTIMIZATION_PARAMETERS.map(param => (
                                    <Chip
                                        key={param.value}
                                        label={param.label}
                                        onClick={() => togglePriority(param.value)}
                                        color={localConfig.optimization_priority?.includes(param.value as OptimizationParam) ? 'primary' : 'default'}
                                        variant={localConfig.optimization_priority?.includes(param.value as OptimizationParam) ? 'filled' : 'outlined'}
                                    />
                                ))}
                            </Stack>

                            {/* Parameter Floors */}
                            <Accordion>
                                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                    <Typography variant="h6">Minimum Parameter Values (Floors)</Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Box display="flex" flexDirection="column" gap={2}>
                                        <Box display="flex" gap={2}>
                                            <TextField
                                                label="Min Context Size"
                                                type="number"
                                                value={localConfig.parameter_floors?.n_ctx || 0}
                                                onChange={(e) => handleFloorChange('n_ctx', parseInt(e.target.value))}
                                                helperText="Minimum context window size (tokens)"
                                                sx={{ flex: 1 }}
                                            />
                                            <TextField
                                                label="Min Batch Size"
                                                type="number"
                                                value={localConfig.parameter_floors?.n_batch || 0}
                                                onChange={(e) => handleFloorChange('n_batch', parseInt(e.target.value))}
                                                helperText="Minimum batch size for processing"
                                                sx={{ flex: 1 }}
                                            />
                                        </Box>
                                        <Box display="flex" gap={2}>
                                            <TextField
                                                label="Min Micro-batch Size"
                                                type="number"
                                                value={localConfig.parameter_floors?.n_ubatch || 0}
                                                onChange={(e) => handleFloorChange('n_ubatch', parseInt(e.target.value))}
                                                helperText="Minimum micro-batch size"
                                                sx={{ flex: 1 }}
                                            />
                                            <TextField
                                                label="Min GPU Layers"
                                                type="number"
                                                value={localConfig.parameter_floors?.n_gpu_layers || 0}
                                                onChange={(e) => handleFloorChange('n_gpu_layers', parseInt(e.target.value))}
                                                helperText="Minimum layers on GPU (0 = CPU only)"
                                                sx={{ flex: 1 }}
                                            />
                                        </Box>
                                    </Box>
                                </AccordionDetails>
                            </Accordion>

                            {/* Crash Prevention */}
                            <Accordion sx={{ mt: 2 }}>
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
                                                    checked={localConfig.crash_prevention?.enable_preallocation_test || false}
                                                    onChange={(e) => handleCrashPreventionChange('enable_preallocation_test', e.target.checked)}
                                                />
                                            }
                                            label="Enable Memory Preallocation Test"
                                        />
                                        <FormControlLabel
                                            control={
                                                <Switch
                                                    checked={localConfig.crash_prevention?.enable_graceful_degradation || false}
                                                    onChange={(e) => handleCrashPreventionChange('enable_graceful_degradation', e.target.checked)}
                                                />
                                            }
                                            label="Enable Graceful Degradation"
                                        />
                                        <Box display="flex" gap={2}>
                                            <TextField
                                                label="Memory Buffer (MB)"
                                                type="number"
                                                value={localConfig.crash_prevention?.memory_buffer_mb || 0}
                                                onChange={(e) => handleCrashPreventionChange('memory_buffer_mb', parseInt(e.target.value))}
                                                helperText="Memory buffer to prevent system OOM"
                                                sx={{ flex: 1 }}
                                            />
                                            <TextField
                                                label="Timeout (seconds)"
                                                type="number"
                                                value={localConfig.crash_prevention?.timeout_seconds || 0}
                                                onChange={(e) => handleCrashPreventionChange('timeout_seconds', parseInt(e.target.value))}
                                                helperText="Maximum time for initialization"
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
                        💡 How Parameter Optimization Works
                    </Typography>
                    <Typography variant="body2" paragraph>
                        Parameter optimization automatically finds the best memory configuration for your hardware:
                    </Typography>
                    <Box component="ul" sx={{ pl: 2, mt: 1 }}>
                        <Typography component="li" variant="body2">
                            <strong>Binary Search:</strong> Fast and precise, ideal for stable systems
                        </Typography>
                        <Typography component="li" variant="body2">
                            <strong>Conservative Increment:</strong> Gradual optimization, safer for large models
                        </Typography>
                        <Typography component="li" variant="body2">
                            <strong>Crash Prevention:</strong> Tests memory allocation before full initialization
                        </Typography>
                        <Typography component="li" variant="body2">
                            <strong>Per-Model Configuration:</strong> Different strategies for different model sizes
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