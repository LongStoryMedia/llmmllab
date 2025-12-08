import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { Box, Typography, TextField, Button, Switch, FormControlLabel, Alert, Paper, Select, MenuItem, FormControl, InputLabel, Accordion, AccordionSummary, AccordionDetails, Divider, Tooltip, IconButton } from '@mui/material';
import { ExpandMore as ExpandMoreIcon, Info as InfoIcon, Tune as TuneIcon, Memory as MemoryIcon, Security as SecurityIcon } from '@mui/icons-material';
import { useConfigContext } from '../../context/ConfigContext';
import { updateConfig } from '../../api';
import { useAuth } from '../../auth';
import { getToken } from '../../api';
import { ParameterTuningStrategyValues } from '../../types/ParameterTuningStrategy';
import { getAllParameterDisplayInfo, createDefaultPerformanceParameter } from '../../utils/parameterUtils';
const OPTIMIZATION_STRATEGIES = [
    { value: ParameterTuningStrategyValues.BINARY_SEARCH, label: 'Binary Search', description: 'Fast, precise optimization for stable systems' },
    { value: ParameterTuningStrategyValues.CONSERVATIVE_INCREMENT, label: 'Conservative Increment', description: 'Gradual increase, safer for large models' },
    { value: ParameterTuningStrategyValues.EXPONENTIAL_BACKOFF, label: 'Exponential Backoff', description: 'Advanced strategy for complex scenarios' }
];
// All parameter configurations now dynamically generated from type-safe utilities
const OPTIMIZATION_PARAMETERS = getAllParameterDisplayInfo();
const OPERATORS = [
    { value: '+', label: 'Add (+)', description: 'Add modifier to parameter value' },
    { value: '-', label: 'Subtract (-)', description: 'Subtract modifier from parameter value' },
    { value: '*', label: 'Multiply (*)', description: 'Multiply parameter value by modifier' },
    { value: '/', label: 'Divide (/)', description: 'Divide parameter value by modifier' }
];
// All default parameter configurations now handled by parameterUtils.ts
// Default performance parameter creation now handled by utility function
const DEFAULT_CRASH_PREVENTION = {
    enable_preallocation_test: true,
    memory_buffer_mb: 1024,
    timeout_seconds: 120,
    enable_graceful_degradation: true
};
const DEFAULT_OPTIMIZATION_CONFIG = {
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
    const [localConfig, setLocalConfig] = useState(DEFAULT_OPTIMIZATION_CONFIG);
    const [saveStatus, setSaveStatus] = useState(null);
    const [isSaving, setIsSaving] = useState(false);
    // Load current optimization config from user config
    useEffect(() => {
        if (config?.parameter_optimization) {
            const serverConfig = config.parameter_optimization;
            setLocalConfig({
                enabled: serverConfig.enabled || false,
                parameters: serverConfig.parameters || DEFAULT_OPTIMIZATION_CONFIG.parameters,
                crash_prevention: serverConfig.crash_prevention || DEFAULT_CRASH_PREVENTION
            });
        }
    }, [config]);
    const handleEnabledChange = (enabled) => {
        setLocalConfig(prev => ({ ...prev, enabled }));
    };
    const handleParameterChange = (index, updatedParameter) => {
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
    const handleRemoveParameter = (index) => {
        setLocalConfig(prev => ({
            ...prev,
            parameters: prev.parameters.filter((_, i) => i !== index)
        }));
    };
    const handleCrashPreventionChange = (field, value) => {
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
            const success = await updateConfig(getToken(auth.user), updatedConfig);
            if (success) {
                setSaveStatus({
                    success: true,
                    message: 'Parameter optimization settings saved successfully!'
                });
            }
            else {
                setSaveStatus({
                    success: false,
                    message: 'Failed to save parameter optimization settings.'
                });
            }
        }
        catch (error) {
            setSaveStatus({
                success: false,
                message: error instanceof Error ? error.message : 'Failed to save settings'
            });
        }
        finally {
            setIsSaving(false);
        }
    };
    return (_jsxs(Box, { children: [_jsxs(Box, { display: "flex", alignItems: "center", gap: 1, mb: 3, children: [_jsx(TuneIcon, { color: "primary" }), _jsx(Typography, { variant: "h5", children: "Parameter Optimization" }), _jsx(Tooltip, { title: "Automatically find optimal LLM parameters for your hardware", children: _jsx(IconButton, { size: "small", children: _jsx(InfoIcon, {}) }) })] }), saveStatus && (_jsx(Alert, { severity: saveStatus.success ? 'success' : 'error', sx: { mb: 2 }, onClose: () => setSaveStatus(null), children: saveStatus.message })), _jsxs(Box, { display: "flex", flexDirection: "column", gap: 3, children: [_jsxs(Paper, { sx: { p: 3 }, children: [_jsxs(Typography, { variant: "h6", gutterBottom: true, children: [_jsx(MemoryIcon, { sx: { mr: 1, verticalAlign: 'middle' } }), "Optimization Configuration"] }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.enabled, onChange: (e) => handleEnabledChange(e.target.checked) }), label: "Enable Parameter Optimization", sx: { mb: 2 } }), localConfig.enabled && (_jsxs(_Fragment, { children: [_jsx(Divider, { sx: { my: 3 } }), _jsx(Typography, { variant: "h6", gutterBottom: true, children: "Performance Parameters" }), _jsx(Typography, { variant: "body2", color: "textSecondary", gutterBottom: true, children: "Configure individual parameter optimization settings:" }), localConfig.parameters.map((param, index) => (_jsx(Paper, { sx: { p: 2, mb: 2, bgcolor: 'grey.50' }, children: _jsxs(Box, { display: "flex", flexDirection: "column", gap: 2, children: [_jsxs(Box, { display: "flex", gap: 2, alignItems: "center", children: [_jsxs(FormControl, { sx: { minWidth: 200 }, children: [_jsx(InputLabel, { children: "Parameter" }), _jsx(Select, { value: param.parameter_name, onChange: (e) => handleParameterChange(index, {
                                                                        ...param,
                                                                        parameter_name: e.target.value
                                                                    }), label: "Parameter", children: OPTIMIZATION_PARAMETERS.map(p => (_jsx(MenuItem, { value: p.value, children: _jsxs(Box, { children: [_jsx(Typography, { children: p.label }), _jsx(Typography, { variant: "caption", color: "textSecondary", children: p.description })] }) }, p.value))) })] }), _jsx(TextField, { label: "Priority", type: "number", value: param.priority, onChange: (e) => handleParameterChange(index, {
                                                                ...param,
                                                                priority: parseInt(e.target.value) || 1
                                                            }), inputProps: { min: 1, max: 10 }, helperText: "Lower = higher priority", sx: { width: 150 } }), _jsxs(FormControl, { sx: { minWidth: 200 }, children: [_jsx(InputLabel, { children: "Strategy" }), _jsx(Select, { value: param.tuning_strategy, onChange: (e) => handleParameterChange(index, {
                                                                        ...param,
                                                                        tuning_strategy: e.target.value
                                                                    }), label: "Strategy", children: OPTIMIZATION_STRATEGIES.map(strategy => (_jsx(MenuItem, { value: strategy.value, children: _jsxs(Box, { children: [_jsx(Typography, { children: strategy.label }), _jsx(Typography, { variant: "caption", color: "textSecondary", children: strategy.description })] }) }, strategy.value))) })] }), _jsx(Button, { color: "error", onClick: () => handleRemoveParameter(index), disabled: localConfig.parameters.length <= 1, children: "Remove" })] }), _jsxs(Box, { display: "flex", gap: 2, children: [_jsx(TextField, { label: "Max Attempts", type: "number", value: param.max_search_attempts, onChange: (e) => handleParameterChange(index, {
                                                                ...param,
                                                                max_search_attempts: parseInt(e.target.value) || 1
                                                            }), inputProps: { min: 1, max: 20 }, sx: { width: 150 } }), _jsx(TextField, { label: "Floor (Min Value)", type: "number", value: param.floor, onChange: (e) => handleParameterChange(index, {
                                                                ...param,
                                                                floor: parseFloat(e.target.value) || 0
                                                            }), sx: { flex: 1 } }), _jsxs(FormControl, { sx: { width: 120 }, children: [_jsx(InputLabel, { children: "Operator" }), _jsx(Select, { value: param.operator, onChange: (e) => handleParameterChange(index, {
                                                                        ...param,
                                                                        operator: e.target.value
                                                                    }), label: "Operator", children: OPERATORS.map(op => (_jsx(MenuItem, { value: op.value, children: op.label }, op.value))) })] }), _jsx(TextField, { label: "Modifier", type: "number", value: param.modifier, onChange: (e) => handleParameterChange(index, {
                                                                ...param,
                                                                modifier: parseFloat(e.target.value) || 1
                                                            }), sx: { width: 150 } }), _jsx(TextField, { label: "Max Value", type: "number", value: param.max_value, onChange: (e) => handleParameterChange(index, {
                                                                ...param,
                                                                max_value: parseFloat(e.target.value) || 1000
                                                            }), sx: { flex: 1 } })] })] }) }, index))), _jsx(Button, { variant: "outlined", onClick: handleAddParameter, sx: { mt: 1 }, children: "Add Parameter" }), _jsxs(Accordion, { sx: { mt: 3 }, children: [_jsx(AccordionSummary, { expandIcon: _jsx(ExpandMoreIcon, {}), children: _jsxs(Typography, { variant: "h6", children: [_jsx(SecurityIcon, { sx: { mr: 1, verticalAlign: 'middle' } }), "Crash Prevention Settings"] }) }), _jsx(AccordionDetails, { children: _jsxs(Box, { display: "flex", flexDirection: "column", gap: 2, children: [_jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.crash_prevention.enable_preallocation_test, onChange: (e) => handleCrashPreventionChange('enable_preallocation_test', e.target.checked) }), label: "Enable Memory Preallocation Test" }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.crash_prevention.enable_graceful_degradation, onChange: (e) => handleCrashPreventionChange('enable_graceful_degradation', e.target.checked) }), label: "Enable Graceful Degradation" }), _jsxs(Box, { display: "flex", gap: 2, children: [_jsx(TextField, { label: "Memory Buffer (MB)", type: "number", value: localConfig.crash_prevention.memory_buffer_mb, onChange: (e) => handleCrashPreventionChange('memory_buffer_mb', parseInt(e.target.value) || 512), helperText: "Memory buffer to prevent system OOM", inputProps: { min: 512, max: 8192 }, sx: { flex: 1 } }), _jsx(TextField, { label: "Timeout (seconds)", type: "number", value: localConfig.crash_prevention.timeout_seconds, onChange: (e) => handleCrashPreventionChange('timeout_seconds', parseInt(e.target.value) || 30), helperText: "Maximum time for initialization", inputProps: { min: 30, max: 300 }, sx: { flex: 1 } })] })] }) })] })] }))] }), _jsxs(Paper, { sx: { p: 3, bgcolor: 'background.default' }, children: [_jsx(Typography, { variant: "h6", gutterBottom: true, children: "\uD83D\uDCA1 How Dynamic Parameter Optimization Works" }), _jsx(Typography, { variant: "body2", paragraph: true, children: "Each parameter can now be configured with individual optimization strategies:" }), _jsxs(Box, { component: "ul", sx: { pl: 2, mt: 1 }, children: [_jsxs(Typography, { component: "li", variant: "body2", children: [_jsx("strong", { children: "Priority:" }), " Lower numbers get optimized first"] }), _jsxs(Typography, { component: "li", variant: "body2", children: [_jsx("strong", { children: "Strategy:" }), " Choose binary search, conservative increment, or exponential backoff per parameter"] }), _jsxs(Typography, { component: "li", variant: "body2", children: [_jsx("strong", { children: "Floor/Max:" }), " Set individual min/max bounds for each parameter"] }), _jsxs(Typography, { component: "li", variant: "body2", children: [_jsx("strong", { children: "Operator/Modifier:" }), " Control how values are adjusted during optimization"] })] })] })] }), _jsx(Box, { display: "flex", justifyContent: "flex-end", mt: 3, children: _jsx(Button, { variant: "contained", onClick: handleSave, disabled: isLoading || isSaving, size: "large", children: isSaving ? 'Saving...' : 'Save Parameter Optimization Settings' }) })] }));
};
export default ParameterOptimizationSettings;
