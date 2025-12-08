import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState, useEffect, useCallback } from 'react';
import { Box, Typography, TextField, Button, Switch, FormControlLabel, Alert, Paper, Grid, Select, MenuItem, FormControl, InputLabel, IconButton, Slider, Accordion, AccordionSummary, AccordionDetails, CircularProgress, Chip, Stack } from '@mui/material';
import { Add as AddIcon, Delete as DeleteIcon, Refresh as RefreshIcon, ExpandMore as ExpandMoreIcon, Memory as MemoryIcon } from '@mui/icons-material';
import { useConfigContext } from '../../context/ConfigContext';
import { useAuth } from '../../auth';
import { getDeviceMappings } from '../../api/resources';
const GpuSettings = () => {
    const { config, updatePartialConfig, isLoading } = useConfigContext();
    const { user } = useAuth();
    const [localConfig, setLocalConfig] = useState({
        no_kv_offload: false,
        main_gpu: -1,
        tensor_split: [],
        tensor_split_devices: [],
        split_mode: 'none',
        offload_kqv: true
    });
    const [devices, setDevices] = useState({});
    const [isLoadingDevices, setIsLoadingDevices] = useState(false);
    const [saveStatus, setSaveStatus] = useState(null);
    const [deviceError, setDeviceError] = useState(null);
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
            const response = await getDeviceMappings(user.access_token);
            setDevices(response.devices);
        }
        catch (error) {
            console.error('Failed to load devices:', error);
            setDeviceError(`Failed to load devices: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setIsLoadingDevices(false);
        }
    }, [user?.access_token]);
    useEffect(() => {
        loadDevices();
    }, [loadDevices]);
    const handleChange = (field, value) => {
        setLocalConfig(prev => ({ ...prev, [field]: value }));
    };
    const handleTensorSplitChange = (index, value) => {
        const newTensorSplit = [...(localConfig.tensor_split || [])];
        newTensorSplit[index] = value;
        handleChange('tensor_split', newTensorSplit);
    };
    const handleTensorSplitDeviceChange = (index, deviceId) => {
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
    const removeTensorSplitDevice = (index) => {
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
            }
            else {
                setSaveStatus({
                    success: false,
                    message: 'Failed to save GPU settings.'
                });
            }
        }
        catch (err) {
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
    const getAvailableDevicesForTensorSplit = (currentIndex) => {
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
    const getDeviceName = (deviceId) => {
        return devices[deviceId]?.name || deviceId;
    };
    if (isLoading) {
        return (_jsxs(Box, { sx: { display: 'flex', justifyContent: 'center', alignItems: 'center', p: 4 }, children: [_jsx(CircularProgress, {}), _jsx(Typography, { sx: { ml: 2 }, children: "Loading GPU settings..." })] }));
    }
    return (_jsxs(Box, { sx: { padding: 2 }, children: [_jsxs(Typography, { variant: "h6", gutterBottom: true, sx: { display: 'flex', alignItems: 'center' }, children: [_jsx(MemoryIcon, { sx: { mr: 1 } }), "GPU Configuration"] }), _jsx(Typography, { variant: "body2", color: "text.secondary", sx: { mb: 3 }, children: "Configure GPU memory management, device allocation, and performance optimization settings." }), saveStatus && (_jsx(Alert, { severity: saveStatus.success ? "success" : "error", sx: { mb: 2 }, onClose: () => setSaveStatus(null), children: saveStatus.message })), _jsxs(Accordion, { sx: { mb: 2 }, children: [_jsx(AccordionSummary, { expandIcon: _jsx(ExpandMoreIcon, {}), children: _jsxs(Typography, { variant: "h6", children: ["Available Devices", _jsx(IconButton, { onClick: (e) => {
                                        e.stopPropagation();
                                        loadDevices();
                                    }, size: "small", sx: { ml: 1 }, disabled: isLoadingDevices, children: isLoadingDevices ? _jsx(CircularProgress, { size: 16 }) : _jsx(RefreshIcon, {}) })] }) }), _jsx(AccordionDetails, { children: deviceError ? (_jsxs(Alert, { severity: "error", sx: { mb: 2 }, children: [deviceError, _jsx(Button, { onClick: loadDevices, sx: { ml: 2 }, size: "small", children: "Retry" })] })) : (_jsxs(Grid, { container: true, spacing: 2, children: [Object.entries(devices).map(([key, device]) => (_jsx(Grid, { size: { xs: 12, sm: 6, md: 4 }, children: _jsxs(Paper, { sx: { p: 2 }, children: [_jsx(Typography, { variant: "subtitle2", children: device.name }), _jsxs(Typography, { variant: "body2", color: "text.secondary", children: ["Device ID: ", key] }), _jsx(Typography, { variant: "body2", color: "text.secondary", children: "Memory: Available" })] }) }, key))), Object.keys(devices).length === 0 && !isLoadingDevices && (_jsx(Grid, { size: 12, children: _jsx(Typography, { variant: "body2", color: "text.secondary", children: "No devices found. Click refresh to discover available GPUs and devices." }) }))] })) })] }), _jsxs(Grid, { container: true, spacing: 3, children: [_jsx(Grid, { size: 12, children: _jsxs(Paper, { sx: { p: 3 }, children: [_jsx(Typography, { variant: "h6", gutterBottom: true, children: "Memory Management" }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.no_kv_offload || false, onChange: (e) => handleChange('no_kv_offload', e.target.checked) }), label: "Force KV Cache to CPU (saves VRAM)", sx: { mb: 2, display: 'block' } }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.offload_kqv || false, onChange: (e) => handleChange('offload_kqv', e.target.checked) }), label: "Offload Key/Query/Value tensors to GPU", sx: { mb: 2, display: 'block' } })] }) }), _jsx(Grid, { size: 12, children: _jsxs(Paper, { sx: { p: 3 }, children: [_jsx(Typography, { variant: "h6", gutterBottom: true, children: "Device Selection" }), _jsxs(FormControl, { fullWidth: true, margin: "normal", children: [_jsx(InputLabel, { children: "Main GPU Device" }), _jsxs(Select, { value: localConfig.main_gpu_device_id || '', onChange: (e) => handleChange('main_gpu_device_id', e.target.value), label: "Main GPU Device", children: [_jsx(MenuItem, { value: "", children: _jsx("em", { children: "Auto-select" }) }), getDeviceOptions().map(device => (_jsx(MenuItem, { value: device.value, children: device.label }, device.value)))] })] }), _jsx(TextField, { label: "Main GPU Index", type: "number", value: localConfig.main_gpu ?? -1, onChange: (e) => handleChange('main_gpu', parseInt(e.target.value) || -1), fullWidth: true, margin: "normal", helperText: "GPU device index (-1 for auto-selection, overridden by device ID above)", inputProps: { min: -1 } }), _jsxs(FormControl, { fullWidth: true, margin: "normal", children: [_jsx(InputLabel, { children: "Model Split Mode" }), _jsxs(Select, { value: localConfig.split_mode || 'none', onChange: (e) => handleChange('split_mode', e.target.value), label: "Model Split Mode", children: [_jsx(MenuItem, { value: "none", children: "None - Single device" }), _jsx(MenuItem, { value: "layer", children: "Layer - Split by layers" }), _jsx(MenuItem, { value: "row", children: "Row - Split by tensor rows" })] })] })] }) }), _jsx(Grid, { size: 12, children: _jsxs(Paper, { sx: { p: 3 }, children: [_jsxs(Typography, { variant: "h6", gutterBottom: true, sx: { display: 'flex', alignItems: 'center' }, children: ["Tensor Split Configuration", _jsx(Button, { onClick: addTensorSplitDevice, startIcon: _jsx(AddIcon, {}), size: "small", sx: { ml: 2 }, disabled: Object.keys(devices).length === 0 || (localConfig.tensor_split_devices?.length || 0) >= Object.keys(devices).length, children: "Add Device" })] }), _jsx(Typography, { variant: "body2", color: "text.secondary", sx: { mb: 2 }, children: "Distribute model computation across multiple GPUs. Values must sum to 1.0." }), localConfig.tensor_split && localConfig.tensor_split.length > 0 && (_jsxs(_Fragment, { children: [_jsx(Box, { sx: { mb: 2 }, children: _jsxs(Typography, { variant: "body2", color: isTensorSplitValid() ? 'success.main' : 'error.main', children: ["Current sum: ", getTensorSplitSum().toFixed(3), " ", isTensorSplitValid() ? '✓' : '(must equal 1.0)'] }) }), localConfig.tensor_split.map((split, index) => {
                                            const currentDeviceId = localConfig.tensor_split_devices?.[index];
                                            const availableDevices = getAvailableDevicesForTensorSplit(index);
                                            return (_jsxs(Box, { sx: { mb: 3, p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }, children: [_jsxs(Stack, { direction: "row", spacing: 2, alignItems: "center", sx: { mb: 2 }, children: [_jsxs(Typography, { variant: "subtitle2", sx: { minWidth: 80 }, children: ["Device ", index + 1, ":"] }), _jsxs(FormControl, { sx: { minWidth: 200 }, children: [_jsx(InputLabel, { size: "small", children: "Select Device" }), _jsxs(Select, { value: currentDeviceId || '', onChange: (e) => handleTensorSplitDeviceChange(index, e.target.value), label: "Select Device", size: "small", children: [availableDevices.map(([deviceId, device]) => (_jsxs(MenuItem, { value: deviceId, children: [device.name, " (", deviceId, ")"] }, deviceId))), currentDeviceId && !availableDevices.find(([id]) => id === currentDeviceId) && (_jsxs(MenuItem, { value: currentDeviceId, children: [getDeviceName(currentDeviceId), " (", currentDeviceId, ")"] }))] })] }), _jsx(IconButton, { onClick: () => removeTensorSplitDevice(index), size: "small", color: "error", children: _jsx(DeleteIcon, {}) })] }), _jsxs(Box, { sx: { display: 'flex', alignItems: 'center' }, children: [_jsx(Typography, { sx: { minWidth: 80 }, children: "Split ratio:" }), _jsx(Slider, { value: split, onChange: (_, value) => handleTensorSplitChange(index, value), min: 0, max: 1, step: 0.01, sx: { mx: 2, flex: 1 }, valueLabelDisplay: "auto", valueLabelFormat: (value) => value.toFixed(2) }), _jsx(Typography, { sx: { minWidth: 60, textAlign: 'center' }, children: split.toFixed(2) })] }), currentDeviceId && (_jsx(Chip, { label: `${getDeviceName(currentDeviceId)}: ${(split * 100).toFixed(1)}%`, size: "small", sx: { mt: 1 } }))] }, index));
                                        })] })), (!localConfig.tensor_split || localConfig.tensor_split.length === 0) && (_jsx(Typography, { variant: "body2", color: "text.secondary", children: "No tensor split configured. Device allocation will be determined automatically based on available resources." }))] }) })] }), _jsxs(Box, { sx: { mt: 3, display: 'flex', gap: 2 }, children: [_jsx(Button, { variant: "contained", color: "primary", onClick: handleSave, disabled: isLoading, children: "Save GPU Settings" }), _jsx(Button, { variant: "outlined", onClick: () => setLocalConfig(config?.gpu_config || {
                            no_kv_offload: false,
                            main_gpu: -1,
                            tensor_split: [],
                            tensor_split_devices: [],
                            split_mode: 'none',
                            offload_kqv: true
                        }), disabled: isLoading, children: "Reset Changes" })] })] }));
};
export default GpuSettings;
