import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { Box, Typography, Button, Switch, FormControlLabel, Alert, TextField, Divider } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';
import ImageModelSelector from '../ModelSelector/ImageModelSelector';
const ImageGenerationSettings = () => {
    const { config, updatePartialConfig, isLoading } = useConfigContext();
    const [localConfig, setLocalConfig] = useState({
        enabled: false,
        storage_directory: '',
        max_image_size: 1024,
        retention_hours: 720,
        auto_prompt_refinement: true,
        width: 1024,
        height: 1024,
        inference_steps: 20,
        guidance_scale: 7.5,
        low_memory_mode: false,
        negative_prompt: ''
    });
    const [saveStatus, setSaveStatus] = useState(null);
    useEffect(() => {
        // When user config loads, update local state
        if (config?.image_generation) {
            setLocalConfig({
                enabled: config.image_generation.enabled ?? false,
                storage_directory: config.image_generation.storage_directory ?? '',
                max_image_size: config.image_generation.max_image_size ?? 1024,
                retention_hours: config.image_generation.retention_hours ?? 720,
                auto_prompt_refinement: config.image_generation.auto_prompt_refinement ?? true,
                width: config.image_generation.width ?? 1024,
                height: config.image_generation.height ?? 1024,
                inference_steps: config.image_generation.inference_steps ?? 20,
                guidance_scale: config.image_generation.guidance_scale ?? 7.5,
                low_memory_mode: config.image_generation.low_memory_mode ?? false,
                negative_prompt: config.image_generation.negative_prompt ?? ''
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
    const handleMaxSizeChange = (e) => {
        const value = parseInt(e.target.value);
        if (!isNaN(value)) {
            setLocalConfig({
                ...localConfig,
                max_image_size: Math.min(Math.max(value, 128), 4096) // Enforce min/max values
            });
        }
    };
    const handleRetentionHoursChange = (e) => {
        const value = parseInt(e.target.value);
        if (!isNaN(value)) {
            setLocalConfig({
                ...localConfig,
                retention_hours: Math.min(Math.max(value, 1), 720) // Enforce min/max values
            });
        }
    };
    const handleStorageDirectoryChange = (e) => {
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
            }
            else {
                setSaveStatus({
                    success: false,
                    message: 'Failed to save image generation settings.'
                });
            }
        }
        catch (err) {
            console.error('Error saving image generation settings:', err);
            setSaveStatus({
                success: false,
                message: 'An error occurred while saving settings.'
            });
        }
    };
    if (isLoading) {
        return _jsx(Box, { sx: { padding: 2 }, children: _jsx(Typography, { children: "Loading image generation settings..." }) });
    }
    return (_jsxs(Box, { sx: { padding: 2 }, children: [_jsx(Typography, { variant: "h6", gutterBottom: true, children: "Image Generation Settings" }), saveStatus && (_jsx(Alert, { severity: saveStatus.success ? 'success' : 'error', sx: { mb: 2 }, onClose: () => setSaveStatus(null), children: saveStatus.message })), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.enabled, onChange: handleToggleEnabled }), label: "Enable Image Generation", sx: { mb: 2, display: 'block' } }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.auto_prompt_refinement, onChange: handleToggleAutoPromptRefinement, disabled: !localConfig.enabled }), label: "Enable Automatic Prompt Refinement", sx: { mb: 2, display: 'block' } }), _jsx(TextField, { label: "Storage Directory", value: localConfig.storage_directory, onChange: handleStorageDirectoryChange, fullWidth: true, margin: "normal", disabled: !localConfig.enabled, helperText: "Directory where generated images will be stored" }), _jsx(TextField, { label: "Maximum Image Size", type: "number", value: localConfig.max_image_size, onChange: handleMaxSizeChange, fullWidth: true, margin: "normal", disabled: !localConfig.enabled, inputProps: { min: 128, max: 4096 }, helperText: "Maximum size in pixels (128-4096)" }), _jsx(TextField, { label: "Retention Hours", type: "number", value: localConfig.retention_hours, onChange: handleRetentionHoursChange, fullWidth: true, margin: "normal", disabled: !localConfig.enabled, inputProps: { min: 1, max: 720 }, helperText: "How long to keep generated images (1-720 hours)" }), _jsx(Button, { variant: "contained", color: "primary", sx: { mt: 2 }, onClick: handleSave, disabled: isLoading, children: "Save Image Generation Settings" }), localConfig.enabled && (_jsxs(_Fragment, { children: [_jsx(Divider, { sx: { my: 4 } }), _jsx(Typography, { variant: "h6", gutterBottom: true, children: "Active Model Selection" }), _jsx(Typography, { variant: "body2", color: "text.secondary", sx: { mb: 2 }, children: "Select which model to use for image generation. Only models with TextToImage specialization are shown." }), _jsx(ImageModelSelector, { mode: "TextToImage" })] }))] }));
};
export default ImageGenerationSettings;
