import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { Box, TextField, Typography, Button, Switch, FormControlLabel, Alert, Slider } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';
const SummarizationSettings = () => {
    const { config, updatePartialConfig, isLoading } = useConfigContext();
    const [localConfig, setLocalConfig] = useState({
        enabled: true,
        messages_before_summary: 10,
        summaries_before_consolidation: 5,
        embedding_dimension: 768,
        max_summary_levels: 3,
        summary_weight_coefficient: 0.7
    });
    const [saveStatus, setSaveStatus] = useState(null);
    // Update local state when the user config loads or changes
    useEffect(() => {
        if (config?.summarization) {
            setLocalConfig({
                enabled: config.summarization.enabled !== false,
                messages_before_summary: config.summarization.messages_before_summary ?? 10,
                summaries_before_consolidation: config.summarization.summaries_before_consolidation ?? 5,
                embedding_dimension: config.summarization.embedding_dimension ?? 768,
                max_summary_levels: config.summarization.max_summary_levels ?? 3,
                summary_weight_coefficient: config.summarization.summary_weight_coefficient ?? 0.7
            });
        }
    }, [config]);
    const handleToggleEnabled = () => {
        setLocalConfig({
            ...localConfig,
            enabled: !localConfig.enabled
        });
    };
    const handleWeightChange = (_event, newValue) => {
        setLocalConfig({
            ...localConfig,
            summary_weight_coefficient: newValue
        });
    };
    const handleSave = async () => {
        setSaveStatus(null);
        try {
            // Convert camelCase to snake_case when passing to updatePartialConfig
            const snakeCaseConfig = {
                enabled: localConfig.enabled,
                messages_before_summary: localConfig.messages_before_summary,
                summaries_before_consolidation: localConfig.summaries_before_consolidation,
                embedding_dimension: localConfig.embedding_dimension,
                max_summary_levels: localConfig.max_summary_levels,
                summary_weight_coefficient: localConfig.summary_weight_coefficient
            };
            const success = await updatePartialConfig('summarization', snakeCaseConfig);
            if (success) {
                setSaveStatus({
                    success: true,
                    message: 'Summarization settings saved successfully!'
                });
            }
            else {
                setSaveStatus({
                    success: false,
                    message: 'Failed to save settings.'
                });
            }
        }
        catch (err) {
            setSaveStatus({
                success: false,
                message: `Error: ${err instanceof Error ? err.message : String(err)}`
            });
        }
    };
    if (isLoading) {
        return _jsx(Box, { sx: { padding: 2 }, children: _jsx(Typography, { children: "Loading summarization settings..." }) });
    }
    return (_jsxs(Box, { sx: { padding: 2 }, children: [_jsx(Typography, { variant: "h6", gutterBottom: true, children: "Conversation Summarization Settings" }), saveStatus && (_jsx(Alert, { severity: saveStatus.success ? "success" : "error", sx: { mb: 2 }, onClose: () => setSaveStatus(null), children: saveStatus.message })), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.enabled, onChange: handleToggleEnabled }), label: "Enable Conversation Summarization", sx: { mb: 2, display: 'block' } }), localConfig.enabled && (_jsxs(_Fragment, { children: [_jsx(TextField, { label: "Messages Before Summary", type: "number", value: localConfig.messages_before_summary, onChange: (e) => setLocalConfig({ ...localConfig, messages_before_summary: parseInt(e.target.value) || 10 }), fullWidth: true, margin: "normal", helperText: "Number of messages before generating a summary" }), _jsx(TextField, { label: "Summaries Before Consolidation", type: "number", value: localConfig.summaries_before_consolidation, onChange: (e) => setLocalConfig({ ...localConfig, summaries_before_consolidation: parseInt(e.target.value) || 5 }), fullWidth: true, margin: "normal", helperText: "Number of summaries before consolidating them" }), _jsx(TextField, { label: "Embedding Dimension", type: "number", value: localConfig.embedding_dimension, onChange: (e) => setLocalConfig({ ...localConfig, embedding_dimension: parseInt(e.target.value) || 768 }), fullWidth: true, margin: "normal", helperText: "Dimension of the embedding vectors" }), _jsx(TextField, { label: "Max Summary Levels", type: "number", value: localConfig.max_summary_levels, onChange: (e) => setLocalConfig({ ...localConfig, max_summary_levels: parseInt(e.target.value) || 3 }), fullWidth: true, margin: "normal", helperText: "Maximum depth of summary hierarchy" }), _jsxs(Box, { sx: { mt: 3, mb: 2 }, children: [_jsxs(Typography, { id: "weight-coefficient-slider", gutterBottom: true, children: ["Summary Weight Coefficient: ", localConfig.summary_weight_coefficient.toFixed(2)] }), _jsx(Slider, { value: localConfig.summary_weight_coefficient, onChange: handleWeightChange, "aria-labelledby": "weight-coefficient-slider", step: 0.05, marks: true, min: 0.1, max: 1.0, valueLabelDisplay: "auto" }), _jsx(Typography, { variant: "body2", color: "text.secondary", children: "Weight reduction factor for deeper summaries (lower values give less weight to older summaries)" })] })] })), _jsx(Button, { variant: "contained", color: "primary", sx: { mt: 2 }, onClick: handleSave, children: "Save Summarization Settings" })] }));
};
export default SummarizationSettings;
