import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { Box, Typography, FormControlLabel, Switch, Slider, Alert, Button } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';
const WebSearchSettings = () => {
    const { config, updatePartialConfig, isLoading } = useConfigContext();
    const [localConfig, setLocalConfig] = useState({
        enabled: false,
        auto_detect: true,
        max_results: 3,
        include_results: true,
        engines: [
            "google",
            "bing",
            "duckduckgo",
            "startpage"
        ],
        max_urls_deep: 3,
        categories: ["general"],
        language: "en",
        safesearch: 1
    });
    const [saveStatus, setSaveStatus] = useState(null);
    useEffect(() => {
        // When user config loads, update local state
        if (config?.web_search) {
            setLocalConfig({
                enabled: config.web_search.enabled ?? false,
                auto_detect: config.web_search.auto_detect ?? true,
                max_results: config.web_search.max_results ?? 3,
                include_results: config.web_search.include_results ?? true,
                engines: config.web_search.engines ?? ["google", "bing", "duckduckgo", "startpage"],
                max_urls_deep: config.web_search.max_urls_deep ?? 3,
                categories: config.web_search.categories ?? ["general"],
                language: config.web_search.language ?? "en",
                safesearch: config.web_search.safesearch ?? 1
            });
        }
    }, [config]);
    const handleToggleEnabled = () => {
        setLocalConfig({
            ...localConfig,
            enabled: !localConfig.enabled
        });
    };
    const handleToggleAutoDetect = () => {
        setLocalConfig({
            ...localConfig,
            auto_detect: !localConfig.auto_detect
        });
    };
    const handleToggleIncludeResults = () => {
        setLocalConfig({
            ...localConfig,
            include_results: !localConfig.include_results
        });
    };
    const handleMaxResultsChange = (_event, newValue) => {
        setLocalConfig({
            ...localConfig,
            max_results: newValue
        });
    };
    const handleSave = async () => {
        setSaveStatus(null);
        try {
            const success = await updatePartialConfig('web_search', localConfig);
            if (success) {
                setSaveStatus({
                    success: true,
                    message: 'Web search settings saved successfully!'
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
            console.error('Error saving web search settings:', err);
            setSaveStatus({
                success: false,
                message: 'An error occurred while saving settings.'
            });
        }
    };
    if (isLoading) {
        return _jsx(Box, { sx: { padding: 2 }, children: _jsx(Typography, { children: "Loading web search settings..." }) });
    }
    return (_jsxs(Box, { sx: { padding: 2 }, children: [_jsx(Typography, { variant: "h6", gutterBottom: true, children: "Web Search Settings" }), saveStatus && (_jsx(Alert, { severity: saveStatus.success ? "success" : "error", sx: { mb: 2 }, onClose: () => setSaveStatus(null), children: saveStatus.message })), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.enabled, onChange: handleToggleEnabled }), label: "Enable Web Search", sx: { mb: 2, display: 'block' } }), localConfig.enabled && (_jsxs(_Fragment, { children: [_jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.auto_detect, onChange: handleToggleAutoDetect }), label: "Auto-detect when to search", sx: { mb: 2, display: 'block' } }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.include_results, onChange: handleToggleIncludeResults }), label: "Include search results in responses", sx: { mb: 2, display: 'block' } }), _jsxs(Typography, { id: "max-results-slider", gutterBottom: true, children: ["Maximum search results: ", localConfig.max_results] }), _jsx(Slider, { "aria-labelledby": "max-results-slider", value: localConfig.max_results, onChange: handleMaxResultsChange, step: 1, marks: true, min: 1, max: 5, valueLabelDisplay: "auto", sx: { mb: 3 } })] })), _jsx(Button, { variant: "contained", color: "primary", onClick: handleSave, children: "Save Web Search Settings" })] }));
};
export default WebSearchSettings;
