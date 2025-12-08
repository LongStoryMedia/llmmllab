import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { Box, TextField, Typography, Button, Switch, FormControlLabel, Slider, Alert, Divider, Dialog, DialogTitle, DialogContent, DialogActions } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import MemoryIcon from '@mui/icons-material/Memory';
import { useConfigContext } from '../../context/ConfigContext';
import { useAuth } from '../../auth';
import { clearMemory, nuclearClearMemory } from '../../api/resources';
const RetrievalSettings = () => {
    const { config, updatePartialConfig, isLoading } = useConfigContext();
    const { user, isAdmin } = useAuth();
    const [localConfig, setLocalConfig] = useState({
        enabled: true,
        limit: 5,
        enable_cross_user: false,
        enable_cross_conversation: false,
        similarity_threshold: 0.7,
        always_retrieve: false,
        timeout: 30
    });
    const [saveStatus, setSaveStatus] = useState(null);
    const [memoryCleanupStatus, setMemoryCleanupStatus] = useState(null);
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
                always_retrieve: config.memory.always_retrieve ?? false,
                timeout: config.memory.timeout ?? 30
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
    const handleThresholdChange = (_event, newValue) => {
        setLocalConfig({
            ...localConfig,
            similarity_threshold: newValue
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
        }
        catch (error) {
            setMemoryCleanupStatus({
                success: false,
                message: `Failed to clear memory: ${error instanceof Error ? error.message : 'Unknown error'}`
            });
        }
        finally {
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
        }
        catch (error) {
            setMemoryCleanupStatus({
                success: false,
                message: `Nuclear cleanup failed: ${error instanceof Error ? error.message : 'Unknown error'}`
            });
        }
        finally {
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
                always_retrieve: localConfig.always_retrieve,
                timeout: localConfig.timeout
            };
            const success = await updatePartialConfig('memory', snakeCaseConfig);
            if (success) {
                setSaveStatus({
                    success: true,
                    message: 'Memory retrieval settings saved successfully!'
                });
            }
            else {
                setSaveStatus({
                    success: false,
                    message: 'Failed to save memory retrieval settings.'
                });
            }
        }
        catch (err) {
            console.error('Error saving memory retrieval settings:', err);
            setSaveStatus({
                success: false,
                message: 'An error occurred while saving settings.'
            });
        }
    };
    if (isLoading) {
        return _jsx(Box, { sx: { padding: 2 }, children: _jsx(Typography, { children: "Loading memory retrieval settings..." }) });
    }
    return (_jsxs(Box, { sx: { padding: 2 }, children: [_jsx(Typography, { variant: "h6", gutterBottom: true, children: "Memory Retrieval Settings" }), saveStatus && (_jsx(Alert, { severity: saveStatus.success ? "success" : "error", sx: { mb: 2 }, onClose: () => setSaveStatus(null), children: saveStatus.message })), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.enabled, onChange: handleToggleEnabled }), label: "Enable Memory Retrieval", sx: { mb: 2, display: 'block' } }), localConfig.enabled && (_jsxs(_Fragment, { children: [_jsx(TextField, { label: "Retrieval Limit", type: "number", value: localConfig.limit, onChange: (e) => setLocalConfig({ ...localConfig, limit: parseInt(e.target.value) || 5 }), fullWidth: true, margin: "normal", helperText: "Maximum number of memory items to retrieve" }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.always_retrieve, onChange: handleToggleAlwaysRetrieve }), label: "Always Attempt Memory Retrieval", sx: { mt: 2, display: 'block' } }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.enable_cross_conversation, onChange: handleToggleCrossConversation }), label: "Enable Cross-Conversation Memory Retrieval", sx: { mt: 2, display: 'block' } }), _jsx(FormControlLabel, { control: _jsx(Switch, { checked: localConfig.enable_cross_user, onChange: handleToggleCrossUser }), label: "Enable Cross-User Memory Retrieval", sx: { mt: 2, display: 'block' } }), localConfig.enable_cross_conversation && (_jsxs(Box, { sx: { mt: 3, mb: 2 }, children: [_jsxs(Typography, { id: "similarity-threshold-slider", gutterBottom: true, children: ["Similarity Threshold: ", localConfig.similarity_threshold.toFixed(2)] }), _jsx(Slider, { value: localConfig.similarity_threshold, onChange: handleThresholdChange, "aria-labelledby": "similarity-threshold-slider", step: 0.05, marks: true, min: 0.3, max: 1.0, valueLabelDisplay: "auto" }), _jsx(Typography, { variant: "caption", color: "text.secondary", children: "Higher values require more similar memories (more precise, fewer results)" })] }))] })), isAdmin && (_jsxs(_Fragment, { children: [_jsx(Divider, { sx: { mt: 4, mb: 3 } }), _jsxs(Typography, { variant: "h6", gutterBottom: true, sx: { mt: 3 }, children: [_jsx(MemoryIcon, { sx: { mr: 1, verticalAlign: 'middle' } }), "Memory Management (Admin Only)"] }), memoryCleanupStatus && (_jsx(Alert, { severity: memoryCleanupStatus.success ? "success" : "error", sx: { mb: 2 }, onClose: () => setMemoryCleanupStatus(null), children: memoryCleanupStatus.message })), _jsxs(Box, { sx: { display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }, children: [_jsx(Button, { variant: "outlined", color: "warning", onClick: handleBasicMemoryCleanup, disabled: isCleaningMemory, startIcon: _jsx(DeleteIcon, {}), sx: { minWidth: '200px' }, children: isCleaningMemory ? 'Cleaning...' : 'Clear Memory Cache' }), _jsx(Button, { variant: "outlined", color: "error", onClick: () => setShowNuclearDialog(true), disabled: isCleaningMemory, startIcon: _jsx(DeleteIcon, {}), sx: { minWidth: '200px' }, children: "Nuclear Memory Cleanup" })] }), _jsxs(Typography, { variant: "body2", color: "text.secondary", sx: { mb: 2 }, children: [_jsx("strong", { children: "Clear Memory Cache:" }), " Releases GPU memory and unloads cached models.", _jsx("br", {}), _jsx("strong", { children: "Nuclear Cleanup:" }), " Force-kills all processes and performs aggressive memory cleanup. Use with caution!"] })] })), _jsx(Button, { variant: "contained", color: "primary", sx: { mt: 2 }, onClick: handleSave, disabled: isLoading, children: "Save Memory Settings" }), _jsxs(Dialog, { open: showNuclearDialog, onClose: () => setShowNuclearDialog(false), "aria-labelledby": "nuclear-cleanup-dialog-title", "aria-describedby": "nuclear-cleanup-dialog-description", children: [_jsx(DialogTitle, { id: "nuclear-cleanup-dialog-title", children: "\u26A0\uFE0F Nuclear Memory Cleanup" }), _jsxs(DialogContent, { children: [_jsx(Typography, { id: "nuclear-cleanup-dialog-description", sx: { mb: 2 }, children: "This will forcefully terminate all running processes and perform aggressive memory cleanup." }), _jsxs(Alert, { severity: "warning", sx: { mb: 2 }, children: [_jsx("strong", { children: "Warning:" }), " This action may interrupt running tasks and cause data loss. Only use this if normal memory cleanup fails."] }), _jsx(Typography, { variant: "body2", color: "text.secondary", children: "Are you sure you want to proceed with nuclear memory cleanup?" })] }), _jsxs(DialogActions, { children: [_jsx(Button, { onClick: () => setShowNuclearDialog(false), color: "primary", children: "Cancel" }), _jsx(Button, { onClick: handleNuclearMemoryCleanup, color: "error", variant: "contained", disabled: isCleaningMemory, children: isCleaningMemory ? 'Cleaning...' : 'Proceed' })] })] })] }));
};
export default RetrievalSettings;
