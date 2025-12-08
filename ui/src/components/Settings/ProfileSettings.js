import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { Box, TextField, Typography, Button, Alert, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import { useConfigContext } from '../../context/ConfigContext';
import { useAuth } from '../../auth';
const ProfileSettings = () => {
    const { user } = useAuth();
    const { config, updatePartialConfig, isLoading } = useConfigContext();
    const [preferences, setPreferences] = useState({
        font_size: 14,
        language: 'en',
        notifications_on: true
    });
    const [saveStatus, setSaveStatus] = useState(null);
    // Update local state when the user config loads or changes
    useEffect(() => {
        if (config?.preferences) {
            setPreferences({
                font_size: config.preferences.font_size || 14,
                language: config.preferences.language || 'en',
                notifications_on: config.preferences.notifications_on !== false
            });
        }
    }, [config]);
    const handleSave = async () => {
        setSaveStatus(null);
        try {
            // Convert camelCase to snake_case when passing to updatePartialConfig
            const snakeCasePreferences = {
                font_size: preferences.font_size,
                language: preferences.language,
                notifications_on: preferences.notifications_on
            };
            const success = await updatePartialConfig('preferences', snakeCasePreferences);
            if (success) {
                setSaveStatus({
                    success: true,
                    message: 'Profile settings saved successfully!'
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
        return _jsx(Box, { sx: { padding: 2 }, children: _jsx(Typography, { children: "Loading profile settings..." }) });
    }
    return (_jsxs(Box, { sx: { padding: 2 }, children: [_jsx(Typography, { variant: "h6", gutterBottom: true, children: "Profile Settings" }), saveStatus && (_jsx(Alert, { severity: saveStatus.success ? "success" : "error", sx: { mb: 2 }, onClose: () => setSaveStatus(null), children: saveStatus.message })), _jsx(Typography, { variant: "subtitle1", gutterBottom: true, children: "Account Information" }), _jsx(TextField, { label: "Name", value: user?.profile.name || '', fullWidth: true, margin: "normal", disabled: true, helperText: "Your account name (managed by authentication provider)" }), _jsx(TextField, { label: "Email", value: user?.profile.email || '', fullWidth: true, margin: "normal", disabled: true, helperText: "Your account email (managed by authentication provider)" }), _jsx(Typography, { variant: "subtitle1", gutterBottom: true, sx: { mt: 2 }, children: "Display Preferences" }), _jsx(TextField, { label: "Font Size", type: "number", value: preferences.font_size, onChange: (e) => setPreferences({ ...preferences, font_size: parseInt(e.target.value) || 14 }), fullWidth: true, margin: "normal", helperText: "Font size for chat messages (10-24px)", slotProps: {
                    input: { inputProps: { min: 10, max: 24 } }
                } }), _jsxs(FormControl, { fullWidth: true, margin: "normal", children: [_jsx(InputLabel, { id: "language-select-label", children: "Language" }), _jsxs(Select, { labelId: "language-select-label", id: "language-select", value: preferences.language, onChange: (e) => setPreferences({ ...preferences, language: e.target.value }), label: "Language", children: [_jsx(MenuItem, { value: "en", children: "English" }), _jsx(MenuItem, { value: "es", children: "Spanish" }), _jsx(MenuItem, { value: "fr", children: "French" }), _jsx(MenuItem, { value: "de", children: "German" }), _jsx(MenuItem, { value: "zh", children: "Chinese" }), _jsx(MenuItem, { value: "ja", children: "Japanese" })] })] }), _jsx(Button, { variant: "contained", color: "primary", sx: { mt: 2 }, onClick: handleSave, children: "Save Profile Settings" })] }));
};
export default ProfileSettings;
