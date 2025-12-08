import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { Grid, Typography, useTheme, Tabs, Tab, Box, Paper, Alert, CircularProgress, Button } from '@mui/material';
import { useConfig } from '../../hooks/useConfig';
import ProfileSettings from './ProfileSettings';
import ModelSettings from './ModelSettings';
import SummarizationSettings from './SummarizationSettings';
import MemorySettings from './MemorySettings';
import WebSearchSettings from './WebSearchSettings';
import SecuritySettings from './SecuritySettings';
import RefinementSettings from './RefinementSettings';
import ImageGenerationSettings from './ImageGenerationSettings';
import CircuitBreakerSettings from './CircuitBreakerSettings';
import GpuSettings from './GpuSettings';
import ParameterOptimizationSettings from './ParameterOptimizationSettings';
function TabPanel(props) {
    const { children, value, index, ...other } = props;
    return (_jsx("div", { role: "tabpanel", hidden: value !== index, id: `settings-tabpanel-${index}`, "aria-labelledby": `settings-tab-${index}`, ...other, children: value === index && (_jsx(Box, { sx: { p: 0 }, children: children })) }));
}
function a11yProps(index) {
    return {
        id: `settings-tab-${index}`,
        'aria-controls': `settings-tabpanel-${index}`
    };
}
const tabRoutes = ["profile", "models", "summarization", "retrieval", "websearch", "security", "refinement", "image-generation", "gpu", "parameter-optimization", "circuit-breaker"];
const SettingsTabs = ({ onTabChange, currentTab }) => {
    const theme = useTheme();
    const [tabValue, setTabValue] = useState(tabRoutes.indexOf(currentTab));
    const { isLoading, error, fetchConfig } = useConfig();
    const handleTabChange = (_event, newValue) => {
        setTabValue(newValue);
        onTabChange(tabRoutes[newValue]);
    };
    return (_jsxs(Grid, { container: true, spacing: 3, sx: { padding: theme.spacing(2.5) }, children: [_jsx(Grid, { size: 12, children: _jsx(Typography, { variant: "h4", gutterBottom: true, children: "Settings" }) }), error && (_jsx(Grid, { size: 12, children: _jsxs(Alert, { severity: "error", action: _jsx(Button, { color: "inherit", size: "small", onClick: fetchConfig, children: "Retry" }), children: ["Error loading configuration: ", error.message] }) })), _jsx(Grid, { size: 12, children: isLoading ? (_jsxs(Paper, { sx: { padding: 3, display: 'flex', justifyContent: 'center', alignItems: 'center' }, children: [_jsx(CircularProgress, {}), _jsx(Typography, { sx: { ml: 2 }, children: "Loading settings..." })] })) : (_jsxs(Paper, { sx: { width: '100%' }, children: [_jsxs(Tabs, { value: tabValue, onChange: handleTabChange, indicatorColor: "primary", textColor: "primary", variant: "scrollable", scrollButtons: "auto", "aria-label": "settings tabs", children: [_jsx(Tab, { label: "User Profile", ...a11yProps(0) }), _jsx(Tab, { label: "Models", ...a11yProps(1) }), _jsx(Tab, { label: "Summarization", ...a11yProps(2) }), _jsx(Tab, { label: "Memory Retrieval", ...a11yProps(3) }), _jsx(Tab, { label: "Web Search Settings", ...a11yProps(4) }), _jsx(Tab, { label: "Security", ...a11yProps(5) }), _jsx(Tab, { label: "Refinement", ...a11yProps(6) }), _jsx(Tab, { label: "Image Generation", ...a11yProps(7) }), _jsx(Tab, { label: "GPU Configuration", ...a11yProps(8) }), _jsx(Tab, { label: "Parameter Optimization", ...a11yProps(9) }), _jsx(Tab, { label: "Circuit Breaker", ...a11yProps(10) })] }), _jsxs(Box, { sx: { p: 2 }, children: [_jsx(TabPanel, { value: tabValue, index: 0, children: _jsx(ProfileSettings, {}) }), _jsx(TabPanel, { value: tabValue, index: 1, children: _jsx(ModelSettings, {}) }), _jsx(TabPanel, { value: tabValue, index: 2, children: _jsx(SummarizationSettings, {}) }), _jsx(TabPanel, { value: tabValue, index: 3, children: _jsx(MemorySettings, {}) }), _jsx(TabPanel, { value: tabValue, index: 4, children: _jsx(WebSearchSettings, {}) }), _jsx(TabPanel, { value: tabValue, index: 5, children: _jsx(SecuritySettings, {}) }), _jsx(TabPanel, { value: tabValue, index: 6, children: _jsx(RefinementSettings, {}) }), _jsx(TabPanel, { value: tabValue, index: 7, children: _jsx(ImageGenerationSettings, {}) }), _jsx(TabPanel, { value: tabValue, index: 8, children: _jsx(GpuSettings, {}) }), _jsx(TabPanel, { value: tabValue, index: 9, children: _jsx(ParameterOptimizationSettings, {}) }), _jsx(TabPanel, { value: tabValue, index: 10, children: _jsx(CircuitBreakerSettings, {}) })] })] })) })] }));
};
export default SettingsTabs;
