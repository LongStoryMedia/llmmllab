import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Box, FormControl, InputLabel, Select, MenuItem, Typography } from '@mui/material';
import { useChat } from '../../chat';
import ControlLoader from '../Shared/ControlLoader';
const ModelSelector = ({ onSelect, name, label, optional }) => {
    const { models, isLoading } = useChat();
    return (isLoading ?
        _jsx(ControlLoader, { text: 'Loading models...' }) :
        _jsxs(Box, { sx: { mb: 2, p: 2 }, children: [_jsx(Typography, { variant: "h6", gutterBottom: true, children: label || "Select a Model" }), _jsxs(FormControl, { fullWidth: true, children: [_jsx(InputLabel, { id: "model-select-label", children: "Model" }), _jsxs(Select, { labelId: "model-select-label", id: "model-select", value: name, onChange: onSelect, label: "Model", children: [optional && _jsx(MenuItem, { value: "", children: "None" }), models && models?.map((model) => (_jsx(MenuItem, { value: model.id, children: model.name }, model.name)))] })] })] }));
};
export default ModelSelector;
