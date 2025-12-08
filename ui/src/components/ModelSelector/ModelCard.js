import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Card, CardContent, Typography, Button, useTheme } from '@mui/material';
const ModelCard = ({ modelName, modelDescription, onSelect }) => {
    const theme = useTheme();
    return (_jsx(Card, { variant: "outlined", sx: {
            margin: theme.spacing(1.25),
            cursor: 'pointer'
        }, children: _jsxs(CardContent, { children: [_jsx(Typography, { variant: "h5", component: "div", children: modelName }), _jsx(Typography, { variant: "body2", color: "text.secondary", children: modelDescription }), _jsx(Button, { variant: "contained", color: "primary", onClick: () => onSelect(modelName), sx: { mt: theme.spacing(1) }, children: "Select Model" })] }) }));
};
export default ModelCard;
