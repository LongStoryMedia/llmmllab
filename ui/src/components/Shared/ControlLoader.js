import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Box, Typography, useTheme } from '@mui/material';
import LoadingAnimation from './LoadingAnimation';
const ControlLoader = ({ text = '' }) => {
    const theme = useTheme();
    return (_jsxs(Box, { sx: {
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            p: theme.spacing(2)
        }, children: [_jsx(LoadingAnimation, { size: 24 }), text && (_jsx(Typography, { variant: "body2", color: "text.secondary", sx: { mt: theme.spacing(1) }, children: text }))] }));
};
export default ControlLoader;
