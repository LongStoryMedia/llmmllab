import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Box, Divider, Typography, useTheme, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useAuth } from '../../auth';
import ChatHistory from '../Sidebar/ChatHistory';
import NewChatButton from '../Sidebar/NewChatButton';
import Navigation from './Navigation';
const Sidebar = ({ onClose }) => {
    const { user } = useAuth();
    const theme = useTheme();
    return (_jsxs(Box, { sx: {
            width: 250,
            bgcolor: 'background.paper',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            padding: theme.spacing(2),
            position: 'sticky',
            top: 0,
            alignSelf: 'flex-start'
        }, children: [onClose && (_jsx(IconButton, { onClick: onClose, sx: { position: 'absolute', top: 8, right: 8 }, size: "small", children: _jsx(CloseIcon, {}) })), _jsxs(Typography, { variant: "h6", sx: { mb: theme.spacing(2) }, children: ["Welcome, ", user?.profile.name || 'User'] }), _jsx(Navigation, {}), _jsx(Box, { sx: { mt: theme.spacing(2), mb: theme.spacing(2) }, children: _jsx(NewChatButton, {}) }), _jsx(Divider, { sx: { my: theme.spacing(1) } }), _jsx(Box, { sx: { flex: 1, overflow: 'auto' }, children: _jsx(ChatHistory, {}) })] }));
};
export default Sidebar;
