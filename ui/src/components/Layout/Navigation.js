import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Box, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Divider } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';
import ChatIcon from '@mui/icons-material/Chat';
import ImageIcon from '@mui/icons-material/Image';
import ChecklistIcon from '@mui/icons-material/Checklist';
import SettingsIcon from '@mui/icons-material/Settings';
import HandymanIcon from '@mui/icons-material/Handyman';
const Navigation = () => {
    const location = useLocation();
    const navigationItems = [
        { text: 'Chat', icon: _jsx(ChatIcon, {}), path: '/' },
        { text: 'Images', icon: _jsx(ImageIcon, {}), path: '/images' },
        { text: 'Todos', icon: _jsx(ChecklistIcon, {}), path: '/todos' },
        { text: 'Settings', icon: _jsx(SettingsIcon, {}), path: '/settings' },
        { text: 'Model Profiles', icon: _jsx(HandymanIcon, {}), path: '/model-profiles' }
    ];
    return (_jsxs(Box, { sx: { width: '100%' }, children: [_jsx(List, { children: navigationItems.map((item) => {
                    const isActive = location.pathname === item.path;
                    return (_jsx(ListItem, { disablePadding: true, children: _jsxs(ListItemButton, { component: Link, to: item.path, selected: isActive, sx: {
                                '&.Mui-selected': {
                                    backgroundColor: 'primary.light',
                                    '&:hover': {
                                        backgroundColor: 'primary.light'
                                    }
                                }
                            }, children: [_jsx(ListItemIcon, { sx: { color: isActive ? 'primary.main' : 'inherit' }, children: item.icon }), _jsx(ListItemText, { primary: item.text, sx: { color: isActive ? 'primary.main' : 'inherit' } })] }) }, item.text));
                }) }), _jsx(Divider, {})] }));
};
export default Navigation;
