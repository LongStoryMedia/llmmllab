import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { Box, useTheme, Drawer, Backdrop, styled } from '@mui/material';
import { useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import TopBar from './TopBar';
import GalleryFAB from '../Shared/GalleryFAB';
const MainContainer = styled(Box)(({ theme, overflow }) => ({
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    backgroundColor: theme.palette.background.default,
    color: theme.palette.text.primary,
    position: 'relative',
    overflow: overflow
}));
const ContentContainer = styled(Box)(() => ({
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    paddingTop: '80px', // Account for TopBar height
    minHeight: 0
}));
const MainLayout = ({ children }) => {
    const theme = useTheme();
    const location = useLocation();
    const [drawerOpen, setDrawerOpen] = useState(false);
    // Check if current route is a chat page
    const isChatPage = location.pathname === '/' || location.pathname.startsWith('/chat/');
    const handleDrawerOpen = () => setDrawerOpen(true);
    const handleDrawerClose = () => setDrawerOpen(false);
    return (_jsxs(MainContainer, { overflow: isChatPage ? 'hidden' : 'auto', children: [_jsx(TopBar, { onMenuClick: handleDrawerOpen }), _jsx(Drawer, { open: drawerOpen, onClose: handleDrawerClose, variant: "temporary", ModalProps: { keepMounted: true }, children: _jsx(Sidebar, { onClose: handleDrawerClose }) }), drawerOpen && (_jsx(Backdrop, { open: true, sx: { zIndex: theme.zIndex.drawer - 1, position: 'fixed' } })), _jsx(ContentContainer, { overflow: isChatPage ? 'hidden' : 'auto', children: children }), _jsx(GalleryFAB, {})] }));
};
export default MainLayout;
