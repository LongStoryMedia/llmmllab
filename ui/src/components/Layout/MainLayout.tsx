import React, { useState } from 'react';
import { Box, useTheme, Drawer, Backdrop, styled } from '@mui/material';
import { useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import TopBar from './TopBar';
import GalleryFAB from '../Shared/GalleryFAB';

const MainContainer = styled(Box)<{ overflow: string }>(({ theme, overflow }) => ({
  display: 'flex',
  flexDirection: 'column',
  height: '100vh',
  backgroundColor: theme.palette.background.default,
  color: theme.palette.text.primary,
  position: 'relative',
  overflow: overflow
}));

const ContentContainer = styled(Box)<{ overflow: string }>(({ overflow }) => ({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  overflow: overflow,
  paddingTop: '80px', // Account for TopBar height
  minHeight: 0
}));

const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const theme = useTheme();
  const location = useLocation();
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Check if current route is a chat page
  const isChatPage = location.pathname === '/' || location.pathname.startsWith('/chat/');

  const handleDrawerOpen = () => setDrawerOpen(true);
  const handleDrawerClose = () => setDrawerOpen(false); return (
    <MainContainer overflow={isChatPage ? 'hidden' : 'auto'}>
      <TopBar onMenuClick={handleDrawerOpen} />

      {/* Sidebar as Drawer */}
      <Drawer
        open={drawerOpen}
        onClose={handleDrawerClose}
        variant="temporary"
        ModalProps={{ keepMounted: true }}
      >
        <Sidebar onClose={handleDrawerClose} />
      </Drawer>

      {/* Dim overlay when drawer is open */}
      {drawerOpen && (
        <Backdrop open sx={{ zIndex: theme.zIndex.drawer - 1, position: 'fixed' }} />
      )}

      <ContentContainer overflow={isChatPage ? 'hidden' : 'auto'}>
        {children}
      </ContentContainer>

      {/* Image Gallery Floating Action Button */}
      <GalleryFAB />
    </MainContainer>
  );
};

export default MainLayout;