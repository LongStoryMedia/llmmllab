import React, { useState } from 'react';
import { Box, useTheme, Drawer, Backdrop, styled } from '@mui/material';
import Sidebar from './Sidebar';
import TopBar from './TopBar';
import GalleryFAB from '../Shared/GalleryFAB';
import StageProgressBars from '../Shared/StageProgressBars';
import { useBackgroundContext } from '../../context/BackgroundContext';

const MainContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  height: '100vh',
  backgroundColor: theme.palette.background.default,
  color: theme.palette.text.primary,
  position: 'relative',
  overflow: 'hidden'
}));

const ContentContainer = styled(Box)({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  paddingTop: '80px' // Account for TopBar height
});

const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const theme = useTheme();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const { activeStages } = useBackgroundContext();

  const handleDrawerOpen = () => setDrawerOpen(true);
  const handleDrawerClose = () => setDrawerOpen(false); return (
    <MainContainer>
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

      <ContentContainer>
        {children}
      </ContentContainer>

      {/* Image Gallery Floating Action Button */}
      <GalleryFAB />

      {/* Stage Progress Bars */}
      <StageProgressBars activeStages={activeStages} />
    </MainContainer>
  );
};

export default MainLayout;