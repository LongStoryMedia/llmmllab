import React, { useState, useEffect } from 'react';
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

const ContentContainer = styled(Box)<{ topBarVisible: boolean }>(({ theme, topBarVisible }) => ({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  paddingTop: topBarVisible ? '64px' : '0px', // Account for TopBar height
  transition: theme.transitions.create(['padding-top'], {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen
  })
}));

const TopBarContainer = styled(Box)<{ visible: boolean }>(({ theme, visible }) => ({
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  zIndex: theme.zIndex.appBar,
  transform: visible ? 'translateY(0)' : 'translateY(-100%)',
  transition: theme.transitions.create(['transform'], {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.short,
  }),
}));

const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const theme = useTheme();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [topBarVisible, setTopBarVisible] = useState(true);
  const [lastScrollY, setLastScrollY] = useState(0);
  const { activeStages } = useBackgroundContext();

  const handleDrawerOpen = () => setDrawerOpen(true);
  const handleDrawerClose = () => setDrawerOpen(false);

  // Auto-hide TopBar on scroll
  useEffect(() => {
    let timeoutId: number;
    
    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      
      // Show TopBar when scrolling up or at top
      if (currentScrollY < lastScrollY || currentScrollY < 10) {
        setTopBarVisible(true);
      } else if (currentScrollY > lastScrollY && currentScrollY > 100) {
        // Hide TopBar when scrolling down
        setTopBarVisible(false);
      }
      
      setLastScrollY(currentScrollY);
      
      // Always show TopBar after user stops scrolling
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        setTopBarVisible(true);
      }, 2000);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', handleScroll);
      clearTimeout(timeoutId);
    };
  }, [lastScrollY]);

  return (
    <MainContainer>
      <TopBarContainer visible={topBarVisible}>
        <TopBar onMenuClick={handleDrawerOpen} />
      </TopBarContainer>
      
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
      
      <ContentContainer topBarVisible={topBarVisible}>
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