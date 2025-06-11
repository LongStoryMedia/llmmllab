import React, { useState } from 'react';
import { Box, useTheme, Drawer, Backdrop } from '@mui/material';
import Sidebar from './Sidebar';
import TopBar from './TopBar';

const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const theme = useTheme();
  const [drawerOpen, setDrawerOpen] = useState(false);

  const handleDrawerOpen = () => setDrawerOpen(true);
  const handleDrawerClose = () => setDrawerOpen(false);

  return (
    <Box display="flex" flexDirection="column" height="100%" className="main-layout">
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
      <Box p={theme.spacing(2)} overflow="auto" flexGrow={1}>
        {children}
      </Box>
    </Box>
  );
};

export default MainLayout;