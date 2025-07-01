import React from 'react';
import { Box, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Divider } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';
import ChatIcon from '@mui/icons-material/Chat';
import ImageIcon from '@mui/icons-material/Image';
import SettingsIcon from '@mui/icons-material/Settings';
import HandymanIcon  from '@mui/icons-material/Handyman';

const Navigation: React.FC = () => {
  const location = useLocation();
  
  const navigationItems = [
    { text: 'Chat', icon: <ChatIcon />, path: '/' },
    { text: 'Images', icon: <ImageIcon />, path: '/images' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
    { text: 'Model Profiles', icon: <HandymanIcon />, path: '/model-profiles' }
  ];

  return (
    <Box sx={{ width: '100%' }}>
      <List>
        {navigationItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <ListItem key={item.text} disablePadding>
              <ListItemButton 
                component={Link} 
                to={item.path}
                selected={isActive}
                sx={{
                  '&.Mui-selected': {
                    backgroundColor: 'primary.light',
                    '&:hover': {
                      backgroundColor: 'primary.light'
                    }
                  }
                }}
              >
                <ListItemIcon sx={{ color: isActive ? 'primary.main' : 'inherit' }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText 
                  primary={item.text} 
                  sx={{ color: isActive ? 'primary.main' : 'inherit' }}
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>
      <Divider />
    </Box>
  );
};

export default Navigation;