import React from 'react';
import { AppBar, Toolbar, Typography, IconButton, useTheme } from '@mui/material';
import { Menu as MenuIcon } from '@mui/icons-material';

interface ChatHeaderProps {
  title: string;
  onMenuClick: React.MouseEventHandler<HTMLButtonElement>
}

const ChatHeader = ({ title, onMenuClick }: ChatHeaderProps) => {
  const theme = useTheme();
  
  return (
    <AppBar position="static">
      <Toolbar>
        <IconButton edge="start" color="inherit" onClick={onMenuClick}>
          <MenuIcon />
        </IconButton>
        <Typography variant="h6" sx={{ flexGrow: 1, color: theme.palette.common.white }}>
          {title}
        </Typography>
      </Toolbar>
    </AppBar>
  );
};

export default ChatHeader;