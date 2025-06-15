import React, { useState } from 'react';
import { 
  IconButton, 
  Badge, 
  Menu, 
  MenuItem, 
  Typography,
  Box, 
  Divider,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import NotificationsIcon from '@mui/icons-material/Notifications';
import ImageIcon from '@mui/icons-material/Image';
import ErrorIcon from '@mui/icons-material/Error';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ClearAllIcon from '@mui/icons-material/ClearAll';
import { useBackgroundProcess } from '../../context/BackgroundProcessContext';
import ImageGalleryDrawer from '../Shared/ImageGalleryDrawer';

const BackgroundProcessNotifications: React.FC = () => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [isGalleryOpen, setIsGalleryOpen] = useState(false);
  const { 
    processes, 
    generatedImages, 
    unreadCount, 
    markAllAsRead, 
    removeAllProcesses 
  } = useBackgroundProcess();
  
  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
    markAllAsRead();
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleClearAll = () => {
    removeAllProcesses();
    handleClose();
  };

  const handleOpenImageGallery = () => {
    setIsGalleryOpen(true);
    handleClose();
  };

  // Get the time difference in human-readable format
  const getTimeDifference = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);
    
    if (diffHour > 0) {
      return `${diffHour} hour${diffHour !== 1 ? 's' : ''} ago`;
    } else if (diffMin > 0) {
      return `${diffMin} minute${diffMin !== 1 ? 's' : ''} ago`;
    } else {
      return 'Just now';
    }
  };

  const hasImageGenerationResults = generatedImages.length > 0;
  
  return (
    <>
      <IconButton
        color="inherit"
        onClick={handleClick}
        disabled={processes.length === 0}
      >
        <Badge badgeContent={unreadCount} color="error">
          <NotificationsIcon />
        </Badge>
      </IconButton>
      
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right'
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right'
        }}
        PaperProps={{
          sx: {
            minWidth: 320,
            maxHeight: '60vh'
          }
        }}
      >
        <Box sx={{ px: 2, py: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="subtitle1" fontWeight="bold">
            Background Processes
          </Typography>
          {processes.length > 0 && (
            <IconButton size="small" onClick={handleClearAll} title="Clear all notifications">
              <ClearAllIcon />
            </IconButton>
          )}
        </Box>
        
        <Divider />
        
        {processes.length === 0 ? (
          <MenuItem disabled>
            <Typography variant="body2" color="text.secondary">
              No active processes or notifications
            </Typography>
          </MenuItem>
        ) : (
          <>
            {hasImageGenerationResults && (
              <MenuItem onClick={handleOpenImageGallery}>
                <ListItemIcon>
                  <ImageIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="View Generated Images Gallery" 
                  secondary={`${generatedImages.length} image${generatedImages.length !== 1 ? 's' : ''} available`}
                />
              </MenuItem>
            )}
            
            <Divider />
            
            {processes.map((process) => (
              <MenuItem key={process.id} sx={{ display: 'block', py: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
                  <ListItemIcon>
                    {process.status === 'completed' ? (
                      <CheckCircleIcon color="success" />
                    ) : process.status === 'failed' ? (
                      <ErrorIcon color="error" />
                    ) : (
                      <ImageIcon color="primary" />
                    )}
                  </ListItemIcon>
                  <ListItemText 
                    primary={process.title}
                    secondary={
                      <>
                        <Typography variant="body2" component="span" color="text.secondary">
                          {process.details || process.error || ''}
                        </Typography>
                        <Typography variant="caption" component="p" color="text.secondary" sx={{ mt: 0.5 }}>
                          {getTimeDifference(process.createdAt)}
                        </Typography>
                      </>
                    }
                  />
                </Box>
              </MenuItem>
            ))}
          </>
        )}
      </Menu>
      
      {/* Image Gallery Drawer */}
      <ImageGalleryDrawer
        open={isGalleryOpen}
        onClose={() => setIsGalleryOpen(false)}
        images={generatedImages}
      />
    </>
  );
};

export default BackgroundProcessNotifications;