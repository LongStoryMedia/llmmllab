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
  ListItemText,
  List,
  ListItem,
  LinearProgress,
  Chip,
  ListSubheader,
  Grid,
  Button
} from '@mui/material';
import NotificationsIcon from '@mui/icons-material/Notifications';
import ImageIcon from '@mui/icons-material/Image';
import ClearAllIcon from '@mui/icons-material/ClearAll';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import AutorenewIcon from '@mui/icons-material/Autorenew';
import ImageGalleryDrawer from '../Shared/ImageGalleryDrawer';
import { useBackgroundContext } from '../../context/BackgroundContext';
import { getStageFriendlyName } from '../Shared/StageProgressBars';
import { SocketStageType } from '../../types/SocketStageType';

const BackgroundProcessNotifications: React.FC = () => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [isGalleryOpen, setIsGalleryOpen] = useState(false);
  const { 
    markAllAsRead,
    images,
    unreadNotificationCount,
    longRunningStages,
    errors,
    warnings,
    dismissError,
    dismissWarning
  } = useBackgroundContext();
  
  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleClearAll = () => {
    markAllAsRead();
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

  const hasImageGenerationResults = images.length > 0;
  const hasLongRunningProcesses = longRunningStages.length > 0;
  const hasErrors = errors.length > 0;
  const hasWarnings = warnings.length > 0;
  
  return (
    <>
      <IconButton
        color="inherit"
        onClick={handleClick}
      >
        <Badge badgeContent={unreadNotificationCount} color="error">
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
            maxHeight: '60vh',
            maxWidth: 400
          }
        }}
      >
        <Box sx={{ px: 2, py: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="subtitle1" fontWeight="bold">
            Background Processes
          </Typography>
          {unreadNotificationCount > 0 && (
            <IconButton size="small" onClick={handleClearAll} title="Clear all notifications">
              <ClearAllIcon />
            </IconButton>
          )}
        </Box>
        
        <Divider />
        
        {!hasImageGenerationResults && !hasLongRunningProcesses && !hasErrors && !hasWarnings ? (
          <MenuItem disabled>
            <Typography variant="body2" color="text.secondary">
              No active processes or notifications
            </Typography>
          </MenuItem>
        ) : (
          <>
            {hasLongRunningProcesses && (
              <>
                <List
                  subheader={
                    <ListSubheader>
                      Active Processes
                    </ListSubheader>
                  }
                >
                  {longRunningStages.map((stage) => (
                    <ListItem key={stage.id || stage.stage} sx={{ display: 'block', py: 1 }}>
                      <Grid container spacing={1} alignItems="center">
                        <Grid>
                          <AutorenewIcon color="primary" fontSize="small" />
                        </Grid>
                        <Grid>
                          <Typography variant="body2">{getStageFriendlyName(stage.stage)}</Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Box sx={{ width: '100%', mr: 1 }}>
                              <LinearProgress 
                                variant="determinate" 
                                value={stage.progress} 
                                sx={{ height: 4, borderRadius: 1 }}
                              />
                            </Box>
                            <Typography variant="caption">
                              {Math.round(stage.progress)}%
                            </Typography>
                          </Box>
                          <Typography variant="caption" color="text.secondary">
                            {getTimeDifference(stage.timestamp)}
                          </Typography>
                        </Grid>
                      </Grid>
                    </ListItem>
                  ))}
                </List>
                <Divider />
              </>
            )}

            {hasErrors && (
              <>
                <List
                  subheader={
                    <ListSubheader>
                      Errors
                    </ListSubheader>
                  }
                >
                  {errors.map((error) => (
                    <ListItem 
                      key={error.id} 
                      sx={{ display: 'block', py: 1 }}
                      secondaryAction={
                        <IconButton 
                          edge="end" 
                          aria-label="dismiss" 
                          size="small"
                          onClick={() => dismissError(error.id)}
                        >
                          <CancelIcon fontSize="small" />
                        </IconButton>
                      }
                    >
                      <Grid container spacing={1} alignItems="flex-start">
                        <Grid>
                          <ErrorIcon color="error" fontSize="small" />
                        </Grid>
                        <Grid>
                          <Typography variant="body2">
                            {error.message}
                          </Typography>
                          {error.stage && (
                            <Chip 
                              label={getStageFriendlyName(error.stage as SocketStageType)} 
                              size="small" 
                              sx={{ mt: 0.5 }}
                            />
                          )}
                          <Typography variant="caption" display="block" color="text.secondary">
                            {getTimeDifference(error.timestamp)}
                          </Typography>
                        </Grid>
                      </Grid>
                    </ListItem>
                  ))}
                </List>
                <Divider />
              </>
            )}
            
            {hasWarnings && (
              <>
                <List
                  subheader={
                    <ListSubheader>
                      Warnings
                    </ListSubheader>
                  }
                >
                  {warnings.map((warning) => (
                    <ListItem 
                      key={warning.id} 
                      sx={{ display: 'block', py: 1 }}
                      secondaryAction={
                        <IconButton 
                          edge="end" 
                          aria-label="dismiss" 
                          size="small"
                          onClick={() => dismissWarning(warning.id)}
                        >
                          <CancelIcon fontSize="small" />
                        </IconButton>
                      }
                    >
                      <Grid container spacing={1} alignItems="flex-start">
                        <Grid>
                          <WarningIcon color="warning" fontSize="small" />
                        </Grid>
                        <Grid>
                          <Typography variant="body2">
                            {warning.message}
                          </Typography>
                          {warning.stage && (
                            <Chip 
                              label={getStageFriendlyName(warning.stage as SocketStageType)} 
                              size="small" 
                              sx={{ mt: 0.5 }}
                            />
                          )}
                          <Typography variant="caption" display="block" color="text.secondary">
                            {getTimeDifference(warning.timestamp)}
                          </Typography>
                        </Grid>
                      </Grid>
                    </ListItem>
                  ))}
                </List>
                <Divider />
              </>
            )}
            
            {hasImageGenerationResults && (
              <MenuItem onClick={handleOpenImageGallery}>
                <ListItemIcon>
                  <ImageIcon color="primary" />
                </ListItemIcon>
                <ListItemText 
                  primary="View Generated Images Gallery" 
                  secondary={`${images.length} image${images.length !== 1 ? 's' : ''} available`}
                />
              </MenuItem>
            )}
            
            {unreadNotificationCount > 0 && (
              <Box sx={{ p: 1, display: 'flex', justifyContent: 'center' }}>
                <Button
                  size="small"
                  startIcon={<CheckCircleIcon />}
                  onClick={handleClearAll}
                >
                  Mark all as read
                </Button>
              </Box>
            )}
          </>
        )}
      </Menu>
      
      {/* Image Gallery Drawer */}
      <ImageGalleryDrawer
        open={isGalleryOpen}
        onClose={() => setIsGalleryOpen(false)}
        images={images}
      />
    </>
  );
};

export default BackgroundProcessNotifications;