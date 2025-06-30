import React, { useState } from 'react';
import { Fab, Badge, Tooltip } from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import ImageGalleryDrawer from './ImageGalleryDrawer';
import { useBackgroundContext } from '../../context/BackgroundContext';

const GalleryFAB: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { images } = useBackgroundContext();
  
  const handleOpen = () => {
    setIsOpen(true);
  };

  const handleClose = () => {
    setIsOpen(false);
  };

  return (
    <>
      <Tooltip title="View Image Gallery" placement="left">
        <Fab 
          color="secondary"
          aria-label="Open Image Gallery"
          onClick={handleOpen}
          sx={{ 
            position: 'fixed', 
            bottom: 24, 
            right: 24,
            zIndex: (theme) => theme.zIndex.drawer - 2
          }}
        >
          <Badge 
            badgeContent={images.length} 
            color="error" 
            overlap="circular"
            max={99}
          >
            <ImageIcon />
          </Badge>
        </Fab>
      </Tooltip>

      {/* Image Gallery Drawer */}
      <ImageGalleryDrawer
        open={isOpen}
        onClose={handleClose}
        images={images}
      />
    </>
  );
};

export default GalleryFAB;