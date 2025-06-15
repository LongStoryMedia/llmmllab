import React, { useState } from 'react';
import { Fab, Badge, Tooltip } from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import { useBackgroundProcess } from '../../context/BackgroundProcessContext';
import ImageGalleryDrawer from './ImageGalleryDrawer';

const GalleryFAB: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { generatedImages } = useBackgroundProcess();
  
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
            badgeContent={generatedImages.length} 
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
        images={generatedImages}
      />
    </>
  );
};

export default GalleryFAB;