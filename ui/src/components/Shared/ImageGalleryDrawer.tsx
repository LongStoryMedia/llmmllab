import React, { useState } from 'react';
import { 
  Drawer,
  Box,
  Typography,
  IconButton,
  Grid,
  Card,
  CardMedia,
  CardContent,
  CardActions,
  Button,
  Divider,
  useTheme
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import DownloadIcon from '@mui/icons-material/Download';
import DeleteIcon from '@mui/icons-material/Delete';
import { useBackgroundContext } from '../../context/BackgroundContext';
import { ImageMetadata } from '../../types/ImageMetadata';

interface ImageGalleryDrawerProps {
  open: boolean;
  onClose: () => void;
  images: ImageMetadata[];
}

const ImageGalleryDrawer: React.FC<ImageGalleryDrawerProps> = ({ open, onClose, images }) => {
  const theme = useTheme();
  const { deleteImage } = useBackgroundContext();
  const [selectedImage, setSelectedImage] = useState<number | null>(null);

  const handleDownload = (url: string, name?: string) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = name ? `${name.toLowerCase()}.png` : 'image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Handler for image deletion from the gallery
  const handleRemove = (id: number) => {
    deleteImage(id);
  };
  
  // Handler for selecting/deselecting an image for preview
  const toggleImageSelection = (id: number | null) => {
    setSelectedImage(currentId => currentId === id ? null : id);
  };

  // Get the selected image data
  const selectedImageData = selectedImage 
    ? images.find(img => img.id === selectedImage)
    : null;

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      sx={{
        '& .MuiDrawer-paper': {
          width: '100%',
          maxWidth: 600,
          p: 2
        }
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Generated Images</Typography>
        <IconButton onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </Box>
      
      <Divider sx={{ mb: 2 }} />
      
      {images.length === 0 ? (
        <Typography color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
          No images have been generated yet.
        </Typography>
      ) : (
        <>
          {/* Selected Image Preview */}
          {selectedImageData && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom>Preview</Typography>
              <Card>
                <CardMedia
                  component="img"
                  sx={{ 
                    maxHeight: '500px', 
                    objectFit: 'contain',
                    backgroundColor: 'black'
                  }}
                  image={selectedImageData.view_url || selectedImageData.download_url}
                  alt="Selected image preview"
                />
                <CardActions>
                  <Button 
                    startIcon={<DownloadIcon />}
                    onClick={() => handleDownload(selectedImageData.download_url || '', selectedImageData.created_at.toISOString())}
                  >
                    Download
                  </Button>
                  <Button 
                    startIcon={<DeleteIcon />}
                    color="error"
                    onClick={() => {
                      handleRemove(selectedImage ?? -1);
                      setSelectedImage(null);
                    }}
                  >
                    Remove
                  </Button>
                </CardActions>
              </Card>
            </Box>
          )}

          {/* Image Grid */}
          <Typography variant="subtitle1" gutterBottom>
            Image Gallery
          </Typography>
          <Grid container spacing={2}>
            {images.map((image) => (
              <Grid sx={{xs:12, sm:6, md:4}} key={image.id}>
                <Card 
                  sx={{ 
                    cursor: 'pointer',
                    border: selectedImage === image.id 
                      ? `2px solid ${theme.palette.primary.main}` 
                      : 'none'
                  }}
                  onClick={() => toggleImageSelection(image.id ?? -1)}
                >
                  <CardMedia
                    component="img"
                    height="140"
                    image={image.view_url || image.download_url}
                    alt={`Generated image: ${image.filename}`}
                    sx={{ objectFit: 'cover' }}
                  />
                  <CardContent sx={{ py: 1 }}>
                    <Typography variant="body2" noWrap title={image.filename}>
                      {image.filename}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {new Date(image.created_at).toLocaleString()}
                    </Typography>
                  </CardContent>
                  <CardActions>
                    <Button 
                      size="small" 
                      startIcon={<DownloadIcon />}
                      onClick={(e) => {
                        e.stopPropagation(); // Prevent image selection
                        handleDownload(image.download_url ?? '', image.created_at.toISOString());
                      }}
                    >
                      Download
                    </Button>
                    <Button 
                      size="small" 
                      color="error"
                      startIcon={<DeleteIcon />}
                      onClick={(e) => {
                        e.stopPropagation(); // Prevent image selection
                        handleRemove(image.id ?? -1);
                        if (selectedImage === image.id) {
                          setSelectedImage(null);
                        }
                      }}
                    >
                      Remove
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        </>
      )}
    </Drawer>
  );
};

export default ImageGalleryDrawer;