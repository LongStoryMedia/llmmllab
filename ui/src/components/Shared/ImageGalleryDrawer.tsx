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
import { useBackgroundContext, GeneratedImage } from '../../context/BackgroundContext';

interface ImageGalleryDrawerProps {
  open: boolean;
  onClose: () => void;
  images: GeneratedImage[];
}

const ImageGalleryDrawer: React.FC<ImageGalleryDrawerProps> = ({ open, onClose, images }) => {
  const theme = useTheme();
  const { deleteImage } = useBackgroundContext();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  // Handler for image download
  const handleDownload = (url: string, prompt: string) => {
    const filename = `generated-image-${prompt.slice(0, 20).replace(/[^a-z0-9]/gi, '-')}.png`;
    
    // Create a link and click it
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Handler for image deletion from the gallery
  const handleRemove = (id: string) => {
    deleteImage(id);
  };
  
  // Handler for selecting/deselecting an image for preview
  const toggleImageSelection = (id: string | null) => {
    setSelectedImage(currentId => currentId === id ? null : id);
  };

  // Get the selected image data
  const selectedImageData = selectedImage 
    ? images.find(img => img.id === selectedImage)
    : null;

  // Function to get the display source for an image (base64 data or URL)
  const getImageSource = (image: GeneratedImage): string => {
    // Use base64 data if available, otherwise use the URL
    return image.imageData ? `data:image/png;base64,${image.imageData}` : image.downloadUrl;
  };

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
                  image={getImageSource(selectedImageData)}
                  alt="Selected image preview"
                />
                <CardActions>
                  <Button 
                    startIcon={<DownloadIcon />}
                    onClick={() => handleDownload(selectedImageData.downloadUrl, selectedImageData.prompt)}
                  >
                    Download
                  </Button>
                  <Button 
                    startIcon={<DeleteIcon />}
                    color="error"
                    onClick={() => {
                      handleRemove(selectedImage ?? "");
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
                  onClick={() => toggleImageSelection(image.id)}
                >
                  <CardMedia
                    component="img"
                    height="140"
                    image={getImageSource(image)}
                    alt={`Generated image: ${image.prompt}`}
                    sx={{ objectFit: 'cover' }}
                  />
                  <CardContent sx={{ py: 1 }}>
                    <Typography variant="body2" noWrap title={image.prompt}>
                      {image.prompt}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {new Date(image.createdAt).toLocaleString()}
                    </Typography>
                  </CardContent>
                  <CardActions>
                    <Button 
                      size="small" 
                      startIcon={<DownloadIcon />}
                      onClick={(e) => {
                        e.stopPropagation(); // Prevent image selection
                        handleDownload(image.downloadUrl, image.prompt);
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
                        handleRemove(image.id);
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