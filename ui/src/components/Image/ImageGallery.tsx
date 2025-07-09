import React from 'react';
import { 
  Box,
  Card,
  CardMedia,
  Grid,
  Typography,
  Paper,
  Divider,
  IconButton,
  useTheme
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import DeleteIcon from '@mui/icons-material/Delete';
import { ImageMetadata } from '../../types/ImageMetadata';
import config from '../../config';

interface ImageGalleryProps {
  images: ImageMetadata[];
  selectedImage: number | null;
  onSelectImage: (id: number | null) => void;
  onDeleteImage: (id: number) => void;
  onDownloadImage: (url?: string, name?: string) => void;
}

const ImageGallery: React.FC<ImageGalleryProps> = ({
  images,
  selectedImage,
  onSelectImage,
  onDeleteImage,
  onDownloadImage
}) => {
  const theme = useTheme();

  const ensureFullUrl = (url: string) => {
    return url.startsWith('http') ? url : `${config.server.baseUrl}${url}`;
  };

  const getImgViewUrl = (image: ImageMetadata) => {
    return ensureFullUrl(`${image.view_url || image.download_url}`);
  };

  const getImgDownloadUrl = (image: ImageMetadata) => {
    return ensureFullUrl(image.download_url ?? '');
  };

  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 2, 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column'
      }}
    >
      <Typography variant="h6" gutterBottom>
        Image Gallery
      </Typography>
      <Divider sx={{ mb: 2 }} />
      
      <Box sx={{ flexGrow: 1, overflow: 'auto', maxHeight: 'calc(100vh - 200px)' }}>
        {images.length === 0 ? (
          <Typography color="text.secondary" align="center" sx={{ py: 4 }}>
            No images have been generated yet.
          </Typography>
        ) : (
          <Grid container spacing={2}>
            {images.map((image) => {
              return (
                <Grid key={image.id} sx={{ width: { xs: '100%', sm: '50%', lg: '33.33%' }, padding: 1 }}>
                  <Card
                    sx={{
                      cursor: 'pointer',
                      border: selectedImage === image.id
                        ? `2px solid ${theme.palette.primary.main}` 
                        : 'none'
                    }}
                    onClick={() => onSelectImage(image.id ?? -1)}
                  >
                    <CardMedia
                      component="img"
                      height="140"
                      image={getImgViewUrl(image)}
                      alt={`Generated image: ${image.filename}`}
                      sx={{ objectFit: 'cover' }}
                    />
                    <Box sx={{ 
                      display: 'flex', 
                      justifyContent: 'space-between',
                      p: 1
                    }}>
                      <Typography variant="body2" noWrap>
                        {new Date(image.created_at).toLocaleString()}
                      </Typography>
                      <Box>
                        <IconButton 
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDownloadImage(getImgDownloadUrl(image), image.created_at.toISOString());
                          }}
                        >
                          <DownloadIcon fontSize="small" />
                        </IconButton>
                        <IconButton 
                          size="small" 
                          color="error"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteImage(image.id ?? -1);
                            if (selectedImage === image.id) {
                              onSelectImage(null);
                            }
                          }}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Box>
                    </Box>
                  </Card>
                </Grid>
              )
            })}
          </Grid>
        )}
      </Box>
    </Paper>
  );
};

export default ImageGallery;
