import React, { useEffect, useState } from 'react';
import { 
  Box, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Typography, 
  SelectChangeEvent,
  Alert,
  Snackbar
} from '@mui/material';
import { Model } from '../../types/Model';
import { useAuth } from '../../auth';
import { getModels } from '../../api/model';
import { getHeaders, req } from '../../api/base';
import { getToken } from '../../api';
import ControlLoader from '../Shared/ControlLoader';

interface ImageModelSelectorProps {
  onModelChange?: (modelId: string) => void;
  mode: 'TextToImage' | 'ImageToImage';
}

const ImageModelSelector: React.FC<ImageModelSelectorProps> = ({ onModelChange, mode }) => {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [snackbar, setSnackbar] = useState<{open: boolean, message: string, severity: 'success' | 'error'}>({
    open: false,
    message: '',
    severity: 'success'
  });
  
  const auth = useAuth();

  // Load models from API
  useEffect(() => {
    const fetchModels = async () => {
      try {
        setIsLoading(true);
        const allModels = await getModels(getToken(auth.user));
        
        // Filter models to only include those with TextToImage specialization
        const imageModels = allModels.filter(model => 
          model.details?.specialization === mode
        );
        
        setModels(imageModels);
        
        // If we have models, try to find the active one if any
        // This could be enhanced if the API provides this information in the future
      } catch (error) {
        console.error('Failed to fetch image models:', error);
        setSnackbar({
          open: true,
          message: 'Failed to load image models',
          severity: 'error'
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchModels();
  }, [auth.user, mode]);

  const handleModelChange = async (event: SelectChangeEvent) => {
    const modelId = event.target.value;
    setSelectedModel(modelId);
    
    try {
      // Make API request to set the active image model
      await req({
        method: 'PUT',
        path: `api/models/image/${modelId}`,
        headers: getHeaders(getToken(auth.user))
      });
      
      setSnackbar({
        open: true,
        message: 'Active image model updated successfully',
        severity: 'success'
      });
      
      if (onModelChange) {
        onModelChange(modelId);
      }
    } catch (error) {
      console.error('Failed to update active image model:', error);
      setSnackbar({
        open: true,
        message: 'Failed to update active image model',
        severity: 'error'
      });
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  if (isLoading) {
    return <ControlLoader text='Loading image models...' />;
  }

  return (
    <Box sx={{ mb: 2, p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Select Image Generation Model
      </Typography>
      
      {models.length === 0 ? (
        <Alert severity="info">No text-to-image models available</Alert>
      ) : (
        <FormControl fullWidth>
          <InputLabel id="image-model-select-label">Image Model</InputLabel>
          <Select
            labelId="image-model-select-label"
            id="image-model-select"
            value={selectedModel}
            onChange={handleModelChange}
            label="Image Model"
          >
            {models.map((model) => (
              <MenuItem key={model.model} value={model.id ?? model.model} title={model.details.description || ''}>
                {model.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      )}
      
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity} 
          sx={{ width: '100%' }}
          variant="filled"
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ImageModelSelector;
